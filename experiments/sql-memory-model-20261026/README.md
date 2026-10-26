# 显存建模 v0：不上卡计算原训练方案的分项占用

用户提出“显存建模”：在上卡之前，通过模型预测每部分大概占多少。本轮已实现纯Python/标准库计算器，不导入torch、不加载权重、不启动NPU训练。对象是原81.22秒、双卡前后各32层、BF16计算/梯度、CPU FP32 master/m/v、4GiB梯度驻留、48层重算的文本训练方案；压缩版不使用。

## 总模型：形状 + 生命周期 + 实现

对每份实际存储：

`bytes_i = product(shape_i) * bytes_per_element(dtype_i)`

对rank r的某执行时刻：

`Allocated_r(t) = persistent_r + sum(unique live dynamic storages at t) + allocator-visible workspace_r(t)`

`PeakAllocated_r = max_t Allocated_r(t)`

`PhysicalHBM_r(t) = Allocated_r(t) + cached/reserved-unused_r(t) + allocator-external_runtime_r(t)`

不能把每类张量各自的最大值相加；view/transpose等共享同一storage时只算一次。使用batch后真实padding长度，不用原始平均长度代替。两个rank取各自峰值，且可能不是同一时间。

模型不仅输入“27B”，还必须知道微批B、填充长度S、隐藏维H、中间维I、词表V、Q/K头数与head_dim、dtype、切层方式、重算策略、保存反向张量的规则、注意力/损失是否融合、梯度驻留/卸载策略及调度。相同架构换实现，中间张量和workspace都会变。

## 已编码的数学项

变量来自仓库中的公开模型config：H=5120、I=17408、V=248320、Q=24、K=4、D=256。

### 权重与状态

架构逐层计数参数P，文本两段分别13,447,996,672 / 13,448,001,792个参数。BF16权重`2P`；每段FP32主权重`4P`、Adam m/v合计`8P`在CPU，不加入HBM。参数公式另用历史851张量CSV独立核对，完整一致，未用总字节常数替代架构计算。

梯度驻留按有效预算输入，当前每卡4GiB。这是跨微批的累加结果，不代表当前反向梯度全都留在NPU。

### FFN前向保留项

四份BF16的B×S×I结果：gate、up、SiLU与门控乘积。每组：

`F(B,S)=4*2*B*S*I = 8BSI`。

当前峰值生命周期模板：2卡15组、3卡8组，所以B16/S256分别7.96875/4.25GiB。组数来自当前峰值生命周期校准，不是所有流水线都适用的常数；当前2卡同时持有两个微批的图，反向已释放部分结果。

### RMSNorm

当前非融合实现中，隐藏维归一化仍存活的FP32中间结果、BF16输出和小项：

`Norm(H) = 10BSH + 4BS + 4H`。

Q/K归一化的BF16结果已被rotary消费释放，当前仍存活项是：

`HeadNorm(N,D) = 8BSND + 4BSN + 4D`。

每个未重算普通注意力层的保留项：

`R = 2Norm(H) + HeadNorm(Q,D) + HeadNorm(K,D)`。

乘15/8组，3卡再加最后一层Norm(H)，得到9.148092/5.074329GiB，与快照对应分类一致。系数来自当前实现和存活规则，不等于所有RMSNorm算法必需这些存储。采用融合内核时要更换规则，而不是继续用旧系数。

### eager普通注意力

扣除已单列的Q/K norm后，每组：

`A(B,S) = 12BSQD + 6BQS²`。

前一项覆盖位置编码、展开K/V、上下文及门控相关存储；后一项是FP32与BF16注意力概率。当前峰值乘15/8组，得到6.328125/3.375GiB。序列翻倍，后一项变四倍，因此不能把整个模型动态显存简单按长度线性缩放。

### 非融合词表损失

一份FP32 B×S×V为`4BSV`。当前loss峰值有一份前向log-softmax结果，以及NLL/log-softmax两份临时反向梯度，共：

`LossPeak = 3*4BSV = 12BSV`。

B16/S256时11.3671875GiB。词表输出梯度与参数梯度分开计；融合或分块交叉熵会改变这条存活规则。

### 其他项和校准项

2卡当前普通反向临时梯度模板：`4HI + 2BSI + 2BSH`，约0.503906GiB。两份HxI的BF16权重梯度相关缓冲、B×S×I及B×S×H临时梯度；存活时刻由模板限定。

隐藏状态/边界/流水线张量按51/24份BF16 B×S×H计。51/24和15/8都是当前调度校准值，不是已实现了任意调度的自动静态推导。

checkpoint/GDN重算合计和小控制项目前按参考快照的BS比例估计。这部分明确标记`empirical_BL_scaling`，尚未把每一个GDN后端kernel的workspace写成独立公式。额外workspace、runtime及allocator slack都可显式输入；没有runtime/slack时不输出物理HBM总量。不能把未知workspace默认解释为“所有配置都为零”。

## 校准与未参与校准的长度检查

| 配置 | 2卡模型 | 2卡历史实测 | 3卡模型 | 3卡历史实测 |
| --- | ---: | ---: | ---: | ---: |
| B16，峰值涉及S256，校准参考 | 56.55950 | 56.55997 | 54.05388 | 54.05422 |
| B16，16个微批全部S192，独立长度检查 | 49.36949 | 49.37045 | 47.59177 | 47.59237 |

单位GiB。第二行取已存在的原版第二轮记录，并由原`groups(step)`与dynamic64/min128规则核对16个padding长度全为192，没有再上卡运行。它没有参与256参考模板的校准；误差约0.989/0.611MiB。

这验证了同一实现、同一批量和调度下一个额外长度点。它不验证任意batch、任意长度、任意重算/融合策略。256参考接近实测不是独立泛化测试，不能把所有小误差宣传为通用预测精度。

CPU上的其他推演（尚未上卡验证）：B16/S128约42.44/41.27GiB；B16/S512约87.96/81.31GiB。这里仍使用当前峰值模板；形状变化可能切换峰值时刻、tiling和workspace，不能宣称新配置已跑通。B8推演对应16micro/global128，不冒充固定global256对照。

## 使用

在仓库根目录运行，不需训练环境：

```powershell
python experiments/sql-memory-model-20261026/memory_model.py --micro-batch 16 --sequence 192 --resident-gib 4
python experiments/sql-memory-model-20261026/validate_reference.py
```

输出包括每类字节数/GiB、公式、来源类型和总峰值模板。`--runtime-gib`、`--allocator-slack-gib`都提供时，才额外输出带这些输入的物理HBM估算；`--extra-workspace-gib`补充后端workspace预算。预留缓存和运行时应按同时刻或明示估计处理，不能用不同时间的两个峰值之差伪装精确余量。

[公式与历史检查JSON](../../evidence/2026-10-26/sql-memory-model-validation.json)、[默认分项输出](../../evidence/2026-10-26/sql-memory-model-reference.json)。

## 当前完成程度

已经能在CPU上对当前方案做分项计算和长度/微批的受限推演，避免每个参数组合都先上卡碰OOM。当前是**形状公式 + 已校准生命周期模板**，不是仅输入模型config就能精确覆盖所有框架的通用模拟器。

若继续扩展，下一步是把保存/释放/view别名、checkpoint重算、1F1B排程写成自动生命周期图，逐事件求max；再为GDN和后端workspace建立规则库。这样才能减少对15/8、51/24等已观测模板的依赖。该扩展本轮未冒充已完成，也未启动新训练。

官方方法参考：[PyTorch meta device](https://docs.pytorch.org/docs/main/meta.html)支持只保留形状与类型的抽象分析；[saved_tensors_hooks](https://docs.pytorch.org/docs/main/autograd.html)说明前向保存反向所需张量的行为；[分配与预留口径](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-memory-management)区分allocated和reserved。当前计算器直接使用数学规则，没有宣称已经集成meta图追踪，NPU开销仍以实际后端规则为准。
