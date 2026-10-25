# FP32、BF16混合精度与压缩训练对照

已补齐[双卡混合精度20步](bf16-dual-training.md)：修复路径采用本进程`TASK_QUEUE_ENABLE=0`，双卡分配峰值29.90/27.76 GiB，同样第2–20步平均53.88秒，单卡45.76秒。当前模型分布实现节省单卡容量，没有提速；队列设置差异保留。2/3卡已释放。

最新结果：[BF16计算 + CPU FP32主权重/m/v的20步单卡验证](bf16-master-training.md)已完成，NPU峰值57.31 GiB，2/3卡已释放。该新实现保留派生BF16计算缓存，替代下面历史失败组的按需转换路径；不能将旧失败配置与新成功配置混用。纯FP32后续[CPU主权重/流水优化](cpu-master-offload-optimization.md)达到44.51秒/步。以下保留最初三组实验的协议和结果。

当前完整FP32基准优化已完成4步：后3步平均161.76→76.35秒，约2.12倍。原FP32状态和更新公式不变；64 MiB窗口与CPU亲和配置详见[offload优化结果](fp32-offload-optimization.md)。2/3卡已释放。

最新补采：[CPU↔NPU与编解码时间报告](codec-timing-report.md)。一次完整压缩更新325.29秒，其中权重codec全流程7.42秒、量化Adam路径297.74秒；旧FP32完整状态基准后3步CPU↔NPU传输113.31秒/步。完整FP32和压缩组均已完成4步，BF16后续基准仍未通过训练前数值校验，不将其写成有效对照。

2026-10-24新授权。使用物理2/3号910B2，三组均训练原模型全部26,895,998,464个文本参数；不训练视觉/MTP，不使用LoRA，不改原checkpoint。

| 组 | 权重驻留 | 计算 | Adam历史状态 |
|---|---|---|---|
| FP32 | NPU未压缩FP32主权重 | FP32，关闭HF32 | CPU完整FP32 m/v |
| BF16混合精度 | NPU未压缩FP32主权重 | 按需BF16线性计算与autocast，敏感计算FP32 | CPU完整FP32 m/v |
| 压缩B128 | NPU ENEC逐位校验权重，失败原字节回退 | FP32，关闭HF32 | NPU NF4 m/log-v各4 bit；微小参数FP32 |

完整状态对照获准使用主机内存卸载；压缩组继续不卸载。BF16组不常驻完整第二份BF16权重。按需转换矩阵参与计算，反向矩阵乘也使用BF16，结果转回FP32供更新；转换不能恢复已经发生的计算舍入。

## 共同协议

- 同一已验证ENEC归档解码初始权重；原BF16输入先精确扩展FP32，没有改动原文件。
- 同一既有独立Alpaca-GPT4中文32训练/8验证划分。按完整token长度与消息哈希排序，4次更新使用最短4条完整训练样本；全部8条验证独立，GSM8K不进入训练。
- batch=1、梯度checkpointing、梯度即算即更新释放、无梯度累积/全局裁剪。lr=1e-5、beta=(.9,.999)、eps=1e-8、wd=.01、种子20261024。
- 验证统一用FP32计算，以观察训练后权重差异；BF16组另提供BF16执行验证。
- 记录逐步loss、全部参数更新覆盖、每卡分配/预留、主机RSS和完整状态对照的复制/CPU Adam分段耗时。保存模型参数的确定性抽样到服务器私有目录，仅输出聚合误差，不向Git上传权重数值。

## 比较口径

使用同一自定义autograd训练harness，完整对照改为未压缩FP32权重与CPU FP32 Adam，BF16对照改变计算精度。不是调优过的DeepSpeed/ZeRO吞吐基线。CPU卸载没有重叠传输；速度必须结合复制和CPU更新时间解释，不能当作这些框架的性能上限。

FP32 CPU Adam与NPU Adam可能有设备算术差异；相同dtype不能保证逐位一致。输出层按行分块累加也改变求和分组。四步/八验证样本能给出初期偏差和容量对照，不能认证长期同质量。

## 前置检查

`smoke_compare.py`实际核对两卡、三模式的前向/输入梯度、首次更新和两步全坐标更新；完整对照核对m/v位于CPU且dtype为FP32。

`bench_resident.py`复用此前在线原型的`resident.py`。复现时将该目录加入PYTHONPATH，链接此前已验证ENEC作者122变体生成的`libonline_codec.so`；不改作者kernel。

执行结果与失败待实际运行后更新；未完成的组不记为有效对比。

## 初始化问题与修复

首个FP32对照在CPU m/v分配阶段显著变慢，尚无完整更新。600秒自动栈停在`CpuMoments.__init__`的`torch.zeros`；进程无swap、cgroup无内存限额事件，但memory PSI full约47%–55%，匿名大页约75.6 GiB，系统compact_stall持续增长。512 MiB匿名映射+MADV_HUGEPAGE+memset独立探针：启用THP两次6.591/0.109秒，当前进程禁用THP后约0.215–0.233秒；该微测试有波动，不当作一般吞吐结论。

终止自己的尚未完成初始化进程，保留`train_compare_initial.py`与中止记录；重跑只在导入torch前调用`prctl(PR_SET_THP_DISABLE,1)`，状态记录PR_GET_THP_DISABLE。三组使用同一进程局部策略，不修改共享内核配置；不改变数值精度或算法。

第二次启动在训练前遇脚本命名错误：远端队列脚本`queue.py`覆盖Python标准库queue，导入时执行调度等待；0次更新。改名`run_sequence.py`后第三次启动正常，PR_GET_THP_DISABLE为true。此问题与数值算法无关，作为设置失败保留。

CPU卸载窗口检查：16,777,216坐标、两次FP32 Adam更新，1M窗口约0.342秒、16M窗口约0.406秒；两者最终FP32位串相同。微测试未发现放大窗口的收益，完整对照保持原1M窗口；不据此外推整模型速度。

## 由实测调整的主要参照

CPU Adam辅助组首步688.43秒，其中CPU Adam440.82秒、NPU→CPU复制75.92秒、CPU→NPU复制37.02秒。为避免将未融合CPU实现开销当作完整FP32训练的必然代价，增加完整FP32 m/v原字节CPU驻留、分窗口回NPU执行FP32 Adam的对照；当前CPU组继续完成，不丢弃结果。

16,777,216坐标两次更新的微测试：CPU更新0.587秒，NPU窗口更新0.311秒；后者权重、m、v与完整驻留NPU的FP32 Adam逐位相同。两卡/三模式小型前向、输入梯度与两次完整更新检查也通过。该微测试与CPU组同时运行，不作为整模型性能数据。

后续主要对照采用`train_compare_npu.py`：FP32原始权重+完整状态卸载/NPU更新；BF16混合精度+完整状态卸载/NPU更新；原压缩B128/NPU更新。新增`bench_resident_npu.py`保持旧CPU组源码不变。各组均4步/8验证，同批样本；完整m/v不量化，只有压缩组仍用4-bit状态。微小参数对照也将完整m/v往返NPU更新。

BF16输出层分块乘积先按BF16舍入，再FP32累计；这不是与标准整矩阵AMP逐位一致的声明。三组共同使用HF eager attention与原分块输出层harness，均不代表已调优生产框架的吞吐上限。

`analyze_probes.py --replay`在CPU按各自真实抽样梯度轨迹重放完整FP32 Adam，用于区分单次更新差异与模型梯度轨迹差异；不冒充另一次完整模型训练，也不向Git输出抽样权重。

首个整模型NPU更新步492.22秒，其中显式NPU Adam段12.95秒、写回CPU状态359.82秒。检查128 MiB写回小测试：CPU切片直接接收0.0294秒，先生成CPU临时张量再CPU复制0.0383秒，位串相同但未提速，故未改路径。实际第2/3步降至190.72/154.56秒，表明首步复制开销不能代表后续；没有凭微测试推断整模型带宽或更改实现。

## 复现主要对照

使用相同CANN/torch-npu训练环境，设置`ASCEND_RT_VISIBLE_DEVICES=2,3`，将之前在线实验的`resident.py`所在目录加入PYTHONPATH，将已验证的`libonline_codec.so`复制到当前脚本目录。仅在已获准且实时空闲的卡上顺序执行：

```bash
python smoke_npu_offload.py --output "$RUN/smoke.json"
python train_compare_npu.py --model "$MODEL_DIR" --archives "$ENEC_ARCHIVE_ROOT" \
  --data "$INDEPENDENT_SPLIT_DIR" --output "$RUN/fp32-npuadam-run1" \
  --mode fp32 --steps 4 --eval-count 8
# 进程退出、确认卡释放后，分别以bf16 / compressed启动独立新输出目录。
```

CPU更新辅助控制使用`train_compare.py --mode fp32`，其余参数相同。没有自动保存几百GiB完整模型/优化器检查点；保留了状态、每步事件和服务器私有的抽样参数轨迹供聚合核对。原始checkpoint只读。

主要精度比较：

```bash
python analyze_probes.py --reference "$RUN/fp32-npuadam-run1/parameter-probes.json" \
  --candidate "$RUN/compressed-run1/parameter-probes.json" \
  --output "$RUN/compressed-analysis.json" --replay
python summarize_comparison.py --root "$RUN" --output "$RUN/comparison-results.json"
```

抽样轨迹每步覆盖52,928个坐标，按张量大小加权仅用于估计，不等于读取并比较全部268.96亿参数。CPU辅助组的4步真实梯度重放中，抽样更新与完整FP32 Adam逐位数值一致（最大绝对差0）。验证逐token正确率不是整题准确率。源码证据同时保留执行原始字节SHA及LF规范化SHA，便于Git换行规范化后核对。

## 初次全局AMP配置失真：不作为BF16基准

`bf16-npuadam-run1`虽完成4次更新，但初始未更新时同一条样本loss已为9.26035（FP32为1.54175），最终BF16执行验证loss9.53685、逐token正确率3.70%；转回FP32执行验证loss2.00859。不能把它描述为正常BF16训练的精度损失。

审计Transformers5.14.1的`torch_chunk_gated_delta_rule`：query/key/value/beta/g明确转FP32后，核心矩阵乘仍受外层autocast影响；`value.dtype`还决定递推状态dtype。CPU核心小测试实际记录到：朴素AMP内部矩阵乘输出BF16，保护后输出FP32，保护结果与无AMP的原FP32核心逐位一致。说明最初只测线性层是不够的。

新增`bf16_precision_guards.py`，仅在本次进程保护chunk/recurrent gated-delta函数，保持投影/FFN BF16，主权重/Adam FP32。不改共享Transformers安装。压缩组完成后在空闲NPU做同样核心检查，再以`train_compare_bf16_guarded.py`重跑`bf16-guarded-run1`。增加训练前FP32/BF16完整8样本验证，loss差过大则停止并保存逐层隐藏状态差异诊断，不再先跑4步才发现异常。主对照汇总只使用修正后的组；异常完整记录保留在setup证据。
