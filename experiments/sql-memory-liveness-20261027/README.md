# 前四项显存生命周期：真实模型的meta执行

2026-10-27。已完成当前固定训练路径的框架张量级实现和8个单卡峰值对比；不等于完整物理HBM建模已完成。CPU/meta运行，没有NPU初始化、加载模型权重或启动真实训练。原81.22秒训练入口的三份源码LF哈希保持不变，压缩版本不使用。

## 实际实现了什么

1. **中间结果生命周期**：运行实际Transformers Qwen3.5文本前后向和原生autograd。`storage_tracker.py`通过TorchDispatchMode观察ATen输出，用StorageImpl native弱引用判断存活，不保留tensor/storage强引用。SavedVariable及Python引用实际决定保存/释放；不是填入快照中的张量份数。
2. **流水线和重计算叠加**：按原前后32层、48层非重入checkpoint、micro16和16个微批顺序执行。rank0存在上一批和当前批前向图叠加，rank1逐微批反向；沿实际checkpoint early stop追踪重算分配。不假设激活都能同时释放，也不加入测得的重算常数。
3. **临时梯度和副本**：实际执行自动求导、参数梯度回调、驻留累加及清空param.grad。根据固定版本TorchNPU源码补充bias-free线性层2D/3D的联合matmul反向输出布局；由原生AccumulateGrad自动产生布局不匹配时的复制，没有手动添加170MiB。CPU接收槽/FP32主权重/m/v不计为HBM。
4. **共享存储和布局**：按storage而非逻辑tensor去重。view/expand/detach共享不重复计费，非连续reshape/contiguous真正分配时才计费；记录shape/stride及分配来源。没有覆盖任意动态resize或未知后端实现的承诺。

GDN沿原`community_gdn.py`调用固定VeOmni的Python前后向分配包装；`launch_stub.py`保留empty/zeros/contiguous及save_for_backward，只跳过Triton设备kernel启动。设备上下文/RNG仅作meta适配，保留checkpoint保存张量机制；目标核数24/48是输入假设。跳过kernel不表示内部workspace为零。

## 预测与比较隔离

预测只读配置、实现源码和各微批padding长度；文件审计禁止读取evidence、calibration.json、权重、原快照和DB。封存inputs及各预测SHA256后，`compare_sealed.py`才在独立进程打开历史实测。

第一版6个结果发现抵消后，原三组已经用于诊断，修正后的复测不能称作未见过的独立验证。额外2GiB驻留策略的两个预测也先封存，之后才读取该策略的实测峰值；这是本轮新增策略检查，仍不是新上卡实验或任意模型泛化验证。

## 峰值对比

预测为逻辑storage字节；实测为历史NPU框架allocated峰值，包含分配对齐。单位GiB；有符号误差=预测减实测。step1/2/3使用每卡4GiB梯度驻留，cache2-step1使用2GiB，输入长度表随证据导出。

| 输入案例 | 物理卡 | 预测GiB | 历史实测GiB | 误差MiB |
|---|---:|---:|---:|---:|
| step1 | 2 | 56.559499 | 56.559970 | -0.482300 |
| step1 | 3 | 54.053881 | 54.054222 | -0.348869 |
| step2 | 2 | 49.369724 | 49.370453 | -0.745972 |
| step2 | 3 | 47.591772 | 47.592371 | -0.613029 |
| step3 | 2 | 52.978245 | 52.978974 | -0.745972 |
| step3 | 3 | 54.053881 | 54.054480 | -0.613029 |
| cache2-step1 | 2 | 54.559499 | 54.559973 | -0.485718 |
| cache2-step1 | 3 | 52.053881 | 52.054225 | -0.352287 |

8项最大绝对误差约0.746MiB、0.001476%，仅适用于以上固定版本/路径。该数字不是完整物理HBM误差，也不是未知workspace已经解决的证据。

## 分项核对与第一版错误

第一版总误差最大0.136%，但step1卡2重算项多136MiB、临时梯度少170MiB，存在抵消。原始结果保留在`sql-memory-liveness-comparison.json`，不能用其小总误差声称分项准确。

安装包git_version为TorchNPU `5dd8ef3f9b375b5ae4a83538d5785754148c3302`，其gitlink对应op-plugin `8b9c8534fa41eff367a41c155843daa530ab3a08`。源码的matmul联合反向先产生other梯度，转置权重对应的other梯度仍按其形状连续分配；普通meta拆分mm路径与此不同。`npu_linear_meta.py`据此实现元数据规则，实际生成梯度布局并让原生累加器决定副本。版本、源码链接和SHA见`backend-rule-sources.json`，没有按实测加字节补偿。

修正后step1卡2各自峰值的分组如下。这是各自峰值组成对比，不是逐地址/同时间戳事件对齐。

| 分组 | 预测GiB | 实测GiB | 误差MiB |
|---|---:|---:|---:|
| weight | 25.048845768 | 25.049038410 | -0.197266 |
| resident_gradient | 4.000000000 | 4.000013828 | -0.014160 |
| ffn_forward | 7.968750000 | 7.968750000 | 0.000000 |
| rmsnorm_forward | 9.148092270 | 9.148092270 | 0.000000 |
| attention_forward | 6.328125000 | 6.328125000 | 0.000000 |
| checkpoint_recompute | 1.567512870 | 1.567513466 | -0.000610 |
| loss_forward | 0.000000000 | 0.000000000 | 0.000000 |
| backward_temporary | 0.503906250 | 0.503907681 | -0.001465 |
| other_boundary_inputs | 1.994266748 | 1.994267941 | -0.001221 |

此前136/170MiB抵消已消除。卡3的FFN、norm、attention、loss及loss梯度大项也相符；完整数据见native-comparison。剩余总差包含分配对齐、快照重建未分类对账和跨训练步控制标量等口径差异，不把它们统一命名为workspace：本模拟逐案例从新状态开始，原多步脚本会跨步保留norm标量；没有用其测量值补预测。

## 验证和适用边界

存储别名/复制、SavedVariable保存释放、原地累加、真实非重入checkpoint重算，以及NPU线性反向布局的2D/3D小测均通过。最后一项验证元数据和复制行为，不验证数值内核。测试证据为`sql-memory-liveness-tests.json`，环境/VeOmni源文件哈希随预测摘要导出。

未覆盖：

- CANN或编译器内部kernel workspace：Triton host包装显式分配可见，kernel内部申请尚未知。
- 驱动/运行时基础占用、通信库额外HBM、allocator reserved/碎片/缓存池策略；没有把其默认成0。
- 任意异步流完成时间和record_stream延迟释放：当前采用原路径同流复用语义，没有建立通用时序模型。
- 其他版本、bias线性层、不同注意力分支/融合内核、变长GDN、不同调度未经验证。padding eager mask分支是显式输入假设。

本轮能给出当前路径的张量级存储峰值及组成，不足以直接断言换一套训练框架后总HBM是否放得下。下一步应先建立逐算子的workspace查询/源码规则与allocator预算，再扩展配置，不能用驱动占用常数回填后宣称全白盒。

## 分配粒度修正（2026-10-27 追加）

`storage_tracker.py` 原先按**申请字节**记账，而实测峰值是**分配字节**。逐块比对实测轨迹的 `requested_bytes` 与 `bytes` 后确认规则分两类：初始化块（权重、buffer、驻留）在申请值之上多一个 512 边界（即使已对齐）；运行时中间块只向上取整到 512（对齐的不变）。峰值、分项和逐块记录都改为分配字节，`peak_requested_bytes` 另存申请值备查。

重算 8 项（`native-alloc.private`、`evidence/2026-10-27/sql-memory-liveness-alloc-comparison.json`）后：最大绝对误差 0.746→0.533 MiB、0.001476%→0.001055%。

残差可拆为三块：实测侧 `allocator_block_reconciliation`（280,576 / 141,312 B）、模型自身 2,048 / 512 B、step2 起的 276,480 / 276,992 B。

早前"全局严格 512 边界、残差 2,560 B"的结论是聚合巧合（全局多算约 503 KB ≈ 未归类差额 280 KB + 真实粒度 223 KB），**已作废**。

step2 起的 270 KiB 尚未定位：原 `train_memory.py` 的跨步状态（Adam/master/梯度累加）都在 CPU，GPU 上没有对应常驻张量，`off.reset()` 每步清累加器，因此不能归因于跨步残留。不影响第一步结论，但多步预测暂不成立。

## 代码和证据

`run_native_predictions.py`生成修正版8项预测；`compare_sealed.py`先验封存哈希再读观测；`export_predictions.py`剔除逐storage原始轨迹。CPU原始日志/全量轨迹留在私有目录，公开证据不含地址、训练文本、权重或私有路径。

[修正版预测摘要](../../evidence/2026-10-27/sql-memory-liveness-native-predictions.json) · [修正版对比](../../evidence/2026-10-27/sql-memory-liveness-native-comparison.json) · [测试](../../evidence/2026-10-27/sql-memory-liveness-tests.json) · [第一版抵消记录](../../evidence/2026-10-27/sql-memory-liveness-comparison.json)
