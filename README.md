# Qwen27B Ascend 训练日志

最新通信调度试验已完成并未采用：大梯度就绪即发实测96.98秒/256条，慢于保留配置95.74秒；采样显示归约与计算重叠减少。见[结果及拒绝依据](experiments/sql-reduce-bucket-20261025/README.md)。

**最新固定256条配置为95.74秒/256条**：相比上一轮101.28秒，耗时再降5.47%；通过将最终微批的原生范数/梯度卸载延至反向下发结束，减少逐块等待。BF16计算与CPU FP32主权重、Adam状态保持，短验证结果一致。见[最终微批调度报告与重放入口](experiments/sql-boundary-overlap-20261025/README.md)。2/3卡已释放。下面保留各阶段历史记录，固定256条性能以本段为准。

**上一轮固定256条调度优化：107.97 → 101.28秒/256条，时间缩短6.19%。** 保持相同样本、顺序、padding、更新次数及BF16计算/CPU FP32 master/m/v。有效组合是有界32槽CPU梯度接收与原生归约事件阈值2→4；同768条短验证一致，每卡allocated峰值42.87GiB。选中[配置与重放入口](experiments/sql-reduce-schedule-20261025/README.md)已保存，其他候选及未采用原因一并记录。2/3卡已释放，未自动Git推送。以下107.97秒及global128段落属于此前记录，不是最新固定256性能配置。

**最初的固定256条瓶颈分析**：global256、每卡micro16、累积8，无采样稳态107.97秒/256条。改变更新频率的结果单列，不能当作纯系统加速。本轮Profiler显示114.15秒窗口里计算覆盖51.80秒，另有26.20秒通信未与计算重叠、14.93秒仅异步复制、9.73秒CPUAdam，其余11.49秒。优先研究反向梯度通信与CPU/NPU往返，详见[固定批量瓶颈报告](experiments/sql-bottleneck-20261025/README.md)。采样任务正常退出，2/3已释放。

记录我们在Ascend 910B2上学习和验证Qwen后训练的过程。这里维护训练环境、数据划分、可复现配置、实验代码和结果；推理部署与性能调优继续由原推理调优仓库维护。

**此前按验证目标选择的SQL短训练默认：每卡16条、全局128条、累积4轮。** 固定同768条数据比较全局64/128/256，以之前短验证水平为目标（loss<=0.86），128在已测检查点中286.37秒达标，256为368.50秒、64为349.10秒；不是长训练全局最优。相同后512条相对旧micro8/global256由382.95秒降至244.62秒，吞吐+56.55%。256仍有更高固定数据吞吐，64在更严格验证目标下可能更快，完整比较与限制见[批量选择报告](experiments/sql-batch-tuning-20261025/README.md)。[默认配置](experiments/sql-batch-tuning-20261025/selected-profile.json)和[启动入口](experiments/sql-batch-tuning-20261025/run_selected.py)已保存，FP32 master/m/v保留，2/3卡已释放。

**纯文本SQL公开任务对照完成**：改用原始Qwen3.5-27B、相同两份SQL数据来源、长度上限512、全局batch256，双910B2完成3次全参更新，后两轮平均191.47秒/256条（1.337条/秒）。作者16×H200公布17.8万条3小时6分，折算约16.05秒/256条，整机吞吐约为我们的11.93倍；我们按短跑外推一遍约36.98小时，未实际跑完整epoch。作者预处理未完全公开，长度分布也有差异，不当纯硬件同工作量实测。FP32 master/m/v保留，验证和源码审计通过，2/3卡已释放。见[完整SQL对照报告](experiments/sql-throughput-20261025/README.md)。

[公开训练配置与批量选择](research/training-config-comparison-20261025.md)：核对8×H100和16×GB200的卡数、微批量、全局批量、精度与重算设置，解释我们batch2约25.50秒、batch32约27.58秒为何应按吞吐区分。

各轮数据按相同批量、样本窗口和验证精度整理于[训练对比总表](experiments/training-comparison-20261025.md)，包含FP32优化、压缩驻留、20步BF16、标准ZeRO-3和失败记录；[汇总JSON](evidence/2026-10-25/training-runs-overview.json)由既有证据生成。

**后续双卡吞吐优化完成**：同batch=2/相同8样本4更新，双卡41.64→30.13秒/更新，吞吐提升1.382倍；最终验证loss1.504419923、263/433，与原组接近。固定batch=32时，从每卡4样本93.16秒优化到每卡16样本34.88秒、68.15有效输入token/s。BF16计算、CPU FP32主权重/m/v，调microbatch、部分重算、NUMA、预取/更新分组及CPU总24→48核；不同batch分开比较。Profiler与源码配置已保存，2/3卡已释放，见[双卡优化报告](experiments/dual-throughput-20261025/README.md)。

**CPU回传流水与社区训练融合已实测**：同batch2由30.13→25.50秒/更新（吞吐+18.2%）；同batch32由34.88→27.58秒、86.20输入token/s（吞吐+26.5%）。FP32 master/m/v不压缩，接入固定版本VeOmni GDN训练前后向与锁页BF16异步回传。融合使用BF16中间计算，非原FP32分解逐位等价；短验证batch32仍271/433、loss1.447144，长期收敛未验证。Profiler计算任务144031→33199、MatMul调用数不变，CPUAdam约10.7秒仍是主要开销。失败、FP64误差复核、源码/SQL与资源释放已记录，见[本轮报告](experiments/training-fusion-20261025/README.md)。

**2026-10-25标准双进程ZeRO-3对照完成**：BF16计算、CPU FP32主权重/m/v、固定全局batch=2，同样8条短训练样本各4次更新。后3步单卡累积2样本62.49秒，双卡每卡1样本41.64秒，训练吞吐约1.50倍；NPU分配峰值59.66 GiB→每卡35.77 GiB。两组最终独立验证loss均1.504420、263/433正确token。处理了单卡整张梯度转FP32的OOM，分块转换位级校验通过；单卡仍有allocator压力，不代表硬件速度上限。2/3卡已释放，见[ZeRO-3实验](experiments/zero3-offload-20261025/README.md)。

**双卡混合精度20步也已完成**：复现并通过本进程`TASK_QUEUE_ENABLE=0`绕过无效stream/event错误，模型分布在2/3卡，CPU保留FP32 master/m/v。同样第2–20步平均双卡53.88秒、单卡45.76秒，双卡耗时增加17.7%；每卡分配峰值29.90/27.76 GiB。统一FP32验证loss双卡1.53579、单卡1.53838。当前单进程模型分片没有提速；不能视为双卡数据并行/张量并行上限。2/3卡已释放，详见[双卡结果与错误定位](experiments/fp32-precision-compare-20261024/bf16-dual-training.md)。

最新混合精度短跑已完成：**单张910B2、BF16计算权重 + CPU FP32主权重/m/v，20次全参数更新**。同样前4条样本的第2–4步平均42.43秒（旧双卡纯FP32 44.51秒），NPU分配峰值57.31 GiB，CPU RSS约302.95 GiB。20步后独立8条验证以FP32/BF16执行的loss分别1.53838/1.53195；这是同一训练后权重的执行对照，不能代替长期训练质量验证。原始模型只读，本次未存新完整checkpoint，2/3卡已释放。[混合精度结果与失败记录](experiments/fp32-precision-compare-20261024/bf16-master-training.md)。

当前最快完整FP32短跑采用CPU常驻主权重/m/v、DeepSpeed CPUAdam NEON和有界传输/更新流水。相同4步/8验证下，后3步 **76.35→44.51秒/步，约1.72倍加速**，相对最初161.76秒约3.63倍；末步53.37→42.98秒，保留预热变化。权重、梯度、计算、m/v均FP32；流水与同步CPU版4步抽样梯度/更新/权重差均0，但CPU与原NPU数学有微小FP32舍入差异。各组验证loss约1.53496、逐token正确率60.74%。NPU峰值每卡约50.9 GiB，CPU峰值约303 GiB，额外主权重约100 GiB RAM；2/3卡已释放。[最新报告与证据](experiments/fp32-precision-compare-20261024/cpu-master-offload-optimization.md)。这是短样本协议实测，不是长期生产吞吐或质量认证。

前一轮保持NPU Adam数学不变，扩大m/v窗口并调整CPU亲和，161.76→76.35秒/步；逐步抽样差0。[上一轮报告](experiments/fp32-precision-compare-20261024/fp32-offload-optimization.md)。

压缩驻留路线也已完成：ENEC无损权重+有损4-bit B128 m/v（约27.414 GiB）在2/3卡跑通4次完整更新，无CPU/NVMe运行时状态卸载、无整模型BF16副本。后3步约321.88秒，短验证loss1.53325，不等于更新无损。另一次[真实更新计时](experiments/fp32-precision-compare-20261024/codec-timing-report.md)显示，权重ENEC全流程7.42秒、量化Adam路径297.74秒；当前不将该路径当最快方案。早期2步checkpoint独立保存，B64两次OOM，详见[在线原型记录](experiments/fp32-online-20261024/README.md)。当时BF16组未完成，后续有效混合精度结果见上方新报告。

前置完成全部文本权重ENEC归档：100.195 GiB降至87.106 GiB（含回退块与映射），2265个保存分块全部回读核验通过。[压缩实验](experiments/fp32-enec-20261024/README.md)。

## 2026-10-24 研究结论

已整理并归档 [Adam 状态显存研究 Stage1–13](research/2026-10-24-adam-state-memory/README.md)。严格逐位一致、长期训练、m/v 替代预算 ≤30 GiB 目前未同时达成；允许近似的 4-bit B128 约 27.406 GiB 是该归档当时的条件设计；后续在线短跑另见上方，长期目标质量仍未证明。有限历史因子重放只在固定窗口、相同显式 FP32 算术下精确；真实成熟小 MLP 状态无损帧仍约 83%，不是 15%。

该研究归档提交只包含已完成的 CPU 研究、失败/更正、轻量源码和 [下一次 benchmark 协议](research/2026-10-24-adam-state-memory/NEXT_BENCHMARK.txt)，归档提交当时没有重跑实验或恢复NPU训练；其后用户另行授权的硬件实验按日志单独记录。原始大归档在研究对话中另行交付，Git 中保留文件名和 SHA256。详见 [当天记录](journal/2026-10-24.md)。

2026-10-24归档了后续[混合精度、权重压缩与纠正位研究](research/adam-fp32-lossless/README.md)。仅做CPU位级检查及现成公开11M模型FP32 AdamW状态的离线编码，所有被测恢复方法均逐位还原；所测完整表示按本模型参数量外推仍约163–174 GiB，未达到m/v合计30 GiB目标。这不是Qwen实测，也不是所有无损算法不可能的证明；没有启动训练或NPU任务。

## 历史LoRA基线结果

**Qwen3.8-27B已在单张910B2上完成BF16 LoRA训练基线。** 原权重冻结，不量化、不做CPU卸载；这次不是全参数训练。

| 项目 | 结果 |
|---|---|
| 框架 | ms-swift 4.6.0、Transformers 5.14.1、torch-npu 2.10.0.post4 |
| 训练卡 | 单张910B2，物理2号卡 |
| 数据与更新 | 独立32条训练/8条验证；8次更新，batch=1、累积4，最大128 token |
| LoRA | rank=4；q_proj、v_proj、in_proj_qkv；可训练约446万参数 |
| 验证loss | 2.1690 → 1.9409 |
| PyTorch显存峰值 | 分配51.56 GiB，预留51.68 GiB，不含驱动开销 |
| 参数保存 | 160个LoRA张量全部更新，适配器约17.85 MB |
| 结束状态 | 2026-10-23 18:32:12确认自己的训练进程退出，2号卡无活动进程 |

这次验证了短序列LoRA训练可行。验证集只有8条，尚未据此评估通用能力，该LoRA基线当时未测试全参数训练或更长序列。原有GSM8K推理评测题没有进入训练。

## 阅读入口

- [2026-10-24研究归档与验证记录](journal/2026-10-24.md)
- [后续讨论汇总：混合精度、梯度窗口与30 GiB预算](research/adam-fp32-lossless/discussion-summary.md)
- [纠正位容量研究及可复核脚本](research/adam-fp32-lossless/correction-bit-report.md)

- [首个基线的完整配置、结果与复现方法](experiments/qwen27b-swift-sft/README.md)
- [2026-10-23训练日志](journal/2026-10-23.md)
- [FP32 Adam显存讨论与论文线索](research/2026-10-23-fp32-adam-memory.md)
- [过滤后的逐步指标与证据](evidence/2026-10-23/qwen27b-swift-sft.json)
- [训练入口与参数更新审计](experiments/qwen27b-swift-sft/train_baseline.py)
- [独立数据划分](experiments/qwen27b-swift-sft/prepare_data.py)
- [基线启动参数](experiments/qwen27b-swift-sft/run_baseline.sh)

原权重、数据全文、适配器、完整日志和服务器私有路径留在原运行环境，不上传本仓库。仓库中的脚本通过环境变量指定模型和实验目录。
