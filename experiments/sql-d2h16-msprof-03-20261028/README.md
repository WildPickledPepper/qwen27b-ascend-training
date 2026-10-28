# 固定16槽msprof诊断：0/3卡采样完成

2026-10-28。结论：本轮观察到的跨microbatch长断粮，主要在等下一份可卸载梯度的源计算完成，不是DMA本身把数据传了几百毫秒。0号15次长等待合计约9.370秒，实际DMA约0.0905秒；3号15次合计约12.498秒，实际DMA总计约38.66微秒。限定为本轮>100ms的长等待，不代表所有CPU时间或整网瓶颈。

## 固定配置与正确性

两张卡全程16×128MiB锁页接收槽，各2GiB，硬断言拒绝其他数值。rank0物理0/NUMA6/CPU144–167，rank1物理3/NUMA4/CPU96–119。Qwen3.5-27B文本全参、PyTorch/Transformers自定义两段1F1B、global256/micro16/16micro、SQL原768条顺序与padding、48层非重入重算、BF16计算与累加、CPU FP32 master/m/v、4/8GiB驻留、512MiB非驻留待搬预算、512token连续化loss不变。不改源模型、共享库或原默认入口。

两卡开始前均确认空闲。先tiny三步，全部grad/master/m/v逐位通过，末步设备事件与Level2 DB/Text导出验证；再完整三步，末步采样。完整三步范数与既有对照逐步完全相同，前后短验证一致，最终loss0.8530882597、正确204/224token，FP32 m/v容量215,167,987,712字节。未逐张量比较完整27B所有状态，不作长期质量结论。

首步683.005秒，线程栈确认大量时间在CPUAdam首次创建FP32 m/v，伴随内存压力；第二步98.022秒，末步采样91.428秒。换了物理卡/NUMA、系统负载及采样配置，不能把这些时间与旧3/4结果作干净速度比值，更不能把采样比第二步短当优化收益。第三步后的解析另约140–146秒，不计作训练时间。

## ms工具及校验

Ascend PyTorch Profiler CPU+NPU、Level2、AiCoreNone、record_shapes/profile_memory/with_stack关闭、data_simplification=False，DB+Text完整导出。主分析rank1框架目录存在profiler_info结束信息、ASCEND_PROFILER_OUTPUT、trace_view.json、kernel_details.csv及原生DB，Stop/解析检查通过。两卡原生DB约165/167MB；原始性能文件与逐块诊断留服务器，不入Git。

只读SQL依照ascend-profiler-db-explorer参考表结构：MEMCPY_INFO关联TASK得真实DMA起止/字节，CANN_API看下发时刻，PYTORCH_API中的自定义梯度hook范围配对host时钟，COMPUTE_TASK_INFO/TASK与COMMUNICATION_OP检查源就绪之前在做什么。计算和通信概览沿用技能原始CTE；时间窗为硬件分段问题，用已有文档字段查询。

大复制被硬件拆成1或2个DMA子任务，不能按任务条数直接一对一匹配CPU槽。按CANN connectionId分组后，0卡7784次复制/365,911,384,064字节，3卡6720次/301,487,038,464字节，与逐块记录和原对照字节完全一致。分别保存DMA子任务活动时间和整个复制包络；示例小块只有一个任务。

源就绪事件在计算流记录，复制流先wait_event再放起止标记。事件包络还可能包含主机提交间隙，最终DMA时长以原生TASK为准。host范围/hook配对的偏差中位绝对值约15–16微秒，但存在6–8毫秒离群；device事件时钟用各复制任务前后包络局部约束，不虚报亚微秒跨时钟精度。长等待按约毫秒/秒解读，2.5微秒DMA直接来自单个硬件任务，不靠host跨钟相减。

## 一处真实1.34秒断粮

3号卡等待`model.norm.weight`，形状[5120]，BF16梯度仅10KiB：

| 时间点（步起点后） | 发生的事 |
|---|---|
| 18.92394秒 | CPU用完已有数据，开始等下一份；快照16空槽、0份ready |
| 约18.899–19.456秒 | HCCL任务，包含等对端；与本次CPU等待交叠约0.532秒 |
| 随后至20.264秒 | 下一micro前向、输出头相关反向；硬件计算任务覆盖约0.792秒 |
| 19.86928秒 | `aclrtMemcpyAsyncWithCondition`已调用，下发约29微秒；在流里等待源计算 |
| 20.26418秒 | 计算流的源就绪标记完成，DMA紧接着执行 |
| 实际DMA | 10KiB，单任务约0.0025毫秒，即2.5微秒 |
| 20.26472秒 | CPU收到完成通知并取到数据，继续累加 |

卡间任务含Notify_Wait，最长约556毫秒，不是有效数据持续传556毫秒。计算任务包括MatMul、GDN前向、LogSoftmax等；CPU范围也显示pipeline/stage1/forward和随后backward。由于目标是最终norm的权重梯度，此前要先完成该micro前向和输出头反向，不能把GDN内部名为recompute的前向内核误认作整层checkpoint重算。

通信和计算覆盖可能有少量重叠，数字不能直接相加作独立分解。这里已能清楚区分：复制调用提前下发了，但源梯度还不可用；并非CPU忘记提交，也不是DMA持续传1.34秒。

补充：[Insight实际视图、CPU/D2H重叠与16槽全满证据](insight-overlap.md)；[截图所选窗口NPU连续性](npu-selected-window.md)：3号NPU计算覆盖99.8%，0号50.7%并有部分主线程卸载背压重叠停顿。以下全槽空结论针对micro交界长等待，不能推广到整步；其他阶段确实存在15槽就绪+1槽CPU处理中，复制线程因无空槽而等待。

## 为什么所有槽都空，且反复发生

一轮16个micro，3号恰好15次>100ms长断粮，序号350、699、1048…5236，相差349，全部等待每个新micro的`model.norm.weight`；15次直接快照均16空槽。CPU已处理完上一micro需要卸载的梯度，但下一micro仍处于通信/前向/输出头反向，下一份梯度还没就绪。驻留在NPU的梯度也不会逐micro交给CPU，所以梯度供应本来是分段的。

0号也有15次同类断粮，序号397、793…5941，相差396，目标是每个新micro的`model.layers.31.mlp.down_proj.weight`首块128MiB；14次全空、最后一次9空槽/7待完成。典型一次等739.7毫秒，其中源就绪前733.2毫秒，实际128MiB DMA仅6.26毫秒。源之前的计算覆盖约515.7毫秒，HCCL覆盖约215.2毫秒，说明也在等下一轮前向和跨卡依赖推进。

| 本次15处长等待合计 |0号|3号|
|---|---:|---:|
| CPU长等待总计 |9.370秒|12.498秒|
| 源就绪之前 |9.274秒|12.492秒|
| 实际DMA包络 |0.0905秒|0.0000387秒|
| DMA结束后至CPU取到 |0.0054秒|0.00685秒|

源就绪后至DMA开始的差额处于跨时钟误差量级，只能称基本紧接着开始。大块传输也做了全量统计：128MiB DMA中位数0号8.42毫秒、3号11.09毫秒。这不证明硬件带宽达到理想上限，但足以排除把上述几百毫秒到1秒长等待全算成实际传输。

## 下载与其他条件

用户说明同期在服务器下载模型。观察到curl约10MiB/s，采样阶段观察点磁盘I/O压力接近0、内存压力已恢复；不能据此断言下载全程无影响，也不能把已由设备计算/通信记录解释的长断粮直接归因于下载。首步初始化的内存压力单列，不混为DMA故障。没有停止下载或他人任务。

## 结论与交付

固定16槽足以支持多缓冲，但缓冲只能保存已经产出的梯度。此次全槽空的长等待主要是上游梯度生成有阶段空档；下一步若继续优化应研究micro间调度、卡间依赖与输出头计算，不能再默认靠加槽或调D2H带宽解决。本轮仅完成定位，不启动新优化实验。

0/3卡运行结束后npu-smi确认无本次或其他活动进程，均已释放。

- [msprof SQL结果、逐段统计、协议与清理检查](../../evidence/2026-10-28/d2h16-03-msprof-results.json)
- [tiny逐位检查](../../evidence/2026-10-28/d2h16-03-tiny-exact.json)
- [实测图对应的原始窗口](../../evidence/2026-10-28/d2h16-03-figure-case.json)
- [实测PNG](../../artifacts/2026-10-28/d2h16-measured-starvation.png) / [可放大HTML](../../artifacts/2026-10-28/d2h16-measured-starvation.html)

复现：环境提供既有MODEL_DIR、DATA_DIR、CPU_ADAM_EXTENSION、VEOMNI_SOURCE、TINY_REFERENCE，独立run_sequence.py先检查0/3空闲，tiny逐位通过再完整采样；分析命令`python analyze_msprof.py --folder <profile16.private> --output <report.json>`。只读原生DB，无需截图识别前端；图像来自查询的实际时间戳。源文件、原生DB、逐块诊断SHA保存在证据中。
