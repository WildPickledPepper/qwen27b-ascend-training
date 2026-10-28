# 固定16槽的D2H启动/传输诊断（进行中）

2026-10-28用户明确要求：固定16×128MiB CPU锁页接收槽，共2GiB；不再改变容量。独立ready-pipeline诊断副本，驻留4/8GiB、非驻留待搬512MiB、原SQL768条/global256/micro16/48重算/BF16/CPU FP32主权重-m-v/512token连续化loss及CPU/NUMA规则保持。3/4空闲才运行。没有优化改动，不修改默认/共享库/源模型。

先tiny三步（末步采样），验证所有grad/master/m/v逐位一致和跨流Event计时/导出可用，再27B三步末步采样。Level2、CPU+NPU、AiCoreNone、Db+Text，保留原始数据；不以采样步时间比较性能。

逐块唯一sequence+offset，记录producer hook/admit、D2H提交/完成通知、CPU get/add以及真实槽状态。计算流source-ready event、复制流在wait_event之后记录的start/end timing event，区分源就绪后排队与设备复制区间。计时读取在profiler停止、全部任务完成后进行，不在CPU供数路径上批量查询。profiling instrumentation影响时序，必须用msprof TASK/MEMCPY_INFO/CANN_API交叉核对，不能再把completion等待全叫纯DMA。

两个时间轴分开：host monotonic相对步起点、设备事件相对device anchor；不未经映射直接相减。槽快照有微小bookkeeping竞态，另保留queue大小/状态及数据已就绪数量核对。完成先校验采样Stop和导出件，再只读SQL分析、归档、释放3/4。
