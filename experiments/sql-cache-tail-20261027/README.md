# 梯度缓存尾部额外3.87秒：分段计时诊断（进行中）

用户要求解释上一轮4/4→4/8GiB尾部12.64→16.51秒。旧计时同时覆盖CPUAdam、FP32梯度转换/裁剪、FP32→锁页BF16、H2D调用、双缓冲等待和最终device同步/跨卡barrier，不能据旧值归因。

独立train_timed.py只插入host monotonic计时，训练数学/更新顺序不改。每个更新chunk分别计梯度转换、CPUAdam、staging等待、BF16 staging转换、H2D copy调用和event record；尾部最后event等待、device同步、barrier另计，保存每rank每chunk记录。H2D调用墙钟不是设备纯DMA时间，异步流水时不能把host阶段计时解释为设备忙时。CPU/meta预测不涉及本轮。

物理3/4空闲检查后，同配置global256/micro16/原SQL768条/48重算/BF16/CPU FP32 master-m-v/512token连续化loss，缓存4/4与4/8各3步，首步lr0预热、后两步比较。此处重复对照只为获取旧日志缺失的细分计时，不宣称原多3.87秒必然复现或必然由缓存导致。若差异消失，明确原差异不能作缓存因果证据。

原入口及上一轮源码保持，结束检查3/4空闲；只清理本实验自身任务。分阶段Git归档。首次两组均未启用profiler，避免把采样开销混入正常比较。
