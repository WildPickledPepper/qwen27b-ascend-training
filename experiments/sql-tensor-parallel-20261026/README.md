# 双卡张量并行评估

用户授权在完成流水线梯度卸载优化后评估TP。流水线2GiB梯度驻留已完成，固定256条达到85.09秒；TP仅顺序使用实时空闲2/3卡，不动其他任务/共享库/原checkpoint。

## 小模型兼容性门槛已通过

环境torch2.10.0+cpu、torch-npu2.10.0.post4、Transformers5.14.1、DeepSpeed0.19.8。本机Qwen3.5有14条HF TP规则，AutoTP可读取，普通注意力/FFN分片；GDN采用`colwise_gather_output`，投影权重分片后收齐激活，核心GDN计算在每rank保留完整。

实际小模型为4层（3 GDN + 1普通注意力），同一批数据TP2、DP1、micro16、累积16，原生ZeRO-1 CPU FP32主权重/Adam m/v，BF16计算/累积。输出与未切分模型比较，所有参数的重建梯度逐项比较，随后完成一次原生CPU优化器更新。正式路径关闭观察钩子，同样通过；这不是27B速度结果，也不是位级一致证明。

原生第一次检查被Q/K norm梯度差异拦下：每rank相对误差约64%–70%，两个rank相加后相对参考约1%，其余参数通过。注册状态和字典均显示钩子存在，但观察到实际回调0次；提前移动模型到NPU或读取前全设备同步均未解决。在优化器初始化结束后，仅对这些标记为TP梯度求和的共享参数重新调用公开`register_hook`绑定SUM回调，实际执行16次，全部梯度检查通过。不改共享安装，不放宽门槛。具体何处丢失回调连接尚未追到框架源码行，不能泛称所有DeepSpeed/Ascend版本都有该问题。

## 27B首次容量试验

原micro16/global256/48层选择性重算的TP2+ZeRO-1 CPU卸载，完成初始化/验证后在第一个训练micro前向OOM；尚无整步速度。失败日志和原入口保留，不能把未完成任务记作慢或快。

下一步保持micro16、global256和原样本/padding/精度，改64层全部激活重算，先tiny验证再27B短跑。该系统设置增加重算量，必须与流水线48层重算分开说明，不宣称所有实现细节相同。

官方机制参考：[DeepSpeed AutoTP training](https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/docs/_tutorials/autotp-training.md)。规则存在不等于该硬件完整训练已验证，以上结论以实际小模型/容量结果为准。
