# 压缩FFN权重，将空间分给梯度累积

2026-10-26用户授权把已验证BF16无损压缩接入训练，用释放的HBM扩大梯度累积，实测净速度。只在实时空闲物理2/3卡独立实现，原81.22秒入口、原checkpoint和其他任务不动。

固定原768条/顺序/长度/padding、global256/micro16/16流水线micro、48层选择性重算、BF16计算与保序梯度累积、CPU FP32 master/m/v、相同全局裁剪/Adam/学习率/更新边界。

## 先集成FFN，避免一次改变所有模块

每卡96个FFN gate/up/down矩阵，原BF16 15.9375GiB，压缩约11.9GiB。只替换这部分存储，GDN、普通attention和embedding/lm_head保持原生权重。初始从已逐帧验证的归档读取；所有载入矩阵再次在NPU按int16位型与当前原生权重完整比较，随后释放原矩阵存储。

实际初始化：2卡原FFN17,112,760,320 bytes，带guard payload12,823,577,408 bytes；框架allocated26,896,201,728→22,607,050,240 bytes。3卡payload12,776,564,704 bytes，allocated26,896,212,480→22,560,044,032 bytes。释放的是实际NPU存储，不把离线压缩率冒充线上显存下降。元数据和校验guard已计入payload，解压/校验workspace仍需另计。

将梯度驻留4→7GiB，剩余约1GiB空间用于解码矩阵、workspace和checkpoint重算的临时权重，正式峰值仍须实测。全部权重归档总共省12.63GiB并不意味着本次每卡增加6GiB：本次只采用FFN子集。

## 原生算子与autograd

每个FFN Linear前解码原BF16矩阵，调用原生Linear，结束后换成保留原形状的零步长元数据占位，不让完整权重持续驻留。非checkpoint部分通过`saved_tensors_hooks`保存权重的记录编号、形状/stride/offset，反向需要时解码重建视图，避免autograd把原矩阵留住。checkpoint内层hook仍遵循原框架机制，重算时同样调用模块解码；退出/提前停止重算也释放权重。

仅在原全局更新边界，CPU FP32 Adam按原顺序更新master，原样转换BF16分块后传回NPU，重新编码并逐位验证；作者codec位差/膨胀即原字节回退。没有量化权重、梯度、主权重或m/v，没有提前更新、改写matmul/反向求和顺序，也没有运行时从CPU取旧计算权重。

作者BF16 kernel复用已验证静态库；只链接宿主桥，解码提交到当前PyTorch流、不逐帧额外同步。编码读取Header/字节验证保留必要同步，其全部开销纳入训练整步。

依据[PyTorch 2.10 saved_tensors_hooks文档](https://docs.pytorch.org/docs/2.10/autograd.html#torch.autograd.graph.saved_tensors_hooks)，未压缩Tensor的pass-through返回detach以避免图引用环。tiny首次完整状态门槛通过后补上这一不改变值的引用处理；额外CPU autograd检查与正式源代码哈希分别归档，不能说该行已包含在首次tiny源码里。

## 当前验证状态

tiny连续三步全部梯度、CPU FP32 master、Adam m/v与原流水线逐位相同，独立首步参考和范数也通过；checkpoint/非checkpoint两种路径实际执行。27B初始全部压缩矩阵位比较通过，已进入完整训练。前三轮正常（首轮预热）、第四轮单独ms采样，尚不宣称收益。

原始权重、归档、完整日志、构建产物只留服务器。完成后提交过滤证据与可移植脚本、释放自身任务。
