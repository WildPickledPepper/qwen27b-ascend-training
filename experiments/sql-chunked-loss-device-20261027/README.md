# Chunked loss在物理3/4号910B2的实测

2026-10-27已完成。最终512-token分块＋连续梯度版，物理4号卡allocated峰值54.054480→49.835765GiB，省4.218715GiB（约7.80%）；reserved峰值56.269531→50.351562GiB，少5.917969GiB。物理3号卡allocated峰值56.559970GiB不变，因此两卡中的最大单卡峰值不变。后两轮平均77.98599→80.02179秒，慢2.61%，不是加速方案；保留为省显存候选，不替换原81.22秒入口。

## 固定配置与配对结果

框架是自定义PyTorch/Transformers两段1F1B流水线，前后各32层；复用DeepSpeed CPUAdam组件，非ZeRO-3。固定BF16计算/梯度累积、CPU FP32主权重/m/v、4GiB驻留、global256/micro16/16micro、48/64层非重入激活重计算、原SQL768条/顺序/padding。只改训练输出层/CE分块及grad连续化，普通验证路径保持；未改更新次数、学习率、CPU状态精度或缓存预算。原模型只读，无新27B checkpoint。

本轮3号卡作为rank0，NUMA4/CPU96–119；4号卡作为rank1，NUMA0/CPU0–23。使用同卡组新普通loss对照，历史2/3卡81.22秒仅作出处，不用于本轮比值。每组3次更新，首轮lr0预热，后两轮计时；256条/更新，处理完同768条，不是完整数据集收敛测试。

| 每次更新 | 普通loss秒 | 最终分块秒 | 普通loss rank1 allocated GiB | 分块rank1 allocated GiB |
|---|---:|---:|---:|---:|
| 1 | 175.900 | 129.040 | 54.054222 | 49.835507 |
| 2 | 78.098 | 79.894 | 47.592371 | 46.205680 |
| 3 | 77.874 | 80.150 | 54.054480 | 49.835765 |

较短的第二轮输入以192长度为主，显存降幅小于含256长度的轮次，不能跨轮相减。allocated和reserved分别取各自峰值，reserved包含allocated，不相加；物理驱动/通信占用不在这些数值里。

## 验证结果和精度范围

两组初始验证、768条样本哈希/长度/顺序、851个参数、48层重算、每卡4GiB驻留、FP32 m/v总215167987712字节、每步梯度D2H字节与驻留累加次数均核对一致。tiny三步比较全部master/m/v/梯度，最大整体梯度relative L2约0.0026151、master约3.452e-7。真实输出层短输入检查通过。

普通loss短验证loss0.855101466、205/224正确回答token；分块版0.853088260、204/224。loss更低但正确token少1个，不据8条验证判断更好/等价。分块BF16权重梯度累加改变求和顺序，不是无损更新；完整长期收敛未验证。

## 初版为什么失败，怎么修正

初版虽然切开了logits/CE，但NPU原生自动求导返回的权重梯度stride[1,248320]非连续。独立完整尺寸观测将峰值直接定位到`dw.add_(gw)`：操作前当前allocated约7.187567GiB、此前峰值7.690498GiB；操作后当前量不变，峰值跳到14.292061GiB，临时增量7.104494GiB。它相当三份梯度大小，不仅凭差额断言内部确切副本数量。

因此初版整模型rank1峰值反而54.572093GiB、后两轮85.51283秒，未采用。旧meta只把add视为原地操作，没有覆盖后端临时分配，原“省6.35GiB”预测在这一路径上被实测推翻。

最终`chunked_loss_contiguous.py`在每块部分梯度生成后显式`gw.contiguous()`，随后同顺序BF16累加。梯度stride变为[5120,1]；同尺寸单测峰值降到9.555733GiB，加法不再抬高峰值。这份显式连续化副本有成本，故最终只省4.22GiB。没有改Adam、量化或混入缓存扩容。

## 更新模型的检验

不按实测添加7.10GiB常数。用包含显式contiguous的同一实现重新CPU/meta追踪完整16micro，先封存预测SHA，再读取对应整模型观测。第一组rank1预测49.835371971GiB、实测49.835507393GiB，差约0.139MiB。算法是根据此前布局诊断选择的，不能包装成盲选新算法验证；后续跨步小残差、其他后端workspace/driver/reserve仍未通用建模。

## 复现、资源和证据

新目录使用`run_replay.py`，环境提供MODEL_DIR、DATA_DIR、CPU_ADAM_EXTENSION、VEOMNI_SOURCE，按3/4拓扑运行；先初始化/输出层/tiny完整状态门槛，再普通loss和最终分块各3步。它是最后有效方案的重放入口；本次没有额外重放已经完成的实验。

`run_initial_sequence.py`、`train_initial.py`和初版结果保留失败；`resume_sequence.py`/`run_contiguous.py`保留本轮实际分阶段控制。所有核心训练/梯度源码SHA在结果中核验，报告脚本单独记录SHA。原始日志、快照和权重不提交。

最终3/4号卡均已无活动进程，未向其他卡进程发信号，原81.22秒入口三份LF哈希未变。

- [最终实测与校验](../../evidence/2026-10-27/sql-chunked-loss-device-contiguous-results.json)
- [新meta预测](../../evidence/2026-10-27/sql-chunked-loss-contiguous-meta-prediction.json)
- [被否决初版](../../evidence/2026-10-27/sql-chunked-loss-device-first-results.json)
- [原布局分配观测](../../evidence/2026-10-27/sql-chunked-loss-layout-diagnostic.json)
- [连续布局分配观测](../../evidence/2026-10-27/sql-chunked-loss-layout-contiguous-diagnostic.json)

## 前期阶段记录（以下为当时状态）

# Chunked loss在物理3/4号910B2的实测（进行中）

用户2026-10-27授权在3/4号卡实测。两卡初始化、BF16矩阵乘、HCCL和P2P通过；输出层小形状及真实V248320/H5120、短token前后向通过；tiny流水线三步完整梯度/master/m/v对比通过。完整27B配对实验进行中，没有显存收益或速度完成声明。

固定原SQL768条、顺序/padding/global256/micro16/16micro、48层非重入重计算、4GiB BF16梯度驻留、CPU FP32主权重与Adam m/v。独立`train_chunked.py`由原81.22秒入口派生；只改变训练loss路径及卡组/对应CPU-NUMA绑定。3号卡NUMA4/CPU96–119，4号卡NUMA0/CPU0–23。验证仍用完整普通loss与logits，方便同口径统计；每次更新边界和全局有效回答token加权保持。

普通loss和512-token分块各3次更新，第一步lr0预热，后两步计时；记录两rank每步allocated/reserved峰值、范数、卸载字节、验证与数据哈希。必须在同卡组对照，不把历史2/3卡81.22秒直接作速度参照。新完整模型保存不在范围内，原模型/共享环境不改，完成释放自身任务。

## 预检与失败

初版tiny使用原逐元素allclose（rtol2%、atol1e-7），lm_head.weight因BF16分块求和顺序改变失败，最大绝对差1.220703125e-4；rank0其他梯度逐位一致。原始失败保留在`train_initial.py`与服务器日志，未绕过后直接跑27B。

修正仅涉及tiny验证口径：lm_head保留逐元素比较结果，增加并使用与输出层预检相同的整体relative L2<2%及有限值检查；其他参数维持原逐元素门槛。没有修改BF16累加数学。tiny三步所有张量对照最大整体梯度relative L2约0.0026151、master约3.452e-7；不等于无损或长期收敛认证。

真实输出层尺寸短输入loss绝对差1.91e-6，输入/权重梯度relative L2约3.47e-6/6.07e-4。详见[预检证据](../../evidence/2026-10-27/sql-chunked-loss-device-preflight.json)。

实现来自已建模的`chunked_loss.py`：前向分块线性+FP32 CE，反向逐块重算、BF16部分权重梯度累加；不保留所有块logits计算图，但仍有完整lm_head梯度和临时部分梯度。不使用新精度、压缩或新优化器。只验证这套普通PyTorch实现，不冒称NPU融合loss内核。

## 新目录复现协议

将本目录代码与固定模型/数据/CPUAdam/VeOmni环境放入新实验目录，指定`MODEL_DIR`、`DATA_DIR`、`CPU_ADAM_EXTENSION`、`VEOMNI_SOURCE`，运行`python run_sequence.py`。它先检查3/4卡空闲，依次初始化/输出层/两个tiny/完整状态比较，然后才完整27B对照与512分块；每阶段均有超时和失败即停。`run_initial_sequence.py`/`train_initial.py`保留首次失败代码，不用于新运行；`resume_sequence.py`是本轮保留初始失败后续跑入口。运行时source-manifest记录源码SHA，日志和私有快照不提交。

本轮既有对照从头运行，无隐含继续训练；tiny实验用随机小模型，完整模型从原权重各自开始。没有保存27B更新后checkpoint。

## 第一轮整模型结果：未采用，布局修正版在测

三步普通loss与初版512分块均完成，同一3/4卡、同768条数据/顺序/padding，参数、FP32状态字节、梯度D2H量和驻留累加次数一致。普通loss后两步约77.99秒，初版分块约85.51秒，慢9.65%；rank1 allocated峰值54.054480→54.572093GiB，增加0.517613GiB。reserved有所下降不能等同活跃内存下降。验证loss变化约-0.002013，只是8条短验证。初版不采用，不将原6.35GiB预测冒充实测。

随后在完整输出头形状B16/S256/H5120/V248320单独同步观测：自动求导产生的BF16权重梯度stride[1,248320]非连续。第一次`dw.add_(gw)`前当前allocated约7.187567GiB、此前峰值7.690498GiB；加法后当前allocated不变，瞬时峰值14.292061GiB，额外临时空间7.104494GiB，恰相当三份梯度大小。仅凭这一差额不声称内部一定有三个特定副本，但证据将峰值直接定位到非连续梯度加法调用；meta对原地add没有记录到这些后端临时分配。

`chunked_loss_contiguous.py`只在取得部分梯度后添加`gw.contiguous()`，保持BF16和分块顺序，显式复制后再累加。同尺寸单独测试中梯度stride[5120,1]连续，加法不再增加峰值，整体峰值14.292061→9.555733GiB。明确连续化也需要一份完整临时梯度，不能当成零成本。正在重新通过输出层、tiny及整模型三步验证；不修改原入口、缓存预算或精度。

[首轮结果](../../evidence/2026-10-27/sql-chunked-loss-device-first-results.json) · [非连续加法观测](../../evidence/2026-10-27/sql-chunked-loss-layout-diagnostic.json) · [连续化观测](../../evidence/2026-10-27/sql-chunked-loss-layout-contiguous-diagnostic.json)
