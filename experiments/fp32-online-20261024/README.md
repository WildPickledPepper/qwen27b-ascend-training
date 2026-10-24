# 纯FP32压缩驻留训练原型（2/3号910B2）

2026-10-24。用户授权将此前思路接成整模型短训练。此原型使用原27B模型的全部文本参数26,895,998,464；不加载视觉/MTP、不使用LoRA、不保留整模型BF16计算副本、不用CPU/NVMe保存运行时权重或m/v后备。

## 执行方式

- **权重**：载入已逐帧核验的ENEC FP32归档，压缩字节始终在NPU。按矩阵/分块解压为FP32参与计算；更新后在NPU重新压缩并逐位比较，失败或膨胀则保留FP32原字节。只有32-byte帧头等布局元数据传到CPU；权重检查不转移到CPU。
- **m/v**：每参数一个uint8，低4 bit为NF4 m、高4 bit为保留真零码的log-v；每块3个FP32元数据共12 bytes。随机舍入。FP32 NPU实现插值/概率，与CPU研究的FP64概率实现不是逐位等价端口；它属于获准的有损状态方案。
- **微小参数**：norm、卷积等非Linear/Embedding参数保持普通FP32权重与FP32 m/v，避免大量微小压缩对象。真实字节另计。
- **梯度**：自定义线性层反向先计算旧权重对应的输入梯度，再更新对应权重并释放权重梯度。常规投影暂存一个完整参数梯度，约最大356 MB；embedding/lm_head特别按块处理，避免生成约4.74 GiB的单参数梯度。微小参数用post-accumulate hook更新并清梯度。
- **激活**：Transformer层使用reentrant gradient checkpointing，完整样本batch=1；不做梯度累积或全局梯度裁剪。两卡按原归档张量分工，激活必要时点对点传递。
- **计算**：FP32；禁用matmul/conv HF32，不开autocast或GradScaler。lm_head分块累加会改变求和分组，不能宣称整条训练与其他FP32实现逐位一致。

自定义autograd中实际权重保存在WeightStore，模块里的标量proxy仅用于建立反向依赖。因此不能用`model.parameters()`中的proxy个数冒充训练规模。成功关卡要求实际store坐标与微小参数合计等于全部文本参数，每一步各store更新坐标恰好等于其参数数，所有微小参数钩子完成，且没有残留Parameter.grad。

不使用CPU权重/优化器卸载。初次读取checkpoint以及训练完成后的保存属于输入/输出，不能拿磁盘在训练中补容量。容量或数值异常会停止本次私有运行；原checkpoint不写回，不宣称失败步是完整更新。

## 前置验证

两卡初始化及跨设备梯度复制通过；小型压缩线性层前向和输入梯度与普通FP32实现最大绝对差为0。首次Adam更新相对同梯度FP32参考最大绝对差约1.86e-9/4.66e-10；首次更新从零矩开始，**不等于验证后续量化状态无误差**。embedding重复token梯度及全坐标更新也通过。`smoke.py`为实际检查入口。

## 数据与短跑协议

复用此前独立Alpaca-GPT4中文32训练/8验证划分。按完整token长度和消息哈希排序，取最短两条训练样本（28、31 token）、独立最短验证样本（49 token）；只在assistant token计算因果语言模型loss。不截断成残缺对话，不使用GSM8K。学习率1e-5，AdamW beta=(0.9,0.999)、eps=1e-8、wd=0.01；B128优先，B64随后对照。

短跑验证的是容量、反向与全参数更新，不是27B训练质量或收敛认证。最终状态、loss、峰值、耗时与失败保留在当日日志及过滤证据中。新checkpoint是本原型的分块格式，不是可以直接交给vLLM/HF的标准权重目录。

## 复现

先配置既有独立torch/torch-npu/Transformers环境和CANN，建立ASCEND缓存目录；仅使用获准物理2、3卡。作者kernel链接自此前已核验的ENEC122构建，宿主桥不重新编写压缩kernel。

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3
python build_bridge.py --author-build "$ENEC_AUTHOR_BUILD"
python smoke.py --output "$RUN/smoke.json"
python train.py --model "$MODEL_DIR" --archives "$ENEC_ARCHIVE_ROOT" \
  --data "$INDEPENDENT_SPLIT_DIR" --output "$RUN/b128" --block 128 --steps 2 --save
```

输出目录必须新建。训练后保存只输出新目录，不合并原模型。未经当轮明确请求不自动Git提交/推送权重或新实验。

## 已完成的B128结果

- 修复前尝试：allocator比例0.97时在第一步codec校验临时分配处OOM，0个完整更新。
- 修复：比例0.995，复用64 MiB解压校验缓冲；不改变dtype，不卸载状态。
- 重跑：2个完整全参数更新，每步26,895,998,464参数完成，残留Parameter.grad=0；320.94/316.98秒。
- 验证单样本loss：4.169747829→2.856882095。两步各用不同训练样本，不能把两个train loss直接当同一样本下降曲线。
- 权重帧及guard在第2步后87.25857 GiB；m/v实际27.41423 GiB。两步最高分配峰值logical0/1为58.50972/58.45382 GiB（以证据bytes为准）；reserved及设备总占用另计。
- 新checkpoint保存完成，两个分块文件的索引连续覆盖与长度核对通过；未做完整checkpoint重载续训，不能宣称恢复路径已验证。整次949.02秒，退出后物理2/3卡无进程。

[过滤后的实际状态与事件证据](../../evidence/2026-10-24/fp32-online-results.json)。B64相同协议两次首步反向OOM，结果见下文。

## B64容量重跑记录

首次B64使用相同窗口与allocator比例，在首步第13层附近分配新压缩帧时OOM：活跃59.47 GiB、预留60.47 GiB、设备剩余59.43 MiB，申请56 MiB；0个完整更新。重跑只设置`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`以控制分配碎片；设备所装torch-npu的`NPUAllocatorConfig.h`和`npu/memory.py`确认该接口支持。模型、数据、FP32计算、优化器窗口不变。重跑仍在同一位置OOM，申请56 MiB时设备仅剩60.65 MiB、预留60.47 GiB；0个完整更新。可扩展段配置没有在本实现下解决容量问题。两次失败均保留，不能将部分反向计为完整更新。

## 结束状态与适用范围

10:28:02（Asia/Shanghai）核对物理2/3卡均无活动进程，基础占用3413/3414 MiB；其他既有服务仍在。此次未自动Git提交/推送。

已跑通的是B128、batch=1、28/31 token的两步文本全参数训练。真实参数总量与逐步更新覆盖已审计。尚未验证长序列、长期收敛、量化状态相对完整FP32 Adam的质量、检查点重载续训或生产吞吐。每步约5.3分钟，当前是容量原型。权重编解码逐位校验与4-bit m/v近似要分别理解；不能称为整体无损Adam训练。
