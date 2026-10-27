# 前四项显存生命周期：真实模型的meta执行

用户要求实现中间结果生命周期、流水线/重算叠加、临时梯度与副本、别名/布局跟踪，并检验预测精度。本轮使用原不压缩4GiB方案，独立CPU/meta进程，原训练入口和其他任务不改。不启动NPU初始化，不加载权重，不读取旧快照/校准表生成预测。

## 实现

- `predict_meta.py`实例化实际Transformers Qwen3.5文本模型的meta参数，沿原两段切层、48层非重入checkpoint规则和16micro调度执行真实autograd。前向输入只有形状；mask采用当前有padding的eager分支形状。源码中的原地梯度累加、param.grad=None、驻留选择规则保留，CPU接收槽不算HBM。
- `storage_tracker.py`使用TorchDispatchMode观察算子输出；用StorageImpl的native弱引用检查存储是否仍存活，不保留tensor或storage强引用。保存给反向的native SavedVariable、checkpoint重算和Python引用因此参与实际meta存活关系。按storage去重，记录形状/stride及分配/释放事件，逐事件求峰值，不填15/8、51/24。
- GDN沿原`community_gdn.py`调用固定VeOmni的Python前后向分配包装；`launch_stub.py`只跳过Triton设备启动，保留empty/zeros/contiguous等显式张量操作和save_for_backward。utils仅提供meta可用的连续化/设备上下文替代，目标核数是输入假设24/48，不从显存快照取得。
- checkpoint只把meta不支持的设备RNG/上下文选择转到CPU上下文，保留非重入重算及early stop。没有覆盖官方保存张量hook，避免改变checkpoint保存机制。
- 真正CANN/Triton编译器内部workspace、驱动/分配器预留仍不属于本轮前四项覆盖范围，保留未知；NPU独有反向布局/复制可能与meta不同，要通过误差分析识别，不能拿快照数回填。

## 预测与验证隔离

预测进程的文件审计拒绝打开evidence目录、calibration.json、safetensors、原始快照和DB。输入仅公开模型配置、真实实现源码、训练方案和16个微批的padding长度；长度是输入形状，不是显存测量结果。`run_predictions.py`先写并哈希封存所有预测，比较器再在独立进程读取历史观测。源码、输出及配置哈希归档，禁止为了贴合观测改系数。

小例子验证已通过：view保留底层存储、非连续reshape复制、expand/detach共享、原生SavedVariable保留与反向释放、param.grad清空释放、原地累加无额外结果存储。tiny元数据模型前后向已通过；真实27B结构的三组形状正在运行，尚无完成或精度声明。

所有存储ID仅为本次运行的编号，导出不含真实地址/权重/数据全文/私有路径。原始CPU运行日志仅在服务器私有目录保存。
