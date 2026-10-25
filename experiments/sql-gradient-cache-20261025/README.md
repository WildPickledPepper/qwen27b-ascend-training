# 12GiB NPU梯度驻留：实测变慢，不采用

固定两张910B2、256条/更新、每卡16条、累积8。将148个梯度分片共12GiB留在NPU，复用原生DeepSpeed GPU累积与CPU FP32边界卸载，不修改逐微批相加顺序或Adam。

小模型两rank、3更新、每更新8微批的完整FP32梯度/master/m/v逐位一致。整模型相同768条、相同token及padding计数；第768条验证loss0.8581519723、204/224，与原生一致。稳态同256条却从107.97秒变成126.66秒，慢17.31%，所以没有替换原配置。

每rank梯度逻辑复制量484.13→277.97GB，减少42.58%。Profiler异步复制独占区间14.93→7.48秒，说明确实省了复制。但是出现aclrtMallocPhysical、aclrtFreePhysical各11289次，主机累计14.90/13.45秒；无计算/通信/复制/CPUAdam覆盖区间11.49→37.35秒。前向墙钟从约17秒增至约35秒，CPUAdam仍约10秒。应先解决内存池物理页申请/释放开销，而不能用复制字节下降宣称提速。

allocated峰值每卡54.87GiB看似仍有空间，但原来的reserved内存池已接近整卡容量，新增常驻梯度使临时分配频繁回收/映射。容量规划只按allocated峰值减法不够。第4次更新是Profiler专用，重复首256条，比原Profiler第3轮padding多约8%，所以不同采样窗口不能当严格同工作量测速；实际性能比较使用无采样第2、3轮，物理内存管理API开销由采样直接记录。

`gradient_cache.py`只重定向实例内梯度缓冲，不改共享库。初次源码保护检查误看了观测包装器，在小模型加载后即停止；调整安装顺序后通过，失败原件保留在服务器独立目录。原模型只读，未存新完整checkpoint；任务正常结束、2/3卡释放，其他服务未动。

证据位于`evidence/2026-10-25/sql-gradient-cache-{probe,results,profile,resource-release}.json`。`profile`含只读SQL与Stop/导出检查。ZeRO分片与卸载背景参考[DeepSpeed官方文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)。

后续沿同一目标研究CPU BF16梯度累积与锁页双缓冲，避免挤占NPU内存池；结果单独记录，不与本失败方案混合。
