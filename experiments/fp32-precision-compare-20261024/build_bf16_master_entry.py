"""Derive explicit BF16-compute / FP32-master training from the audited harness."""
from pathlib import Path
root = Path(__file__).resolve().parent
source = (root/'train_cpu_pipeline.py').read_text(encoding='utf-8')

def replace(old, new):
    global source
    assert old in source, old
    source = source.replace(old, new)

replace('import bench_cpu_pipeline\n', 'import bench_cpu_pipeline\nimport bench_bf16_master\nfrom bench_bf16_master import precision_context,evaluation_context\n')
replace('a.mode=="fp32" and a.window==16777216 and a.steps==4;configure(a.mode);bench_cpu_pipeline.install()',
        'a.mode=="bf16" and a.window==16777216 and 4<=a.steps<=20;configure("fp32");bench_bf16_master.install()')
replace("'bf16_model_copy':False", "'bf16_model_copy':True")
replace("'bf16 autocast with FP32 master weights'", "'explicit BF16 linear matmuls, FP32 outputs/other operations; persistent BF16 linear weights + CPU FP32 master'")
replace('bench_cpu_pipeline.install_store(store,is_embedding=isinstance(old,torch.nn.Embedding))',
        'bench_bf16_master.install_store(store,is_embedding=isinstance(old,torch.nn.Embedding))')
replace("result.sort(key=lambda x:(x[0],x[1]));assert len(result)>=count\n    return result[:count]",
        "result.sort(key=lambda x:(x[0],x[1]));assert len(result)>=min(count,4)\n    return result[:count]")
replace("train=get_examples(tokenizer,Path(a.data)/'train.jsonl',a.steps)",
        "train_pool=get_examples(tokenizer,Path(a.data)/'train.jsonl',a.steps)\n    train=[train_pool[i%len(train_pool)] for i in range(a.steps)]\n    status['training_unique_examples']=len(train_pool)\n    status['training_repeats_if_needed']=len(train_pool)<a.steps")
replace("    for step,example in enumerate(train,1):", """    native_before=run_eval(val[:2],canonical=False)
    status['validation_before_bf16_preflight']=native_before
    reference_loss=sum(r['loss']*r['assistant_tokens'] for r in baseline['examples'][:2])/sum(r['assistant_tokens'] for r in baseline['examples'][:2])
    assert abs(native_before['loss']-reference_loss)<0.15, (native_before['loss'],reference_loss)
    log('bf16_preflight_passed',metrics=native_before,reference_fp32_loss=reference_loss,memory=memory())
    for step,example in enumerate(train,1):""")
replace("        for s in stores.values():s.enabled=False\n    final=", """        for s in stores.values():s.enabled=False
        if step%5==0:
            log('validation_progress',step=step,metrics=run_eval(val[:2],canonical=False))
    final=""")
replace("pipeline_runtime=dict(bench_cpu_pipeline.RUNTIME),", "pipeline_runtime=dict(bench_cpu_pipeline.RUNTIME),bf16_matmul_counts=dict(bench_bf16_master.COUNTS),")
source = source.replace('Four-step full-parameter controlled FP32/offload, BF16/offload and compressed comparison.',
                        'Explicit BF16 linear compute with CPU FP32 master/Adam; up to 20 full-parameter steps.')
(root/'train_bf16_master.py').write_text(source, encoding='utf-8', newline='\n')
