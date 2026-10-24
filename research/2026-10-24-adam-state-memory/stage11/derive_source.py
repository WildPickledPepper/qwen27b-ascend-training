from pathlib import Path
root=Path(__file__).resolve().parent
source=root/'provenance/stage10_factor_replay.c'
if not source.exists():source=root.parent/'stage10/factor_replay.c'
s=source.read_text()
for n,v in [('D','32u'),('H','128u'),('KOUT','16u'),('MB','4u')]:s=s.replace(f'#define {n} {v}',f'#ifndef {n}\n#define {n} {v}\n#endif')
s=s.replace('typedef struct { uint32_t step,accumulation,microbatch,format;float data[FACTOR_FLOATS]; } Record;', '''#ifndef MIXED_X
#define MIXED_X 0
#endif
#if MIXED_X
typedef uint16_t XFactor;
#else
typedef float XFactor;
#endif
/* The input dataset is BF16-native in BOTH storage modes. Only X storage changes.
   Activations, dZ, dY and bias gradients are originally FP32 and remain FP32. */
typedef struct { uint32_t step,accumulation,microbatch,format;
 XFactor x[ACC*MB*D]; float activation[ACC*MB*H],dz[ACC*MB*H],dy[ACC*MB*KOUT],bias_gradient[H+KOUT];
} Record;''')
start=s.index('static inline float sample(');end=s.index('static inline float*weights',start)
s=s[:start]+'''static inline float load_x(XFactor x){
#if MIXED_X
 return word_float(((uint32_t)x)<<16);
#else
 return x;
#endif
}
static inline XFactor store_x(float x){
#if MIXED_X
 return (XFactor)(word(x)>>16);
#else
 return x;
#endif
}
static inline float sample(uint32_t seed,uint32_t t,uint32_t mb,uint32_t b,uint32_t i){
 uint32_t u=mix(seed^(t*0x9e3779b9u)^(mb*0x85ebca6bu)^(b*0xc2b2ae35u)^(i*0x27d4eb2fu));
 /* Construct the source datum directly as a BF16 word, not rounding an FP32 factor. */
 uint32_t bf=0x3f00u|(u&127u)|((u&128u)<<8);return word_float(bf<<16);
}
static inline float target(const XFactor*x,uint32_t j){float q=load_x(x[(j+7)%D]);return add(add(mul(.25f,load_x(x[j%D])),mul(.5f,mul(q,q))),-.1f);}
''' + s[end:]
start=s.index('static inline float*fx(');end=s.index('/* Captures actual',start)
s=s[:start]+'''static inline XFactor*fx(Record*f,uint32_t a){return f->x+a*MB*D;}
static inline float*fa(Record*f,uint32_t a){return f->activation+a*MB*H;}
static inline float*fz(Record*f,uint32_t a){return f->dz+a*MB*H;}
static inline float*fy(Record*f,uint32_t a){return f->dy+a*MB*KOUT;}
static inline float*bias(Record*f){return f->bias_gradient;}

''' + s[end:]
s=s.replace('float*x=fx(f,a),*activation=fa(f,a),*dz=fz(f,a),*dy=fy(f,a);','XFactor*x=fx(f,a);float*activation=fa(f,a),*dz=fz(f,a),*dy=fy(f,a);')
s=s.replace('x[b*D+i]=sample(r->seed,t,a,b,i);','x[b*D+i]=store_x(sample(r->seed,t,a,b,i));')
s=s.replace('mul(x[b*D+i],w[W1+i*H+h])','mul(load_x(x[b*D+i]),w[W1+i*H+h])')
s=s.replace('for(uint32_t b=0;b<MB;b++)micro=add(micro,mul(fx(f,a)[b*D+row],fz(f,a)[b*H+col]));','for(uint32_t k=0;k<MB;k++){uint32_t b=r->mode==3?MB-1-k:k;micro=add(micro,mul(load_x(fx(f,a)[b*D+row]),fz(f,a)[b*H+col]));}')
s=s.replace('for(uint32_t b=0;b<MB;b++)micro=add(micro,mul(fa(f,a)[b*H+row],fy(f,a)[b*KOUT+col]));','for(uint32_t k=0;k<MB;k++){uint32_t b=r->mode==3?MB-1-k:k;micro=add(micro,mul(fa(f,a)[b*H+row],fy(f,a)[b*KOUT+col]));}')
s=s.replace('if(mode>2||','if(mode>3||')
s=s.replace('\\\"factor_float_bytes\\\":%u,','\\\"factor_float_bytes\\\":%u,\\\"native_input_storage_bits\\\":%u,\\\"original_fp32_factor_bytes_per_step\\\":%u,')
s=s.replace('PARAMS,sizeof(Record),4u,4*(H+KOUT),','PARAMS,sizeof(Record),4u,(unsigned)(8*sizeof(XFactor)),4u*(ACC*MB*(H+H+KOUT)+H+KOUT),4*(H+KOUT),')
(root/'width_replay.c').write_text(s)
