#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <malloc.h>
#include <fenv.h>
#include "bias_constants.h"
#ifndef D
#define D 32u
#endif
#ifndef H
#define H 128u
#endif
#ifndef KOUT
#define KOUT 16u
#endif
#ifndef MB
#define MB 4u
#endif
#define ACC 2u
#define MAX_STEPS 16u
#define W1 0u
#define B1 (D*H)
#define W2 (B1+H)
#define B2 (W2+H*KOUT)
#define PARAMS (B2+KOUT)
#define PER_MB (MB*(D+H+H+KOUT))
#define FACTOR_FLOATS (ACC*PER_MB+H+KOUT)
#define FORMAT_VERSION 0xF320001u
_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24,"binary32 required");
typedef struct { float w,m,v; } State;
#ifndef MIXED_X
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
} Record;
typedef struct {
    uint32_t mode,seed,steps,block,capacity,completed,status,rejected_step;
    uint64_t capture_calls,rebuild_calls,reconstructed_coordinates,gradient_products,moment_transitions;
    uint64_t gradient_digest,state_digest,factor_digest;
    uint64_t reject_weights_before,reject_weights_after,reject_history_before,reject_history_after;
    double last_loss;
} Run;
typedef struct {
    State*states;float*gradients;Record*factors;
    uint64_t gradient_hash[MAX_STEPS],state_hash[MAX_STEPS];
} Proof;
typedef struct {float b1,b2,o1,o2,eps,lr,wd,one;uint32_t c1[17],c2[17];} Constants;
static const Constants C={.9f,.999f,.1f,.001f,1e-8f,.003f,.01f,1.f,C1_WORDS,C2_WORDS};

static inline float add(float a,float b){volatile float r=a+b;return r;}
static inline float mul(float a,float b){volatile float r=a*b;return r;}
static inline float divide(float a,float b){volatile float r=a/b;return r;}
static inline float word_float(uint32_t u){float f;memcpy(&f,&u,4);return f;}
static inline uint32_t word(float f){uint32_t u;memcpy(&u,&f,4);return u;}
static inline uint64_t hash_word(uint64_t h,uint32_t u){return (h^u)*UINT64_C(1099511628211);}
static uint64_t hash_bytes(const void*ptr,size_t n){const unsigned char*b=ptr;uint64_t h=UINT64_C(1469598103934665603);for(size_t i=0;i<n;i++)h=(h^b[i])*UINT64_C(1099511628211);return h;}
static inline uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
static inline float signed_unit(uint32_t x){return divide(add((float)(x&65535u),-32768.f),32768.f);}
static inline float initial(uint32_t seed,uint32_t i){if(i>=B1 && i<W2)return .01f;if(i>=B2)return 0.f;return mul(signed_unit(mix(seed^(i*0x9e3779b9u)^0xa011u)),i<W2?.1f:.05f);}
static inline float load_x(XFactor x){
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
static inline float*weights(Run*r){return (float*)(r+1);}
static inline Record*records(Run*r){return (Record*)(weights(r)+PARAMS);}
static inline float*workspace(Run*r){return (float*)(records(r)+(r->mode?r->capacity:1));}
static inline XFactor*fx(Record*f,uint32_t a){return f->x+a*MB*D;}
static inline float*fa(Record*f,uint32_t a){return f->activation+a*MB*H;}
static inline float*fz(Record*f,uint32_t a){return f->dz+a*MB*H;}
static inline float*fy(Record*f,uint32_t a){return f->dy+a*MB*KOUT;}
static inline float*bias(Record*f){return f->bias_gradient;}

/* Captures actual historical activations/backprop factors before any weight update. */
__attribute__((noinline)) static void capture(Run*r,Record*f,uint32_t t){
    float*w=weights(r);f->step=t;f->accumulation=ACC;f->microbatch=MB;f->format=FORMAT_VERSION;
    for(uint32_t i=0;i<H+KOUT;i++)bias(f)[i]=0.f;
    double loss=0;
    for(uint32_t a=0;a<ACC;a++){
        XFactor*x=fx(f,a);float*activation=fa(f,a),*dz=fz(f,a),*dy=fy(f,a);
        for(uint32_t b=0;b<MB;b++)for(uint32_t i=0;i<D;i++)x[b*D+i]=store_x(sample(r->seed,t,a,b,i));
        for(uint32_t b=0;b<MB;b++)for(uint32_t h=0;h<H;h++){
            float z=0.f;
            for(uint32_t i=0;i<D;i++)z=add(z,mul(load_x(x[b*D+i]),w[W1+i*H+h]));
            z=add(z,w[B1+h]);activation[b*H+h]=z>0.f?z:0.f;
        }
        for(uint32_t b=0;b<MB;b++)for(uint32_t j=0;j<KOUT;j++){
            float z=0.f;
            for(uint32_t h=0;h<H;h++)z=add(z,mul(activation[b*H+h],w[W2+h*KOUT+j]));
            z=add(z,w[B2+j]);float error=add(z,-target(x+b*D,j));
            dy[b*KOUT+j]=divide(error,(float)(ACC*MB*KOUT));loss+=(double)error*(double)error/(2.0*ACC*MB*KOUT);
        }
        for(uint32_t b=0;b<MB;b++)for(uint32_t h=0;h<H;h++){
            float z=0.f;
            for(uint32_t j=0;j<KOUT;j++)z=add(z,mul(dy[b*KOUT+j],w[W2+h*KOUT+j]));
            dz[b*H+h]=activation[b*H+h]>0.f?z:0.f;
        }
        /* Bias is not silently omitted: accumulate MB internally, then ACC groups. */
        for(uint32_t h=0;h<H;h++){
            float micro=0.f;for(uint32_t b=0;b<MB;b++)micro=add(micro,dz[b*H+h]);bias(f)[h]=add(bias(f)[h],micro);
        }
        for(uint32_t j=0;j<KOUT;j++){
            float micro=0.f;for(uint32_t b=0;b<MB;b++)micro=add(micro,dy[b*KOUT+j]);bias(f)[H+j]=add(bias(f)[H+j],micro);
        }
    }
    r->capture_calls++;r->last_loss=loss;
}

/* Full output and tiles use this same explicit K reduction and MB grouping. */
__attribute__((noinline)) static void reconstruct(Run*r,Record*f,uint32_t lo,uint32_t count,float*out){
    for(uint32_t n=0;n<count;n++){
        uint32_t i=lo+n;
        if(i>=B1 && i<W2){out[n]=bias(f)[i-B1];continue;}
        if(i>=B2){out[n]=bias(f)[H+i-B2];continue;}
        float total=0.f;
        for(uint32_t a=0;a<ACC;a++){
            float micro=0.f;
            if(i<W2){uint32_t row=i/H,col=i%H;for(uint32_t k=0;k<MB;k++){uint32_t b=r->mode==3?MB-1-k:k;micro=add(micro,mul(load_x(fx(f,a)[b*D+row]),fz(f,a)[b*H+col]));}}
            else {uint32_t row=(i-W2)/KOUT,col=(i-W2)%KOUT;for(uint32_t k=0;k<MB;k++){uint32_t b=r->mode==3?MB-1-k:k;micro=add(micro,mul(fa(f,a)[b*H+row],fy(f,a)[b*KOUT+col]));}}
            total=add(total,micro);
        }
        out[n]=total;r->gradient_products+=(uint64_t)MB*ACC;
    }
    r->rebuild_calls++;r->reconstructed_coordinates+=count;
}

static inline void moment(float*m,float*v,float g){*m=add(mul(C.b1,*m),mul(C.o1,g));*v=add(mul(C.b2,*v),mul(C.o2,mul(g,g)));}
__attribute__((noinline)) static float update_weight(float w,float m,float v,uint32_t t){
    float den=add(__builtin_sqrtf(divide(v,word_float(C.c2[t]))),C.eps);
    float d=divide(divide(m,word_float(C.c1[t])),den);
    return add(mul(w,add(C.one,-mul(C.lr,C.wd))),-mul(C.lr,d));
}
static inline void record_value(Run*r,Proof*p,uint32_t t,uint32_t i,float g,float m,float v){
    State st={weights(r)[i],m,v};r->gradient_digest=hash_word(r->gradient_digest,word(g));
    r->state_digest=hash_word(hash_word(hash_word(r->state_digest,word(st.w)),word(m)),word(v));
    if(p){p->states[(t-1)*PARAMS+i]=st;p->gradients[(t-1)*PARAMS+i]=g;}
}

__attribute__((noinline)) static void algorithm(Run*r,Proof*proof){
    float*w=weights(r),*ws=workspace(r);for(uint32_t i=0;i<PARAMS;i++)w[i]=initial(r->seed,i);
    r->gradient_digest=r->state_digest=r->factor_digest=UINT64_C(1469598103934665603);
    if(!r->mode)for(uint32_t i=0;i<2*PARAMS;i++)ws[i]=0.f;
    for(uint32_t t=1;t<=r->steps;t++){
        if(r->mode && t>r->capacity){
            /* Atomic rejection before capture, inputs, factors or model update. */
            r->reject_weights_before=hash_bytes(w,4*PARAMS);r->reject_history_before=hash_bytes(records(r),sizeof(Record)*r->capacity);
            r->status=1;r->rejected_step=t;
            r->reject_weights_after=hash_bytes(w,4*PARAMS);r->reject_history_after=hash_bytes(records(r),sizeof(Record)*r->capacity);return;
        }
        Record*f=records(r)+(r->mode?t-1:0);capture(r,f,t);
        r->factor_digest=hash_word(hash_word(r->factor_digest,(uint32_t)hash_bytes(f,sizeof(Record))),(uint32_t)(hash_bytes(f,sizeof(Record))>>32));
        if(proof)proof->factors[t-1]=*f;
        if(!r->mode){
            float*m=ws,*v=m+PARAMS,*g=v+PARAMS;reconstruct(r,f,0,PARAMS,g);
            for(uint32_t i=0;i<PARAMS;i++){moment(&m[i],&v[i],g[i]);w[i]=update_weight(w[i],m[i],v[i],t);record_value(r,proof,t,i,g[i],m[i],v[i]);}
            r->moment_transitions+=PARAMS;
        }else{
            float*m=ws,*v=m+r->block,*g=v+r->block;
            for(uint32_t lo=0;lo<PARAMS;lo+=r->block){
                uint32_t count=PARAMS-lo;if(count>r->block)count=r->block;
                for(uint32_t n=0;n<count;n++)m[n]=v[n]=0.f;
                for(uint32_t s=0;s<t;s++){
                    reconstruct(r,records(r)+s,r->mode==1?0:lo,r->mode==1?PARAMS:count,g);
                    for(uint32_t n=0;n<count;n++){float gg=g[r->mode==1?lo+n:n];moment(&m[n],&v[n],gg);}
                    r->moment_transitions+=count;
                }
                for(uint32_t n=0;n<count;n++){float gg=g[r->mode==1?lo+n:n];w[lo+n]=update_weight(w[lo+n],m[n],v[n],t);record_value(r,proof,t,lo+n,gg,m[n],v[n]);}
            }
        }
        r->completed=t;if(proof){proof->gradient_hash[t-1]=r->gradient_digest;proof->state_hash[t-1]=r->state_digest;}
    }
}

static uint64_t nanos(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return(uint64_t)t.tv_sec*1000000000u+t.tv_nsec;}
__attribute__((noinline)) static void validate_dump(Run*r,size_t request,const char*prefix){
    Run spec=*r;Proof proof={0};proof.states=malloc((size_t)spec.steps*PARAMS*sizeof(State));proof.gradients=malloc((size_t)spec.steps*PARAMS*4);proof.factors=malloc((size_t)spec.steps*sizeof(Record));
    if(!proof.states||!proof.gradients||!proof.factors)exit(3);
    memset(r,0,request);r->mode=spec.mode;r->seed=spec.seed;r->steps=spec.steps;r->block=spec.block;r->capacity=spec.capacity;algorithm(r,&proof);
    if(r->status!=spec.status||r->completed!=spec.completed||r->gradient_digest!=spec.gradient_digest||r->state_digest!=spec.state_digest||r->factor_digest!=spec.factor_digest){fprintf(stderr,"cold validation mismatch\n");exit(4);}
    char path[512];FILE*f;
    snprintf(path,sizeof(path),"%s.states",prefix);f=fopen(path,"wb");if(!f)exit(3);fwrite(proof.states,sizeof(State),(size_t)r->completed*PARAMS,f);fclose(f);
    snprintf(path,sizeof(path),"%s.gradients",prefix);f=fopen(path,"wb");if(!f)exit(3);fwrite(proof.gradients,4,(size_t)r->completed*PARAMS,f);fclose(f);
    snprintf(path,sizeof(path),"%s.factors",prefix);f=fopen(path,"wb");if(!f)exit(3);fwrite(proof.factors,sizeof(Record),r->completed,f);fclose(f);
    free(proof.states);free(proof.gradients);free(proof.factors);
}

int main(int argc,char**argv){
    if(argc!=10){fprintf(stderr,"mode seed steps block capacity stack_bytes optimizer_budget prefix repeats\n");return 2;}
    uint32_t mode=atoi(argv[1]),seed=atoi(argv[2]),steps=atoi(argv[3]),block=atoi(argv[4]),capacity=atoi(argv[5]),repeats=atoi(argv[9]);
    uint64_t stack=strtoull(argv[6],0,10),budget=strtoull(argv[7],0,10);
    if(mode>3||!steps||steps>MAX_STEPS||!block||block>PARAMS||capacity>MAX_STEPS||!repeats||fegetround()!=FE_TONEAREST)return 2;
    size_t history_bytes=sizeof(Record)*(mode?capacity:1),work_bytes=mode?4*(2*block+(mode==1?PARAMS:block)):12*PARAMS;
    size_t request=sizeof(Run)+4*PARAMS+history_bytes+work_bytes;
    size_t preflight_optimizer=request+32+sizeof(C)+128+stack-4*PARAMS;
    if(budget && preflight_optimizer>budget){printf("{\"status\":\"budget_rejected\",\"optimizer_preflight_bytes\":%zu,\"budget\":%llu}\n",preflight_optimizer,(unsigned long long)budget);return 0;}
    Run*r=calloc(1,request);if(!r)return 3;size_t usable=malloc_usable_size(r),charged=usable+16+sizeof(C)+128+stack,opt_charged=charged-4*PARAMS;
    if(budget && opt_charged>budget){free(r);puts("{\"status\":\"budget_rejected_after_allocation\"}");return 0;}
    uint64_t sum=0,minimum=UINT64_MAX,maximum=0;
    for(uint32_t n=0;n<repeats;n++){
        memset(r,0,request);r->mode=mode;r->seed=seed;r->steps=steps;r->block=block;r->capacity=capacity;
        uint64_t a=nanos();algorithm(r,NULL);uint64_t elapsed=nanos()-a;sum+=elapsed;if(elapsed<minimum)minimum=elapsed;if(elapsed>maximum)maximum=elapsed;
    }
    printf("{\"status\":\"%s\",\"mode\":%u,\"seed\":%u,\"requested_steps\":%u,\"completed_steps\":%u,\"rejected_step\":%u,\"block\":%u,\"capacity_steps\":%u,\"parameters\":%u,\"record_bytes\":%zu,\"factor_float_bytes\":%u,\"native_input_storage_bits\":%u,\"original_fp32_factor_bytes_per_step\":%u,\"bias_vector_bytes_per_step\":%u,\"history_reserved_bytes\":%zu,\"history_used_bytes\":%zu,\"unused_history_bytes_charged\":%zu,\"workspace_bytes\":%zu,\"header_bytes\":%zu,\"current_weight_bytes\":%u,\"raw_FP32_m_v_bytes\":%u,\"request_bytes\":%zu,\"malloc_usable_bytes\":%zu,\"constant_reserve_bytes\":%zu,\"static_stack_reserve\":%llu,\"charged_algorithm_bytes\":%zu,\"optimizer_replacement_bytes\":%zu,\"nanoseconds_mean\":%.3f,\"nanoseconds_min\":%llu,\"nanoseconds_max\":%llu,\"capture_calls\":%llu,\"rebuild_calls\":%llu,\"reconstructed_coordinates\":%llu,\"gradient_products\":%llu,\"moment_transitions\":%llu,\"gradient_digest\":\"%016llx\",\"state_digest\":\"%016llx\",\"factor_digest\":\"%016llx\",\"atomic_weights_unchanged\":%s,\"atomic_history_unchanged\":%s,\"last_preupdate_halfMSE\":%.12g}\n",
        r->status?"capacity_rejected":"complete",mode,seed,steps,r->completed,r->rejected_step,block,capacity,PARAMS,sizeof(Record),4u,(unsigned)(8*sizeof(XFactor)),4u*(ACC*MB*(H+H+KOUT)+H+KOUT),4*(H+KOUT),history_bytes,(size_t)(mode?r->completed:1)*sizeof(Record),mode?history_bytes-r->completed*sizeof(Record):0,work_bytes,sizeof(Run),4*PARAMS,8*PARAMS,request,usable,sizeof(C)+128,(unsigned long long)stack,charged,opt_charged,(double)sum/repeats,(unsigned long long)minimum,(unsigned long long)maximum,(unsigned long long)r->capture_calls,(unsigned long long)r->rebuild_calls,(unsigned long long)r->reconstructed_coordinates,(unsigned long long)r->gradient_products,(unsigned long long)r->moment_transitions,(unsigned long long)r->gradient_digest,(unsigned long long)r->state_digest,(unsigned long long)r->factor_digest,r->reject_weights_before==r->reject_weights_after?"true":"false",r->reject_history_before==r->reject_history_after?"true":"false",r->last_loss);
    if(strcmp(argv[8],"-")) validate_dump(r,request,argv[8]);
    free(r);
    return 0;
}
