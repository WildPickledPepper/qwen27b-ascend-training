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
_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24, "Requires binary32");

typedef struct { float w,m,v; } State;
typedef struct { uint32_t t,i,j,phase; State prior; float gradient; } Frame;
typedef struct { uint32_t t,i,age; State value; } Entry;
typedef struct {
    uint32_t p,t,q,tau,mode,filled,top,maxdepth,age,status;
    uint64_t calls,hits,updates,cap,checksum;
    float invp;
    State returned;
} Run;
typedef struct {
    float b1,b2,o1,o2,eps,lr,wd,initial_step,initial_offset,one;
    uint32_t c1[9],c2[9];
} Constants;
static const Constants K={.9f,.999f,.1f,.001f,1e-8f,.01f,.01f,.125f,.5f,1.f,C1_WORDS,C2_WORDS};

static inline float add(float a,float b) { volatile float r=a+b; return r; }
static inline float mul(float a,float b) { volatile float r=a*b; return r; }
static inline float divide(float a,float b) { volatile float r=a/b; return r; }
static inline float word_float(uint32_t u) { float f; memcpy(&f,&u,4);return f; }
static inline float initial(uint32_t i) { return add(mul((float)(i+1),K.initial_step),-K.initial_offset); }
static inline float coefficient(Run*r,uint32_t i,uint32_t j) {return i==j?add(K.one,r->invp):r->invp;}

__attribute__((noinline)) static State transition(State old,float g,uint32_t t) {
    State n;
    n.m=add(mul(K.b1,old.m),mul(K.o1,g));
    n.v=add(mul(K.b2,old.v),mul(K.o2,mul(g,g)));
    float denominator=add(__builtin_sqrtf(divide(n.v,word_float(K.c2[t]))),K.eps);
    float d=divide(divide(n.m,word_float(K.c1[t])),denominator);
    n.w=add(mul(old.w,add(K.one,-mul(K.lr,K.wd))),-mul(K.lr,d));
    return n;
}

static inline Frame*frames(Run*r) { return (Frame*)(r+1); }
static inline void*cache(Run*r) { return frames(r)+(r->t+1); }
static inline size_t cache_bytes(Run*r) { return r->mode==2?r->q*sizeof(Entry):r->mode==3?r->q*sizeof(State):0; }
static inline float*output_weights(Run*r) { return (float*)((unsigned char*)cache(r)+cache_bytes(r)); }

static int lookup(Run*r,uint32_t t,uint32_t i,State*out) {
    if(r->mode==2) {
        Entry*e=cache(r);
        for(uint32_t n=0;n<r->q;n++)if(e[n].t && e[n].t==t && e[n].i==i) {
            e[n].age=++r->age;*out=e[n].value;r->hits++;return 1;
        }
    } else if(r->mode==3 && t==r->tau && i<r->filled) {
        *out=((State*)cache(r))[i];r->hits++;return 1;
    }
    return 0;
}
static void insert(Run*r,uint32_t t,uint32_t i,State value) {
    if(r->mode!=2 || !r->q)return;
    Entry*e=cache(r);uint32_t slot=0;
    for(uint32_t n=0;n<r->q;n++) {
        if(!e[n].t){slot=n;break;}
        if(e[n].age<e[slot].age)slot=n;
    }
    e[slot]=(Entry){t,i,++r->age,value};
}

/* Explicit heap-arena DFS frames, no C recursion, no full historical vector. */
__attribute__((noinline)) static int query(Run*r,uint32_t t,uint32_t i) {
    Frame*s=frames(r);r->top=0;s[0]=(Frame){.t=t,.i=i};
    for(;;) {
        Frame*f=&s[r->top];
        if(r->top+1>r->maxdepth)r->maxdepth=r->top+1;
        if(f->phase==0) {
            if(r->calls>=r->cap){r->status=1;return 0;}
            r->calls++;
            if(lookup(r,f->t,f->i,&r->returned))goto returned;
            if(!f->t){r->returned=(State){initial(f->i),0.f,0.f};goto returned;}
            f->phase=1;s[++r->top]=(Frame){.t=f->t-1,.i=f->i};continue;
        }
        if(f->phase==1){f->prior=r->returned;f->gradient=0.f;f->j=0;f->phase=2;}
        if(f->phase==2) {
            if(f->j==r->p) {
                r->returned=transition(f->prior,f->gradient,f->t);r->updates++;
                insert(r,f->t,f->i,r->returned);goto returned;
            }
            if(f->j==f->i) {
                f->gradient=add(f->gradient,mul(coefficient(r,f->i,f->j),f->prior.w));f->j++;continue;
            }
            f->phase=3;s[++r->top]=(Frame){.t=f->t-1,.i=f->j};continue;
        }
        if(f->phase==3) {
            f->gradient=add(f->gradient,mul(coefficient(r,f->i,f->j),r->returned.w));f->j++;f->phase=2;continue;
        }
returned:
        if(!r->top)return 1;
        r->top--;
    }
}

static inline void record(Run*r,State v) {
    uint32_t words[3];memcpy(words,&v,sizeof(words));
    for(unsigned k=0;k<3;k++){r->checksum^=words[k];r->checksum*=UINT64_C(1099511628211);}
}

__attribute__((noinline)) static void algorithm(Run*r,State*capture) {
    r->checksum=UINT64_C(1469598103934665603);r->invp=divide(K.one,(float)r->p);
    if(r->mode==0) {
        float*w=(float*)(r+1),*m=w+r->p,*v=m+r->p,*g=v+r->p;
        for(uint32_t i=0;i<r->p;i++){w[i]=initial(i);m[i]=v[i]=0.f;}
        for(uint32_t t=1;t<=r->t;t++) {
            for(uint32_t i=0;i<r->p;i++) {
                g[i]=0.f;
                for(uint32_t j=0;j<r->p;j++)g[i]=add(g[i],mul(coefficient(r,i,j),w[j]));
            }
            for(uint32_t i=0;i<r->p;i++) {
                State n=transition((State){w[i],m[i],v[i]},g[i],t);
                w[i]=n.w;m[i]=n.m;v[i]=n.v;r->updates++;
            }
        }
        for(uint32_t i=0;i<r->p;i++) {
            State st={w[i],m[i],v[i]};record(r,st);if(capture)capture[i]=st;
        }
    } else {
        if(r->mode==3)for(uint32_t i=0;i<r->q;i++) {
            if(!query(r,r->tau,i))return;
            ((State*)cache(r))[i]=r->returned;r->filled++;
        }
        for(uint32_t i=0;i<r->p;i++) {
            if(!query(r,r->t,i))return;
            output_weights(r)[i]=r->returned.w;record(r,r->returned);if(capture)capture[i]=r->returned;
        }
    }
}

static uint64_t nanos(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000u+t.tv_nsec;}
__attribute__((noinline)) static void dump_validation(Run*r,size_t request,const char*path) {
    /* Validation-only cold rerun with raw-output capture; never benchmark memory. */
    Run spec=*r;
    State*capture=malloc(spec.p*sizeof(State));if(!capture)exit(3);
    memset(r,0,request);r->p=spec.p;r->t=spec.t;r->q=spec.q;r->tau=spec.tau;r->mode=spec.mode;r->cap=spec.cap;
    algorithm(r,capture);
    if(r->status || r->checksum!=spec.checksum){fprintf(stderr,"cold validation mismatch\n");exit(4);}
    FILE*f=fopen(path,"wb");if(!f){perror("fopen");exit(3);}
    fwrite(capture,sizeof(State),r->p,f);fclose(f);free(capture);
}

int main(int argc,char**argv) {
    if(argc!=11){fprintf(stderr,"mode P T Q tau cap stack_bytes budget dump_path repeats\n");return 2;}
    unsigned mode=atoi(argv[1]),p=atoi(argv[2]),t=atoi(argv[3]),q=atoi(argv[4]),tau=atoi(argv[5]);
    uint64_t cap=strtoull(argv[6],0,10),stack=strtoull(argv[7],0,10),budget=strtoull(argv[8],0,10);unsigned repeats=atoi(argv[10]);
    if(!p||t>8||mode>3||q>p||!repeats||fegetround()!=FE_TONEAREST || (mode==3 && (!tau||tau>=t)))return 2;
    size_t cb=mode==2?q*sizeof(Entry):mode==3?q*sizeof(State):0;
    size_t request=sizeof(Run)+(mode==0?4*p*sizeof(float):(t+1)*sizeof(Frame)+cb+p*sizeof(float));
    /* Conservative allocator padding/header screening, then exact usable check. */
    size_t preflight=request+32+sizeof(K)+64+stack;
    if(budget && preflight>budget) {
        printf("{\"status\":\"budget_rejected\",\"request\":%zu,\"conservative_preflight\":%zu,\"budget\":%llu}\n",request,preflight,(unsigned long long)budget);return 0;
    }
    Run*r=calloc(1,request);if(!r)return 3;
    size_t usable=malloc_usable_size(r),charged=usable+16+sizeof(K)+64+stack;
    if(budget && charged>budget){free(r);printf("{\"status\":\"budget_rejected_after_allocator_check\"}\n");return 0;}
    uint64_t minimum=UINT64_MAX,maximum=0,sum=0;
    for(unsigned k=0;k<repeats;k++) {
        memset(r,0,request);r->p=p;r->t=t;r->q=q;r->tau=tau;r->mode=mode;r->cap=cap;
        uint64_t a=nanos();algorithm(r,NULL);uint64_t elapsed=nanos()-a;
        if(elapsed<minimum) minimum=elapsed;
        if(elapsed>maximum) maximum=elapsed;
        sum+=elapsed;
        if(r->status)break;
    }
    printf("{\"status\":\"%s\",\"mode\":%u,\"P\":%u,\"T\":%u,\"Q\":%u,\"tau\":%u,\"calls\":%llu,\"hits\":%llu,\"updates\":%llu,\"max_depth\":%u,\"checksum\":\"%016llx\",\"nanoseconds_min\":%llu,\"nanoseconds_max\":%llu,\"nanoseconds_mean\":%.3f,\"request_bytes\":%zu,\"malloc_usable_bytes\":%zu,\"allocator_envelope_bytes\":%zu,\"constant_bytes\":%zu,\"static_call_stack_reserve\":%llu,\"charged_algorithm_bytes\":%zu,\"dense_current_state_payload\":%u,\"cache_bytes\":%zu,\"output_weight_bytes\":%u,\"header_bytes\":%zu,\"frame_bytes\":%zu,\"entry_bytes\":%zu}\n",
        r->status?"operation_capped":"complete",mode,p,t,q,tau,(unsigned long long)r->calls,(unsigned long long)r->hits,(unsigned long long)r->updates,r->maxdepth,(unsigned long long)r->checksum,(unsigned long long)minimum,(unsigned long long)maximum,(double)sum/(r->status?1:repeats),request,usable,usable+16,sizeof(K)+64,(unsigned long long)stack,charged,12*p,cb,4*p,sizeof(Run),sizeof(Frame),sizeof(Entry));
    if(!r->status && strcmp(argv[9],"-")!=0)dump_validation(r,request,argv[9]);
    free(r);return 0;
}
