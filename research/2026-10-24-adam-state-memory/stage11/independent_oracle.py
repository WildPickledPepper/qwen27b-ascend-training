"""Independent NumPy elementwise FP32 oracle; never calls C or BLAS GEMM."""
from pathlib import Path
import json, struct
import numpy as np
D,H,O,MB,ACC=64,64,64,4,2
P=D*H+H+H*O+O
B1=D*H;W2=B1+H;B2=W2+H*O
f=np.float32

def mix(v):
    v=np.asarray(v,dtype=np.uint32)
    v=v^(v>>np.uint32(16));v=v*np.uint32(0x7feb352d)
    v=v^(v>>np.uint32(15));v=v*np.uint32(0x846ca68b)
    return v^(v>>np.uint32(16))

def signed(v):
    return ((v&np.uint32(65535)).astype(np.float32)-f(32768))/f(32768)

def initialize(seed):
    i=np.arange(P,dtype=np.uint32)
    w=signed(mix(np.uint32(seed)^(i*np.uint32(0x9e3779b9))^np.uint32(0xa011)))
    w*=np.where(i<W2,f(.1),f(.05));w[B1:W2]=f(.01);w[B2:]=f(0)
    return w

def oracle(seed,steps,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    w=initialize(seed);m=np.zeros(P,dtype=np.float32);v=m.copy()
    states=[];gradients=[];factors=[];losses=[]
    one=f(1);beta1=f(.9);beta2=f(.999);o1=f(.1);o2=f(.001);lr=f(.003);wd=f(.01);eps=f(1e-8)
    # Match the explicitly declared bias-correction reference: binary64 decimal powers, rounded once.
    c1=np.array([1-.9**t for t in range(steps+1)],dtype=np.float32)
    c2=np.array([1-.999**t for t in range(steps+1)],dtype=np.float32)
    for t in range(1,steps+1):
        g=np.zeros(P,dtype=np.float32);bg=np.zeros(H+O,dtype=np.float32);parts=[[],[],[],[]];loss=0.
        for a in range(ACC):
            b=np.arange(MB,dtype=np.uint32)[:,None];i=np.arange(D,dtype=np.uint32)[None,:]
            u=mix(np.uint32(seed)^np.uint32((t*0x9e3779b9)&0xffffffff)^np.uint32((a*0x85ebca6b)&0xffffffff)^(b*np.uint32(0xc2b2ae35))^(i*np.uint32(0x27d4eb2f)))
            x=((np.uint32(0x3f00)|(u&np.uint32(127))|((u&np.uint32(128))<<np.uint32(8)))<<np.uint32(16)).view(np.float32)
            pre=np.zeros((MB,H),dtype=np.float32)
            w1=w[:B1].reshape(D,H);w2=w[W2:B2].reshape(H,O)
            for k in range(D):pre=pre+x[:,k,None]*w1[k,None,:]
            pre=pre+w[None,B1:W2];act=np.where(pre>f(0),pre,f(0))
            y=np.zeros((MB,O),dtype=np.float32)
            for k in range(H):y=y+act[:,k,None]*w2[k,None,:]
            y=y+w[None,B2:];q=x[:,(np.arange(O)+7)%D]
            tgt=(f(.25)*x[:,np.arange(O)%D]+f(.5)*(q*q))+f(-.1)
            err=y-tgt;dy=err/f(ACC*MB*O)
            loss+=np.sum(err.astype(np.float64)**2)/(2.*ACC*MB*O)
            dz=np.zeros((MB,H),dtype=np.float32)
            for k in range(O):dz=dz+dy[:,k,None]*w2[None,:,k]
            dz=np.where(act>f(0),dz,f(0))
            bm=np.zeros(H,dtype=np.float32);bo=np.zeros(O,dtype=np.float32)
            for k in range(MB):bm=bm+dz[k];bo=bo+dy[k]
            bg[:H]=bg[:H]+bm;bg[H:]=bg[H:]+bo
            g1=np.zeros((D,H),dtype=np.float32);g2=np.zeros((H,O),dtype=np.float32)
            for k in range(MB):g1=g1+x[k,:,None]*dz[k,None,:];g2=g2+act[k,:,None]*dy[k,None,:]
            g[:B1]=g[:B1]+g1.ravel();g[W2:B2]=g[W2:B2]+g2.ravel()
            for group,values in zip(parts,[x.ravel(),act.ravel(),dz.ravel(),dy.ravel()]):group.append(values)
        g[B1:W2]=bg[:H];g[B2:]=bg[H:]
        m=beta1*m+o1*g;v=beta2*v+o2*(g*g)
        den=np.sqrt(v/c2[t])+eps;delta=(m/c1[t])/den
        w=w*(one-lr*wd)-lr*delta
        states.append(np.stack([w,m,v],axis=1).astype('<f4').tobytes())
        gradients.append(g.astype('<f4').tobytes())
        factors.append(struct.pack('<4I',t,ACC,MB,0xF320001)+np.concatenate([v for group in parts for v in group]+[bg]).astype('<f4').tobytes())
        losses.append(float(loss))
    for suffix,records in [('states',states),('gradients',gradients),('factors',factors)]:
        out.with_suffix('.'+suffix).write_bytes(b''.join(records))
    return dict(seed=seed,steps=steps,parameters=P,last_preupdate_halfMSE=losses[-1],numpy=np.__version__,scope='Independent elementwise ordered FP32 implementation; same prescribed operation order, not hardware GEMM or PyTorch parity.')

if __name__=='__main__':
    import sys
    print(json.dumps(oracle(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]),indent=2))
