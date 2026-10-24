import ctypes,time,json,sys
lib=ctypes.CDLL(None,use_errno=True)
lib.mmap.restype=ctypes.c_void_p
lib.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
lib.madvise.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
lib.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
lib.memset.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
disabled=int(sys.argv[1]);assert lib.prctl(41,disabled,0,0,0)==0
size=512<<20;times=[]
for i in range(2):
 addr=lib.mmap(None,size,3,34,-1,0);assert addr not in [None,ctypes.c_void_p(-1).value]
 assert lib.madvise(addr,size,14)==0
 t=time.monotonic();lib.memset(addr,0,size);elapsed=time.monotonic()-t;times.append(elapsed)
 print(json.dumps({'disabled':disabled,'iteration':i,'seconds':elapsed}),flush=True)
 assert lib.munmap(addr,size)==0
