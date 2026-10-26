// Host bridge to pinned author kernels. All weight/state buffers are supplied by PyTorch NPU tensors.
#include "snec_utils.h"
#include <acl/acl.h>
#include <cstring>
#include <stdexcept>
extern "C" void enec_compress(Header*,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
extern "C" void enec_merge(Header*,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void enec_decompress(Header*,void*,uint8_t*,uint8_t*);
static thread_local char error_message[1024];
static void check(aclError e) { if(e) throw std::runtime_error("ACL " + std::to_string(e)); }
extern "C" const char* online_error() { return error_message; }
extern "C" uint64_t online_capacity(uint32_t n) {
    uint32_t padded=(n+32767u)/32768u*32768u;
    return getFinalbufferSize(padded,1024,32768);
}
extern "C" int online_decompress(void* stream, int logical_device, const void* header,
                                 void* compressed, void* output) {
    try {
        int device=-1; check(aclrtGetDevice(&device));
        if(device != logical_device || device<0 || device>1) throw std::runtime_error("Wrong current logical NPU");
        Header h{}; memcpy(&h,header,sizeof(h));
        if(h.dataType!=0 || h.dataBlockSize!=32768 || h.threadBlockNum!=48 || h.tileLength!=16)
            throw std::runtime_error("Unsupported saved ENEC header");
        enec_decompress(&h,stream,static_cast<uint8_t*>(compressed),static_cast<uint8_t*>(output));
        return 0; // Same PyTorch stream orders the decoder and its consumer.
    } catch(const std::exception& e) { snprintf(error_message,sizeof(error_message),"%s",e.what()); return 1; }
}
extern "C" int online_compress(void* stream, int logical_device, void* source, uint32_t n,
                               void* scratch, void* packed, void* histogram, void* sizes,
                               void* header_out) {
    try {
        int device=-1; check(aclrtGetDevice(&device));
        if(device != logical_device || device<0 || device>1) throw std::runtime_error("Wrong current logical NPU");
        if(!n || n%32768 || n>(64u<<20)) throw std::runtime_error("Require <=64 MiB aligned BF16 chunk");
        Header h{};h.dataBlockSize=32768;h.dataBlockNum=n/32768;h.threadBlockNum=48;
        h.totalUncompressedBytes_Origin=n;h.totalUncompressedBytes=n;
        h.tileLength=16;h.dataType=0;h.mblLength=4;h.options=3;
        enec_compress(&h,stream,static_cast<uint8_t*>(source),static_cast<uint8_t*>(scratch),
                      static_cast<uint8_t*>(packed),static_cast<uint8_t*>(histogram),static_cast<uint8_t*>(sizes));
        enec_merge(&h,stream,static_cast<uint8_t*>(scratch),static_cast<uint8_t*>(packed),
                   static_cast<uint8_t*>(histogram),static_cast<uint8_t*>(sizes),32768*((h.dataBlockNum+47)/48));
        check(aclrtSynchronizeStream(stream));
        check(aclrtMemcpy(header_out,sizeof(Header),packed,sizeof(Header),ACL_MEMCPY_DEVICE_TO_HOST));
        return 0;
    } catch(const std::exception& e) { snprintf(error_message,sizeof(error_message),"%s",e.what()); return 1; }
}
