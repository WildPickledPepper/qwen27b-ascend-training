// ACL host wrapper only. Author compression/decompression kernels are unchanged.
#include "snec_utils.h"
#include <acl/acl.h>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" void enec_compress(Header*,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);
extern "C" void enec_merge(Header*,void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t);
extern "C" void enec_decompress(Header*,void*,uint8_t*,uint8_t*);
static aclrtContext context = nullptr;
static aclrtStream stream = nullptr;
static char error_message[1024];
static void check(aclError c) { if (c) throw std::runtime_error("ACL error " + std::to_string(c)); }
extern "C" const char* enec_error() { return error_message; }
extern "C" int enec_init(int device) {
    try {
        if (device != 2 && device != 3) throw std::runtime_error("Only physical cards 2 and 3 authorized");
        check(aclInit(nullptr)); check(aclrtSetDevice(device));
        check(aclrtCreateContext(&context, device)); check(aclrtSetOpExecuteTimeOut(30));
        check(aclrtCreateStream(&stream)); return 0;
    } catch (const std::exception& e) { snprintf(error_message,sizeof(error_message),"%s",e.what()); return 1; }
}
extern "C" int enec_close() {
    try {
        if (stream) check(aclrtDestroyStream(stream)); stream = nullptr;
        if (context) check(aclrtDestroyContext(context)); context = nullptr;
        check(aclFinalize()); return 0;
    } catch (const std::exception& e) { snprintf(error_message,sizeof(error_message),"%s",e.what()); return 1; }
}
// metrics: compression, merge, decompression ms; live device bytes; free,total HBM.
extern "C" int enec_roundtrip(const void* source, uint32_t bytes, void* restored,
                              void* encoded_host, uint32_t host_capacity,
                              uint32_t* encoded_bytes, double* metrics) {
    std::vector<void*> allocations;
    aclrtEvent begin = nullptr, end = nullptr;
    try {
        if (!stream || !bytes || bytes % 2 || bytes > (256u << 20)) throw std::runtime_error("Invalid BF16 chunk");
        check(aclrtSetCurrentContext(context));
        uint32_t padded = (bytes + 32767u) / 32768u * 32768u;
        Header h{}; h.dataBlockSize = 32768; h.dataBlockNum = padded / 32768;
        h.threadBlockNum = 48; h.totalUncompressedBytes_Origin = bytes;
        h.totalUncompressedBytes = padded; h.tileLength = 16; h.dataType = 0;
        h.mblLength = 4; h.options = 3;
        size_t live = 0;
        auto allocate = [&](size_t n) { void* q = nullptr; check(aclrtMalloc(&q,n,ACL_MEM_MALLOC_HUGE_FIRST));
            allocations.push_back(q); live += n; check(aclrtMemset(q,n,0,n)); return static_cast<uint8_t*>(q); };
        size_t cap = getFinalbufferSize(padded,1024,32768);
        auto src = allocate(padded), scratch = allocate(cap), packed = allocate(cap);
        auto histogram = allocate(48*256*4), sizes = allocate(48*32), output = allocate(padded);
        check(aclrtMemcpy(src,padded,source,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
        check(aclrtCreateEvent(&begin)); check(aclrtCreateEvent(&end));
        auto timed = [&](auto launch) { check(aclrtRecordEvent(begin,stream)); launch(); check(aclrtRecordEvent(end,stream));
            check(aclrtSynchronizeStream(stream)); float ms=0; check(aclrtEventElapsedTime(&ms,begin,end)); return double(ms); };
        metrics[0] = timed([&]{ enec_compress(&h,stream,src,scratch,packed,histogram,sizes); });
        uint32_t blockBytes = 32768*((h.dataBlockNum+47)/48);
        metrics[1] = timed([&]{ enec_merge(&h,stream,scratch,packed,histogram,sizes,blockBytes); });
        Header merged{}; check(aclrtMemcpy(&merged,sizeof(merged),packed,sizeof(merged),ACL_MEMCPY_DEVICE_TO_HOST));
        if (merged.dataType != 0 || merged.totalCompressedBytes < 32 || merged.totalCompressedBytes > cap ||
            merged.totalCompressedBytes > host_capacity || merged.totalUncompressedBytes_Origin != bytes)
            throw std::runtime_error("Invalid compressed header/capacity");
        *encoded_bytes = merged.totalCompressedBytes;
        metrics[2] = timed([&]{ enec_decompress(&merged,stream,packed,output); });
        check(aclrtMemcpy(restored,bytes,output,bytes,ACL_MEMCPY_DEVICE_TO_HOST));
        check(aclrtMemcpy(encoded_host,host_capacity,packed,*encoded_bytes,ACL_MEMCPY_DEVICE_TO_HOST));
        metrics[3] = double(live); size_t free=0,total=0; check(aclrtGetMemInfo(ACL_HBM_MEM,&free,&total));
        metrics[4] = double(free); metrics[5] = double(total);
        check(aclrtDestroyEvent(begin)); begin=nullptr; check(aclrtDestroyEvent(end)); end=nullptr;
        for (auto q:allocations) check(aclrtFree(q)); return 0;
    } catch (const std::exception& e) {
        snprintf(error_message,sizeof(error_message),"%s",e.what());
        if(begin) aclrtDestroyEvent(begin); if(end) aclrtDestroyEvent(end);
        for(auto q:allocations) aclrtFree(q); return 1;
    }
}

// Reload a saved frame; no original tensor is supplied to the decoder.
extern "C" int enec_restore(const void* encoded, uint32_t bytes, void* restored, uint32_t output_capacity) {
    void* packed = nullptr; void* output = nullptr;
    try {
        check(aclrtSetCurrentContext(context));
        if (!stream || bytes < sizeof(Header)) throw std::runtime_error("Short saved frame");
        Header h{}; memcpy(&h,encoded,sizeof(h));
        if(h.dataType != 0 || h.dataBlockSize != 32768 || h.threadBlockNum != 48 || h.tileLength != 16 ||
           h.totalCompressedBytes != bytes || h.totalUncompressedBytes_Origin > output_capacity ||
           h.totalUncompressedBytes > (256u<<20) || h.totalUncompressedBytes < h.totalUncompressedBytes_Origin ||
           uint64_t(h.dataBlockNum)*32768 != h.totalUncompressedBytes)
            throw std::runtime_error("Saved ENEC header invalid");
        size_t cap = getFinalbufferSize(h.totalUncompressedBytes,1024,32768);
        if(bytes > cap) throw std::runtime_error("Saved ENEC capacity invalid");
        check(aclrtMalloc(&packed,cap,ACL_MEM_MALLOC_HUGE_FIRST)); check(aclrtMemset(packed,cap,0,cap));
        check(aclrtMalloc(&output,h.totalUncompressedBytes,ACL_MEM_MALLOC_HUGE_FIRST));
        check(aclrtMemset(output,h.totalUncompressedBytes,0,h.totalUncompressedBytes));
        check(aclrtMemcpy(packed,cap,encoded,bytes,ACL_MEMCPY_HOST_TO_DEVICE));
        enec_decompress(&h,stream,static_cast<uint8_t*>(packed),static_cast<uint8_t*>(output));
        check(aclrtSynchronizeStream(stream));
        check(aclrtMemcpy(restored,output_capacity,output,h.totalUncompressedBytes_Origin,ACL_MEMCPY_DEVICE_TO_HOST));
        check(aclrtFree(packed)); packed=nullptr; check(aclrtFree(output)); output=nullptr; return 0;
    } catch(const std::exception& e) {
        snprintf(error_message,sizeof(error_message),"%s",e.what());
        if(packed) aclrtFree(packed); if(output) aclrtFree(output); return 1;
    }
}
