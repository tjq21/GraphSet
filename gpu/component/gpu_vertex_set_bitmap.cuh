#pragma once
#include <cstdint>
#include <cub/cub.cuh>

#include "gpu_const.cuh"
#include "utils.cuh"
#define DEBUG

__device__ constexpr uint32_t mask[32] = {0x0,      0x1,       0x3,        0x7,       0xF,       0x1F,       0x3F,       0x7F,
                               0xFF,     0x1FF,     0x3FF,      0x7FF,     0xFFF,     0x1FFF,     0x3FFF,     0x7FFF,
                               0xFFFF,   0x1FFFF,   0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,   0x3FFFFF,   0x7FFFFF,
                               0xFFFFFF, 0x1FFFFFF, 0x3FFFFFF,  0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF};

class GPUVertexSet_Bitmap{
public:
// TODO: lazy update?
    GPUVertexSet_Bitmap(const GPUVertexSet_Bitmap&) = delete;
    GPUVertexSet_Bitmap(const GPUVertexSet_Bitmap&&) = delete;
    GPUVertexSet_Bitmap& operator=(const GPUVertexSet_Bitmap&) = delete;

    __device__ void construct(uint32_t v_cnt){
        // vertex_count = v_cnt;
        size = (v_cnt + 31) >> 5;
        non_zero_cnt = 0;
        data = (uint32_t *)malloc(size * sizeof(uint32_t));
        // gpuErrchk(cudaMalloc((void **)&data, size * sizeof(uint32_t)));
    }

    __device__ void destroy(){
        free(data);
        // gpuErrchk(cudaFree(data));
    }

    __device__ void init(uint32_t input_size, uint32_t* input_data){
        __shared__ uint32_t tmp_val[THREADS_PER_BLOCK];
        
        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id

        uint32_t *output = tmp_val + wid * THREADS_PER_WARP;

        clear();
        for(uint32_t i = 0; i < input_size; i++){
            uint32_t id = input_data[i];
            atomicOr(&data[id >> 5], 1 << (id & 31));
        }
        non_zero_cnt = input_size;
    }

    __device__ int get_size() const {
        return non_zero_cnt;
    }
    __device__ void set_data_ptr(uint32_t *ptr){
        free(data);
        data = ptr;
    }
    
    __device__ void insert(uint32_t id){
        atomicOr(&data[id >> 5], 1 << (id & 31));
        __threadfence_block();
    }

    __device__ void insert_and_update(uint32_t id){
        uint32_t index = id >> 5;
        uint32_t tmp_data = data[index];
        uint32_t offset = 1 << (id & 31);
        if ((tmp_data & offset) == 0) {
            ++non_zero_cnt;
            atomicOr(&data[index], offset);
            // data[index] = tmp_data | offset;
        }
    }

    __device__ void erase(uint32_t id){
        atomicAnd(&data[id >> 5], ~(1 << (id & 31)));
        // data[id >> 5] &= ~(1 << (id & 31));
        __threadfence_block();
    }

    __device__ void erase_and_update(uint32_t id){
        uint32_t index = id >> 5;
        uint32_t tmp_data = data[index];
        uint32_t offset = 1 << (id & 31);
        if (tmp_data & offset) {
            --non_zero_cnt;
            atomicAnd(&data[index], ~offset);
            // data[index] = tmp_data & (~offset);
        }
    }
    
    inline __device__ void clear(){
        non_zero_cnt = 0;
        memset((void *)data, 0, size * sizeof(uint32_t));
    }
    
    inline __device__ bool has_data(uint32_t id) const {
        uint32_t tmp_data = data[id >> 5];
        uint32_t offset = 1 << (id & 31);
        return (tmp_data & offset);
    }

    static __device__ int subtraction_size(const GPUVertexSet_Bitmap& vset1, const GPUVertexSet_Bitmap& vset2, uint32_t min_vertex = 0xffffffff);

    __device__ void build_vertex_set(const GPUSchedule* schedule, const GPUVertexSet_Bitmap* vertex_set, uint32_t* input_data, uint32_t input_size, int prefix_id)
    {
        int father_id = schedule->get_father_prefix_id(prefix_id);
        if (father_id == -1)
        {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                init(input_size, input_data);
            __threadfence_block();
        }
        else
        {
            bool only_need_size = schedule->only_need_size[prefix_id];
            if(only_need_size) {
                if (threadIdx.x % THREADS_PER_WARP == 0)
                    init(input_size, input_data);
                __threadfence_block();
                if(input_size > vertex_set[father_id].get_size())
                    this->non_zero_cnt -= subtraction_size(*this, vertex_set[father_id]);
                else
                    this->non_zero_cnt = vertex_set[father_id].get_size() - subtraction_size(vertex_set[father_id], *this);
            }
            else {
                intersect_and_update(vertex_set[father_id], input_data, input_size);
            }
        }
    }

    /**
     * @brief `*this = intersect(*this, other)`
     * @todo speedup with paralellism?
    */
    __device__ void intersection_with(const GPUVertexSet_Bitmap& other){
        for(int i = 0; i < size; i++){
            data[i] &= other.data[i];
        }
        auto tmp_nzc = calculate_non_zero_cnt();
        non_zero_cnt = tmp_nzc;
        __threadfence_block();
    }

    /**
     * @brief `*this = intersect(vset, input_data)`
     * @todo speedup with paralellism?
    */
    __device__ uint32_t intersect_and_update(const GPUVertexSet_Bitmap& vset, uint32_t *input_data, uint32_t data_size){
        clear();
        uint32_t nzc = 0;
        for(int i = 0; i < data_size; i++){
            if(vset.has_data(input_data[i])){
                this->insert(input_data[i]);
                nzc++;
            }
        }
        non_zero_cnt = nzc;
        __threadfence_block();
    }

    /**
     * @return The i-th element
     * @note To be deprecated after CSR changes into bitmap.
    */
    __device__ uint32_t get_data(uint32_t i) const {
        uint32_t id = 0;
        for(i++; i > 0; id++){
            if(this->has_data(id)) i--;
        }
        return id;
    }

    __device__ uint32_t get_first() const {
        __shared__ bool found[THREADS_PER_BLOCK];

        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id

        bool *output = found + wid * THREADS_PER_WARP;

        uint32_t idx = 0;

        for(idx = 0; idx < size; idx += THREADS_PER_WARP){
            uint32_t data_idx = idx + lid;
            if(data_idx < size && data[data_idx]){
                found[lid] = true;
                break;
            }
        }

        #pragma unroll
        for(uint32_t i = 0; i < THREADS_PER_WARP; i++){
            if(found[idx + i]){
                uint32_t tmp_data = data[idx + i];
                return (idx + i) << 5 + __ffs(tmp_data);
            }
        }
        return UINT32_MAX;
    }

    __device__ uint32_t get_next(uint32_t id) const {
        __shared__ bool found[THREADS_PER_BLOCK];

        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id

        bool *output = found + wid * THREADS_PER_WARP;

        for(id++; id % 32; id++){
            if(this->has_data(id)) return id;
        }
        uint32_t idx = id >> 5;

        for(idx = 0; idx < size; idx += THREADS_PER_WARP){
            uint32_t data_idx = idx + lid;
            if(data_idx < size && data[data_idx]){
                found[lid] = true;
                break;
            }
        }

        #pragma unroll
        for(uint32_t i = 0; i < THREADS_PER_WARP; i++){
            if(found[idx + i]){
                uint32_t tmp_data = data[idx + i];
                return (idx + i) << 5 + __ffs(tmp_data);
            }
        }
        return UINT32_MAX;
    }

#ifdef DEBUG
    __device__ void print() const {
        // printf("Block %d, thread %d: size = %u, ", blockIdx.x, threadIdx.x, non_zero_cnt);
        if(threadIdx.x == 0){
            printf("size = %u, ", non_zero_cnt);
            for(int i = 0; i < size; i++){
                __threadfence_block();
                printf("[%x] ", data[i]);
                __threadfence_block();
                for(int j = 0, _ = 1; j < 32; j++, _ <<= 1){
                    if(data[i] & _) printf("%d ", i * 32 + j);
                }
            }
            printf("\n");
        }
    }
#endif

    uint32_t pat2emb[7];
    // pat2emb[i] means the corresponding vertex in the embedding of the i-th vertex in the pattern

private:
    // uint32_t vertex_count;
    uint32_t size;
    uint32_t *data;
    uint32_t non_zero_cnt;
    
    __device__ uint32_t calculate_non_zero_cnt(){
        // warp reduce version
        typedef cub::WarpReduce<uint32_t> WarpReduce;
        __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
        int wid = threadIdx.x / THREADS_PER_WARP; // warp id
        int lid = threadIdx.x % THREADS_PER_WARP; // lane id
        uint32_t sum = 0;
        for (int index = 0; index < size; index += THREADS_PER_WARP)
            if (index + lid < size)
                sum += __popc(data[index + lid]);
#ifdef DEBUG1
        printf("sum = %d, data[%d] = %x\n", sum, lid, data[lid]);
#endif
        __syncwarp();
        uint32_t aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
        __syncwarp();
        // brute force version
        // uint32_t aggregate = 0;
        // for(int index = 0; index < size; index++){
        //     aggregate += __popc(data[index]);
        // }
        return aggregate;
    }

};

__device__ int GPUVertexSet_Bitmap::subtraction_size(const GPUVertexSet_Bitmap& vset1, const GPUVertexSet_Bitmap& vset2, uint32_t min_vertex){
    // if(min_vertex == 0)
    //     return 0;

    // warp reduce version
    typedef cub::WarpReduce<uint32_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[WARPS_PER_BLOCK];
    int wid = threadIdx.x / THREADS_PER_WARP; // warp id
    int lid = threadIdx.x % THREADS_PER_WARP; // lane id
    // printf("wid = %d, lid = %d\n", wid, lid);
    uint32_t sum = 0;
    if(min_vertex > (vset1.size << 5)){
        for (int index = 0; index < vset1.size; index += THREADS_PER_WARP)
            if (index + lid < vset1.size){
                sum += __popc(vset1.data[index + lid] & (~vset2.data[index + lid]));
            }
    }
    else{
        uint32_t complete_block_cnt = min_vertex / 32;
        uint32_t incomplete_block_num = min_vertex % 32;
        for (int index = 0; index < complete_block_cnt; index += THREADS_PER_WARP)
            if (index + lid < complete_block_cnt){
                sum += __popc(vset1.data[index + lid] & (~vset2.data[index + lid]));
            }
        uint32_t tmp1 = (vset1.data[complete_block_cnt] & mask[incomplete_block_num]);
        uint32_t tmp2 = (vset2.data[complete_block_cnt] & mask[incomplete_block_num]);
        if(lid == 0)
            sum += __popc(tmp1 & (~tmp2));
    }
    // printf("sum = %u\n", sum);
    uint32_t aggregate = WarpReduce(temp_storage[wid]).Sum(sum);
    // printf("aggregate = %u\n", aggregate);
    return aggregate;
}
