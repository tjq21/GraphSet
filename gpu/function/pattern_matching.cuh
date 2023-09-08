#pragma once

#include "../component/gpu_device_context.cuh"
#include "../component/gpu_schedule.cuh"
#include "../component/gpu_vertex_set.cuh"
#include "../component/gpu_vertex_set_bitmap.cuh"

constexpr int MAX_DEPTH = 7; // 非递归pattern matching支持的最大深度

struct PatternMatchingDeviceContext : public GraphDeviceContext {
    GPUSchedule *dev_schedule;
    unsigned long long *dev_sum;
    unsigned long long *dev_cur_edge;
    size_t block_shmem_size;
    e_index_t *task_start;
    e_index_t *dev_new_order;
    int devices_num;
    int devices_no;
    void init(const Graph *_g, const Schedule_IEP &schedule, int total_devices = 1, int no_devices = 0) {
        g = _g;
        // prefix + subtraction + tmp + extra (n-2)
        int num_vertex_sets_per_warp = schedule.get_total_prefix_num() + schedule.get_size();

        size_t size_edge = g->e_cnt * sizeof(uint32_t);
        size_t size_vertex = (g->v_cnt + 1) * sizeof(e_index_t);
#ifdef ARRAY
        size_t size_tmp = VertexSet::max_intersection_size * num_total_warps * (schedule.get_total_prefix_num() + 2) *
                          sizeof(uint32_t); // prefix + subtraction + tmp
#else
        size_t size_tmp = (g->v_cnt + 31) / 32 * num_total_warps * (schedule.get_total_prefix_num() + 2) * sizeof(uint32_t);
#endif
        uint32_t *edge_from = new uint32_t[g->e_cnt];
        for (uint32_t i = 0; i < g->v_cnt; ++i) {
            for (e_index_t j = g->vertex[i]; j < g->vertex[i + 1]; ++j)
                edge_from[j] = i;
        }
        task_start = new e_index_t[total_devices + 1];
        
        devices_num = total_devices;
        devices_no = no_devices; 

        e_index_t *new_order = new e_index_t[g->e_cnt];        

        // g->reorder_edge(schedule, new_order, task_start, total_devices);
        size_t size_new_order = sizeof(e_index_t) * g->e_cnt;
        
        if(no_devices == 0) {
            log("Memory Usage:\n");
            log("  Graph (GB): %.3lf \n", (size_edge + size_vertex) / (1024.0 * 1024 * 1024));
            log("  Global memory usage (GB): %.3lf \n", (size_new_order + size_edge + size_tmp) / (1024.0 * 1024 * 1024));
            log("  Total memory usage (GB): %.3lf \n", (size_edge + size_vertex + size_new_order + size_edge + size_tmp) / (1024.0 * 1024 * 1024) * total_devices);
#ifdef ARRAY
            log("  Shared memory for vertex set per block: %ld bytes\n",
                num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet) +
                    schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int));
#else
            log("  Shared memory for vertex set per block: %ld bytes\n",
                num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet_Bitmap) +
                    schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int));
#endif
        }
        gpuErrchk(cudaMalloc((void **)&dev_new_order, size_new_order));
        gpuErrchk(cudaMemcpy(dev_new_order, new_order, size_new_order, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void **)&dev_edge, size_edge));
        gpuErrchk(cudaMalloc((void **)&dev_edge_from, size_edge));
        gpuErrchk(cudaMalloc((void **)&dev_vertex, size_vertex));
        gpuErrchk(cudaMalloc((void **)&dev_tmp, size_tmp));

        gpuErrchk(cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_edge_from, edge_from, size_edge, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));

        unsigned long long sum = 0;
        gpuErrchk(cudaMalloc((void **)&dev_sum, sizeof(sum)));
        gpuErrchk(cudaMemcpy(dev_sum, &sum, sizeof(sum), cudaMemcpyHostToDevice));
        unsigned long long cur_edge = 0;
        gpuErrchk(cudaMalloc((void **)&dev_cur_edge, sizeof(cur_edge)));
        gpuErrchk(cudaMemcpy(dev_cur_edge, &cur_edge, sizeof(cur_edge), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMallocManaged((void **)&dev_schedule, sizeof(GPUSchedule)));
        dev_schedule->create_from_schedule(schedule);



#ifdef ARRAY
        block_shmem_size = num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet) +
                           schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int);
#else
        block_shmem_size = num_vertex_sets_per_warp * WARPS_PER_BLOCK * sizeof(GPUVertexSet_Bitmap) +
                           schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int);
#endif

        dev_schedule->ans_array_offset = block_shmem_size - schedule.in_exclusion_optimize_vertex_id.size() * WARPS_PER_BLOCK * sizeof(int);

        delete[] edge_from;
        delete[] new_order;
    }
    void destroy() {
        gpuErrchk(cudaFree(dev_edge));
        gpuErrchk(cudaFree(dev_edge_from));
        gpuErrchk(cudaFree(dev_vertex));
        gpuErrchk(cudaFree(dev_tmp));

        gpuErrchk(cudaFree(dev_new_order));

        gpuErrchk(cudaFree(dev_schedule->adj_mat));
        gpuErrchk(cudaFree(dev_schedule->father_prefix_id));
        gpuErrchk(cudaFree(dev_schedule->last));
        gpuErrchk(cudaFree(dev_schedule->next));
        gpuErrchk(cudaFree(dev_schedule->loop_set_prefix_id));
        gpuErrchk(cudaFree(dev_schedule->restrict_last));
        gpuErrchk(cudaFree(dev_schedule->restrict_next));
        gpuErrchk(cudaFree(dev_schedule->restrict_index));

        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_vertex_id));
        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_coef));
        gpuErrchk(cudaFree(dev_schedule->in_exclusion_optimize_flag));

        gpuErrchk(cudaFree(dev_schedule));
    }
};

/**
 * @brief 最终层的容斥原理优化计算。
 */
#ifdef ARRAY
__device__ void GPU_pattern_matching_final_in_exclusion(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                        GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
#else
__device__ void GPU_pattern_matching_final_in_exclusion(const GPUSchedule *schedule, GPUVertexSet_Bitmap *vertex_set, GPUVertexSet_Bitmap &subtraction_set,
                                                        GPUVertexSet_Bitmap &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
#endif
    int last_pos = -1;
    long long val;

    extern __shared__ char ans_array[];
    int *ans = ((int *)(ans_array + schedule->ans_array_offset)) + schedule->in_exclusion_optimize_vertex_id_size * (threadIdx.x / THREADS_PER_WARP);

    for (int i = 0; i < schedule->in_exclusion_optimize_vertex_id_size; ++i) {
#ifdef ARRAY
        if (schedule->in_exclusion_optimize_vertex_flag[i]) {
            ans[i] = vertex_set[schedule->in_exclusion_optimize_vertex_id[i]].get_size() - schedule->in_exclusion_optimize_vertex_coef[i];
        } else {
            ans[i] = unordered_subtraction_size(vertex_set[schedule->in_exclusion_optimize_vertex_id[i]], subtraction_set);
        }
#else
        if (schedule->in_exclusion_optimize_vertex_flag[i]) {
            ans[i] = vertex_set[schedule->in_exclusion_optimize_vertex_id[i]].get_size() - schedule->in_exclusion_optimize_vertex_coef[i];
        } else {
            ans[i] = GPUVertexSet_Bitmap::subtraction_size(vertex_set[schedule->in_exclusion_optimize_vertex_id[i]], subtraction_set);
        }
#endif
    }

    for (int pos = 0; pos < schedule->in_exclusion_optimize_array_size; ++pos) {

        if (pos == last_pos + 1)
            val = ans[schedule->in_exclusion_optimize_ans_pos[pos]];
        else {
            if (val != 0)
                val = val * ans[schedule->in_exclusion_optimize_ans_pos[pos]];
        }
        if (schedule->in_exclusion_optimize_flag[pos]) {
            last_pos = pos;
            local_ans += val * schedule->in_exclusion_optimize_coef[pos];
        }
    }
}

/**
 * @brief 用于 vertex_induced 的计算（好像没怎么测过）
 * @param out_buf Output.
 * @param in_buf Prefix of v_depth.
 * @param partial_embedding Vertices already explored in a embedding, subtraction_set.
 * @param vp Depth.
 */
#ifdef ARRAY
__device__ void remove_anti_edge_vertices(GPUVertexSet &out_buf, const GPUVertexSet &in_buf, const GPUSchedule &sched,
                                          const GPUVertexSet &partial_embedding, int vp, const uint32_t *edge, const e_index_t *vertex) {
#else
__device__ void remove_anti_edge_vertices(GPUVertexSet_Bitmap &out_buf, const GPUVertexSet_Bitmap &in_buf, const GPUSchedule &sched,
                                          const GPUVertexSet_Bitmap &partial_embedding, int vp, const uint32_t *edge, const e_index_t *vertex) {
#endif

    __shared__ uint32_t block_out_offset[THREADS_PER_BLOCK];
    __shared__ uint32_t block_out_size[WARPS_PER_BLOCK];

#ifdef ARRAY
    auto d_out = out_buf.get_data_ptr();
    auto d_in = in_buf.get_data_ptr();
    int n_in = in_buf.get_size();
#endif

    int warp = threadIdx.x / THREADS_PER_WARP;
    int lane = threadIdx.x % THREADS_PER_WARP;
    auto out_offset = block_out_offset + warp * THREADS_PER_WARP;
    auto &out_size = block_out_size[warp];

    if (lane == 0)
        out_size = 0;

    /**
     * for vertex v1 in in_buf:
     *   produce_output = true;
     *   for u < vp and !(u, vp):
     *     v = partial_embedding.get_data(u)
     *     if v1 in N(v): produce_output = false, break
     *   d_out.insert(v1)
    */
#ifdef ARRAY
    for (int nr_done = 0; nr_done < n_in; nr_done += THREADS_PER_WARP) {
        int i = nr_done + lane;
        bool produce_output = false;

        if (i < n_in) {
            produce_output = true;
            for (int u = 0; u < vp; ++u) {
                if (sched.adj_mat[u * sched.get_size() + vp] != 0)
                    continue;

                auto v = partial_embedding.get_data(u);
                e_index_t v_neighbor_begin, v_neighbor_end;
                get_edge_index(v, v_neighbor_begin, v_neighbor_end);
                int m = v_neighbor_end - v_neighbor_begin; // m = |N(v)|

                if (binary_search(&edge[v_neighbor_begin], m, d_in[i])) {
                    produce_output = false;
                    break;
                }
            }
        }
        out_offset[lane] = produce_output;
        __threadfence_block();

#pragma unroll
        // out_offset[lane] = sum(out_offset[lane] + [lane-1] + ... + [0])
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            uint32_t v = lane >= s ? out_offset[lane - s] : 0;
            out_offset[lane] += v;
            __threadfence_block();
        }

        if (produce_output) {
            uint32_t offset = out_offset[lane] - 1;
            d_out[out_size + offset] = d_in[i];
        }

        if (lane == THREADS_PER_WARP - 1)
            out_size += out_offset[THREADS_PER_WARP - 1];
    }

    if (lane == 0)
        out_buf.init(out_size, d_out);
    __threadfence_block();
#else
    for(uint32_t v1 = in_buf.get_first(); v1 != UINT32_MAX; v1 = in_buf.get_next(v1)){
        bool produce_output = true;
        for (int u = 0; u < vp; ++u) {
            if (sched.adj_mat[u * sched.get_size() + vp] != 0)
                continue;

            auto v = partial_embedding.get_data(u);
            e_index_t v_neighbor_begin, v_neighbor_end;
            get_edge_index(v, v_neighbor_begin, v_neighbor_end);
            int m = v_neighbor_end - v_neighbor_begin; // m = |N(v)|

            if (binary_search(&edge[v_neighbor_begin], m, v1)) {
                produce_output = false;
                break;
            }
        }
        if(produce_output)
            out_buf.insert(v1);
    }
#endif
}


/**
 * @brief 以模板形式伪递归的计算函数
 *
 */
template <int depth>
#ifdef ARRAY
__device__ void GPU_pattern_matching_func(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set, GPUVertexSet &tmp_set,
                                          unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
#else
__device__ void GPU_pattern_matching_func(const GPUSchedule *schedule, GPUVertexSet_Bitmap *vertex_set, GPUVertexSet_Bitmap &subtraction_set, GPUVertexSet_Bitmap &tmp_set,
                                          unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex, uint32_t *pat2emb) {
#endif
    if(threadIdx.x % THREADS_PER_WARP == 0)
        printf("GPU_PatterMatchingFunc<%d>, schedule.size = %u, IEP_opt_num = %u\n", depth, schedule->get_size(), schedule->get_in_exclusion_optimize_num());
    if (depth == schedule->get_size() - schedule->get_in_exclusion_optimize_num()) {
        GPU_pattern_matching_final_in_exclusion(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        // if(threadIdx.x % 32 == 0) printf("after in_exclusion<%d>, local_ans = %llu\n", depth, local_ans);
        if(threadIdx.x % THREADS_PER_WARP == 0) printf("return 1\n");
        return;
    }

    int loop_set_prefix_id = schedule->get_loop_set_prefix_id(depth);
    auto vset = &vertex_set[loop_set_prefix_id];

#ifdef ARRAY
    int loop_size = vset->get_size();

    auto loop_data_ptr = vset->get_data_ptr();
    uint32_t min_vertex = 0xffffffff;
    // for all restrictions that points to depth (in pattern), get the mininum index in the partial embedding where v_index maps to res.first
    // The newly explored vertex ought to be smaller than all restricted vertices.
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > subtraction_set.get_data(schedule->get_restrict_index(i)))
            min_vertex = subtraction_set.get_data(schedule->get_restrict_index(i));

    // can be optimized via code generation
    if (schedule->is_vertex_induced) {
        GPUVertexSet &diff_buf = vertex_set[schedule->get_total_prefix_num() + depth];
        remove_anti_edge_vertices(diff_buf, vertex_set[loop_set_prefix_id], *schedule, subtraction_set, depth, edge, vertex);
        loop_data_ptr = diff_buf.get_data_ptr();
        loop_size = diff_buf.get_size();
        vset = &diff_buf;
    }

    if (depth == schedule->get_size() - 1 && schedule->get_in_exclusion_optimize_num() == 0) {
        /*
        for (int i = 0; i < loop_size; ++i)
        {
            uint32_t x = vset->get_data(i);
            bool flag = true;
            for (int j = 0; j < subtraction_set.get_size(); ++j)
                if (subtraction_set.get_data(j) == x)
                    flag = false;
            if (flag && threadIdx.x == 0)
                printf("%d %d %d %d\n", subtraction_set.get_data(0), subtraction_set.get_data(1), subtraction_set.get_data(2), x);
        }
        return;*/
        int size_after_restrict = lower_bound(loop_data_ptr, loop_size, min_vertex);
        // int size_after_restrict = -1;
        local_ans += unordered_subtraction_size(*vset, subtraction_set, size_after_restrict);
        return;
    }
    for (int i = 0; i < loop_size; ++i) {
        uint32_t v = loop_data_ptr[i];
        if (min_vertex <= v)
            break;
        if (subtraction_set.has_data(v))
            continue;
        e_index_t l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == schedule->get_break_size(prefix_id)) {
                is_zero = true;
                break;
            }
        }
        if (is_zero)
            continue;
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.push_back(v);
            __threadfence_block();
        }
        GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex);
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.pop_back();
            __threadfence_block();
        }
    }
#else
    uint32_t min_vertex = 0xffffffff;
    // for all restrictions that points to depth (in pattern), get the mininum index in the partial embedding where v_index maps to res.first
    // The newly explored vertex ought to be smaller than all restricted vertices.
    for (int i = schedule->get_restrict_last(depth); i != -1; i = schedule->get_restrict_next(i))
        if (min_vertex > pat2emb[schedule->get_restrict_index(i)])
            min_vertex = pat2emb[schedule->get_restrict_index(i)];

    // can be optimized via code generation
    if (schedule->is_vertex_induced) {
        GPUVertexSet_Bitmap &diff_buf = vertex_set[schedule->get_total_prefix_num() + depth];
        remove_anti_edge_vertices(diff_buf, vertex_set[loop_set_prefix_id], *schedule, subtraction_set, depth, edge, vertex);
        vset = &diff_buf;
    }
    if (depth == schedule->get_size() - 1 && schedule->get_in_exclusion_optimize_num() == 0) {
        // int size_after_restrict = lower_bound(loop_data_ptr, loop_size, min_vertex);
        local_ans += GPUVertexSet_Bitmap::subtraction_size(*vset, subtraction_set, min_vertex);
        if(threadIdx.x % THREADS_PER_WARP == 0) printf("return 2\n");
        return;
    }
    /**
     * for v in loop_data_ptr:
     *   if(min_vertex <= v) break;
     *   if(subtraction_set.has_data(v)) continue;
     *   for prefix in schedule->get_last(depth)(array):
     *     vertex_set[prefix_id].build_vertex_set(self, v's neighbor);
     *     if(vertex_set[prefix_id].get_non_zerp_cnt() == breaksize) {iszero=true, break}
    */
    // printf("Starting...size of vset=%d, global_tid=%d\n", vset->get_size(), blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
    // __syncwarp();
    for (uint32_t v = vset->get_first(); v != UINT32_MAX; v = vset->get_next(v)) {
        if(min_vertex <= v){
            break;
        }
        if (subtraction_set.has_data(v)){
            printf("has %u, continue1\n", v);
            continue;
        }
        e_index_t l, r;
        get_edge_index(v, l, r);
        bool is_zero = false;
        for (int prefix_id = schedule->get_last(depth); prefix_id != -1; prefix_id = schedule->get_next(prefix_id)) {
            vertex_set[prefix_id].build_vertex_set(schedule, vertex_set, &edge[l], r - l, prefix_id);
            if (vertex_set[prefix_id].get_size() == schedule->get_break_size(prefix_id)) {
                is_zero = true;
                break;
            }
        }
        if (is_zero){
            printf("iszero, continue2\n");
            continue;
        }
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0){
                subtraction_set.insert_and_update(v);
                pat2emb[depth] = v;
            }
            __threadfence_block();
        }
        GPU_pattern_matching_func<depth + 1>(schedule, vertex_set, subtraction_set, tmp_set, local_ans, edge, vertex, pat2emb);
        if (depth + 1 != MAX_DEPTH) {
            if (threadIdx.x % THREADS_PER_WARP == 0)
                subtraction_set.erase_and_update(v);
            __threadfence_block();
        }
    }
#endif
}

/**
 * @brief 模板递归的边界
 *
 */
template <>
#ifdef ARRAY
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule *schedule, GPUVertexSet *vertex_set, GPUVertexSet &subtraction_set,
                                                     GPUVertexSet &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex) {
    // assert(false);
}
#else
__device__ void GPU_pattern_matching_func<MAX_DEPTH>(const GPUSchedule *schedule, GPUVertexSet_Bitmap *vertex_set, GPUVertexSet_Bitmap &subtraction_set,
                                                     GPUVertexSet_Bitmap &tmp_set, unsigned long long &local_ans, uint32_t *edge, e_index_t *vertex, uint32_t *pat2emb) {
    // assert(false);
}
#endif