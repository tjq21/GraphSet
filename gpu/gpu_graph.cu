#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

#include "common.h"
#include "dataloader.h"
#include "graph.h"
#include "motif_generator.h"
#include "schedule_IEP.h"
#include "vertex_set.h"
#include "timeinterval.h"

#include "component/gpu_device_detect.cuh"
#include "src/gpu_pattern_matching.cuh"


TimeInterval tmpTime;

bool cmp_degree_gt(std::pair<int,int> a,std::pair<int,int> b) {
    return a.second > b.second;
}

void pattern_matching(Graph *g, const Schedule_IEP &schedule_iep) {
    tmpTime.check();
    PatternMatchingDeviceContext *context;
    gpuErrchk(cudaMallocManaged((void **)&context, sizeof(PatternMatchingDeviceContext)));
    context->init(g, schedule_iep);

#if USE_ARRAY == 1
    uint32_t buffer_size = VertexSet::max_intersection_size;
#else
    uint32_t buffer_size = (g->v_cnt + 31) / 32;
#endif
    int max_active_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, gpu_pattern_matching, THREADS_PER_BLOCK, context->block_shmem_size);
    fprintf(stderr, "max number of active warps per SM: %d\n", max_active_blocks_per_sm * WARPS_PER_BLOCK);

    tmpTime.print("Prepare time cost");
    tmpTime.check();

    unsigned long long sum = 0;

    gpu_pattern_matching<<<num_blocks, THREADS_PER_BLOCK, context->block_shmem_size>>>(g->e_cnt, buffer_size, context);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(&sum, context->dev_sum, sizeof(sum), cudaMemcpyDeviceToHost));

    sum /= schedule_iep.get_in_exclusion_optimize_redundancy();

    double counting_time_cost = tmpTime.get_time();
    FILE *fp;
    fp = fopen("../../cmake_gen/counting_time_cost.txt", "w");
    fprintf(fp, "%.6f\n", counting_time_cost);
    fclose(fp);

    fprintf(stderr, "Pattern count: %llu\n", sum);

    context->destroy();
    gpuErrchk(cudaFree(context));
}

int main(int argc, char *argv[]) {
    get_device_information();
    Graph *g;
    DataLoader D;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s graph_file pattern_size pattern_string <1/0 for enable iep or not>\n", argv[0]);
        return 1;
    }

    int enable_iep = 1;
    if(argc >= 5) {
        enable_iep = atoi(argv[4]);
        if(enable_iep != 0 && enable_iep != 1) {
            fprintf(stderr, "Usage: %s graph_file pattern_size pattern_string <1/0 for enable iep or not>\n", argv[0]);
            return 1;
        } else {
            fprintf(stderr, "Enable iep: %d\n", enable_iep);
        }
    }

    using std::chrono::system_clock;
    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);
    if (!ok) {
        fprintf(stderr, "data load failure :-(\n");
        return 0;
    }

#if SORT == 1
    fprintf(stderr, "Sort: true\n");

    std::pair<int,int> *rank = new std::pair<int,int>[g->v_cnt], *e = new std::pair<int, int>[g->e_cnt];
    int* degree = new int [g->v_cnt];
    e_index_t tmp_e = 0;
    for(v_index_t i = 0; i < g->v_cnt; i++){
        degree[i] = g->vertex[i+1] - g->vertex[i];
        for(e_index_t j = g->vertex[i]; j < g->vertex[i+1]; j++){
            e[tmp_e++] = std::make_pair(i, g->edge[j]);
        }
    }

    int *new_id = new v_index_t[g->v_cnt];
    for(int i = 0; i < g->v_cnt; ++i) rank[i] = std::make_pair(i,degree[i]);
    std::sort(rank, rank + g->v_cnt, cmp_degree_gt);
    for(v_index_t i = 0; i < g->v_cnt; ++i) new_id[rank[i].first] = i;
    for(e_index_t i = 0; i < g->e_cnt; ++i) {
        e[i].first = new_id[e[i].first];
        e[i].second = new_id[e[i].second];
    }

    delete[] rank;
    delete[] new_id;
#endif

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    fprintf(stderr, "Load data success! time: %g seconds\n", load_time.count() / 1.0e6);

    int pattern_size = atoi(argv[2]);
    const char *pattern_str = argv[3];

    Pattern p(pattern_size, pattern_str);

    printf("pattern = ");
    p.print();

    fprintf(stderr, "Max intersection size %d\n", VertexSet::max_intersection_size);

    bool is_pattern_valid;
    Schedule_IEP schedule_iep(p, is_pattern_valid, 1, 1, enable_iep, g->v_cnt, g->e_cnt, g->tri_cnt);
    if (!is_pattern_valid) {
        fprintf(stderr, "pattern is invalid!\n");
        return 1;
    }

    pattern_matching(g, schedule_iep);

    return 0;
}
