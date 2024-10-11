#include <iostream>
#include <vector>
#include <cuda.h>
#include <sys/time.h>
using namespace std;

#define MAX_ITER 1000  // Maximum number of iterations
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-5
#define FULL_MASK 0xffffffff

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__device__ void criticalSum(float* blockSum, const float value){
    atomicAdd(blockSum, value);
    return;
}

__global__ void find_sum(const int start, const int end, const float* arr, float* result, const int num_nodes, const int out_degree){
    const int index = threadIdx.x + start;
    register float value = 0.f;
    const int laneId = threadIdx.x % 32;
    const int n = end - start + 1;
        
    // handle last n%32 elements seperately
    if(threadIdx.x >= n / 32 * 32){
        if(laneId == 0) for(int i = index; i <= end; i++) value += arr[i];
    }else{
        value = arr[index];
        for(int offset = 16; offset >= 1; offset /= 2) value += __shfl_down_sync(FULL_MASK, value, offset);
    }

    __shared__ float blockSum;
    if(threadIdx.x == 0)  blockSum = 0.0f;
    __syncthreads();
    if(laneId == 0) criticalSum(&blockSum, value);
    __syncthreads();
    if(threadIdx.x == 0) *result = ( (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * blockSum) / out_degree;
}

__global__ void pageRankKernel(const int *row_ptr, const int *col_idx, const int* out_degree, float *new_contribution, const float *old_contribution, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        const int start = row_ptr[v];
        const int end = row_ptr[v+1] - 1;
        const int in_degree = end - start + 1;
        if(in_degree < 32*2){
            register float total_contribution = 0.0f;
            // let u = col_idx[j], then u->v is an edge in the graph
            for (int j = start; j <= end; j++) total_contribution += old_contribution[col_idx[j]];
            new_contribution[v] = ( (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution) / out_degree[v];
        }else if(in_degree <= 1024){
            // computing contribution
            find_sum <<< 1, in_degree>>> (start, end, old_contribution, (new_contribution + v), num_nodes, out_degree[v]);
            // cudaDeviceSynchronize();
        }else{
            printf("Currently out of scope\n");
        }
    }
}

__global__ void finalPageRankKernel(const int* row_ptr, const int* col_idx, float* rank, const float* old_contribution, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v < num_nodes){
        register float total_contribution = 0.0f;
        for(int j = row_ptr[v]; j < row_ptr[v + 1]; j++) total_contribution += old_contribution[col_idx[j]];
        rank[v] = (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution;
    }
}

__global__ void initializeContribution (float* contribution, const int* out_degree, const int num_nodes){
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) contribution[u] = 1.0f / num_nodes / out_degree[u];
}

float* pageRank(const int *row_ptr, const int *col_idx, const int* out_degree, int num_nodes, int num_edges) {
    int num_blocks = (num_nodes + 255) / 256;
    float random_contribution = (1.0f - DAMPING_FACTOR) / num_nodes;
    
    // d_contribution[u] = page_rank value contributed by u to its out neighbours.
    float *d_old_contribution;
    cudaMalloc(&d_old_contribution, num_nodes * sizeof(float)); 
    
    // d_out_degree[u] = out_degree[u] stored inside gpu
    int *d_out_degree;
    cudaMalloc(&d_out_degree, num_nodes * sizeof(int));
    cudaMemcpyAsync(d_out_degree, out_degree, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    
    // intializinf the contribution[u] = (1/n)/(out_degree[u])
    initializeContribution <<<num_blocks, 256>>> (d_old_contribution, d_out_degree, num_nodes);

    // allocating and copying CSR format to GPU
    int *d_row_ptr, *d_col_idx;
    cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMemcpyAsync(d_row_ptr, row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_col_idx, col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    // rank values
    float* d_new_contribution;
    cudaMalloc(&d_new_contribution, num_nodes * sizeof(float));

    bool new_flag = true;
    // computing page rank values
    for (int i = 0; i < MAX_ITER - 1; i++) {
        if(new_flag) pageRankKernel <<<num_blocks, 256>>> (d_row_ptr, d_col_idx, d_out_degree, d_new_contribution, d_old_contribution, num_nodes);
        else pageRankKernel <<<num_nodes, 256>>> (d_row_ptr, d_col_idx, d_out_degree, d_old_contribution, d_new_contribution, num_nodes);
        // cudaDeviceSynchronize();
        new_flag = !(new_flag);
    }

    float *rank = (float*)malloc(num_nodes * sizeof(float));
    if(new_flag){
        finalPageRankKernel <<< num_nodes, 256 >>> (d_row_ptr, d_col_idx, d_new_contribution, d_old_contribution, num_nodes);
        cudaMemcpy(rank, d_new_contribution, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    }else{
        finalPageRankKernel <<< num_nodes, 256 >>> (d_row_ptr, d_col_idx, d_old_contribution, d_new_contribution, num_nodes);
        cudaMemcpy(rank, d_old_contribution, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_old_contribution);
    cudaFree(d_new_contribution);
    cudaFree(d_out_degree);

    return rank;
}

void print_page_rank(float* rank, int num_nodes){
    printf("Final page rank values:\n");
    for(int u = 0; u < num_nodes; u++) printf("pageRank[%d] = %f\n", u, rank[u]);
}
// assumptions:
// 1) the graph is unweighted, directed.
// 2) the graph may have multiple edges, self loops.

int main() {
    int num_nodes, num_edges;
    scanf("%d %d", &num_nodes, &num_edges);

    vector <vector <int>> in_neighbours(num_nodes);
    int* out_degree = (int*)calloc(num_nodes, sizeof(int));

    for(int edge = 0; edge < num_edges; edge++){
        int u, v;
        scanf("%d %d", &u, &v);
        in_neighbours[v].push_back(u);
        out_degree[u]++;
    }

    int in_neighbour_index[num_nodes + 1];  // Row array in CSR format
    int in_neighbour[num_edges];            // Col array in CSR format

    int edge = 0;
    in_neighbour_index[0] = 0;
    for(int v = 0; v < num_nodes; v++){
        for(int& u: in_neighbours[v]) in_neighbour[edge++] = u;
        in_neighbour_index[v+1] = in_neighbour_index[v] + in_neighbours[v].size();
    }

    double t1 = rtclock();
    // Call PageRank function
    float* rank = pageRank(in_neighbour_index, in_neighbour, out_degree, num_nodes, num_edges);
    double t2 = rtclock();

    print_page_rank(rank, num_nodes);
    printf("Consumed time: %f\n", t2 - t1);

    return 0;
}