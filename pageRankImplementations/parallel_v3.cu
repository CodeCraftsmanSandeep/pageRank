#include <iostream>
#include <vector>
#include <cuda.h>
#include <sys/time.h>
using namespace std;

#define MAX_ITER 1000  // Maximum number of iterations
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-5

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__ void pageRankKernel(const int *row_ptr, const int *col_idx, const float *contribution, float *rank, int num_nodes) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        float total_contribution = 0.0f;
        for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++) total_contribution += contribution[col_idx[j]];  // let u = col_idx[j] then u -> v is a edge in the graph
        rank[v] = (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution;
    }
}

__global__ void initializeContribution (float* contribution, int* out_degree, int num_nodes){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) contribution[u] = 1.0f / num_nodes / out_degree[u];
}

__global__ void precompute_contribution(const float* rank, float* contribution, const int* out_degree, int num_nodes){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) contribution[u] = rank[u] / out_degree[u];
}

void pageRank(const int *row_ptr, const int *col_idx, const int* out_degree, int num_nodes, int num_edges) {
    int num_blocks = (num_nodes + 255) / 256;
    
    // d_contribution[u] = page_rank value contributed by u to its out neighbours.
    float *d_contribution;
    cudaMalloc(&d_contribution, num_nodes * sizeof(float)); 
    
    // d_out_degree[u] = out_degree[u] stored inside gpu
    int *d_out_degree;
    cudaMalloc(&d_out_degree, num_nodes * sizeof(int));
    cudaMemcpyAsync(d_out_degree, out_degree, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    
    // intializinf the contribution[u] = (1/n)/(out_degree[u])
    initializeContribution <<<num_blocks, 256>>> (d_contribution, d_out_degree, num_nodes);

    // allocating and copying CSR format to GPU
    int *d_row_ptr, *d_col_idx;
    cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMemcpyAsync(d_row_ptr, row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_col_idx, col_idx, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    // rank values
    float* d_rank;
    cudaMalloc(&d_rank, num_nodes * sizeof(float));

    // computing page rank values
    for (int i = 0; i < MAX_ITER - 1; i++) {
        pageRankKernel <<<num_blocks, 256>>> (d_row_ptr, d_col_idx, d_contribution, d_rank, num_nodes);
        precompute_contribution <<<num_blocks, 256>>> (d_rank, d_contribution, d_out_degree, num_nodes);
    }
    pageRankKernel <<<num_blocks, 256>>> (d_row_ptr, d_col_idx, d_rank, d_contribution, num_nodes);

    float rank[num_nodes];
    cudaMemcpy(rank, d_rank, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final page rank values:\n");
    for(int u = 0; u < num_nodes; u++) printf("pageRank[%d] = %f\n", u, rank[u]);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_rank);
    cudaFree(d_contribution);
    cudaFree(d_out_degree);
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
    pageRank(in_neighbour_index, in_neighbour, out_degree, num_nodes, num_edges);
    double t2 = rtclock();
    
    printf("Consumed time: %f\n", t2 - t1);

    return 0;
}