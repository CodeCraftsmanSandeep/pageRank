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

__global__ void pageRankKernel(const int *row_ptr, const int *col_idx, const float *rank, float *new_rank, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        register float sum = 0.0f;
        
        // u->v is edge in graph
        // u = col_idx[2*j]
        // out_degre = col_idx[2*j + 1]
        for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++) sum += rank[ col_idx[2*j] ] / col_idx[2*j + 1];
        new_rank[v] = (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
    }
}

__global__ void initializePageRank(float* rank, const int num_nodes){
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) rank[u] = 1.0f / num_nodes;
}

float* pageRank(const int *row_ptr, const int *col_idx, int num_nodes, int num_edges) {
    int num_blocks = (num_nodes + 255) / 256;
    float *d_rank, *d_new_rank;
    int *d_row_ptr, *d_col_idx;

    cudaMalloc(&d_rank, num_nodes * sizeof(float)); 
    initializePageRank <<<num_blocks, 256>>> (d_rank, num_nodes);

    cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, 2*num_edges * sizeof(int));
    cudaMalloc(&d_new_rank, num_nodes * sizeof(float));

    cudaMemcpyAsync(d_row_ptr, row_ptr, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_col_idx, col_idx, 2*num_edges * sizeof(int), cudaMemcpyHostToDevice);

    bool is_old = true;
    for (int i = 0; i < MAX_ITER; i++) {
        if(is_old) pageRankKernel<<<num_blocks, 256>>>(d_row_ptr, d_col_idx, d_rank, d_new_rank, num_nodes);
        else  pageRankKernel<<<num_blocks, 256>>>(d_row_ptr, d_col_idx, d_new_rank, d_rank, num_nodes);
        is_old = !(is_old);
    }

    float* rank = (float*)malloc(num_nodes * sizeof(float));
    if(is_old) cudaMemcpy(rank, d_rank, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(rank, d_new_rank, num_nodes * sizeof(float), cudaMemcpyDeviceToHost); 

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_rank);
    cudaFree(d_new_rank);

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
    int in_neighbour[2*num_edges];            // Col array in CSR format

    int edge = 0;
    in_neighbour_index[0] = 0;
    for(int v = 0; v < num_nodes; v++){
        for(int& u: in_neighbours[v]){
            in_neighbour[2*edge] = u;
            in_neighbour[2*edge + 1] = out_degree[u];
            edge++;
        }
        in_neighbour_index[v+1] = in_neighbour_index[v] + in_neighbours[v].size();
    }
    double t1 = rtclock();
    // Call PageRank function
    float* rank = pageRank(in_neighbour_index, in_neighbour, num_nodes, num_edges);
    double t2 = rtclock();

    print_page_rank(rank, num_nodes);
    printf("Consumed time: %f\n", t2 - t1);

    return 0;
}