#include <iostream>
#include <vector>
#include <cuda.h>
#include <sys/time.h>
using namespace std;

#define MAX_ITER 1000 // Maximum number of iterations
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-6

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__ void pageRankKernel(const int *row_ptr, const int *col_idx, const int* out_degree, float *new_contribution, const float *old_contribution, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        register float total_contribution = 0.0f;
        // let u = col_idx[j] then u -> v is a edge in the graph
        for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++) total_contribution += old_contribution[col_idx[j]];  
        new_contribution[v] = ( (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution) / out_degree[v];
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
    #define C 1
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) contribution[u] = C * 1.0f / num_nodes / out_degree[u];
    #undef C
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

    // initialize page rank contribution values
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

        new_flag = !(new_flag);
    }

    float *rank = new float[num_nodes];
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

struct Node{
    int vertex;
    struct Node* next;

    Node(int vertex): vertex(vertex), next(nullptr) {} 
};

// assumptions:
// 1) the graph is unweighted, directed.
// 2) the graph may have multiple edges, self loops.

float* computePageRank(const int num_nodes, const int num_edges, pair <int, int>* edges){

    Node** in_neighbours = new Node*[num_nodes];
    Node** tail = new Node*[num_nodes];

    for(int u = 0; u < num_nodes; u++){
        tail[u] = in_neighbours[u] = nullptr;
    }

    int* out_degree = (int*)calloc(num_nodes, sizeof(int));

    for(int edge = 0; edge < num_edges; edge++){
        int u, v;
        u = edges[edge].first;
        v = edges[edge].second;

        if(tail[v] == nullptr){
            tail[v] = in_neighbours[v] = new Node(u);
        }else{
            tail[v]->next = new Node(u);
            tail[v] = tail[v]->next;
        }
        out_degree[u]++;
    }
    delete[] tail;

    int* in_neighbour_index = new int[num_nodes + 1];  // Row array in CSR format
    int* in_neighbour = new int[num_edges];            // Col array in CSR format
    
    int edge = 0;
    in_neighbour_index[0] = 0;
    for(int v = 0; v < num_nodes; v++){
        int end = in_neighbour_index[v];
        
        Node* trav = in_neighbours[v];
        while(trav != nullptr){
            in_neighbour[edge++] = trav->vertex;
            end++;
            Node* next_ptr = trav->next;
            delete trav;
            trav = next_ptr;
        }
        in_neighbour_index[v+1] = end;
    }
    delete[] in_neighbours;


    // Call PageRank function
    float* rank = pageRank(in_neighbour_index, in_neighbour, out_degree, num_nodes, num_edges);
    
   return rank;
}

void printPageRank(const int numNodes, const float* pageRank){
    printf("\nFinal pageRank values:\n");
    for(int u = 0; u < numNodes; u++) printf("pageRank[%d] = %.6f\n", u, pageRank[u]);
    return;
}

int main() {
    // Scanning |V|, |E|
    int numNodes, numEdges;
    scanf("%d %d", &numNodes, &numEdges);

    // Storing all edges in heap memory
    std::pair <int, int>* edges = (std::pair <int, int>*)malloc(numEdges * sizeof(std::pair <int, int>));
    for(int i = 0; i < numEdges; i++) scanf("%d %d", &edges[i].first, &edges[i].second);

    // Here currently the problem is only computing pageRank values, and not producing sorted order of nodes w.r.t pageRank values
    double start_time = rtclock();
    float* pageRank = computePageRank(numNodes, numEdges, edges);
    double end_time = rtclock();

    printPageRank(numNodes, pageRank);
    printf("Time consumed: %f\n", end_time - start_time);

    return 0;
}
