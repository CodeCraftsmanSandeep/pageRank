#include <iostream>
#include <vector>
#include <sys/time.h>
using namespace std;


double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

#define MAX_ITER 100  // Maximum number of iterations
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-6

float* pageRank(const int *row_ptr, const int *col_idx, const int* out_degree, int num_nodes, int num_edges) {
    #define C 1

    float* new_page_rank = (float*)malloc(num_nodes * sizeof(float));
    float* page_rank = (float*)malloc(num_nodes * sizeof(float));
    
    for(int u = 0; u < num_nodes; u++) page_rank[u] = C * 1.0f / num_nodes;

    for(int iter = 0; iter < MAX_ITER; iter++){
        for(int v = 0; v < num_nodes; v++){
            float sum = 0.0f;
            for(int j = row_ptr[v]; j < row_ptr[v+1]; j++){
                int u = col_idx[j];
                sum += page_rank[u] / out_degree[u];
            }
            new_page_rank[v] = (1.0f - DAMPING_FACTOR) + DAMPING_FACTOR * sum;
        }
        for(int u = 0; u < num_nodes; u++) page_rank[u] = new_page_rank[u];
    }
    return page_rank;
    #undef C
}

float* computePageRank(const int num_nodes, const int num_edges, pair <int, int>* edges){
    vector <vector <int>> in_neighbours(num_nodes);
    int* out_degree = (int*)calloc(num_nodes, sizeof(int));

    for(int edge = 0; edge < num_edges; edge++){
        int u, v;
        u = edges[edge].first;
        v = edges[edge].second;

        in_neighbours[v].push_back(u);
        out_degree[u]++;
    }

    int *in_neighbour_index = (int*)malloc((num_nodes + 1) * sizeof(int)); // Row array in CSR format
    int *in_neighbour = (int*)malloc((num_edges)*sizeof(int));             // Col array in CSR format


    int edge = 0;
    in_neighbour_index[0] = 0;
    for(int v = 0; v < num_nodes; v++){
        for(int& u: in_neighbours[v]) in_neighbour[edge++] = u;
        in_neighbour_index[v+1] = in_neighbour_index[v] + in_neighbours[v].size();
    }
    in_neighbours.clear();
    in_neighbours.shrink_to_fit();

    return pageRank(in_neighbour_index, in_neighbour, out_degree, num_nodes, num_edges);
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
