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

#define MAX_ITER 1000  // Maximum number of iterations
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-5

void pageRank(const int *row_ptr, const int *col_idx, const int* out_degree, int num_nodes, int num_edges) {
    float new_page_rank[num_nodes];
    float page_rank[num_nodes];

    // Initialization
    for(int u = 0; u < num_nodes; u++) page_rank[u] = 1.0f / num_nodes;

    // Page rank computation
    for(int iter = 0; iter < MAX_ITER; iter++){
        for(int v = 0; v < num_nodes; v++){
            float sum = 0.0f;
            for(int j = row_ptr[v]; j < row_ptr[v+1]; j++){
                int u = col_idx[j];
                sum += page_rank[u] / out_degree[u];
            }
            new_page_rank[v] = (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        }
        // copying page rank to make it useful for next iteration
        for(int u = 0; u < num_nodes; u++) page_rank[u] = new_page_rank[u];
    }

    printf("Final page rank values:\n");
    for(int u = 0; u < num_nodes; u++) printf("pageRank[%d] = %f\n", u, page_rank[u]);
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

    double start_time = rtclock();
    // Call PageRank function
    pageRank(in_neighbour_index, in_neighbour, out_degree, num_nodes, num_edges);
    double end_time = rtclock();
    printf("Consumed time: %.6f\n", end_time - start_time);

    return 0;
}