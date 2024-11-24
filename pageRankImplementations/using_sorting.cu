#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/memory.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#define MAX_ITER 1000
#define DAMPING_FACTOR 0.85
#define THRESHOLD 1e-6

__device__ int numNodes;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__ void pageRankKernel(const int *row_ptr, const int *col_idx, const int* out_degree, const float *old_contribution, float *new_contribution, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        register float total_contribution = 0.0f;
        for (int j = row_ptr[v]; j < row_ptr[v + 1]; j++) total_contribution += old_contribution[col_idx[j]];  // let u = col_idx[j] then u -> v is a edge in the graph
        new_contribution[v] = ( (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution) / out_degree[v];
    }
}

__global__ void finalPageRankKernel(const int* row_ptr, const int* col_idx, const float* old_contribution, float* rank, const int num_nodes) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v < num_nodes){
        register float total_contribution = 0.0f;
        for(int j = row_ptr[v]; j < row_ptr[v + 1]; j++) total_contribution += old_contribution[col_idx[j]];
        rank[v] = (1.0f - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * total_contribution;
    }
}

__global__ void initializeContribution (float* contribution, const int* out_degree, const int num_nodes){
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if(u < num_nodes) contribution[u] = (1.0) / num_nodes / out_degree[u];
}

__global__ void sumReduction(const int start, const int end, const int* inNeighbour, const float* oldContribution, float* newContribution, const int u) {
    __shared__ float sharedMem[32];  // Shared memory for the warp-level reduction

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane = threadIdx.x % 32;  // Thread's lane within a warp
    const int warpId = threadIdx.x / 32;
    const int n = end - start;

    // Step 1: Grid strided loops
    float sum = 0;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        sum += oldContribution[inNeighbour[start + i]];
    }

    // there is a speed up observed when uprolling is done as give below 5 lines
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFF0000, sum, 8);
    sum += __shfl_down_sync(0xFF000000, sum, 4);
    sum += __shfl_down_sync(0xF0000000, sum, 2);
    sum += __shfl_down_sync(0x30000000, sum, 1);

    // Step 3: Store the result of each warp in shared memory
    if (lane == 0) sharedMem[warpId] = sum;

    __syncthreads();  // Ensure all warps have written to shared memory

    // Step 4: Perform final reduction across warps (if blockDim.x > 32)
    if (warpId == 0) {
        sum = (lane < ((blockDim.x + 31) / 32)) ? sharedMem[lane] : 0;

        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFF0000, sum, 8);
        sum += __shfl_down_sync(0xFF000000, sum, 4);
        sum += __shfl_down_sync(0xF0000000, sum, 2);
        sum += __shfl_down_sync(0x30000000, sum, 1);

        if (lane == 0) atomicAdd(newContribution + u, sum);
    }
}

__global__ void storePageRankValues(int *indexToVertexMap, float* renamedPageRank, float* actualPageRank){
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < numNodes){
        actualPageRank[indexToVertexMap[index]] = renamedPageRank[index];
     }
}

__global__ void findContribution(const int start, float* contribution, const int* outDegree, const int numNodes){
    const int u = start + threadIdx.x + blockDim.x * blockIdx.x;
    if(u < numNodes) contribution[u] /= outDegree[u]; 
}

// Should try exploiting openMp instead of below kernel as it is non-coelaced memeory access
__global__ void setMapping(const int *vertices, int *indexOfVertex, const int numNodes){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // non-coelaced access
    if(tid < numNodes) indexOfVertex[vertices[tid]] = tid;
}

#define NUM_THREADS 1024
#define GAMMA 3*1024
#define NUM_STREAMS 10

float* computePageRank(const int numNodes, const int numEdges, const std::pair <int, int>* edges){
    float* pageRank = (float*)calloc(numNodes, sizeof(float));

    cudaMemcpyToSymbol(::numNodes, &numNodes, sizeof(int));


    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Vertex mapping arrays
    thrust::device_vector <int> dVertices(numNodes);
    thrust::device_vector <int> dIndexOfVertex(numNodes);
    thrust::host_vector <int> indexOfVertex(numNodes);

    // inDegree of vertices
    thrust::host_vector <int> inDegree(numNodes + 1, 0); 
    thrust::device_vector <int> dInDegree(numNodes+1);

    // Row vector in CSR format
    thrust::device_vector <int> dIndexVector(numNodes + 1);
    thrust::host_vector <int> indexVector(numNodes + 1);

    // Column vector in CSR format
    thrust::device_vector <int> dInNeighbour(numEdges);
    thrust::host_vector <int> inNeighbour(numEdges);

    thrust::sequence(thrust::cuda::par.on(stream1), dVertices.begin(), dVertices.end());

    // Creating more streams
    cudaStream_t stream2, stream3, stream4;
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Computing inDegree on host
    for(int e = 0; e < numEdges; e++) inDegree[edges[e].second]++;

    // Copying inDegree to device
    // thrust::copy(thrust::cuda::par.on(stream2),inDegree.begin(), inDegree.end(), dInDegree.begin()) 
    cudaMemcpyAsync(thrust::raw_pointer_cast(dInDegree.data()), inDegree.data(), (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice, stream2);

    // Wait for stream1 work items to complete
    cudaStreamSynchronize(stream1);

    // Sort based on inDegree
    thrust::sort_by_key(thrust::cuda::par.on(stream2), dInDegree.begin(), dInDegree.end() - 1, dVertices.begin());
    cudaStreamSynchronize(stream2);

    // Copy back the inDegree to host
    // thrust::copy(thrust::cuda::par.on(stream2), dInDegree.begin(), dInDegree.end(), inDegree.begin());
    cudaMemcpyAsync(inDegree.data(), thrust::raw_pointer_cast(dInDegree.data()), numNodes * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    // Finding gammaPoint using binary search
    int gammaPoint = thrust::lower_bound(dInDegree.begin(), dInDegree.end() - 1, GAMMA) - dInDegree.begin();

    // Operations on rowVector
    thrust::exclusive_scan(thrust::cuda::par.on(stream1), dInDegree.begin(), dInDegree.end(), dIndexVector.begin());
    cudaStreamSynchronize(stream2);

    thrust::transform(
        thrust::cuda::par.on(stream1),
        dInDegree.begin(), dInDegree.end(),
        dIndexVector.begin(),
        dInDegree.begin(),
        [] __device__ (int inDegree, int index) { return -1 * inDegree + index; }
    );

    // Copy back indices to host
    // thrust::copy(thrust::cuda::par.on(stream1), dInDegree.begin(), dInDegree.end(), indexVector.begin());
    cudaMemcpyAsync(indexVector.data(), thrust::raw_pointer_cast(dInDegree.data()), (numNodes + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    // Getting index of each vertex
    setMapping <<< (numNodes + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream2 >>> (thrust::raw_pointer_cast(dVertices.data()), thrust::raw_pointer_cast(dIndexOfVertex.data()), numNodes);

    // Copying the index back to host
    // thrust::copy(thrust::cuda::par.on(stream2), dIndexOfVertex.begin(), dIndexOfVertex.end(), indexOfVertex.begin());
    cudaMemcpyAsync(indexOfVertex.data(), thrust::raw_pointer_cast(dIndexOfVertex.data()), (numNodes) * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
 
    // cudaFreeAsync(stream2);

    thrust::host_vector <int> outDegree(numNodes, 0);
    thrust::device_vector <int> dOutDegree(numNodes);

    for(int e = 0; e < numEdges; e++){
        int uIndex = indexOfVertex[edges[e].first];
        int vIndex = indexOfVertex[edges[e].second];

        inNeighbour[indexVector[vIndex]++ + inDegree[vIndex]] = uIndex;
        outDegree[uIndex]++;
    }

    int* dIndexVectorPtr = thrust::raw_pointer_cast(dIndexVector.data());
    int* dInNeighbourPtr = thrust::raw_pointer_cast(dInNeighbour.data());
    int* dOutDegreePtr = thrust::raw_pointer_cast(dOutDegree.data());

    cudaMemcpyAsync(dInNeighbourPtr, inNeighbour.data(), numEdges*sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dOutDegreePtr, outDegree.data(), numNodes*sizeof(int), cudaMemcpyHostToDevice, stream2);

    float* contributionArrayOne;
    cudaMallocAsync(&contributionArrayOne, numNodes * sizeof(float), stream2);
    initializeContribution <<< (numNodes + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, stream2 >>> (contributionArrayOne, dOutDegreePtr, numNodes);

    float* contributionArrayTwo;
    cudaMallocAsync(&contributionArrayTwo, numNodes * sizeof(float), stream3);

    float* dPageRank;
    cudaMallocAsync(&dPageRank, numNodes * sizeof(float), stream4);

    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream1);


    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&streams[i]);
    }

    bool useSecondArray = true;

    // the below loops can be unrolled by two times
    if(gammaPoint != numNodes){
        // MAX_ITER - 1 times
        for(int iter = 0; iter < MAX_ITER - 1; iter++){
            if(useSecondArray){
                for(int u = gammaPoint; u < numNodes; u++){
                    sumReduction <<< (inDegree[u] + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, streams[u % NUM_STREAMS] >>> (indexVector[u], indexVector[u+1], dInNeighbourPtr, contributionArrayOne, contributionArrayTwo, u); 
                }   

                cudaDeviceSynchronize();
                findContribution <<< (numNodes - gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream1 >>> (gammaPoint, contributionArrayTwo, dOutDegreePtr, numNodes);
                pageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, dOutDegreePtr, contributionArrayOne, contributionArrayTwo, numNodes);
            }else{
                for(int u = gammaPoint; u < numNodes; u++){
                    sumReduction <<< (inDegree[u] + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, streams[u % NUM_STREAMS] >>> (indexVector[u], indexVector[u+1], dInNeighbourPtr, contributionArrayTwo, contributionArrayOne, u); 
                }   

                cudaDeviceSynchronize();
                findContribution <<< (numNodes - gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream1 >>> (gammaPoint, contributionArrayOne, dOutDegreePtr, numNodes);   
                pageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, dOutDegreePtr, contributionArrayTwo, contributionArrayOne, numNodes);
            }   
            useSecondArray = !(useSecondArray);
            cudaDeviceSynchronize();
        }   
        if(useSecondArray){
             for(int u = gammaPoint; u < numNodes; u++){
                 sumReduction <<< (inDegree[u] + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, streams[u % NUM_STREAMS] >>> (indexVector[u], indexVector[u+1], dInNeighbourPtr, contributionArrayOne, contributionArrayTwo, u); 
             }   
            finalPageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, contributionArrayOne, contributionArrayTwo, numNodes);
            cudaDeviceSynchronize();
            storePageRankValues <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >>> (thrust::raw_pointer_cast(dVertices.data()), contributionArrayTwo, dPageRank);
        }else{
            for(int u = gammaPoint; u < numNodes; u++){
                sumReduction <<< (inDegree[u] + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS, 0, streams[u % NUM_STREAMS] >>> (indexVector[u], indexVector[u+1], dInNeighbourPtr, contributionArrayTwo, contributionArrayOne, u); 
            }   
 
            finalPageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, contributionArrayTwo, contributionArrayOne, numNodes);
            cudaDeviceSynchronize();
            storePageRankValues <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS >>> (thrust::raw_pointer_cast(dVertices.data()), contributionArrayOne, dPageRank);
        }   
        cudaMemcpy(pageRank, dPageRank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
        return pageRank;
    }

    // MAX_ITER - 1 times
    for(int iter = 0; iter < MAX_ITER - 1; iter++){
        if(useSecondArray){
            pageRankKernel <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, dOutDegreePtr, contributionArrayOne, contributionArrayTwo, numNodes);
        }else{
            pageRankKernel <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, dOutDegreePtr, contributionArrayTwo, contributionArrayOne, numNodes);
        }
        useSecondArray = !(useSecondArray);
    }

    if(useSecondArray){
        finalPageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, contributionArrayOne, contributionArrayTwo, numNodes);
        storePageRankValues <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (thrust::raw_pointer_cast(dVertices.data()), contributionArrayTwo, dPageRank);
    }else{
        finalPageRankKernel <<< (gammaPoint + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (dIndexVectorPtr, dInNeighbourPtr, contributionArrayTwo, contributionArrayOne, numNodes);
        storePageRankValues <<< (numNodes + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream2 >>> (thrust::raw_pointer_cast(dVertices.data()), contributionArrayOne, dPageRank);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(pageRank, dPageRank, numNodes * sizeof(float), cudaMemcpyDeviceToHost);

    return pageRank;

    // Testing zone
    printf("New of each vertex:\n");
    for(int u = 0; u < numNodes; u++){
        printf("indexOfVertex[%d] = %d\n", u, indexOfVertex[u]);
    }

    printf("\nindexVector (Row vector):\n");
    for(int i = 0; i <= numNodes; i++) printf("%d ", indexVector[i]);
    printf("\n");

    printf("\ninNeighbour (Col vector):\n");
    for(int i = 0; i < numEdges; i++) printf("%d ", inNeighbour[i]);
    printf("\n");

    printf("\nGamma point: %d\n", gammaPoint);
    return pageRank;

    //  #pragma omp parallel for 
    //    for(int i = 0; i < numNodes; i++){
    //        // printf("%d\n", omp_get_thread_num());
    //        indexOfVertex[vertices[i]] = i;
    //     }
}

void printPageRank(const int numNodes, const float* pageRank){
    printf("\nFinal pageRank values:\n");
    for(int u = 0; u < numNodes; u++) printf("pageRank[%d] = %.6f\n", u, pageRank[u]);
    return;
}

// assumptions:
// 1) the graph is unweighted, directed.
// 2) the graph may have multiple edges, self loops.
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
