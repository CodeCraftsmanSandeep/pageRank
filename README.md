# Exploiting Parallelism on GPUs for Graph Algorithms: PageRank Implementation

## Introduction
In many scenarios, we need to determine the relative importance or rank of certain elements, such as the sorted order of weights or the ranking of research papers based on citations. One prominent use case is ranking the nodes (or pages) of a graph based on their importance, and the **PageRank** algorithm is a well-known approach to achieving this. 

The PageRank algorithm was originally introduced to rank web pages in search engines based on their importance, using a graph structure where nodes represent web pages, and edges represent hyperlinks between them.

### Definitions
The following definitions are made in the context of graphs and search engines:
- **page\_rank[p]**: The PageRank value for a node indicates the probability that a random surfer will land on that particular node at any given time during their traversal of the graph.

## Project Overview
This project explores the use of GPU parallelism to accelerate the execution of graph algorithms, with a specific focus on the **PageRank** algorithm. By leveraging CUDA and parallel computing techniques, the aim is to significantly reduce the time required to compute the ranking of nodes in large-scale graphs. The project compares the performance of serial and parallel implementations, demonstrating the performance improvements achieved through GPU parallelism.

## Key Algorithm Implemented
1. **PageRank**  
   - Multiple parallel implementations of the PageRank algorithm have been developed using CUDA to compute the importance of nodes in a graph. 
   - These implementations exploit GPU threads for efficient parallel computation, which results in faster convergence compared to traditional serial approaches.

## Datasets Used
The project utilizes two publicly available graph datasets to test and benchmark the performance of the PageRank algorithm. These datasets represent real-world scenarios, enabling meaningful performance comparisons between serial and parallel implementations.

## Performance Comparisons
This project provides a detailed comparison of execution times between serial and parallel implementations of the PageRank algorithm. By utilizing GPU parallelism, the following improvements were observed:
- Faster convergence for large datasets due to simultaneous processing of multiple nodes.
- Improved scalability as the size of the graph increases.

## Technologies Used
- **CUDA**: Used for GPU programming and parallelism.
- **C++**: Employed for implementing both the serial and parallel versions of the PageRank algorithm.
- **LaTeX**: Used for documentation, including the preparation of the interim thesis.
- **Public graph datasets**: Used for performance benchmarking and testing.


## Results

The project contains both serial and multiple parallel implementations of the PageRank algorithm. Each implementation can be found in the `pageRankImplementations` directory, with file names corresponding to the version described below. The table provides a brief summary of each version, the approach used, and the results observed on two datasets: **Graph 2** and **Facebook**.

| S.No | Code Path                               | Description                                                                                                                                                                                                                   | Graph 2 | Facebook |
|------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------|
| 0    | [`serial.cpp`](pageRankImplementations/serial.cpp)    | This code is the sequential implementation of the PageRank algorithm. It serves as a baseline for benchmarking and ensuring sequential consistency with parallel implementations.                                               | 0.657086| 0.689932 |
| 1    | [`parallel_v1.cu`](pageRankImplementations/parallel_v1.cu) | 1) Launches a CUDA kernel where each thread computes the new PageRank value for a unique vertex.<br>2) Uses an `out_degree` array to store and access the out-degree of each vertex during computation.                          | 0.2277  | 0.190497 |
| 2    | [`parallel_v2.cu`](pageRankImplementations/parallel_v2.cu) | 1) Assigns each CUDA thread a unique vertex to compute its new PageRank value.<br>2) Swaps old and new PageRank values in GPU memory to avoid `cudaMemcpyDeviceToDevice`.<br>3) Improves spatial locality by storing out-degree next to old PageRank values, but this increases space utilization and may negatively impact memory coalescing. | 0.200191| 0.200105 |
| 3    | [`parallel_v3.cu`](pageRankImplementations/parallel_v3.cu) | 1) Each thread computes the new PageRank value for a unique vertex.<br>2) A second kernel reduces floating-point divisions by precomputing contributions.<br>3) Uses separate memory for storing out-degree values.                | 0.172564| 0.16501  |
| 4    | [`parallel_v4.cu`](pageRankImplementations/parallel_v4.cu) | 1) Each thread computes the PageRank contribution of a vertex to its neighbors.<br>2) A separate kernel computes the final PageRank in the last iteration.<br>3) Swaps old and new PageRank values to avoid memory copying.       | 0.161187| 0.160348 |
| 5    | [`parallel_v5.cu`](pageRankImplementations/parallel_v5.cu) | 1) Computes the PageRank contribution of each vertex using a kernel.<br>2) Uses a serial loop or dynamic parallelism (warp-level primitives) for vertices with low in-degree.<br>3) This version currently has race conditions.    | -       | -        |

### Notes

- The **serial** implementation is used as a baseline for performance comparisons.
- Parallel versions progressively optimize performance by minimizing memory transfers, improving spatial locality, and reducing computational overhead.
- Execution times for different datasets (Graph 2 and Facebook) are shown to compare performance.




## Conclusion
This project highlights the significant performance enhancements that can be achieved by leveraging GPU parallelism for graph algorithms, such as PageRank. The comparison between serial and parallel implementations demonstrates the potential of GPUs in efficiently handling large-scale data processing tasks. As this is an interim thesis, further improvements and additional algorithms are planned for future exploration.

## References
1. [The Anatomy of a Large-Scale Hypertextual Web Search Engine by Sergey Brin and Lawrence Page](http://infolab.stanford.edu/~backrub/google.html)