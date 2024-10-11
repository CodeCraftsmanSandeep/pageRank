# Exploiting Parallelism on GPUs for Graph Algorithms

## Project Overview
This project explores the use of GPU parallelism to accelerate the execution of graph algorithms, specifically focusing on the **PageRank** algorithm. By utilizing CUDA and parallel computing techniques, the aim is to significantly reduce computation time for ranking nodes in large graphs. The project presents a comparison between serial and parallel implementations, highlighting the performance improvements achieved through GPU parallelism.

## Key Algorithm Implemented
1. **PageRank**  
   - Multiple parallel implementations of the PageRank algorithm have been developed using CUDA to compute the importance of nodes in a graph.
   - These implementations exploit GPU threads for efficient parallel computation, resulting in faster convergence compared to serial approaches.

## Datasets Used
The project employs two publicly available graph datasets for testing and benchmarking the performance of the PageRank algorithm. These datasets represent real-world scenarios and enable meaningful performance comparisons between serial and parallel executions.

## Performance Comparisons
The project provides a detailed comparison of execution times between serial and parallel implementations of the PageRank algorithm. By leveraging GPU parallelism, the following improvements have been observed:
- Faster convergence for large datasets due to simultaneous processing of multiple nodes.
- Improved scalability as the size of the graph increases.

## Technologies Used
- **CUDA** for GPU programming and parallelism.
- **C++** for implementing both serial and parallel versions.
- **LaTeX** for documentation, including the interim thesis.
- **Public graph datasets** for performance benchmarking.

## Conclusion
This project demonstrates that leveraging GPU parallelism can significantly enhance the performance of graph algorithms such as PageRank. The comparison of serial and parallel implementations illustrates the potential of GPUs in efficiently handling large-scale data processing tasks.

