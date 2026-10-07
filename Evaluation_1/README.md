# Evaluation 1 – Distributed Dataset Statistics using MPI

## Team 6

### Problem Statement

Calculate basic statistics of a distributed dataset using multiple MPI processes.

The statistics calculated are:

- Sum
- Average
- Maximum
- Minimum

## Parallel Model

**MPI (Message Passing Interface)**

MPI is used to distribute the dataset among multiple processes and combine the locally calculated results.

---

## Objectives

1. Implement a sequential solution for dataset statistics.
2. Implement a parallel solution using MPI.
3. Distribute the dataset among multiple MPI processes.
4. Calculate local statistics at each process.
5. Combine local results to obtain global statistics.
6. Test different dataset sizes and process counts.
7. Measure execution time.
8. Calculate speedup and efficiency.
9. Analyze the parallel performance.

---

## Technologies Used

- C++
- MPI
- Open MPI
- VS Code
- macOS Terminal

---

## Dataset

The program generates the dataset automatically.

The values are generated using:

```cpp
data[i] = (i % 100) + 1;
For the performance analysis, a dataset of 100,000 elements was used.
Expected statistics:
Statistic	Value
Sum	5,050,000
Average	50.5
Maximum	100
Minimum	1

Sequential Approach
The sequential program performs all calculations using a single process.
Algorithm
1. Generate the dataset.
2. Calculate the sum.
3. Calculate the average.
4. Find the maximum value.
5. Find the minimum value.
6. Display the results.
7. Measure the execution time.

MPI Parallel Approach
The MPI implementation divides the dataset among multiple processes.
Steps
1. Initialize MPI.
2. Obtain the process rank.
3. Obtain the total number of processes.
4. Generate the dataset.
5. Divide the dataset among processes using MPI_Scatter().
6. Calculate the local sum, maximum and minimum.
7. Combine the local results using MPI_Reduce().
8. Calculate the global average.
9. Display the final results using process 0.
10. Measure execution time.
11. Finalize MPI.
MPI Functions Used
MPI Function	Purpose
MPI_Init()	Initializes the MPI environment
MPI_Comm_rank()	Obtains the rank of the current process
MPI_Comm_size()	Obtains the total number of MPI processes
MPI_Wtime()	Measures MPI execution time
MPI_Scatter()	Distributes portions of the dataset
MPI_Reduce()	Combines local results into global results
MPI_Finalize()	Terminates the MPI environment


Compilation
Sequential Program
g++ sequential.cpp -o sequential
MPI Program
mpic++ mpi_statistics.cpp -o mpi_statistics
Execution
Sequential Execution
./sequential 100000
MPI Execution
Example using 4 processes:
mpirun -np 4 ./mpi_statistics 100000
Here, -np 4 starts four MPI processes.
Experiments Performed
The program was tested with different dataset sizes:
- 1,000 elements
- 10,000 elements
- 100,000 elements
The 100,000-element dataset was used for the main performance analysis with:
- 1 MPI process
- 2 MPI processes
- 4 MPI processes
- 8 MPI processes
Performance Results
For 100,000 elements, the measured sequential execution time was:
0.00346217 seconds
The MPI execution results were:
Configuration	Processes	Execution Time (seconds)
Sequential	1	0.00346217
MPI	1	0.004973
MPI	2	0.004275
MPI	4	0.015596
MPI	8	0.015119


Speedup
Speedup is calculated using:
Speedup = Sequential Execution Time / Parallel Execution Time
The measured speedup values were:
Processes	Speedup
1	0.6962
2	0.8099
4	0.2220
8	0.2290
Efficiency
Efficiency is calculated using:

Efficiency = (Speedup / Number of Processes) × 100
The measured efficiency values were:
Processes	Efficiency
1	69.62%
2	40.49%
4	5.55%
8	2.86%
Performance Analysis
The sequential implementation was faster than the MPI implementation for this experiment.
Increasing the number of MPI processes did not continuously decrease the execution time.
The MPI implementation introduces communication and process-management overhead. Since the experiment was performed locally on a single machine, this overhead affected the overall execution time.
Therefore, the experiment demonstrates that parallel processing does not always guarantee speedup. The workload must be sufficiently large to overcome the overhead introduced by parallel execution.
Result Verification
The calculated statistics were correct for the tested configurations:
Sum     : 5050000
Average : 50.5
Maximum : 100
Minimum : 1

Graphs
The performance graphs generated for the experiment are stored in the graphs folder:
- execution_time_vs_processes.png
- speedup_vs_processes.png
- efficiency_vs_processes.png
Folder Structure
Evaluation_1/
│
├── README.md
│
├── data/
│
├── graphs/
│   ├── execution_time_vs_processes.png
│   ├── speedup_vs_processes.png
│   └── efficiency_vs_processes.png
│
├── presentation/
│   └── Distributed_Dataset_Statistics_MPI.pptx
│
├── report/
│
├── results/
│   └── results.csv
│
└── src/
    ├── sequential.cpp
    └── mpi_statistics.cpp

Conclusion
The Distributed Dataset Statistics problem was implemented using both sequential C++ and MPI.
The MPI implementation demonstrates dataset distribution using MPI_Scatter(), local computation, and combination of results using MPI_Reduce().
Different dataset sizes and MPI process counts were tested. Execution time, speedup and efficiency were measured and analyzed.
The experiment showed that MPI performance depends on workload size and parallelization overhead. For the tested local workload, the sequential implementation performed better than the MPI implementation.

