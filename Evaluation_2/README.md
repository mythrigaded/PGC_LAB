# Evaluation 2 – Parallel Computing

## Team 6

### Problem Statement

Implement and analyze a parallel computing solution using an appropriate parallel programming model. The objective is to improve computational performance by dividing work among multiple processing units.

---

## Parallel Model

**Parallel Programming using MPI / OpenMP**

The parallel programming model is used to divide computational tasks and execute them concurrently to improve performance.

---

## Objectives

1. Understand the selected parallel computing problem.
2. Implement a sequential solution.
3. Implement a parallel solution.
4. Divide the workload among multiple processes or threads.
5. Execute the program with different configurations.
6. Measure execution time.
7. Calculate speedup and efficiency.
8. Compare sequential and parallel performance.
9. Analyze the results.

---

## Technologies Used

- C++
- MPI / OpenMP
- GCC Compiler
- VS Code
- macOS Terminal
- Git and GitHub

---

## Algorithm

1. Initialize the input data.
2. Execute the sequential implementation.
3. Initialize the parallel environment.
4. Divide the workload among processes or threads.
5. Perform local computations.
6. Combine the computed results.
7. Display the final output.
8. Measure execution time.
9. Compare sequential and parallel execution.

---

## Implementation

The experiment consists of two implementations:

### Sequential Approach

The sequential implementation performs all computations using a single process.

### Parallel Approach

The parallel implementation distributes the workload across multiple processes or threads. Each processing unit performs its assigned computation, and the results are combined to obtain the final output.

---

## Compilation and Execution

### Sequential Program

```bash
g++ sequential.cpp -o sequential
./sequential
```

### Parallel Program

Use the appropriate compiler and execution command depending on the parallel programming model.

For MPI:

```bash
mpic++ parallel.cpp -o parallel
mpirun -np 4 ./parallel
```

For OpenMP:

```bash
g++ -fopenmp parallel.cpp -o parallel
./parallel
```

---

## Experiments Performed

The program was tested using different input sizes and parallel configurations.

The execution time was recorded for each configuration to analyze performance.

---

## Performance Analysis

The performance of the sequential and parallel implementations is compared using execution time, speedup, and efficiency.

### Speedup

Speedup is calculated using:

\[
Speedup = \frac{Sequential\ Time}{Parallel\ Time}
\]

### Efficiency

Efficiency is calculated using:

\[
Efficiency = \frac{Speedup}{Number\ of\ Processes} \times 100
\]

---

## Result Verification

The output of the parallel implementation is compared with the sequential implementation to verify correctness.

---

## Graphs

The performance graphs generated for the experiment are stored in the graphs folder.

- execution_time.png
- speedup.png
- efficiency.png

---

## Folder Structure

```text
Evaluation_2/
│
├── README.md
├── data/
├── graphs/
├── report/
├── results/
└── src/
    ├── sequential.cpp
    └── parallel.cpp
```

---

## Conclusion

The experiment demonstrates the implementation of a parallel computing solution and its comparison with a sequential approach. Different input sizes and parallel configurations are tested. Execution time, speedup, and efficiency are used to evaluate performance.

The experiment helps understand how parallel processing distributes computational workloads and how communication overhead and workload size influence overall performance.

**Team 6**
