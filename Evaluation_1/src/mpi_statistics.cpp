#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>

using namespace std;

int main(int argc, char* argv[]) {

    // Start MPI
    MPI_Init(&argc, &argv);

    // Get process ID and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Start execution timer
    double start_time = MPI_Wtime();

    // Dataset size
    int n = 1000;

    // Allow dataset size to be given from command line
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    // Create dataset
    vector<int> data(n);

    // Fill dataset with values from 1 to 100 repeatedly
    for (int i = 0; i < n; i++) {
        data[i] = (i % 100) + 1;
    }

    // Make sure dataset can be divided equally
    if (n % size != 0) {

        if (rank == 0) {
            cout << "Number of processes must divide dataset size." << endl;
        }

        MPI_Finalize();
        return 0;
    }

    // Number of elements handled by each process
    int local_n = n / size;

    // Local array for each process
    vector<int> local_data(local_n);

    // Distribute data among processes
    MPI_Scatter(
        data.data(),
        local_n,
        MPI_INT,
        local_data.data(),
        local_n,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // Calculate local sum
    int local_sum = accumulate(
        local_data.begin(),
        local_data.end(),
        0
    );

    // Calculate local maximum
    int local_max = *max_element(
        local_data.begin(),
        local_data.end()
    );

    // Calculate local minimum
    int local_min = *min_element(
        local_data.begin(),
        local_data.end()
    );

    // Variables for final results
    int global_sum;
    int global_max;
    int global_min;

    // Combine local sums
    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // Combine local maximum values
    MPI_Reduce(
        &local_max,
        &global_max,
        1,
        MPI_INT,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    // Combine local minimum values
    MPI_Reduce(
        &local_min,
        &global_min,
        1,
        MPI_INT,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    // End execution timer
    double end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    // Process 0 displays final results
    if (rank == 0) {

        double average = (double)global_sum / n;

        cout << "Distributed Dataset Statistics" << endl;
        cout << "-------------------------------" << endl;
        cout << "Number of processes : " << size << endl;
        cout << "Dataset size        : " << n << endl;
        cout << "Sum                 : " << global_sum << endl;
        cout << "Average             : " << average << endl;
        cout << "Maximum             : " << global_max << endl;
        cout << "Minimum             : " << global_min << endl;
        cout << "Execution time      : "
             << execution_time << " seconds" << endl;
    }

    // End MPI
    MPI_Finalize();

    return 0;
}