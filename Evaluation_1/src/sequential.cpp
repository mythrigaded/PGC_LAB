#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace chrono;

int main(int argc, char* argv[]) {

    // Dataset size
    int n = 1000;

    // Allow dataset size from command line
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    // Create dataset
    vector<int> data(n);

    // Fill dataset with values from 1 to 100 repeatedly
    for (int i = 0; i < n; i++) {
        data[i] = (i % 100) + 1;
    }

    // Start execution timer
    auto start_time = high_resolution_clock::now();

    // Calculate statistics
    int sum = accumulate(data.begin(), data.end(), 0);

    double average = (double)sum / n;

    int maximum = *max_element(data.begin(), data.end());

    int minimum = *min_element(data.begin(), data.end());

    // End execution timer
    auto end_time = high_resolution_clock::now();

    double execution_time =
        duration<double>(end_time - start_time).count();

    // Display results
    cout << "Sequential Dataset Statistics" << endl;
    cout << "-------------------------------" << endl;
    cout << "Dataset size        : " << n << endl;
    cout << "Sum                 : " << sum << endl;
    cout << "Average             : " << average << endl;
    cout << "Maximum             : " << maximum << endl;
    cout << "Minimum             : " << minimum << endl;
    cout << "Execution time      : "
         << execution_time << " seconds" << endl;

    return 0;
}