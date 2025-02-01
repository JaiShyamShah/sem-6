//write a mpi program using n processes to find 1!+2!+ ...n! using mpi reduce
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, n = 3;  // Compute sum of factorials up to n!
    int local_value, scan_factorial, total_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num = rank + 1;  // Each rank represents a number 1 to n

    // Use MPI_Scan to compute the prefix product (factorial)
    MPI_Scan(&num, &scan_factorial, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    // Ensure ranks beyond n contribute nothing
    if (rank + 1 > n) {
        scan_factorial = 0;
    }

    // Use MPI_Reduce to sum up all factorials at rank 0
    MPI_Reduce(&scan_factorial, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final result
    if (rank == 0) {
        printf("Sum of factorials from 1! to %d! is: %d\n", n, total_sum);
    }

    MPI_Finalize();
    return 0;
}

