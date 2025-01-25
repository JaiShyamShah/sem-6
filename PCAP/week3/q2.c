#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, M;
    int arr[100];                // Array to hold N * M elements (int array)
    int recv_buffer[20];         // Buffer to receive M elements for each process (int array)
    float local_avg, total_avg;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of elements per process (M): ");
        scanf("%d", &M);

        // Read size * M elements into the array (total size = size * M)
        printf("Enter the %d elements:\n", size * M);
        for (int i = 0; i < size * M; i++) {
            scanf("%d", &arr[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the size * M array to all processes
    MPI_Scatter(arr, M, MPI_INT, recv_buffer, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the local average for the M elements each process receives
    int sum = 0;
    for (int i = 0; i < M; i++) {
        sum += recv_buffer[i];
    }
    local_avg = (float)sum / M;

    // Gather all local averages in the root process
    float avg_buffer[100];  // Buffer to gather averages (float array)
    MPI_Gather(&local_avg, 1, MPI_FLOAT, avg_buffer, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Root process calculates the total average
    if (rank == 0) {
        total_avg = 0.0f;
        for (int i = 0; i < size; i++) {
            total_avg += avg_buffer[i];
        }
        total_avg /= size;

        // Print the total average
        printf("Total average: %.2f\n", total_avg);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
