// column wise sum and output matrix
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[4][4];         
    int local_input[4];      
    int local_output[4];     
    int output_matrix[4][4];  

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process initializes the 4x4 matrix
    if (rank == 0) {
        printf("Enter the 4x4 matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Scatter the matrix row-wise to all processes
    MPI_Scatter(matrix, 4, MPI_INT, local_input, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform the scan operation to get the cumulative sum of each row
    MPI_Scan(local_input, local_output, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Gather the rows of the output matrix back to the root process
    MPI_Gather(local_output, 4, MPI_INT, output_matrix, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final output matrix
    if (rank == 0) {
        printf("Modified matrix (after cumulative sum scan operation):\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%d ", output_matrix[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
