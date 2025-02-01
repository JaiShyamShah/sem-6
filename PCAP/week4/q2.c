// write a mpi program to read a 3x3 matrix. 
// enter an element to be searched in the root process. 
// find the number of occurences of this element in the matrix using 3 processes
#include <mpi.h>
#include <stdio.h>

#define ROWS 3
#define COLS 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[ROWS][COLS];  // 3x3 matrix
    int row[COLS];           // Row to be received by each process
    int search_element;      // Element to search for
    int local_count = 0, total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process initializes matrix and takes user input
    if (rank == 0) {
        printf("Enter a 3x3 matrix:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Enter the element to search: ");
        scanf("%d", &search_element);
    }

    // Broadcast search element to all processes
    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix to each process
    MPI_Scatter(matrix, COLS, MPI_INT, row, COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts occurrences of the element in its row
    for (int i = 0; i < COLS; i++) {
        if (row[i] == search_element) {
            local_count++;
        }
    }

    // Reduce all local counts to get the final total at rank 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the total occurrences
    if (rank == 0) {
        printf("The element %d occurs %d times in the matrix.\n", search_element, total_count);
    }

    MPI_Finalize();
    return 0;
}
