#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char s1[100], s2[100], result[200]; 
    int len, chunk;

    if (rank == 0) {
        printf("Enter length: \n");
        scanf("%d", &len);

        printf("Enter S1: \n");
        scanf("%s", s1);

        printf("Enter S2: \n");
        scanf("%s", s2);

        chunk = len / size;
    }

    MPI_Bcast(&chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char sub1[chunk + 1], sub2[chunk + 1], res[2 * chunk + 1];

    MPI_Scatter(s1, chunk, MPI_CHAR, sub1, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, chunk, MPI_CHAR, sub2, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        res[2 * i] = sub1[i];       
        res[2 * i + 1] = sub2[i];   
    }
    res[2 * chunk] = '\0';  

    MPI_Gather(res, 2 * chunk, MPI_CHAR, result, 2 * chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result[2 * len] = '\0'; 
        printf("Final interleaved string: %s\n", result);
    }

    MPI_Finalize();
    return 0;
}

