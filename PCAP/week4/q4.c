#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, i, j, loc = 0;
    char str[40], res[40], ch[40];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if(rank == 0) {
        printf("Enter the string: ");
        scanf("%s", str);
    }

    MPI_Scatter(str, 1, MPI_CHAR, ch, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    for(i = 1; i <= rank; i++) {
        ch[i] = ch[0];
    }

    if(rank == 0) {
        res[0] = ch[0]; 
        loc = 1;

        for(i = 1; i < size; i++) {
            MPI_Recv(ch, i+1, MPI_CHAR, i, i, MPI_COMM_WORLD, &status); 
            for(j = 0; j < i+1; j++) {  
                res[loc] = ch[j];
                loc++;
            }
        }
        res[loc] = '\0';
        printf("Modified string is: %s\n", res);
    } else {
        MPI_Send(ch, rank+1, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}