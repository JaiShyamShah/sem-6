#include<stdio.h>
#include<stdlib.h>
#include "mpi.h"

int factorial(int n){
    int res=1;
    for(int i=n; i>=1; i--){
        res = res*i;
    }
    return res;
}

int main( int argc, char* argv[] ){
    int rank, size, N, A[10], R[10], c, copy;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // R00T Process
    if(rank==0){
        printf("Using the number of elements as no. of processes\n");
        N = size;

        for(int i = 0 ; i<N ; i++){
            scanf("%d", &A[i]);
        }
    }
    
    // ?! This statement is not part of RANK 0's execution only.
    // RecvBuff is c since recieving processes are only recieveing ONE ELEMENT and not multiple.
    MPI_Scatter(A, 1, MPI_INT, &c, 1, MPI_INT, 0, MPI_COMM_WORLD);

    fprintf(stdout, "Process %d recieved value %d\n", rank, c);
    fflush(stdout);
    copy = c;
    c = factorial(c);

    MPI_Gather(&c, 1, MPI_INT, R, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d calculated factorial of %d -> %d\n", rank, copy, c);

    if (rank==0){   
    int sum=0;
    for (int i = 0 ; i<N ; i++){
        // printf("\n\nELEMENT %d: %d", i, R[i]);
        sum+=R[i];
    }

    printf("\nRoot Prints Sum of Factorials: %d\n", sum);}
    MPI_Finalize();
    return 0;
}
