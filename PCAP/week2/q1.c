#include"mpi.h"
#include<stdio.h>
#include<string.h>
void reverse_word(char* word) {
    int len = strlen(word);
    for (int i = 0; i < len / 2; i++) {
        char temp = word[i];
        word[i] = word[len - i - 1];
        word[len - i - 1] = temp;
    }
}

int main(int argc, char*argv[]){
int rank,size;
char word[25];
int len;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);
MPI_Status status;
if(rank==0){
printf("Enter a word:");
scanf("%s",word);
len=strlen(word);
MPI_Ssend(&len,1,MPI_INT,1,1,MPI_COMM_WORLD);
MPI_Ssend(word,len,MPI_CHAR,1,1,MPI_COMM_WORLD);
MPI_Recv(word, len, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
printf("received word:%s",word);
}
else if (rank == 1) { 
MPI_Recv(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
MPI_Recv(word, len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);
word[len]='\0';
reverse_word(word);
printf("received word:%s",word);
MPI_Ssend(word, len, MPI_CHAR, 0, 2, MPI_COMM_WORLD);

    }
MPI_Finalize();
return 0;
}
