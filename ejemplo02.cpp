#include <mpi.h>
#include <stdio.h>  // Incluimos la biblioteca para printf
#include <string>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    int data[100];

    // Utilizamos printf para imprimir el mensaje
    //printf("Rank %d of %d processes\n", rank, nprocs);

    if(rank == 0){
        std::printf("total ranks:%d\n ", nprocs);


        for (int i = 0; i<100; i++){
            data[i] = i;
        }


        for(int i = 1; i < nprocs; i++){
            std::printf("RANK_0 enviando datos a RANK_%d\n ",i);
            MPI_Send(data //datos
                    , 100 //count
                    , MPI_INT  //tipo de dato
                    , 1 //rank-destino
                    , 0 //TAG
                    , MPI_COMM_WORLD //grupo
            );
        }


    }else{

        std::printf("RANK_%d recibiendo datos\n ", rank);

        MPI_Recv(data //datos
                , 100 //count
                , MPI_INT //tipo de dato
                , 0  //rank-origen
                , 0  //TAG
                , MPI_COMM_WORLD //grupo
                , MPI_STATUS_IGNORE //estado
        );

        std::string str = "";

        for(int i = 0; i<10; i++){
            str = str + std::to_string(data[i]) +",";
        }

        std::printf("RANK_%d recibiendo datos ==>%s\n", rank, str.c_str());

    }

    MPI_Finalize();

    return 0;
}