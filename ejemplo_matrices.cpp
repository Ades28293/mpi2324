

#include <iostream>
#include <mpi.h>
#include <vector>
#include <functional>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <string>




#define MATRIX_DIMENSION 25

//MATRIZ POR VECTOR


void matrix_mult(double* A, double* b, double* c, int rows, int cols){
    for(int i = 0; i < rows; i++){
        double tmp = 0;
        for(int j = 0; j < cols; j++){
            tmp = tmp + A[i*cols+j]*b[j];
        }
        c[i] = tmp;
    }
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int row_per_rank;

    //memoria q vamos a solicitar
    int rows_alloc = MATRIX_DIMENSION;

    //padding llenar de 0
    int padding = 0;

    if(MATRIX_DIMENSION%nprocs!=0){

        rows_alloc = std::ceil((double)MATRIX_DIMENSION/nprocs)*nprocs;
        padding=rows_alloc-MATRIX_DIMENSION;
    }

    //siempre va hacer exacto y si no es redondea hacia arriba para q sea exacta
    //es divisible para el numero de procesos
    row_per_rank=rows_alloc/nprocs;

    //no debo inicializar la matriz o leer desde el disco
    //debe leer en el rank 0 ya que es el root y para q todos los ranks no tenga la matriz completa
    if(rank==0){
        //imprimir informacion
        std::printf("Dimension: %d, row_alloc: %d , row_per_rank %d, padding: %d\n",
                    MATRIX_DIMENSION,rows_alloc,row_per_rank,padding);


        //el aumento de columnas 25 a 28 3 de relleno
        //matriz A
        std::vector<double>A(MATRIX_DIMENSION*rows_alloc);

        //el vector de b es 25  la fila de la matriz tiene 25 y la columna 28
        std::vector<double>b(MATRIX_DIMENSION);

        std::vector<double>c(rows_alloc);

        for(int i=0;i<MATRIX_DIMENSION;i++){

            for(int j=0;j<MATRIX_DIMENSION;j++){
                int index=i*MATRIX_DIMENSION+j;
                A[index]=i;
            }

        }


        for(int i=0;i<MATRIX_DIMENSION;i++){
            b[i]=1;
        }

        //enviar la Matriz A
        //el rank 0 no se envia a si mismo xq ya tiene la matriz completa
        MPI_Scatter(A.data(),MATRIX_DIMENSION*row_per_rank,MPI_DOUBLE,//ENVIO
                    MPI_IN_PLACE,0,MPI_DOUBLE,//RECIVE
                    0,MPI_COMM_WORLD);//GRUPO Q CORDINA


        //enviar vecorB
        MPI_Bcast(b.data(),MATRIX_DIMENSION,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //realizar el calculo c=A x b
        matrix_mult(A.data(),b.data(),c.data(),row_per_rank,MATRIX_DIMENSION);



        //RECIBIR RESPUESTAS PARCIALES
        //rank 0 tiene las 7 primeras filas no va enviar nd
        MPI_Gather(MPI_IN_PLACE,0,MPI_DOUBLE,
                   c.data(),row_per_rank,MPI_DOUBLE,
                   0,MPI_COMM_WORLD);

        //AQUI CONTROLO LA BASURA YA Q LLEGA 7 en total 28 pero 3 son basura
        //para eso redimenciono para q llegue 25
        c.resize(MATRIX_DIMENSION);
        std::printf("resultado: \n");

        for(int i=0;i<MATRIX_DIMENSION;i++){
            std::printf("%.0f,",c[i]);
        }

    } else{
                //pedazo de matriz
                std::vector<double> A_local(MATRIX_DIMENSION*row_per_rank);

                std::vector<double>b_local(MATRIX_DIMENSION);
                std::vector<double>c_local(row_per_rank);


            MPI_Scatter(nullptr,0,MPI_DOUBLE,//Envio
                            A_local.data(),MATRIX_DIMENSION*row_per_rank,MPI_DOUBLE,
                            0,MPI_COMM_WORLD);

            std::printf("Rank_%d: [%.0f..%.0f]\n",rank,A_local[0],A_local.back());

            //RECIVO EL VECTOR B
        MPI_Bcast(b_local.data(),MATRIX_DIMENSION,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //realizar el calculo c=A x b
        //tener cuidado con el utlimo ya q hay 3 filas de relleno
        int row_per_rank_temp=row_per_rank;
        if(rank==nprocs-1){
            row_per_rank_temp=MATRIX_DIMENSION-rank*row_per_rank;
            //es lo mismo
           // row_per_rank_temp=MATRIX_DIMENSION-padding;

        }

        matrix_mult(A_local.data(),b_local.data(),c_local.data(),row_per_rank_temp,MATRIX_DIMENSION);

        //envio del calculo es decir el resultado
        //envio 7
        //LOS RANK
        MPI_Gather(c_local.data(),row_per_rank,MPI_DOUBLE,
                   nullptr,0,MPI_DOUBLE,//INGNORANDO
                   0,MPI_COMM_WORLD);


    }




    MPI_Finalize();

}