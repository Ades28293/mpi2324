

#include <mpi.h>
#include <vector>
#include <functional>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <string>


std::vector<int> read_file() {
    std::fstream fs("./datos.txt", std::ios::in);

    std::string line;

    std::vector<int> ret;

    while (std::getline(fs, line)) {
        ret.push_back(std::stoi(line));
    }
    fs.close();
    return ret;
}

int sumar(int *tmp, int n) {
    int suma = 0;
    for (int i = 0; i < n; i++) {
        suma += tmp[i];
    }
    return suma;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        std::printf("Numero total de ranks: %d\n", nprocs);

        std::vector<int> datos = read_file();

        int rango = datos.size();
        int blocksize = std::ceil(rango / nprocs);

        std::printf("Rango: %d, Bloque: %d\n", rango, blocksize);

        int *data = datos.data();

        for (int rank_id = 1; rank_id < nprocs; rank_id++) {
            std::printf("RANK_0 enviando datos a RANK_%d\n ", rank_id);
            int start = (rank_id - 1) * blocksize;
            MPI_Send(&data[start], blocksize, MPI_INT, rank_id, 0, MPI_COMM_WORLD);
        }

        int suma_ranks[nprocs];

        suma_ranks[0] = sumar(data + ((nprocs - 1) * blocksize), rango - ((nprocs - 1) * blocksize));

        for (int rank_id = 1; rank_id < nprocs; rank_id++) {
            MPI_Recv(&suma_ranks[rank_id], 1, MPI_INT, rank_id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::string str = "";
        for (int i = 0; i < nprocs; i++) {
            str = str + std::to_string(suma_ranks[i]) + ", ";
        }
        std::printf("sumas parciales: %s\n", str.c_str());

        int sumaTotal = sumar(suma_ranks, nprocs);

        std::printf("Suma total: %d\n", sumaTotal);

    } else {
        MPI_Status status;
        int buffer;

        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &buffer);

        int *data = new int[buffer];

        MPI_Recv(data, buffer, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int suma_parcial = sumar(data, buffer);

        MPI_Send(&suma_parcial, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}