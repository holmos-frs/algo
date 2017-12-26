#include <iostream>
#include <mpi.h>

using std::cout;
using std::endl;

int work_data[] = {2, 22, 22, 222, 3, 4, 5, 4, 4, 34, 1};

void slave_program(void) {
    int number;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cout << "Received number " << number << endl;
}

void master_program(void) {
    int number = 255;
    cout << "Sending number " << number << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);

    cout << "This is " << processor_name << ", rank " << world_rank << " out of " << world_size << "." << endl;

    if(world_rank == 0) {
	master_program();
    } else {
	slave_program();
    }

    MPI_Finalize();

    return 0;
}

