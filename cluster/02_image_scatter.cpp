#include <iostream>
#include <cassert>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mpi.h>

using std::cout;
using std::cerr;
using std::endl;
using namespace cv;

int work_data[] = {1,2,3,4,5,6,7};
float *image;

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

    if(world_rank==0) {
	/* Read the image from a file */
	image = (float *) malloc(1024*1024*sizeof(float));
	if(image == NULL) {
	    cerr << "unale to allocate image on root" << endl;
	    exit(-1);
	}
	Mat imag;
	imag = imread("../holmos_raw.png", 0);
	cout << "Matrix is continuous? " << imag.isContinuous() << endl;
	assert(imag.isContinuous() == true);
	for(int i=0; i<1024*1024; i++) {
	    image[i] = imag.at<char>(i) / 255.0;
	}
	cout << "Pixel 1: " << image[0] << endl;

    }

    cout << "Ready for debug:\t"<<world_rank<<"\t"<<getpid()<<endl;
    bool val=true;

    int data_size = 1024*1024;
    if((data_size % world_size) != 0) {
	cerr << "Error, data_size can't be distributed evenly over a cluster of "
	    << world_size << " processors." << endl;
	exit(-1);
    }
    float *image_fragment = (float *) malloc(1024*1024*sizeof(float)/world_size);
    if(image_fragment == NULL) {
	cout << "image fragment allocation failed" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(image, data_size / world_size, MPI_FLOAT,
	    image_fragment, data_size/world_size, MPI_FLOAT,
	    0, MPI_COMM_WORLD);
    
    
    for(int i=0; i<(data_size / world_size); i++) {
	image_fragment[i] += 0.1;
    }

    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(image_fragment, data_size / world_size, MPI_FLOAT,
	    image, data_size/world_size, MPI_FLOAT,
	    0, MPI_COMM_WORLD);

    if(world_rank == 0)
	cout << "1st Pixel: " << image[0] << endl;


    MPI_Finalize();
    free(image);
    free(image_fragment);

    return 0;
}

