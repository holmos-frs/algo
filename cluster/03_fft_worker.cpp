/******************************************************************************
 * CALCULATE A FFT WITH A CLUSTER
 ******************************************************************************
 * Constraints:
 *  - if the image size exceeds the int size
 *
 */
#include <iostream>
#include <string>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fftw3-mpi.h>

using namespace std;
using namespace cv;

/* MPI Variables */
int world_size, world_rank;
int *send_counts = NULL, *displs = NULL, *row_assignment;
double t0, t1;

/* FFTW Variables */
const ptrdiff_t n0 = 1024, n1 = 1024;
ptrdiff_t my_starting_row, my_row_count, my_buffer_size;
fftwf_plan plan;
fftwf_complex *buffer;
float *input, *image;

int main(int argc, char **argv) {
    /* Initialize the MPI cluster */
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /* Initialize FFTW */
    fftwf_mpi_init();
    my_buffer_size = fftwf_mpi_local_size_2d(n0, n1, MPI_COMM_WORLD,
	    &my_row_count, &my_starting_row);
    buffer = fftwf_alloc_complex(my_buffer_size);
    plan = fftwf_mpi_plan_dft_2d(n0, n1, buffer, buffer, MPI_COMM_WORLD,
	    FFTW_FORWARD, FFTW_ESTIMATE);

    row_assignment = new int[2];
    row_assignment[0] = my_starting_row;
    row_assignment[1] = my_row_count;
    cout << world_rank << ": " << my_starting_row << " " << my_row_count << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank != 0) {
	/* We need to tell the root process our row assignment */
	MPI_Send(row_assignment, 2, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
	/* As the root process, we need to gather all row assignments and create
	 * a sendcounts and displs buffer
	 */
	send_counts = new int[world_size];
	displs = new int[world_size];
	
	/* Add the root process' rows */
	displs[0] = row_assignment[0] * n1;
	send_counts[0] = row_assignment[1] * n1;

	for(int i=1; i<world_size; i++) {
	    /* For each processor, collect his row assignment and convert it
	     * into bytes */
	    MPI_Recv(row_assignment, 2, MPI_LONG, i, 0, MPI_COMM_WORLD,
		    MPI_STATUS_IGNORE);
	    displs[i] = row_assignment[0] * n1;
	    send_counts[i] = row_assignment[1] * n1;
	}
	for(int i=0; i<world_size; i++) {
	    cout << "Processor " << i << ": " << displs[i] << "\t"
		<< send_counts[i] << endl;
	}
    }

    /* The root process needs to load the image from a file */
    if(world_rank == 0) {
	image = new float[n0 * n1];
	Mat img;
	img = imread("../holmos_raw.png", 0);
	assert(img.cols == n0);
	assert(img.rows = n1);
	
	for(int i=0; i<n0*n1; i++)
	    image[i] = img.at<char>(i) / 255.0;
	cout << "Pixel 1 Proc 1: " << image[0] << endl
	    << "Pixel 1 Proc 2: " << image[512*1024] << endl;
    }
    t0 = MPI_Wtime();
    int frames=0;
    while(frames<300) {

	/* Allocate the input buffer */
	input = new float[my_row_count * n1];

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatterv(image, send_counts, displs, MPI_FLOAT,
		input, my_row_count * n1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	/* Now fill the data buffer for the FFT */
	memset(buffer, 0x00, my_buffer_size * sizeof(fftwf_complex));
	for(int row=my_starting_row; row<my_row_count; row++)
	    for(int column=0; column<n1; column++)
		buffer[row*n1+column][0] = input[row*n1+column];

	MPI_Barrier(MPI_COMM_WORLD);
	fftwf_execute(plan);
	frames++;
    }
    t1 = MPI_Wtime();
    cout << "FPS: " << frames / (t1 - t0) << endl;
    /*
	if(world_rank == 0) {
	    cout << "Took " << t1 - t0 << "s" << endl;
	    * Double the send_counts and displs array, as we send back a complex
	     * array with real and imaginary part *
	    for(int i=0; i<world_size; i++) {
		send_counts[i] *= 2;
		displs[i] *= 2;
	    }
	}
    Mat magnitude(my_row_count, n1, CV_32FC1);
    for(int y=0; y<my_row_count; y++)
	for(int x=0; x<n1; x++)
	    magnitude.at<float>(y*n1+x) = sqrt(
		    buffer[y*n1+x][0] * buffer[y*n1+x][0] +
		    buffer[y*n1+x][1] * buffer[y*n1+x][1]);
    string filename("test"+to_string(world_rank)+".png");
    imwrite(filename, magnitude);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(buffer, my_row_count * n1, MPI_FLOAT,
	    image, send_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(world_rank == 0) {
	Mat magspec(n0, n1, CV_32FC1);
	cout << "FFT 1st pixel: " << image[0] << endl;

	for(int row=0; row<n0; row++) {
	    for(int col=0; col<n1; col++) {
		float val= sqrt(
		    image[2*(row*n0+col)+0] * image[2*(row*n0+col)+0] +
		    image[2*(row*n0+col)+1] * image[2*(row*n0+col)+1]);
		magspec.at<float>(row*n0+col) = val / 2.0;
	    }
	}
	imwrite("test2.png", magspec);
    }*/

    fftwf_destroy_plan(plan);
    fftwf_mpi_cleanup();
    MPI_Finalize();

    return 0;
}
