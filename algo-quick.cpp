/* C++ Implementation of the Beckmann-Algorithm
 *
 * This uses the FFTW and the OpenCV library.
 * Compilation command:
 * 		g++ -fopenmp -O2 -Wall -ggdb algo-quick.cpp -o algo-quick -lfftw3f \
 *          -lfftw3f_threads -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
 *          -lopencv_highgui
 *
 * To run, place a holmos_raw.png alongside the executable. To exit, press q.
 */

#include <iostream>
#include <fstream>
#include <complex>
#include <fftw3.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace cv;

/* Complex two-dimensional arrays stored in row-major format
 * image_space: Input image with imaginary part set to zero
 * fourier_space: Fourier-Transform (frequency space) of the input image
 * cropped_fourier: satellite (frequency space) cropped
 */
complex<float> *image_space, *fourier_space, *cropped_fourier;

/* The location of the clipping region used to extract the satellite in the
 * frequency transform
 */
int rect_y = 492;
int rect_x = 940;
int rect_w = 70;
int frames = 0;

/* Two time points to measure the program execution time */
chrono::steady_clock::time_point t1, t2;

/* Returns a magnitude spectrum for a given two-dimensional complex array.
 * -> The absolute values of the complex numbers multiplied by a factor
 * The input must be in row-major format
 */
Mat get_magnitude_spectrum(complex<float> *data, int square_size, float fac) {
	/* Allocate a new Mat object with the corresponding size, make it a 32-bit
	 * floating array with only one channel
	 */
    Mat magspec(square_size, square_size, CV_32FC1);

    /* Iterate over each element and set the value in the matrix to the
     * absolute value multiplied by factor
     */
    for(int i=0; i<square_size*square_size; i++) {
		magspec.at<float>(i) = abs(data[i]) * fac;
    }
    return magspec;
}

/* Center the zero-frequency in a two-dimensional complex array in row-major
 * format.
 */
void fftshift(complex<float> *data, int square_size) {
	/* Allocate a temporary object */
    complex<float> tmp;
    int index1, index2;
    int half = floor(square_size / 2);

    /* Swap upper-left and bottom-right quadrant */
    for(int y=0; y<half; y++) {
		for(int x=0; x<half; x++) {
			/* index1: position in the upper-left quadrant
			 * index2: position in the bottom-right quadrant
			 */
		    index1 = y * square_size + x;
		    index2 = (y + half) * square_size + x + half;

		    /* Swap the values using a temporary variable */
		    tmp = data[index1];
		    data[index1] = data[index2];
		    data[index2] = tmp;
		}
    }

    /* Swap bottom-left and upper-right quadrant */
    for(int y=half; y<square_size; y++) {
		for(int x=0; x<half; x++) {
		    index1 = y * square_size + x;
		    index2 = (y - half) * square_size + x + half;
		    tmp = data[index1];
		    data[index1] = data[index2];
		    data[index2] = tmp;
		}
    }
}

int main(int argc, char **argv) {
    t1 = chrono::steady_clock::now();

    /* Initialize the GUI and create sliders */
    namedWindow("frequency spectrum", WINDOW_NORMAL);
    createTrackbar("rect x", "frequency spectrum", &rect_x, 1024);
    createTrackbar("rect y", "frequency spectrum", &rect_y, 1024);
    createTrackbar("rect r", "frequency spectrum", &rect_w, 1024);

    image_space = new complex<float>[1024*1024];
    fourier_space = new complex<float>[1024*1024];
    cropped_fourier = new complex<float>[1024*1024];

    /* Read the input image. Could/Should be replaced with real camera
     * input.
     */
    Mat image = imread("holmos_raw.png", 0);

    /* Two floating-point images for the magnitude spectrum (frequency space)
     * and phase angle (height information), both with the same size as the
     * input image
     */
    Mat magnitude_spectrum(1024, 1024, CV_32F);
    Mat phase_angle(1024, 1024, CV_32FC1);

    /* Copy the input image into a complex array with the imaginary part set
     * to zero. Division by 255 to convert into floating point: input image
     * is a 8-bit integer array with values 0...255.
     */
    for(int i=0; i<1024*1024; i++) {
		image_space[i].real(static_cast<float>(image.at<char>(i)) / 255.0L);
		image_space[i].imag(0.0L);
    }

    /* Initialize FFTW, the Fastest Fourier Transform in the West, a fast
     * cpu-based FFT library. Use 4 threads to process the fourier transform.
     * the f in fftwf indicates single-precision (float, not double) values.
     */
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4);

    /* In FFTW, a FFT must be planned before executed. Here we create two plans,
     * one to transform the input image into frequency space so we can extract
     * the information needed and the other to obtain the phase-angle by 
     * performing the transform backwards.
     * FFTW can plan the transform extensively(FFTW_MEASURE) or just quickly
     * (FFTW_ESTIMATE). Latter will usually be less performant.
     * 
     * The reinterpret_cast is dangerous because FFT has defined its own data-
     * type for complex numbers (fftwf_complex), but we defined a C++11 complex
     * array to make use of the builtin operations. According to the FFT
     * documentation the two are binary compatible, meaning we can trick the
     * compiler to think it deals with a fftw_complex, even though is a
     * complex<float>.
     */
    fftwf_plan p1, p2;
    p1 = fftwf_plan_dft_2d(1024, 1024, 
	    reinterpret_cast<fftwf_complex*>(image_space),
	    reinterpret_cast<fftwf_complex*>(fourier_space),
	    FFTW_FORWARD, FFTW_ESTIMATE);
    p2 = fftwf_plan_dft_2d(1024, 1024,
	    reinterpret_cast<fftwf_complex*>(cropped_fourier),
	    reinterpret_cast<fftwf_complex*>(image_space),
	    FFTW_BACKWARD, FFTW_ESTIMATE);

    /* Main loop */
    while(true) {

    	/* Read the image into a complex array with the imaginary part set to
    	 * zero. See above for additional notes
    	 * "#pragma omp parallel for" makes use of OpenMP to parallelize the
    	 * loop.
    	 */
    	#pragma omp parallel for
		for(int i=0; i<1024*1024; i++) {
		    image_space[i].real(static_cast<float>(image.at<char>(i)) / 255.0L);
		    image_space[i].imag(0.0L);
		}

		/* Execute the forward transform of the image and center its zero-
		 * frequency
		 */
		fftwf_execute(p1);
		fftshift(fourier_space, 1024);

		/* Reset the cropped_fourier array to 0 */
		memset(cropped_fourier, 0, sizeof(complex<float>)*1024*1024);

		
		/* Copy the region of interest into the cropped_fourier array and
		 * center it.
		 * "#pragma omp parallel for collapse(2)" paralellizes the nested
		 * for loop.
		 */
		#pragma omp parallel for collapse(2)
		for(int y=rect_y; y<(rect_y+rect_w); y++) {
		    for(int x=rect_x; x<(rect_x+rect_w); x++) {
		    	int new_y, new_x;
				new_y = 512 + y - rect_y - floor(rect_w * 0.5);
				new_x = 512 + x - rect_x - floor(rect_w * 0.5);

				/* Check if the coordinates exeed the image space and set it to
				 * black in that case to avoid SEGFAULT (bad memory access)
				 */
				if(y > 1024 or x > 1024) {
					cropped_fourier[new_y*1024+new_x] = 0.0f;
				} else {
					cropped_fourier[new_y*1024+new_x] = fourier_space[y*1024+x];
				}
		    }
		}

		/* Show the magnitude spectrum of our newly cropped frequency space */
		imshow("magspec crop", get_magnitude_spectrum(cropped_fourier, 1024, 1/300.0));

		/* shift the array again to center the high frequencies as needed by
		 * FFTW and then transform a backwards transformation
		 */
		fftshift(cropped_fourier, 1024);
		fftwf_execute(p2);

		/* Calculate the phase angle for each complex element of the image
		 * space and normalize it, as arg() returns a value between -π and
		 * π.
		 * Then copy that value into the phase_angle image.
		 */
		#pragma omp parallel for
		for(int i=0; i<1024*1024; i++) {
			float angle;
		    angle = arg(image_space[i]);
		    angle += M_PI;
		    angle /= 2.0*M_PI;
		    phase_angle.at<float>(i) = angle;
		}

		/* Visualize the frequency space as magnitude spectrum and highlight
		 * the selected region of interst
		 */
		magnitude_spectrum = get_magnitude_spectrum(fourier_space, 1024, 1/300.0f);
		rectangle(magnitude_spectrum, Point2f(rect_x, rect_y), Point2f(rect_x+rect_w, rect_y+rect_w), 255);

		/* Flip the image upside down for additional coherence with the
		 * original Beckmann implementation
		 */
		flip(phase_angle, phase_angle, 0);

		/* Show the spectrum and the phase angle, also increase the frame
		 * counter
		 */
		imshow("frequency spectrum", magnitude_spectrum);
		imshow("angle", phase_angle);
		frames++;
		
		/* waitKey processes the OpenCV GUI events and returns a keycode if
		 * the user pressed a keyboard key during the specified timeout (1ms).
		 * If that key was q, break the loop
		 */
		if((waitKey(1) & 0xff) == 'q') break;
    }

    /* Calculate the time spent calculating and the frames per second */
    t2 = chrono::steady_clock::now();
	float msecs = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	cout << "Took " << msecs << "ms for " << frames << " frames, FPS: " << (frames / (msecs / 1000)) << endl;

	/* Do a little bit of cleanup */
    fftwf_destroy_plan(p1);
    fftwf_destroy_plan(p2);

    /* Everything went well, signal that the exit was a sucess (0) */

    return 0;
}
