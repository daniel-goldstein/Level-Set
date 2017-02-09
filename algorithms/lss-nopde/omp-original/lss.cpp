/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Level Set Segmentation for Image Processing 
 *  
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <list>
#include <omp.h>

#include "lib/imghandler.h"
#include "lib/evcontour.h"

using namespace std;

// To only have 1 image
bool drawBorder = true;
	
int main(int argc, char* argv[]) {

	// Hard Coding Struct to print same way as other implementations (to compare images)
	struct Color colors[32];

	for (int i = 0; i < 32; i++){
		colors[i] = randomColor();
	}

	colors[0].r = 204;
	colors[0].g = 0;
	colors[0].b = 0;

	colors[1].r = 204;
	colors[1].g = 102;
	colors[1].b = 0;

	colors[2].r = 204;
	colors[2].g = 204;
	colors[2].b = 0;

	colors[3].r = 102;
	colors[3].g = 204;
	colors[3].b = 0;

	colors[4].r = 0;
	colors[4].g = 204;
	colors[4].b = 0;

	colors[5].r = 0;
	colors[5].g = 204;
	colors[5].b = 102;

	colors[6].r = 0;
	colors[6].g = 204;
	colors[6].b = 204;

	colors[7].r = 0;
	colors[7].g = 102;
	colors[7].b = 204;

	colors[8].r = 0;
	colors[8].g = 0;
	colors[8].b = 204;

	colors[9].r = 102;
	colors[9].g = 0;
	colors[9].b = 204;

	// Files needed
	char* imageFile = NULL;
	char* labelFile = NULL;
	char* paramFile = NULL;

	for(int i = 1 ; i < argc ; i++) {
		if(strcmp(argv[i], "--image") == 0) {
			if(i + 1 < argc)
				imageFile = argv[++i];
		} else if(strcmp(argv[i], "--labels") == 0) {
			if(i + 1 < argc)
				labelFile = argv[++i];
		} else if(strcmp(argv[i], "--params") == 0) {
			if(i + 1 < argc)
				paramFile = argv[++i];
		} else if(strcmp(argv[i], "--max_reps") == 0) {
			if(i + 1 < argc)
				modMaxIter(atoi(argv[++i]));
		} else if(strcmp(argv[i], "--draw_border") == 0) {
			drawBorder = true;
		}
	}
	
	if(imageFile == NULL || labelFile == NULL || paramFile == NULL) {
		cerr << "Missing one or more arguments. " << endl;
		exit(1);
	}

	bool produceOutput = true;

        // Load Intensity Image
	image<unsigned char>* input = loadPGM(imageFile);
	const int HEIGHT = input->height();
	const int WIDTH = input->width();
	const int SIZE = HEIGHT*WIDTH*sizeof(char);

	unsigned char* intensity = new unsigned char[HEIGHT*WIDTH];
	for (int j = 0 ; j < SIZE ; j++){
		intensity[j] = input->data[j];
	}

        // Load connected component labels
	input = loadPGM(labelFile);

	unsigned char* labels = new unsigned char[HEIGHT*WIDTH];
	for (int j = 0 ; j < SIZE ; j++){
		labels[j] = input->data[j];
	}

	// Load parameters from parameter file
	ifstream paramStream;
	paramStream.open(paramFile);

	if(paramStream.is_open() != true) {
		cerr << "Could not open '" << paramFile << "'." << endl;
		exit(1);
	}
	
	int targetLabels[1024];
	int lowerIntensityBounds[1024];
	int upperIntensityBounds[1024];

	int numLabels = 0;
	
	while(paramStream.eof() == false) {
		char line[16];
		paramStream.getline(line, 16);
		
		if(paramStream.eof() == true)
			break;

		if(numLabels % 3 == 0)
			targetLabels[numLabels/3] = strtol(line, NULL, 10);
		else if(numLabels % 3 == 1)
			lowerIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		else
			upperIntensityBounds[numLabels/3] = strtol(line, NULL, 10);
		
		numLabels++;
	}
	
	if(numLabels % 3 == 0)
		numLabels /= 3;
	else {
		cerr << "Number of lines in " << paramFile << " is not divisible by 3 and it should." << endl;
		exit(1);
	}
	paramStream.close();


        // Allocate arrays for speed and phi
		// NumLabels specify the amount of labels we will analize
		// so we need that number * SIZE given each label will
		// need 1 SIZE array for speed and phi
	signed char* speed = new signed char [numLabels*SIZE];
	signed char*   phi = new signed char [numLabels*SIZE];

	double analysis_tbegin = omp_get_wtime();
	double analysis_tend;
		
	// Launch function for image segmentation
#pragma omp parallel for
	for (int j = 0 ; j < numLabels ; j++){
		evolveContour(intensity, labels, speed, phi, HEIGHT, WIDTH, targetLabels,
			      numLabels, lowerIntensityBounds, upperIntensityBounds, j);
	}
	
	analysis_tend = omp_get_wtime();
	double elapsed_secs = analysis_tend - analysis_tbegin;

	printf("\nTotal Execution Time for LSS algorithm was: %f ms\n\n", 1000*elapsed_secs);
	
	// Output RGB images
	if(produceOutput == true) {

		Color color;
	
		srand(1000);
	
		// Create 1 image for all labels
		image<Color> output = image<Color>(WIDTH, HEIGHT, true);
		image<Color>* im = &output;
		
		// Initialize image (same as input)
		for(int i = 0 ; i < HEIGHT ; i++) {
			for(int j = 0 ; j < WIDTH ; j++){
				color.r = intensity[i*WIDTH+j];
				color.g = intensity[i*WIDTH+j];
				color.b = intensity[i*WIDTH+j];
				im->access[i][j] = color;
			}
		}
		for(int k = 0 ; k < numLabels ; k++) {
			Color randomcolor = colors[k];
			for(int i = 0 ; i < HEIGHT ; i++) {
				for(int j = 0 ; j < WIDTH ; j++){					
					if (phi[k*HEIGHT*WIDTH+i*WIDTH+j] == -1){
						color = randomcolor;
						im->access[i][j] = color;
					} else if (phi[k*HEIGHT*WIDTH+i*WIDTH+j] == -3){
						color.r = randomcolor.r + 50;
						color.g = randomcolor.g + 50;
						color.b = randomcolor.b + 50;
						im->access[i][j] = color;
					}
				}
			}
		}	
		char filename[64];
		sprintf(filename, "result.ppm");
		savePPM(im, filename);
	}
	
	/* Free arrays */

        return 0;
}
