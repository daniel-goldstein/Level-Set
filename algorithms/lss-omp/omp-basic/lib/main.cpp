
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

#include "imghandler.h"
#include "config.h"

typedef struct __cpuData {
	
	int height;
	int width;
	int gridXSize;
	int gridYSize;
	int size;	

	unsigned char* intensity;
	unsigned char* labels;
	
	signed char* result;
	
	int targetLabels[MAX_LABELS_PER_IMAGE];
	int lowerIntensityBounds[MAX_LABELS_PER_IMAGE];
	int upperIntensityBounds[MAX_LABELS_PER_IMAGE];
	int numLabels;
	
} cpuData;

cpuData cpu;

extern void evolveContour(unsigned char* intensity, 
		   unsigned char* labels, 
		   signed char* phi, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels,
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j);

extern void modMaxIter (int value);

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
		}
	}
	
	if(imageFile == NULL || labelFile == NULL || paramFile == NULL) {
		cerr << "Missing one or more arguments. " << endl;
		exit(1);
	}

	bool produceOutput = true;

        // Load Intensity Image
	image<unsigned char>* intensityInput = loadPGM(imageFile);
	
        // Load connected component labels
	image<unsigned char>* labelInput     = loadPGM(labelFile);
	
	cpu.height = intensityInput->height();
	cpu.width  = intensityInput->width();
	
	cpu.gridXSize = 1 + (( cpu.width - 1) / TILE_SIZE);
	cpu.gridYSize = 1 + ((cpu.height - 1) / TILE_SIZE);
	
	int XSize = cpu.gridXSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	int YSize = cpu.gridYSize*BLOCK_TILE_SIZE*THREAD_TILE_SIZE;
	
	cpu.size = XSize*YSize;

	cpu.intensity = new unsigned char[cpu.size];
	cpu.labels    = new unsigned char[cpu.size];
		
	for (int y = 0 ; y < YSize ; y++){
		for (int x = 0 ; x < XSize ; x++){
			int newLocation =  y*XSize + x;
			
			if (x < cpu.width && y < cpu.height) {
				cpu.intensity[newLocation] = intensityInput->data[y*cpu.width + x];
				cpu.labels[newLocation]    =     labelInput->data[y*cpu.width + x];
			} else{
				// Necessary in case image size is not a multiple of TILE_SIZE
				cpu.intensity[newLocation] = 0;
				cpu.labels[newLocation]    = 0;
			}
		}
	}
	
	// Load parameters from parameter file
	ifstream paramStream;
	paramStream.open(paramFile);

	if(paramStream.is_open() != true) {
		cerr << "Could not open '" << paramFile << "'." << endl;
		exit(1);
	}
	
	cpu.numLabels = 0;
	
	while(paramStream.eof() == false) {
		char line[16];
		paramStream.getline(line, 16);
		
		if(paramStream.eof() == true)
			break;

		if(cpu.numLabels % 3 == 0)
			cpu.targetLabels[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		else if(cpu.numLabels % 3 == 1)
			cpu.lowerIntensityBounds[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		else
			cpu.upperIntensityBounds[cpu.numLabels/3] = 
				strtol(line, NULL, 10);
		
		cpu.numLabels++;
	}
	
	if(cpu.numLabels % 3 == 0) {
		cpu.numLabels /= 3;
		if(cpu.numLabels > MAX_LABELS_PER_IMAGE){
			cerr << "Exceeded maximum number of objects per image." << endl;
			exit(1);
		}
	} else {
		cerr << "Number of lines in " << paramFile << " is not divisible by 3 and it should. " << endl;
		exit(1);
	}
	paramStream.close();

	cpu.result = new signed char [cpu.size*cpu.numLabels];
	
	#if defined(VERBOSE)
		cout << "Finished Processing files with " << cpu.numLabels 
		     << " Objects in the image of size " << cpu.height
		     << "x" << cpu.width << endl;
	#endif

	double analysis_tend;
	double analysis_tbegin = omp_get_wtime();
		
	// Launch function for image segmentation
#pragma omp parallel for
	for (int j = 0 ; j < cpu.numLabels ; j++){
		evolveContour(cpu.intensity,
				cpu.labels, 
				cpu.result, 
				cpu.height, 
				cpu.width, 
				cpu.targetLabels,
				cpu.numLabels, 
				cpu.lowerIntensityBounds, 
				cpu.upperIntensityBounds, 
				j);
	}
		
	analysis_tend = omp_get_wtime();
	double elapsed_secs = analysis_tend - analysis_tbegin;

	printf("\nTotal Execution Time for LSS algorithm was: %f ms\n\n", 1000*elapsed_secs);
	
	// Output RGB images
	if(produceOutput == true) {

		Color color;
	
		srand(1000);
	
		// Create 1 image for all labels
		image<Color> output = image<Color>(cpu.width, cpu.height, true);
		image<Color>* im = &output;
		
		// Initialize image (same as input)
		for(int i = 0 ; i < cpu.height ; i++) {
			for(int j = 0 ; j < cpu.width ; j++){
				color.r = cpu.intensity[i*cpu.width+j];
				color.g = cpu.intensity[i*cpu.width+j];
				color.b = cpu.intensity[i*cpu.width+j];
				im->access[i][j] = color;
			}
		}
		for(int k = 0 ; k < cpu.numLabels ; k++) {
			Color randomcolor = colors[k];
			for(int i = 0 ; i < cpu.height ; i++) {
				for(int j = 0 ; j < cpu.width ; j++){					
					if (cpu.result[k*cpu.height*cpu.width+i*cpu.width+j] == -1){
						color = randomcolor;
						im->access[i][j] = color;
					} else if (cpu.result[k*cpu.height*cpu.width+i*cpu.width+j] == -3){
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
	
	// Free resources and end the program
	free(cpu.intensity);
	free(cpu.labels);
	free(cpu.result);

        return 0;
}
