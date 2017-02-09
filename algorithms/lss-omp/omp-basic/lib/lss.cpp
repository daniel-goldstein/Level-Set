/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Level Set Segmentation for Image Processing 
 *  
 */
 
#include "lss.h"

using namespace std;

void modMaxIter (int value){
	max_iterations = value;
}

void evolveContour(unsigned char* intensity, 
		   unsigned char* labels, 
		   signed char* phi, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels,
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j) {

	// Note: j is the label counter
        phi       = &phi      [j*HEIGHT*WIDTH];
	
	char *phiTemp = new char [HEIGHT*WIDTH];

	int gridXSize = 1 + (( WIDTH - 1) / TILE_SIZE);
	int gridYSize = 1 + ((HEIGHT - 1) / TILE_SIZE);
	
	int globalFinishedVariable = 0;
	int *globalBlockIndicator = new int [gridXSize*gridYSize];

	
	// Timing variables
	double processing_tbegin;
	double processing_tend;
	
	double total_tbegin = omp_get_wtime();
	double total_tend;
	
	double initializing_tbegin = omp_get_wtime();
	double initializing_tend;

	
	// Step 1: Initialize
	#pragma omp parallel for
	for (int bx = 0; bx < gridXSize; bx++){
		for (int by = 0; by < gridYSize; by++){
			for (int tx = 0; tx < TILE_SIZE; tx++){
				for (int ty = 0; ty < TILE_SIZE; ty++){
					int xPos = bx*TILE_SIZE + tx;
					int yPos = by*TILE_SIZE + ty;
					
					if (intensity[yPos*WIDTH+xPos] >= lowerIntensityBounds[j] &&
					    intensity[yPos*WIDTH+xPos] <= upperIntensityBounds[j]){
						if (labels[yPos*WIDTH+xPos] == targetLabels[j]){
							phiTemp[yPos*WIDTH+xPos] = 3;
						}  else {
							globalBlockIndicator[by*gridXSize + bx] = 1;
							phiTemp[yPos*WIDTH+xPos] = 1;				  
						}
					} else {
						if (labels[yPos*WIDTH+xPos] == targetLabels[j]){
							phiTemp[yPos*WIDTH+xPos] = 4;
							globalBlockIndicator[by*gridXSize + bx] = 1;
						}  else {

							phiTemp[yPos*WIDTH+xPos] = 0;				  
						}
					}
				}
			}
		}
	}
	
	initializing_tend = omp_get_wtime();
	double elapsed_secs = initializing_tend - initializing_tbegin;
	
	/* Continue */
        int iterations = 0;
	processing_tbegin = omp_get_wtime();
	
	
	// Step 2: Evolution
	do {
		globalFinishedVariable = 0;
		iterations++;
		#pragma omp parallel for
		for (int bx = 0; bx < gridXSize; bx++){
			for (int by = 0; by < gridYSize; by++){
				int change = 1;
				int redoBlock = 0;
				while (change){
					change = 0;
					
					for (int tx = 0; tx < TILE_SIZE; tx++){
						for (int ty = 0; ty < TILE_SIZE; ty++){
							int xPos = bx*TILE_SIZE + tx;
							int yPos = by*TILE_SIZE + ty;
							
							// Get border values
							char borderUp;
							char borderDown;
							char borderLeft;
							char borderRight;
							
							if (xPos > 0){
								borderLeft = phiTemp[yPos*WIDTH+(xPos-1)];
							} else {
								borderLeft = 0;
							}
							
							if (xPos < WIDTH-1){
								borderRight = phiTemp[yPos*WIDTH+(xPos+1)];
							} else {
								borderRight = 0;
							}
							
							if (yPos > 0){
								borderDown = phiTemp[(yPos-1)*WIDTH+xPos];
							} else {
								borderDown = 0;
							}
							
							if (yPos < HEIGHT-1){
								borderUp = phiTemp[(yPos+1)*WIDTH+xPos];
							} else {
								borderUp = 0;
							}

							// Algorithm
							if((borderUp     == 3 ||
							    borderDown   == 3 ||
							    borderLeft   == 3 ||
							    borderRight  == 3 ) && 
							    phiTemp[yPos*WIDTH+xPos]  == 1){
								phiTemp[yPos*WIDTH+xPos] = 3;
								globalFinishedVariable = 1;
								change = 1;
							}
							if((borderUp     == 0 ||
							    borderDown   == 0 ||
							    borderLeft   == 0 ||
							    borderRight  == 0 ) && 
							    phiTemp[yPos*WIDTH+xPos]  == 4){
								phiTemp[yPos*WIDTH+xPos] = 0;
								globalFinishedVariable = 1;
								change = 1;
							}
						}
					}
				}
				
				for (int tx = 0; tx < TILE_SIZE; tx++){
					for (int ty = 0; ty < TILE_SIZE; ty++){
						int xPos = bx*TILE_SIZE + tx;
						int yPos = by*TILE_SIZE + ty;
						if (phiTemp[yPos*WIDTH+xPos] == 1 || phiTemp[yPos*WIDTH+xPos] == 4){
							redoBlock = 1;
						}
					}
				}
			}
		}
	} while (globalFinishedVariable && (iterations < max_iterations));
	
	
	// Step 3: Finalize
	#pragma omp parallel for
	for (int bx = 0; bx < gridXSize; bx++){
		for (int by = 0; by < gridYSize; by++){
			for (int tx = 0; tx < TILE_SIZE; tx++){
				for (int ty = 0; ty < TILE_SIZE; ty++){
					int xPos = bx*TILE_SIZE + tx;
					int yPos = by*TILE_SIZE + ty;
					
					// Get border values
					char borderUp;
					char borderDown;
					char borderLeft;
					char borderRight;
					
					if (xPos > 0){
						borderLeft = phiTemp[yPos*WIDTH+(xPos-1)];
					} else {
						borderLeft = 0;
					}
					
					if (xPos < WIDTH-1){
						borderRight = phiTemp[yPos*WIDTH+(xPos+1)];
					} else {
						borderRight = 0;
					}
					
					if (yPos > 0){
						borderDown = phiTemp[(yPos-1)*WIDTH+xPos];
					} else {
						borderDown = 0;
					}
					
					if (yPos < HEIGHT-1){
						borderUp = phiTemp[(yPos+1)*WIDTH+xPos];
					} else {
						borderUp = 0;
					}
					
					if (phiTemp[yPos*WIDTH+xPos] > 2){
						if(borderUp     > 2 &&
						   borderDown   > 2 &&
						   borderLeft   > 2 &&
						   borderRight  > 2 ){
							phi[yPos*WIDTH+xPos] = -3;
						} else 
							phi[yPos*WIDTH+xPos] = -1;
					} else {
						if(borderUp     > 2 ||
						   borderDown   > 2 ||
						   borderLeft   > 2 ||
						   borderRight  > 2 ){
							phi[yPos*WIDTH+xPos] = 1;
						} else 
							phi[yPos*WIDTH+xPos] = 3;		
					}
				}
			}
		}
	}

	if (iterations == max_iterations) printf("Reached max number of iterations %d.\n", max_iterations);
	else printf ("Algorithm converged in %d iterations.\n", iterations);	
	
	printf("Initializing Time: %f\n", elapsed_secs);
	
	processing_tend = omp_get_wtime();
	elapsed_secs = processing_tend - processing_tbegin;
	
	printf("Processing Time: %f\n", elapsed_secs);
	
	total_tend = omp_get_wtime();
	elapsed_secs = total_tend - total_tbegin;
	
	printf("Total Time: %f\n", elapsed_secs);
	
	//free(phiTemp);
	//free(globalBlockIndicator);
}
