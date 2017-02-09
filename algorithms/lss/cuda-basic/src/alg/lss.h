/* 
 * Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 *   
 */
 
#ifndef LSS_H
#define LSS_H

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>

#include "../cuda/timing.h"
#include "../config.h"

int max_iterations = MAX_ITER;

typedef struct __gpuData {
	
	int size;
	
	int* targetLabels;
        int* lowerIntensityBounds;
        int* upperIntensityBounds;

	unsigned int* intensity;
	unsigned int* labels;
        
	signed int* phi;
	signed int* phiOut;
	signed int* phiOnCpu;
	
	signed int* globalBlockIndicator;
	signed int* globalFinishedVariable;
	
	signed int* totalIterations;
	signed int* totalIterationsOnCpu;

} gpuData;

gpuData gpu;

// Modify the value of max_iterations
void modMaxIter (int value);

inline cudaError_t checkCuda(cudaError_t result);

__global__ void evolveContour(unsigned int* intensity, 
			      unsigned int* labels,
			      signed int* phi,
			      signed int* phiOut, 
			      int gridXSize,
			      int gridYSize,
			      int* targetLabels, 
			      int* lowerIntensityBounds, 
			      int* upperIntensityBounds,
			      int max_iterations, 
			      int* globalBlockIndicator,
			      int* globalFinishedVariable,
			      int* totalIterations );

__global__ void lssStep1(unsigned int* intensity, 
			 unsigned int* labels,
			 signed int* phi, 
			 int targetLabel, 
			 int lowerIntensityBound, 
			 int upperIntensityBound,
			 int* globalBlockIndicator);

__global__ void lssStep2(signed int* phi, 
			 int* globalBlockIndicator,
			 int* globalFinishedVariable);
 
__global__ void lssStep3(signed int* phi,
			 signed int* phiOut);
			 
signed int *levelSetSegment(unsigned int *intensity, 
			    unsigned int *labels,
			    int height, 
			    int width,
			    int *targetLabels, 
			    int *lowerIntensityBounds,
			    int *upperIntensityBounds,
			    int numLabels);

#endif
