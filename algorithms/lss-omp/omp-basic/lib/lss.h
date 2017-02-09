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
#include <omp.h>

#include "config.h"

int max_iterations = MAX_ITER;

// Modify the value of max_iterations
void modMaxIter (int value);

void evolveContour(unsigned char* intensity, 
		   unsigned char* labels, 
		   signed char* phi, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels,
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j);

#endif
