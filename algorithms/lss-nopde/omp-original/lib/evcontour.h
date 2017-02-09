/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Evolve Contour Header 
 *  
 */
 
#ifndef EVCONTOUR_H
#define EVCONTOUR_H

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <list>
#include <omp.h>

using namespace std;

#include "imghandler.h"

// Threshold of max number of iterations of analysis
#define MAX_ITER 5000

void modMaxIter (int value);

void evolveContour(unsigned char* intensityDev, 
		   unsigned char* labelsDev, 
                   signed char* speedDev, 
		   signed char* phiDev, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels, 
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j);

#endif