/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Evolve Contour Header 
 *  
 */
 
#include "evcontour.h"
 
int max_iterations = MAX_ITER;

// To indicate when a label analysis should finish		   
int stopCondition[1024];

void modMaxIter (int value){
	max_iterations = value;
}

void evolveContour(unsigned char* intensity, 
		   unsigned char* labels, 
		   signed char* speed, 
		   signed char* phi, 
		   int HEIGHT, 
		   int WIDTH, 
		   int* targetLabels, 
		   int numLabels,
		   int* lowerIntensityBounds, 
		   int* upperIntensityBounds, 
		   int j) {

	// Note: j is the label counter
        //intensity = &intensity[HEIGHT*WIDTH];
        //labels    = &labels   [HEIGHT*WIDTH];
        speed     = &speed    [j*HEIGHT*WIDTH];
        phi       = &phi      [j*HEIGHT*WIDTH];
	
	// Lin and Lout
	list<int> Lin;
	list<int> Lout;
	list<int>::iterator LinIter;
	list<int>::iterator LoutIter; 
	
	// Timing variables
	double processing_tbegin;
	double processing_tend;
	
	double total_tbegin = omp_get_wtime();
	double total_tend;
	
	double initializing_tbegin = omp_get_wtime();
	double initializing_tend;

	// Initialize border
#pragma omp parallel for
	for (int k = 0; k < HEIGHT*WIDTH; k++){
		
		
		// Note: Using x and y so its easier to map into 2D array (row mayor)
		int xPos = k%WIDTH;
		int yPos = k/WIDTH;
				
		// Border can't be considered (due to illegal accesses)
		if (xPos > 0 && yPos > 0 && xPos < WIDTH-1 && yPos < HEIGHT-1) {
		} else {
			// If border, ignore
			// If image value is outside boundary, speed = 0
			speed[yPos*WIDTH+xPos] = -1;
			// Label read is the same as the target label and 
			// all neighbors are the same (inside objective)
			phi[yPos*WIDTH+xPos] = 3;
		}
	}
	
        // Initialize the phy and speed arrays
//#pragma omp parallel for
	for (int k = 0; k < HEIGHT*WIDTH; k++){
		
		
		// Note: Using x and y so its easier to map into 2D array (row mayor)
		int xPos = k%WIDTH;
		int yPos = k/WIDTH;
				
		// Border can't be considered (due to illegal accesses)
		if (xPos > 0 && yPos > 0 && xPos < WIDTH-1 && yPos < HEIGHT-1) {

			// Initialization of Phi
			if(labels[yPos*WIDTH+xPos] != targetLabels[j]) {
				if(labels[yPos*WIDTH+(xPos-1)] != targetLabels[j] && 
				   labels[yPos*WIDTH+(xPos+1)] != targetLabels[j] && 
				   labels[(yPos-1)*WIDTH+xPos] != targetLabels[j] && 
				   labels[(yPos+1)*WIDTH+xPos] != targetLabels[j] ){
					// Label read is different than target label and 
					// all neighbors are also different (far from 
					// objective)
					phi[yPos*WIDTH+xPos] = 3;
				} else {
					// Label read is different than target label and 
					// one or more of the neighbors are the same as
					// label  (near objective)
					phi[yPos*WIDTH+xPos] = 1;
					Lout.push_back(yPos*WIDTH+xPos);			
				}
			} else {
				if(labels[yPos*WIDTH+(xPos-1)] != targetLabels[j] || 
				   labels[yPos*WIDTH+(xPos+1)] != targetLabels[j] ||		   
				   labels[(yPos-1)*WIDTH+xPos] != targetLabels[j] ||
				   labels[(yPos+1)*WIDTH+xPos] != targetLabels[j] ){
					// Label read is the same as the target label but 
					// all neighbors are different (close to objective)
					phi[yPos*WIDTH+xPos] = -1;
					Lin.push_back(yPos*WIDTH+xPos);
				} else {
					// Label read is the same as the target label and 
					// all neighbors are the same (inside objective)
					phi[yPos*WIDTH+xPos] = -3;
				}
			}
			
			// Initialization of Speed
			if(intensity[yPos*WIDTH+xPos] >= lowerIntensityBounds[j] &&
			   intensity[yPos*WIDTH+xPos] <= upperIntensityBounds[j] ){
				// If image value is inside boundary, speed = 1
				speed[yPos*WIDTH+xPos] = 1;
			} else {
				// If image value is outside boundary, speed = -1
				speed[yPos*WIDTH+xPos] = -1;
			}
		}
	}
	
	initializing_tend = omp_get_wtime();
	double elapsed_secs = initializing_tend - initializing_tbegin;
	
	/* Continue */
        int numIterations = 0;
        stopCondition[j] = 1;
	
	processing_tbegin = omp_get_wtime();
	
        while(stopCondition[j]) {
                stopCondition[j] = 0;
                numIterations++;
		
		// Outward evolution: Scan through Lout
		for(LoutIter = Lout.begin(); LoutIter != Lout.end(); ++LoutIter){
			
			int xPos = *LoutIter%WIDTH;
			int yPos = *LoutIter/WIDTH;
				
			// Switch out

			// Delete points from Lout and add them to Lin
			if(speed[yPos*WIDTH+xPos] > 0){
				
				// Update Phi
				phi[yPos*WIDTH+xPos] = -1;
			
				// Add to Lin
				Lin.push_back(yPos*WIDTH+xPos);
				
				//Delete from Lout
				Lout.erase(LoutIter++);
						
				// Update Neighbors (add them to Lout if Phi[y]=3)
				if (xPos > 0 && 
				    yPos > 0 && 
				    xPos < WIDTH-1 && 
				    yPos < HEIGHT-1) {
					if((phi[yPos*WIDTH+(xPos+1)] == 3) && xPos < WIDTH-1){
						phi[yPos*WIDTH+(xPos+1)] = 1;
						Lout.push_front(yPos*WIDTH+(xPos+1));
					}
					if((phi[yPos*WIDTH+(xPos-1)] == 3) && xPos > 0){
						phi[yPos*WIDTH+(xPos-1)] = 1;
						Lout.push_front(yPos*WIDTH+(xPos-1));					
					}
					if((phi[(yPos+1)*WIDTH+xPos] == 3) && yPos < HEIGHT-1){
						phi[(yPos+1)*WIDTH+xPos] = 1;
						Lout.push_front((yPos+1)*WIDTH+xPos);					
					}
					if((phi[(yPos-1)*WIDTH+xPos] == 3) && yPos > 0){
						phi[(yPos-1)*WIDTH+xPos] = 1;
						Lout.push_front((yPos-1)*WIDTH+xPos);					
					}
				}
			}
		}
		
		// Eliminate redundant points in Lin
		for(LinIter = Lin.begin(); LinIter != Lin.end(); ++LinIter){
			
			int xPos = *LinIter%WIDTH;
			int yPos = *LinIter/WIDTH;
			
			if (xPos > 0 && 
			    yPos > 0 && 
			    xPos < WIDTH-1 && 
			    yPos < HEIGHT-1) {
			
				if(phi[yPos*WIDTH+(xPos-1)] < 0 && 
				   phi[yPos*WIDTH+(xPos+1)] < 0 && 
				   phi[(yPos-1)*WIDTH+xPos] < 0 && 
				   phi[(yPos+1)*WIDTH+xPos] < 0 ){
					// Update Phi
					phi[yPos*WIDTH+xPos] = -3;
					
					// Update Lin
					Lin.erase(LinIter++);					
				}
			}
		}
		
		// Inward evolution: Scan through Lin
		for(LinIter = Lin.begin(); LinIter != Lin.end(); ++LinIter){
			
			int xPos = *LinIter%WIDTH;
			int yPos = *LinIter/WIDTH;
				
			// Switch in

			// Delete points from Lin and add them to Lout
			if(speed[yPos*WIDTH+xPos] < 0){
				
				// Update Phi
				phi[yPos*WIDTH+xPos] = 1;
			
				// Add to Lout
				Lout.push_back(yPos*WIDTH+xPos);
				
				//Delete from Lin
				Lin.erase(LinIter++);
				
				// Update Neighbors (add them to Lin if Phi[y]=-3)
				if (xPos > 0 && 
				    yPos > 0 && 
				    xPos < WIDTH-1 && 
				    yPos < HEIGHT-1) {
					if(phi[yPos*WIDTH+(xPos+1)] == -3){
						phi[yPos*WIDTH+(xPos+1)] = -1;
						Lin.push_front(yPos*WIDTH+(xPos+1));
					}
					if(phi[yPos*WIDTH+(xPos-1)] == -3){
						phi[yPos*WIDTH+(xPos-1)] = -1;
						Lin.push_front(yPos*WIDTH+(xPos-1));					
					}
					if(phi[(yPos+1)*WIDTH+xPos] == -3){
						phi[(yPos+1)*WIDTH+xPos] = -1;
						Lin.push_front((yPos+1)*WIDTH+xPos);					
					}
					if(phi[(yPos-1)*WIDTH+xPos] == -3){
						phi[(yPos-1)*WIDTH+xPos] = -1;
						Lin.push_front((yPos-1)*WIDTH+xPos);					
					}
				}
			}
		}
		
		// Eliminate redundant points in Lout
		for(LoutIter = Lout.begin(); LoutIter != Lout.end(); ++LoutIter){
			
			int xPos = *LoutIter%WIDTH;
			int yPos = *LoutIter/WIDTH;
			
			if (xPos > 0 && 
			    yPos > 0 && 
			    xPos < WIDTH-1 && 
			    yPos < HEIGHT-1) {
			
				if(phi[yPos*WIDTH+(xPos-1)] > 0 && 
				   phi[yPos*WIDTH+(xPos+1)] > 0 && 
				   phi[(yPos-1)*WIDTH+xPos] > 0 && 
				   phi[(yPos+1)*WIDTH+xPos] > 0 ){
					// Update Phi
					phi[yPos*WIDTH+xPos] = 3;
					
					// Update Lout
					Lout.erase(LoutIter++);					
				}
			}
		}
		
                // Check stopping condition
		for(LoutIter = Lout.begin(); LoutIter != Lout.end(); ++LoutIter){
			
			int xPos = *LoutIter%WIDTH;
			int yPos = *LoutIter/WIDTH;
				
			if(xPos > 0 && xPos < WIDTH-1 && yPos > 0 && yPos < HEIGHT-1){
				if(speed[yPos*WIDTH+xPos] > 0 ){
					stopCondition[j]++; //Don't stop yet		
				}
			}	
		}
		
		for(LinIter = Lin.begin(); LinIter != Lin.end(); ++LinIter){
			
			int xPos = *LinIter%WIDTH;
			int yPos = *LinIter/WIDTH;
				
			if(xPos > 0 && xPos < WIDTH && yPos > 0 && yPos < HEIGHT){
				if(speed[yPos*WIDTH+xPos] < 0 ){
					stopCondition[j]++; //Don't stop yet		
				}
			}	
		}
		
		// Threshold of num iterations reached
		if (numIterations >= max_iterations) {
			stopCondition[j] = 0; 
			printf("Target label %d (intensities: %d-%d) max threshold of %d iterations reached.\n"
			, targetLabels[j]
			, lowerIntensityBounds[j]
			, upperIntensityBounds[j]
			, numIterations);
		} else if(stopCondition[j] == 0 ) {
			printf("Target label %d (intensities: %d-%d) converged in %d iterations.\n"
			, targetLabels[j]
			, lowerIntensityBounds[j]
			, upperIntensityBounds[j]
			, numIterations);
		}
	} //stop condition
	
	printf("Initializing Time: %f\n", elapsed_secs);
	
	processing_tend = omp_get_wtime();
	elapsed_secs = processing_tend - processing_tbegin;
	
	printf("Processing Time: %f\n", elapsed_secs);
	
	total_tend = omp_get_wtime();
	elapsed_secs = total_tend - total_tbegin;
	
	printf("Total Time: %f\n", elapsed_secs);
}
