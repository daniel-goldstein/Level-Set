/* Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Brightness Algorithm Implementation 
 *  
 */
 
#include "brightness.h"

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

using namespace std;

void modBrightness (unsigned int value){
	inc = value;
}

/*
 * Brightness Kernel
 */
__global__ void brightnessAlgorithm(unsigned int *intensity, 
				unsigned int *result,
				unsigned int inc){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	// Read Input Data
	/////////////////////////////////////////////////////////////////////////////////////

	int x = bx<<BTSB;
	x = x + tx;
	x = x<<TTSB;
	int y = by<<BTSB;
	y = y + ty;
	y = y<<TTSB;
	  
	int location = 	(((x>>TTSB)&BTSMask)                ) |
			(((y>>TTSB)&BTSMask) << BTSB        ) |
			((x>>TSB)            << (BTSB+BTSB) ) ;
	location += 	((y>>TSB)<<(BTSB+BTSB))*gridDim.x;
		
	int intensityData = intensity[location];
	
	unsigned char intData1 = intensityData         & 0xFF;
	unsigned char intData2 = (intensityData >>  8) & 0xFF;
	unsigned char intData3 = (intensityData >> 16) & 0xFF;
	unsigned char intData4 = (intensityData >> 24) & 0xFF;
		
	if (intData1 + inc > 255) intData1 = 255;
	else intData1 = intData1 + inc;
	
	if (intData2 + inc > 255) intData2 = 255;
	else intData2 = intData2 + inc;
	
	if (intData3 + inc > 255) intData3 = 255;
	else intData3 = intData3 + inc;
	
	if (intData4 + inc > 255) intData4 = 255;
	else intData4 = intData4 + inc;
	
	int intReturnData = intData1        |
			   (intData2 << 8 ) |
			   (intData3 << 16) |
			   (intData4 << 24);
				
	result[location] = intReturnData;
}

unsigned char *brightness(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){
	
	#if defined(DEBUG)
		printf("Printing input data\n");
		printf("Height: %d\n", height);
		printf("Width: %d\n", width);
	#endif
	
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	#if defined(VERBOSE)
		printf ("Allocating arrays in GPU memory.\n");
	#endif
	
	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif
	
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
	
	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif
	
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
	brightnessAlgorithm<<<dimGrid, dimBlock>>>((unsigned int *)gpu.intensity, 
					      (unsigned int *)gpu.result,
					      inc);
	
	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif
	
	return(gpu.resultOnCPU);

}

unsigned char *brightnessWarmup(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
	brightnessAlgorithm<<<dimGrid, dimBlock>>>((unsigned int *)gpu.intensity, 
					      (unsigned int *)gpu.result,
					      inc);
	
	checkCuda(cudaDeviceSynchronize());

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	return(gpu.resultOnCPU);

}