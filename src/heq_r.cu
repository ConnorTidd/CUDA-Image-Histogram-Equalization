
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
 
#include "config.h"

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     

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
      

__global__ void createHistogram(unsigned char *input, 
    int *histogram,
    unsigned int width,
    unsigned int height) {

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;

    // int pixels = width * height;
    int val = input[location];
    atomicAdd(&(histogram[val]), 1);

}
// Add GPU kernel and functions
// HERE!!!
__global__ void kernel(unsigned char *input, 
                       unsigned char *output,
                       unsigned int *normed_hist,
                       unsigned int width,
                       unsigned int height){

    int x = blockIdx.x*TILE_SIZE+threadIdx.x;
    int y = blockIdx.y*TILE_SIZE+threadIdx.y;

    int location = 	y*TILE_SIZE*gridDim.x+x;
    // data is the copy
    int pixels = width*height;
    int val = input[location];
    unsigned char dumbbutt = (normed_hist[val])*255.0/pixels;
    output[location] = dumbbutt;
}

__global__ void warmup(unsigned char *input, 
                       unsigned char *output){

	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	  
	int location = 	y*(gridDim.x*TILE_SIZE)+x;
	
    output[location] = 0;

}

// NOTE: The data passed on is already padded
void gpu_function(unsigned char *data,  
                  unsigned int height, 
                  unsigned int width){
    
    unsigned char *input_gpu;
    unsigned char *output_gpu;
    unsigned int *normed_hist;
    int *histogram_d;

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&normed_hist , 256*sizeof(unsigned int)));
    checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
    checkCuda(cudaMalloc((void**)&histogram_d , 256 * sizeof(int)));
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
    checkCuda(cudaMemset(normed_hist , 0 , 256*sizeof(unsigned int)));
    checkCuda(cudaMemset(histogram_d , 0 , 256*sizeof(int)));
    // Copy data to GPU
    

    // Execute algorithm

    dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	

     
    // int pixels = height*width;
    printf("\n");



    // call kernel
    // Add more kernels and functions as needed here
    createHistogram<<<dimGrid, dimBlock>>>(input_gpu,
        histogram_d,
        width,
        height);

    int *histogram_host = new int [256];
    checkCuda(cudaMemcpy(histogram_host, histogram_d, 256*sizeof(int), cudaMemcpyDeviceToHost));
    

    unsigned int* cumhistogram = new unsigned int[256];
    cumhistogram[0] = histogram_host[0];
    
    cudaMallocHost(&cumhistogram, sizeof(unsigned int)*256);
    for(int i = 1; i < 256; i = i + 1) {
        cumhistogram[i] = histogram_host[i] + cumhistogram[i-1];
    }
    
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(unsigned char), 
        cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(normed_hist, 
        cumhistogram, 
        256*sizeof(unsigned int), 
        cudaMemcpyHostToDevice));


	checkCuda(cudaDeviceSynchronize());

    // Kernel Call
	#ifdef CUDA_TIMING
         float Ktime;
        TIMER_CREATE(Ktime);
        TIMER_START(Ktime);
    #endif
    
    // Add more kernels and functions as needed here
    kernel<<<dimGrid, dimBlock>>>(input_gpu,
                                  output_gpu,
                                  normed_hist,
                                  width,
                                  height);
    
    // From here on, no need to change anything
    checkCuda(cudaPeekAtLastError());                                     
    checkCuda(cudaDeviceSynchronize());
    
    #ifdef CUDA_TIMING 
        TIMER_END(Ktime);
        printf("Kernel Execution Time: %f ms\n", Ktime);
    #endif
    
    // Retrieve results from the GPU

    
    checkCuda(cudaMemcpy(data, 
        output_gpu, 
        size*sizeof(unsigned char), 
        cudaMemcpyDeviceToHost));
        
    printf("\ntotal is:%d\n", height*width);
    // Free resources and end the program
    //free(mapf);
    //free(data);
	checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));
    checkCuda(cudaFree(normed_hist));
    checkCuda(cudaFree(histogram_d));

    // checkCuda(cudaFree(cumhistogram));
}












































void gpu_warmup(unsigned char *data, 
                unsigned int height, 
                unsigned int width){
    
    unsigned char *input_gpu;
    unsigned char *output_gpu;
     
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
    checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
            
    // Copy data to GPU
    checkCuda(cudaMemcpy(input_gpu, 
        data, 
        size*sizeof(char), 
        cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
    // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    
    warmup<<<dimGrid, dimBlock>>>(input_gpu, 
                                  output_gpu);
                                         
    checkCuda(cudaDeviceSynchronize());
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
    // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
    checkCuda(cudaFree(input_gpu));

}

