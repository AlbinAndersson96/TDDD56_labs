// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <cmath>

const int N = 16; 
const int n = 4;
const int blocksize = 4; 

__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	float *c = new float[N];	
	float *cd;
	const int size = N*sizeof(float);

	float *inputData = new float[n];
	float *outputData = new float[n];
	inputData[0] = 1;
	inputData[1] = 4;
	inputData[2] = 9;
	inputData[3] = 100;
	
	cudaMalloc( (void**)&cd, n*sizeof(float) );

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );

	cudaMemcpy( cd, inputData, n*sizeof(float), cudaMemcpyHostToDevice ); 
	simple<<<dimGrid, dimBlock>>>(cd);

	cudaThreadSynchronize();
	cudaMemcpy( outputData, cd, n*sizeof(float), cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	for (int i = 0; i < n; i++)
		printf("%f ", outputData[i]);
	printf("\n");

	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
