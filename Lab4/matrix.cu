// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

// The index of a thread and its thread ID relate to each other in a straightforward way: 
// for a one-dimensional block, they are the same; 
// for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y*Dx); 
// for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y*Dx + z*Dx*Dy). 
// int i = blockIdx.x * blockDim.x + threadIdx.x;
// int j = blockIdx.y * blockDim.y + threadIdx.y;
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i < N && j < N)
//         C[i][j] = A[i][j] + B[i][j];
// }

#include <stdio.h>
#include <cmath>
#include "milli.h"

__global__
void add_matrix_GPU(float *a, float *b, float *c, int N)
{	
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    //int index = index_x * N + index_y;
    int index = index_y * N + index_x;
    
    c[index] = a[index] + b[index];
}

void add_matrix_CPU(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
    const int N = 4096; // 2^i
    // GPU
    {
        
        float *a_g;
        float *b_g;
        float *c_g;
        
        float *a = (float*)malloc(N*N*sizeof(float));
        float *b = (float*)malloc(N*N*sizeof(float));
        float *c = (float*)malloc(N*N*sizeof(float));
        
        dim3 threadsPerBlock( 16, 16 );
        dim3 numBlocks( N/threadsPerBlock.x, N/threadsPerBlock.y );

        
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                a[i+j*N] = 10 + i;
                b[i+j*N] = (float)j / N;
            }
        
        
        // Allocating space on the GPU
        cudaMalloc( (void**)&a_g, N*N*sizeof(float) );
        cudaMalloc( (void**)&b_g, N*N*sizeof(float) );
        cudaMalloc( (void**)&c_g, N*N*sizeof(float) );
        
        // Sending it to the GPU
        cudaMemcpy( a_g, a, N*N*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( b_g, b, N*N*sizeof(float), cudaMemcpyHostToDevice );
		
		free(a);
		free(b);
        
        // Time start
        cudaEvent_t startTimeEvent;
        cudaEvent_t endTimeEvent;
        float theTime;
        
        cudaEventCreate(&startTimeEvent);
        cudaEventCreate(&endTimeEvent);

        cudaEventRecord(startTimeEvent, 0);

        add_matrix_GPU<<<numBlocks, threadsPerBlock>>>(a_g, b_g, c_g, N);
        cudaDeviceSynchronize();

        // Time end
        cudaEventRecord(endTimeEvent, 0);
        
        cudaEventSynchronize(endTimeEvent);
        //cudaEventSynchronize(startTimeEvent);

        cudaEventElapsedTime(&theTime, startTimeEvent, endTimeEvent);

        // Check for, and print any kernel launch error
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess)
            printf("Cuda error: %s\n", cudaGetErrorString(err));

        // Copying result from the GPU
        
        cudaMemcpy( c, c_g, N*N*sizeof(float), cudaMemcpyDeviceToHost ); 
        
        // Freeing space from the GPU
        cudaFree( a_g );
        cudaFree( b_g );
        cudaFree( c_g );
        
        if (N <= 32) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%0.2f ", c[i+j*N]);
                }
                printf("\n");
		    }
		}
		
		free(c);
        
		
		printf("GPU Kernel executed in %f milliseconds\n", theTime);

    }

    // CPU
    {
        float *a = (float*)malloc(N*N*sizeof(float));
        float *b = (float*)malloc(N*N*sizeof(float));
        float *c = (float*)malloc(N*N*sizeof(float));
    
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                a[i+j*N] = 10 + i;
                b[i+j*N] = (float)j / N;
            }
        int startTime = GetMicroseconds();
		add_matrix_CPU(a, b, c, N);
		int endTime = GetMicroseconds();

		free(a);
		free(b);
		free(c);

		printf("CPU function executed in %f milliseconds\n", (float)(endTime - startTime)/1000);
        
    }
}
