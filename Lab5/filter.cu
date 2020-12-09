// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 40
#define maxKernelSizeY 40

#define KERNEL_SIZE_X 9
#define KERNEL_SIZE_Y 9

#define BLOCK_SIZE 32

__device__
void mapPixel(unsigned char *inputImage, unsigned char *sharedMemory, int sharedIndex, int globalIndex)
{
	sharedMemory[sharedIndex + 0] = inputImage[globalIndex + 0];
	sharedMemory[sharedIndex + 1] = inputImage[globalIndex + 1];
	sharedMemory[sharedIndex + 2] = inputImage[globalIndex + 2];

}

__device__
void mapPixelRed(unsigned char *inputImage, unsigned char *sharedMemory, int sharedIndex, int globalIndex)
{
	sharedMemory[sharedIndex + 0] = 255.0;
	sharedMemory[sharedIndex + 1] = 0;
	sharedMemory[sharedIndex + 2] = 0;

}

__global__
void filterKernel(unsigned char *inputImage, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey) {
	
	extern __shared__ unsigned char sharedMemory[];
	
	int pad_x = blockDim.x + kernelsizex - 1;
	int pad_y = blockDim.y + kernelsizey - 1;
		
	// if (threadIdx.x == 0 && threadIdx.y == 0) {
	// 	printf("%d\n", pixelsPerThread);
	// } 

	int index_x_img = (blockIdx.x * blockDim.x + threadIdx.x); // Thread {0, 0} => img{0, 0}
	int index_y_img = (blockIdx.y * blockDim.y + threadIdx.y);

	int index_shared_x = threadIdx.x + (KERNEL_SIZE_X/2);
	int index_shared_y = threadIdx.y + (KERNEL_SIZE_Y/2);
	
	int halfKernel = kernelsizey/2;
	int index_shared = (index_shared_y * pad_x + index_shared_x) * 3;
	int global_index = (index_y_img * imagesizex + index_x_img) * 3;

	// Read pixels left
	if (threadIdx.x == 0) {
		for(int i = 1; i <= halfKernel; ++i) {
			int indexOOR = (index_shared_y * pad_x + index_shared_x - i) * 3;
			int globalindexOOR = (index_y_img * imagesizex + index_x_img - i) * 3;

			globalindexOOR = max(globalindexOOR, 0);
			mapPixel(inputImage, sharedMemory, indexOOR, globalindexOOR);
		}
	}

	// Read pixels right
	if (threadIdx.x == blockDim.x - 1) {
		for(int i = 1; i <= halfKernel; ++i) {
			int indexOOR = (index_shared_y * pad_x + index_shared_x + i) * 3;
			int globalindexOOR = (index_y_img * imagesizex + index_x_img + i) * 3;
			mapPixel(inputImage, sharedMemory, indexOOR, globalindexOOR);
		}
	}

	// Read pixels down
	if (threadIdx.y == 0) {
		for(int i = 1; i <= halfKernel; ++i) {
			int indexOOR = ((index_shared_y - i) * pad_x + index_shared_x) * 3;
			int globalindexOOR = ((index_y_img - i) * imagesizex + index_x_img) * 3;

			globalindexOOR = max(globalindexOOR, 0);
			mapPixel(inputImage, sharedMemory, indexOOR, globalindexOOR);
		}
	}

	// Read pixels up
	if (threadIdx.y == blockDim.y - 1) {
		for(int i = 1; i <= halfKernel; ++i) {
			int indexOOR = ((index_shared_y + i) * pad_x + index_shared_x) * 3;
			int globalindexOOR = ((index_y_img + i) * imagesizex + index_x_img) * 3;
			mapPixel(inputImage, sharedMemory, indexOOR, globalindexOOR);
		}
	}

	mapPixel(inputImage, sharedMemory, index_shared, global_index);

	__syncthreads();

	int divby = KERNEL_SIZE_X*KERNEL_SIZE_Y;

	float r=0.0, g=0.0, b=0.0;
	for(int i = -(KERNEL_SIZE_Y/2); i <= (KERNEL_SIZE_Y/2); ++i) {
		for(int j = -(KERNEL_SIZE_X/2); j <= (KERNEL_SIZE_X/2); ++j) {
			r += sharedMemory[((index_shared_y + i)*pad_x + (index_shared_x + j))*3 + 0];
			g += sharedMemory[((index_shared_y + i)*pad_x + (index_shared_x + j))*3 + 1];
			b += sharedMemory[((index_shared_y + i)*pad_x + (index_shared_x + j))*3 + 2];
		}
	}

	out[global_index + 0] = r/divby;
	out[global_index + 1] = g/divby;
	out[global_index + 2] = b/divby;
}

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 

  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);
			
			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{


	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

	// 1 thread per pixel in each kernel
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 grid(imagesizex / threadsPerBlock.x, imagesizey / threadsPerBlock.y);
	dim3 grid(imagesizex / threadsPerBlock.x, imagesizey / threadsPerBlock.y);


	filterKernel<<<grid, threadsPerBlock, ((imagesizex/grid.x) + kernelsizex - 1)*((imagesizey/grid.y) + kernelsizey - 1)*3*sizeof(unsigned char)>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	//filterKernel<<<grid, threadsPerBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaDeviceSynchronize();
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros-noisy.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(KERNEL_SIZE_X, KERNEL_SIZE_Y);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}



	//int index_shared_true_start = ((kernelsizey/2) * pad_x + (kernelsizex/2)) * 3;
	/*if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("%d\n", index_shared_true_start);
	}*/

	//int index_shared_x = threadIdx.x + (KERNEL_SIZE_X/2);
	//int index_shared_y = threadIdx.y + (KERNEL_SIZE_Y/2);
	//int index_shared_out = index_shared_true_start + (threadIdx.y * pad_x + threadIdx.x * pixelsPerThread) * 3;

	// int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);

	// if (blockIdx.x == 0) { // Left
	// 	if (blockIdx.y == 0) { // Lower left corner
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);

	// 	} else if (blockIdx.y == gridDim.y - 1) { // Upper left corner
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	} else { // Left
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	}
	// }

	// else if (blockIdx.y == 0 ) { // Lower
	// 	if (blockIdx.x == gridDim.x - 1) { // Lower right corner
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	} else {
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	}

	// }

	// else if (blockIdx.x == gridDim.x - 1) { // Right
	// 	if (blockIdx.y == gridDim.y - 1) { // Upper right corner
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	} else {
	// 		int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 		int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 		mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// 	}
	// }

	// else if (blockIdx.y == gridDim.y - 1) { // Upper
	// 	int special_index_shared = (index_shared_y_temp * pad_x + index_shared_x_temp) * 3;
	// 	int special_index = (index_y_img*imagesizex + index_x_img)*3;
	// 	mapPixel(inputImage, sharedMemory, special_index_shared, special_index, pixelsPerThread);
	// } 
	
	// else {
	// 	/*for (int i = 0; i < pixelsPerThread; ++i) 
	// 	{
	// 		sharedMemory[index_shared + i*3 + 0] = inputImage[index + i*3 + 0];
	// 		sharedMemory[index_shared + i*3 + 1] = inputImage[index + i*3 + 1];
	// 		sharedMemory[index_shared + i*3 + 2] = inputImage[index + i*3 + 2];
	// 	}*/


		// out[index + 0] = sharedMemory[index_shared + 0];
	// out[index + 1] = sharedMemory[index_shared + 1];
	// out[index + 2] = sharedMemory[index_shared + 2];



	// //int pixelsPerThread = (((imagesizex/gridDim.x)*(imagesizey/gridDim.y)*3) / (blockDim.x*blockDim.y)));
	// int index_x_img = (blockIdx.x * blockDim.x + threadIdx.x);
	// int index_y_img = (blockIdx.y * blockDim.y + threadIdx.y);
	// int index = (index_y_img*imagesizex + index_x_img)*3;
	
    // int index_x_shared = threadIdx.x;
	// int index_y_shared = threadIdx.y;
	// int index_shared = (index_y_shared * pad_x + index_x_shared * pixelsPerThread) * 3;

	// int index_shared_true_start = ((kernelsizey/2) * pad_x + kernelsizex/2) * 3;
	// int index_shared_out = index_shared_true_start + (index_y_shared * blockDim.x + index_x_shared) * 3;

	// // if (threadIdx.x < 3 && threadIdx.y == 0) {
	// // 	printf("True index and thread index: {%d, %d}\n", index_shared_true_start, index_shared_out);
	// // }
	
	// Each threads reads 1 pixel (three values)


	// 
	
	// out[index + 0] = sharedMemory[index_shared_out + 0];
	// out[index + 1] = sharedMemory[index_shared_out + 1];
	// out[index + 2] = sharedMemory[index_shared_out + 2];

	// Do operation


	// if (index_x_img < imagesizex && index_y_img < imagesizey) {
	// 	// Filter kernel (simple box filter)
	// 	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	// 	unsigned int sumx, sumy, sumz;
	// 	int dy, dx;
	// 	sumx=0; sumy=0; sumz=0;
	// 	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
	// 		for(dx=-kernelsizex;dx<=kernelsizex;dx++) {
	// 			// Use max and min to avoid branching!
	// 			int yy = min(max(index_y_img+dy, 0), imagesizey-1);
	// 			int xx = min(max(index_x_img+dx, 0), imagesizex-1);
				
	// 			//if (value is inside image but outside of shared kernel)
	// 			int yy_shared = min(max(index_y_shared + dy, 0), blockDim.y - 1);
	// 			int xx_shared = min(max(index_x_shared + dx, 0), blockDim.x - 1);
				
	// 			sumx += sharedMemory[( (yy_shared) * blockDim.x + (xx_shared))*3 + 0];
	// 			sumy += sharedMemory[( (yy_shared) * blockDim.x + (xx_shared))*3 + 1];
	// 			sumz += sharedMemory[( (yy_shared) * blockDim.x + (xx_shared))*3 + 2];
	// 		}
	// 	out[(index_y_img*imagesizex+index_x_img)*3+0] = sumx/divby;
	// 	out[(index_y_img*imagesizex+index_x_img)*3+1] = sumy/divby;
	// 	out[(index_y_img*imagesizex+index_x_img)*3+2] = sumz/divby;
	// }