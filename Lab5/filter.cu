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
#define maxKernelSizeX 10
#define maxKernelSizeY 10

__global__
void filterKernel(unsigned char *inputImage, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey) {
	
	// X by Y kernel
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		//printf("Image section size: {%f, %f}\n", (float)(imagesizex/gridDim.x), (float)(imagesizey/gridDim.y));
	}
	
	extern __shared__ unsigned char sharedMemory[];

	int pad_x = blockDim.x + kernelsizex;
	int pad_y = blockDim.y + kernelsizey;
	int pixelsPerThread = ceil(((float)(pad_x*pad_y)/(blockDim.x*blockDim.y)));

	int index_x_img = (blockIdx.x * blockDim.x + threadIdx.x);
	int index_y_img = (blockIdx.y * blockDim.y + threadIdx.y);
	int index = (index_y_img*imagesizex + index_x_img)*3;

	if (blockIdx.x == 0) {

	}

	if (blockIdx.y == 0 ) {

	}

	if (blockIdx.x == gridDim.x - 1) {

	}

	if (blockIdx.y == gridDim.y -1) {

	}
	














	//int pixelsPerThread = (((imagesizex/gridDim.x)*(imagesizey/gridDim.y)*3) / (blockDim.x*blockDim.y)));
	int index_x_img = (blockIdx.x * blockDim.x + threadIdx.x);
	int index_y_img = (blockIdx.y * blockDim.y + threadIdx.y);
	int index = (index_y_img*imagesizex + index_x_img)*3;
	
    int index_x_shared = threadIdx.x;
	int index_y_shared = threadIdx.y;
	int index_shared = (index_y_shared * pad_x + index_x_shared * pixelsPerThread) * 3;

	int index_shared_true_start = ((kernelsizey/2) * pad_x + kernelsizex/2) * 3;
	int index_shared_out = index_shared_true_start + (index_y_shared * blockDim.x + index_x_shared) * 3;

	// if (threadIdx.x < 3 && threadIdx.y == 0) {
	// 	printf("True index and thread index: {%d, %d}\n", index_shared_true_start, index_shared_out);
	// }
	
	// Each threads reads 1 pixel (three values)
	for (int i = 0; i < pixelsPerThread; ++i) 
	{
		sharedMemory[index_shared + i*3 + 0] = inputImage[index + i*3 + 0];
		sharedMemory[index_shared + i*3 + 1] = inputImage[index + i*3 + 1];
		sharedMemory[index_shared + i*3 + 2] = inputImage[index + i*3 + 2];
	}

	__syncthreads();
	
	out[index + 0] = sharedMemory[index_shared_out + 0];
	out[index + 1] = sharedMemory[index_shared_out + 1];
	out[index + 2] = sharedMemory[index_shared_out + 2];

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

	// for (int i = 0; i < pixelsPerThread; ++i) 
	// {
	// 	out[index + i*3 + 0] = sharedMemory[index_shared_out + i*3 + 0];
	// 	out[index + i*3 + 1] = sharedMemory[index_shared_out + i*3 + 1];
	// 	out[index + i*3 + 2] = sharedMemory[index_shared_out + i*3 + 2];
	// }
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
	dim3 threadsPerBlock(32, 32);
	dim3 grid(imagesizex / threadsPerBlock.x, imagesizey / threadsPerBlock.y);


	filterKernel<<<grid, threadsPerBlock, ((imagesizex/grid.x) + kernelsizex)*((imagesizey/grid.y) + kernelsizey)*3*sizeof(unsigned char)>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
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
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(5, 5);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
