// Laboration in OpenCL. Based on a lab by Jens Ogniewski and Ingemar Ragnemalm 2010-2011.
// Rewritten by Ingemar 2017.
// Very close to the shell for bitonic sort.

// Compilation line for Linux:
// test$ gcc -std=c99 find_max.c -o find_max milli.c CLutilities.c -lOpenCL  -I/usr/local/cuda/include/

// C implementation included.
// The OpenCL kernel is just a placeholder.
// Implement reduction in OpenCL!

// standard utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <CL/cl.h>
  #include <GL/glut.h>
#endif
#include "CLutilities.h"
#include "milli.h"

#define MAXPRINTSIZE 16

// Size of data!
//#define kDataLength 4194304
//#define kDataLength 8388608
//#define kDataLength 16777216
//#define kDataLength 33554432
//#define kDataLength 67108864
#define kDataLength 268435456
//#define kDataLength 1073741824

// #define THREADS 256
// #define PART_SIZE 16384

#define THREADS 1024
#define PART_SIZE 524288
#define MAX_ITERATIONS 10

// #define THREADS 512
// #define PART_SIZE 16384

unsigned int *generateRandomData(unsigned int length)
{
  unsigned int seed;
  struct timeval t_s;
  gettimeofday(&t_s, NULL);
  seed = (unsigned int)t_s.tv_usec;
//  printf("\nseed: %u\n",seed);

  unsigned int *data, i;

  data = (unsigned int *)malloc(length*sizeof(unsigned int));
  if (!data)
  {
    printf("\nerror allocating data.\n\n");
    return NULL;
  }
  srand(seed);
  for (i=0; i<length; i++)
    data[i] = (unsigned int)(rand()%length);
    printf("generateRandomData done.\n\n");
  return data;
}

// ------------ GPU ------------

// Kernel run conveniently packed. Edit as needed, i.e. with more parameters.
// Only ONE array of data.
// __kernel void sort(__global unsigned int *data, const unsigned int length)
void runKernel(cl_kernel kernel, int threads, cl_mem data, cl_mem intermediate, unsigned int length, const unsigned int currentRun)
{
  // threads = 16384, length = 16384
	size_t localWorkSize, globalWorkSize;
	cl_int ciErrNum = CL_SUCCESS;
	
	// Some reasonable number of blocks based on # of threads
	if (threads < THREADS) localWorkSize  = threads;
	else            localWorkSize  = THREADS;
		globalWorkSize = localWorkSize;
	
	// set the args values
	ciErrNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &data); // partData
	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &length); // 16384
  ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &intermediate);
  ciErrNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *) &currentRun);
  //ciErrNum |= clSetKernelArg(kernel, 2, localWorkSize*sizeof(cl_uint), NULL); // 16384
  
  printCLError(ciErrNum,8);
  
	// Run kernel
  cl_event event;
  ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);

	printCLError(ciErrNum,9);
	
	// Synch
	clWaitForEvents(1, &event);
	printCLError(ciErrNum,10);
}

static cl_kernel gpgpuReduction;

int find_max_gpu(unsigned int *data, unsigned int length)
{
  const int outputsPerThread = PART_SIZE / THREADS;
  const int sizeOfBatchOutput = kDataLength / outputsPerThread;

	cl_int ciErrNum = CL_SUCCESS;
	size_t localWorkSize, globalWorkSize;
	cl_mem io_data, intermediate;
	printf("GPU reduction.\n");
  
  int numberOfRuns = kDataLength / PART_SIZE, currentLength = kDataLength;
  if (kDataLength > PART_SIZE) numberOfRuns = (kDataLength / PART_SIZE) + 1; // 131072 times

  // unsigned int maxRuns[numberOfRuns]; // 131072 size (131072 maxes)
  // for(int i = 0; i < numberOfRuns; ++i) {
  //   maxRuns[i] = 0;
  // }

  

  unsigned int partData[PART_SIZE]; // 8192
  io_data = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, PART_SIZE * sizeof(unsigned int), partData, &ciErrNum);
  intermediate = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, numberOfRuns * THREADS * sizeof(unsigned int), NULL, &ciErrNum);

  cl_event eventReadBuffer, eventWriteBuffer;
  ResetMilli();

  const unsigned int MAX_ITERATIONS = 2;
  for(int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    
    if (currentLength > PART_SIZE) numberOfRuns = (currentLength / PART_SIZE) + 1;

    for(int i = 0; i < numberOfRuns; ++i) {
      for(int dataIndex = 0; dataIndex < PART_SIZE; ++dataIndex) {
        partData[dataIndex] = data[i*PART_SIZE + dataIndex];
      }

      ciErrNum = clEnqueueWriteBuffer(commandQueue, io_data, CL_TRUE, 0, PART_SIZE*sizeof(unsigned int), partData, 0, NULL, &eventWriteBuffer);
      clWaitForEvents(1, &eventWriteBuffer);
	    printCLError(ciErrNum,7);

      runKernel(gpgpuReduction, PART_SIZE, io_data, intermediate, PART_SIZE, i);
    }
  }

  unsigned int currentSize = kDataLength / (outputsPerThread * MAX_ITERATIONS);
  ciErrNum = clEnqueueReadBuffer(commandQueue, intermediate, CL_TRUE, 0, currentSize*sizeof(unsigned int), data, 0, NULL, &eventReadBuffer);
  printCLError(ciErrNum,11);
  clWaitForEvents(1, &eventReadBuffer);

    unsigned int max = 0;
    for(int t = 0; t < currentSize; ++t) {
      if (max < data[t]) {
        max = data[t];
      }
    }

    data[0] = max;
  // for(int i = 0; i < numberOfRuns; ++i) {
    
  //   for(int dataIndex = 0; dataIndex < PART_SIZE; ++dataIndex)
  //   {
  //     partData[dataIndex] = data[i*PART_SIZE + dataIndex];
  //   }

  //   ciErrNum = clEnqueueWriteBuffer(commandQueue, io_data, CL_TRUE, 0, PART_SIZE*sizeof(unsigned int), partData, 0, NULL, &eventWriteBuffer);
  //   clWaitForEvents(1, &eventWriteBuffer);
	//   printCLError(ciErrNum,7);
  //   //clFinish(commandQueue);

	//   // ********** RUN THE KERNEL ************
	//   runKernel(gpgpuReduction, PART_SIZE, io_data, intermediate, PART_SIZE);

	//   // Get data
	//   // ciErrNum = clEnqueueReadBuffer(commandQueue, io_data, CL_TRUE, 0, THREADS*sizeof(unsigned int), partData, 0, NULL, &eventReadBuffer);
	//   // printCLError(ciErrNum,11);
  //   // clWaitForEvents(1, &eventReadBuffer);

  //   // for(int t = 0; t < THREADS; ++t) {
  //   //   if (maxRuns[i] < partData[t]) {
  //   //     //printf("New max: %d\n", partData[t]);
  //   //     maxRuns[i] = partData[t];
  //   //   }
  //   // }
  // }
  	// ciErrNum = clEnqueueReadBuffer(commandQueue, intermediate, CL_TRUE, 0, THREADS*sizeof(unsigned int), data, 0, NULL, &eventReadBuffer);
	  // printCLError(ciErrNum,11);
    // clWaitForEvents(1, &eventReadBuffer);

  // //Last pass on CPU
  // unsigned int max = 0;
  // for(int i = 0; i < numberOfRuns - 1; i++)
  // {
  //   //printf("Max %d: %u\n", i, maxRuns[i]);
  //   if(maxRuns[i] > max) {
  //     //printf("New max in last it at index %d: %d\n", i, maxRuns[i]);
  //     max = maxRuns[i];
  //   }
      
  // }
  // data[0] = max;
  
	return ciErrNum;
}

// CPU max finder (sequential)
void find_max_cpu(unsigned int *data, int N)
{
  unsigned int i, m;
  
	m = data[0];
	for (i=0;i<N;i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}
// ------------ main ------------

int main( int argc, char** argv) 
{
  int length = kDataLength; // SIZE OF DATA
  unsigned short int header[2];
  
  // Computed data
  unsigned int *data_cpu, *data_gpu;
  
  // Find a platform and device
  if (initOpenCL()<0)
  {
    closeOpenCL();
    return 1;
  }
  // Load and compile the kernel
  gpgpuReduction = compileKernel("find_max.cl", "find_max");
  //gpgpuReuctionSecond = compileKernel("find_max_second.cl", "find_max_second");

  data_cpu = generateRandomData(length);
  data_gpu = (unsigned int *)malloc (length*sizeof(unsigned int));

  if ((!data_cpu)||(!data_gpu))
  {
    printf("\nError allocating data.\n\n");
    return 1;
  }
  
  // Copy to gpu data.
  for(int i=0;i<length;i++)
    data_gpu[i]=data_cpu[i];
  
  ResetMilli();
  find_max_cpu(data_cpu,length);
  printf("CPU %f\n", GetSeconds());

  //ResetMilli(); // You may consider moving this inside find_max_gpu(), to skip timing of data allocation.
  find_max_gpu(data_gpu,length);
  printf("GPU %f\n", GetSeconds());

  // Print part of result
  for (int i=0;i<MAXPRINTSIZE;i++)
    printf("%d ", data_gpu[i]);
  printf("\n");

  if (data_cpu[0] != data_gpu[0])
    {
      printf("Wrong value at position 0.\n");
      closeOpenCL();
      return(1);
    }
  printf("\nYour max looks correct!\n");
  closeOpenCL();
  if (gpgpuReduction) clReleaseKernel(gpgpuReduction);
  return 0;
}