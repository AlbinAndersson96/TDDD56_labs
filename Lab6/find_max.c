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

// Size of data!
#define kDataLength 4194304
#define MAXPRINTSIZE 16

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
void runKernel(cl_kernel kernel, int threads, cl_mem data, unsigned int length)
{
	size_t localWorkSize, globalWorkSize;
	cl_int ciErrNum = CL_SUCCESS;
	
	// Some reasonable number of blocks based on # of threads
	if (threads<512) localWorkSize  = threads;
	else            localWorkSize  = 512;
		globalWorkSize = threads;
	
	// set the args values
	ciErrNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &data);
	ciErrNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &length);
  
	
  cl_mem o_data;
  unsigned int maxVals[512];
  for(int i = 0; i < 512; i++) maxVals[i] = 0;
  o_data = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 512 * sizeof(unsigned int), maxVals, &ciErrNum);
  ciErrNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &o_data);
  printCLError(ciErrNum,8);
	
	// Run kernel

  cl_event event;

  ciErrNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem),  (void *) &data);
  ciErrNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);

	printCLError(ciErrNum,9);
	
	// Synch
	clWaitForEvents(1, &event);
	printCLError(ciErrNum,10);
}


static cl_kernel gpgpuReduction;
static cl_kernel gpgpuReductionTwo;

int find_max_gpu(unsigned int *data, unsigned int length)
{
	cl_int ciErrNum = CL_SUCCESS;
	size_t localWorkSize, globalWorkSize;
	cl_mem io_data;
	printf("GPU reduction.\n");

  int numberOfRuns = 1;
  if (kDataLength > 16384) numberOfRuns = (kDataLength / 16384) + 1;

  unsigned int maxRuns[numberOfRuns];

  for(int i = 0; i < numberOfRuns; ++i) {
    unsigned int partData[16384];
    //memcpy(partData, data+numberOfRuns*16384*sizeof(unsigned int), 16384);
    for(int dataIndex = 0; dataIndex < 16384; ++dataIndex)
    {
      partData[dataIndex] = data[i*16384 + dataIndex];
    }

  	io_data = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 16384 * sizeof(unsigned int), partData, &ciErrNum);
	  printCLError(ciErrNum,7);

	  // ********** RUN THE KERNEL ************
	  runKernel(gpgpuReduction, 16384, io_data, 16384);

	  // Get data
	  cl_event event;
	  ciErrNum = clEnqueueReadBuffer(commandQueue, io_data, CL_TRUE, 0, 16384 * sizeof(unsigned int), partData, 0, NULL, &event);
	  printCLError(ciErrNum,11);
	  // Synch
	  clWaitForEvents(1, &event);
	  printCLError(ciErrNum,10);

    maxRuns[i] = partData[0];

	  clReleaseMemObject(io_data);
  }

  printf("Data1: %d, Data2: %d\n", maxRuns[0], maxRuns[1]);

  //Last kernel run to find max of maxes
  if(numberOfRuns > 1)
  {
    io_data = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, numberOfRuns * sizeof(unsigned int), maxRuns, &ciErrNum);
	  printCLError(ciErrNum,7);

	    // ********** RUN THE KERNEL ************
	  runKernel(gpgpuReduction, numberOfRuns, io_data, numberOfRuns);

	    // Get data
	  cl_event event;
	  ciErrNum = clEnqueueReadBuffer(commandQueue, io_data, CL_TRUE, 0, numberOfRuns * sizeof(unsigned int), maxRuns, 0, NULL, &event);
	  printCLError(ciErrNum,11);
	    // Synch
	  clWaitForEvents(1, &event);
	  printCLError(ciErrNum,10);

	  clReleaseMemObject(io_data);
  }

  printf("Data1: %d, Data2: %d\n", maxRuns[0], maxRuns[1]);


  data[0] = maxRuns[0];

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

  ResetMilli(); // You may consider moving this inside find_max_gpu(), to skip timing of data allocation.
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
