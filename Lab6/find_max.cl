/*
 * Placeholder OpenCL kernel
 */

#define THREADS 1024
__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  size_t threadIDLocal = get_local_id(0);
  size_t threadIDGlobal = get_global_id(0);
  size_t numberOfThreads = 0;

  // if(16384 < 256)
  // if (length < THREADS) numberOfThreads = length;
  // else numberOfThreads = THREADS; //256

  // Not optimal, but it did not have to be :)
  // 8192 / 1024 = 8
  //size_t numberOfDigitsPerThread = length / get_local_size(0); // Eeach thread is responsible for this many digits (unsigned ints)

  // for(int i = threadID; i < (threadID+1); ++i)
  // {
  if(data[threadIDGlobal] > data[threadIDLocal]) data[threadIDLocal] = data[threadIDGlobal];
    //   data[threadIDGlobal] = data[i];
  // }

  

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (data[0] < data[threadIDLocal]) data[0] = data[threadIDLocal];
  
  //if (get_local_id(0) == 0) {
  //   for (int i = 0; i < get_local_size(0); ++i) {
  //if (data[0] < data[threadIDLocal]) data[0] = data[threadIDLocal];
 // }
}