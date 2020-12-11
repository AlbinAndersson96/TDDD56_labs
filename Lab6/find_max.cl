/*
 * Placeholder OpenCL kernel
 */

#define THREADS 1024
__kernel void find_max(__global unsigned int *data, const unsigned int length, __local unsigned int *tmp)
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
  //if(tmp)
  tmp[threadIDLocal] = data[threadIDGlobal];

  barrier(CLK_GLOBAL_MEM_FENCE);
  
  if (get_local_id(0) == 0) {
    for (int i = 0; i < get_local_size(0); ++i) {
      if (tmp[0] < tmp[i]) tmp[0] = tmp[i];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  data[0] = tmp[0];
}