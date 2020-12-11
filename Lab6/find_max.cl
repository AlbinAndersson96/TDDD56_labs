/*
 * Placeholder OpenCL kernel
 */

#define THREADS 512
__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  size_t threadID = get_local_id(0);
  size_t numberOfThreads = 0;

  // if(16384 < 256)
  // if (length < THREADS) numberOfThreads = length;
  // else numberOfThreads = THREADS; //256

  // Not optimal, but it did not have to be :)
  // 8192 / 256 = 8
  size_t numberOfDigitsPerThread = length / get_local_size(0); // Eeach thread is responsible for this many digits (unsigned ints)

  for(int i = threadID*numberOfDigitsPerThread; i < (threadID+1)*numberOfDigitsPerThread; ++i)
  {
    if(data[threadID] < data[i])
      data[threadID] = data[i];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (data[0] < data[threadID]) data[0] = data[threadID];
}