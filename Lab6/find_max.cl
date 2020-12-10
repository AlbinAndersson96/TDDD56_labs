/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, __global unsigned int *maxVals)
{ 
  unsigned int pos = 0;
  unsigned int val;

  size_t threadID = get_global_id(0);
  size_t numberOfThreads = 0;

  if (length < 512) numberOfThreads = length;
  else numberOfThreads = 512;

  // Not optimal, but it did not have to be :)
  size_t numberOfDigitsPerThread = length / numberOfThreads; // Eeach thread is responsible for this many digits (unsigned ints)

  for(int i = threadID*numberOfDigitsPerThread; i < (threadID+1)*numberOfDigitsPerThread; i++)
  {
    if(maxVals[threadID] < data[i])
      maxVals[threadID] = data[i];
  }

  // Something should happen here:
  // Array split into smaller parts, each run finds max of their respective chunk and adds it to global memory
  // Thread 0 then finds the maimum in global memory after barrier

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (threadID == 0) {
    // Find max from global memory
    for(int i = 0; i < numberOfThreads; ++i)
      if(maxVals[0] < maxVals[i])
        maxVals[0] = maxVals[i];

    data[0] = maxVals[0];
  }
}
