/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  size_t threadID = get_global_id(0);
  size_t numberOfThreads = 0;

  if (length < 256) numberOfThreads = length;
  else numberOfThreads = 256;

  // Not optimal, but it did not have to be :)
  size_t numberOfDigitsPerThread = length / numberOfThreads; // Eeach thread is responsible for this many digits (unsigned ints)

  for(int i = threadID*numberOfDigitsPerThread; i < (threadID+1)*numberOfDigitsPerThread; ++i)
  {
    if(data[threadID] < data[i])
      data[threadID] = data[i];
  }

  if (data[0] < data[threadID]) data[0] = data[threadID];

}
