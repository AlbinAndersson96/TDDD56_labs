/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int val;

  size_t threadID = get_global_id(0);
  size_t numberOfThreads = 0;

  if (length < 512) numberOfThreads = length;
  else numberOfThreads = 512;

  // Not optimal, but it did not have to be :)
  size_t numberOfDigitsPerThread = length / numberOfThreads; // Eeach thread is responsible for this many digits (unsigned ints)

  __global unsigned int *dataStart;
  dataStart = data; // Each threads gets its own section of digits


  // Something should happen here:
  // Array split into smaller parts, each run finds max of their respective chunk and adds it to global memory
  // Thread 0 then finds the maimum in global memory after barrier

  data[threadID] = dataStart + sizeof(unsigned int);

  //barrier();

  // if (threadID == 0) {
  //   // Find max from global memory
  // } 
}
