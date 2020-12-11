/*
 * Placeholder OpenCL kernel
 */

#define THREADS 1024
__kernel void find_max(__global unsigned int *data, const unsigned int length, __local unsigned int *tmp)
{ 
  size_t threadIDLocal = get_local_id(0);

  size_t threadIDGlobal1 = get_global_id(0);
  size_t threadIDGlobal2 = threadIDGlobal1*2;
  
  size_t numberOfDigits = length / get_local_size();

  int biggest = 0;
  for (int i = threadIDLocal; i < numberOfDigits*get_local_size(); i += get_local_size()) {
    biggest = max(biggest, data[i]);
  }

  data[threadIDLocal] = biggest;
  
}