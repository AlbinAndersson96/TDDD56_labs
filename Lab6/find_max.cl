/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  size_t threadIDLocal = get_local_id(0);

  size_t threadIDGlobal1 = get_global_id(0);
  size_t threadIDGlobal2 = threadIDGlobal1*2;
  
  // 8192 / 1024 = 8
  size_t numberOfDigits = length / get_local_size(0);

  int biggest = 0;
  for (int i = threadIDLocal; i < numberOfDigits*get_local_size(0); i += get_local_size(0)) {
    if (biggest < data[i]) biggest = data[i];
  }

  data[threadIDLocal] = biggest;

}