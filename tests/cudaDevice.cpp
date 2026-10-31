// SPDX-License-Identifier: Apache-2.0
//
// The CUDA runtime calls ptcSmoke makes, in a translation unit of their own:
// the CUDA headers' global uint2/float3 clash with anari::math's.

#include "cudaDevice.h"
#include <cuda_runtime_api.h>

int cudaDeviceCount()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

bool makeCudaDeviceCurrent(int device)
{
  return cudaSetDevice(device) == cudaSuccess;
}

int currentCudaDevice()
{
  int device = -1;
  return cudaGetDevice(&device) == cudaSuccess ? device : -1;
}
