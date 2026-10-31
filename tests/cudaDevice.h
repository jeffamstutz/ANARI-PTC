// SPDX-License-Identifier: Apache-2.0
//
// The calling thread's current CUDA device, for ptcSmoke (PTC_TEST_CUDA)

#pragma once

// number of visible CUDA devices; 0 on error
int cudaDeviceCount();
// cudaSetDevice(); whether it succeeded
bool makeCudaDeviceCurrent(int device);
// cudaGetDevice(); -1 on error
int currentCudaDevice();
