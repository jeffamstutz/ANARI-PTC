// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// These throw instead of printing: the library reports errors through its
// caller (ptc turns exceptions into status callback messages).

#define DC_CUDA_ERROR(where, what, rc)                                  \
  std::runtime_error(std::string(where) + "CUDA call (" + (what)        \
      + ") failed with code " + std::to_string(int(rc)) + " (" __FILE__ \
      ":" + std::to_string(__LINE__) + "): " + cudaGetErrorString(rc))

#define CUDA_CHECK( call )                                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess)                                              \
      throw DC_CUDA_ERROR("", #call, rc);                               \
  }

#define CUDA_CALL(call) CUDA_CHECK(cuda##call)

#define CUDA_CHECK2( where, call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess)                                              \
      throw DC_CUDA_ERROR(std::string(where ? where : "") + ": ",       \
                          #call, rc);                                   \
  }

#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t rc = cudaGetLastError();                                \
    if (rc != cudaSuccess)                                              \
      throw DC_CUDA_ERROR("", "cudaDeviceSynchronize", rc);             \
  }
