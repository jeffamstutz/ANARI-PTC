// ======================================================================== //
// Copyright 2024++ Ingo Wald                                               //
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

#include "CUDAFrameWrapper.h"
// std
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
// cuda
#include <cuda.h>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

#include "cuda_helper.h"

namespace ptc {

  // GPU kernels ////////////////////////////////////////////////////////////////
  
  __global__ void writeFrags(dc::DeviceInterface di,
                             uint32_t size_x,
                             uint32_t size_y,
                             const float *d_depth,
                             const void *d_color,
                             ANARIDataType colorType)
  {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= size_x)
      return;
    if (iy >= size_y)
      return;

    const auto offset = ix + iy * size_x;
    const float z = d_depth[offset];

    if (colorType == ANARI_FLOAT32_VEC4) {
      const float *rgba = (const float *)d_color + (offset * 4);
#if 1
      const float r = rgba[0];
      const float g = rgba[1];
      const float b = rgba[2];
      const float a = rgba[3];
#else
      const float a = rgba[0];
      const float b = rgba[1];
      const float g = rgba[2];
      const float r = rgba[3];
#endif
      dc::Fragment frag(z, make_float3(r, g, b), a);
      di.write(ix, iy, frag);
    } else {
      // UFIXED8_RGBA_SRGB stays sRGB-encoded here, exactly (fragments
      // are 8 bit); the compositor decodes it before blending
      const uint32_t rgba = *((const uint32_t *)d_color + offset);
      const float a = ((rgba >> 24) & 0xff) / 255.f;
      const float b = ((rgba >> 16) & 0xff) / 255.f;
      const float g = ((rgba >> 8) & 0xff) / 255.f;
      const float r = ((rgba >> 0) & 0xff) / 255.f;
      dc::Fragment frag(z, make_float3(r, g, b), a);
      di.write(ix, iy, frag);
    }
  }

  // Helper functions /////////////////////////////////////////////////////////

  namespace {

  /*! Makes 'gpu' current on the calling thread while it exists, then makes
      current again whatever context was current before (none included), so
      that the app and the wrapped device never see ptc's GPU. A negative
      'gpu' (no GPU picked yet) does nothing. */
  class DeviceScope
  {
   public:
    explicit DeviceScope(int gpu)
    {
      if (gpu < 0 || cuCtxGetCurrent(&m_previous) != CUDA_SUCCESS)
        return;
      m_restore = true;
      cudaSetDevice(gpu);
    }
    ~DeviceScope()
    {
      if (m_restore)
        cuCtxSetCurrent(m_previous);
    }
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;

   private:
    CUcontext m_previous{nullptr};
    bool m_restore{false};
  };

  } // namespace

  // FrameWrapper definitions ///////////////////////////////////////////////////

  CUDAFrameWrapper
  ::CUDAFrameWrapper(ANARIDevice ptcDevice,
                     ANARIDevice d,
                     ANARIFrame f,
                     FrameWrapperNotificationHandler onObjectDestroy,
                     FrameWrapperStatusHandler reportStatus)
    : FrameWrapper(ptcDevice,d,f,onObjectDestroy,reportStatus)
  {
  }

  CUDAFrameWrapper
  ::~CUDAFrameWrapper()
  {
    CUDAFrameWrapper::cleanup();
  }
  
  void CUDAFrameWrapper::allocate()
  {
    DeviceScope gpu(m_gpu);
    const auto &size = m_bufferSize;
    const size_t numPixels = size_t(size.x) * size.y;

    m_deepComp->resize(size.x, size.y);

    const size_t colorSize = colorPixelSize(m_bufferColorType);
    cudaMalloc((void **)&d_color_in,  numPixels * colorSize);
    cudaMalloc((void **)&d_color_out, numPixels * colorSize);
    cudaMalloc((void **)&d_depth, numPixels * sizeof(*d_depth));
    cudaMalloc((void **)&d_depth_out, numPixels * sizeof(*d_depth_out));
    // without channel.color, composite transparent black so depth still
    // composites; composite() never overwrites it
    if (m_bufferColorType == ANARI_UNKNOWN)
      cudaMemset(d_color_in, 0, numPixels * colorSize);
    CUDA_SYNC_CHECK();
  }

  void CUDAFrameWrapper::stage(const void *color, const float *depth)
  {
    DeviceScope gpu(m_gpu);
    const auto &size = m_bufferSize;
    const size_t numPixels = size_t(size.x) * size.y;
    const size_t colorSize = colorPixelSize(m_bufferColorType);

    cudaMemcpy(d_depth, depth, numPixels * sizeof(float), cudaMemcpyDefault);
    // without color, d_color_in stays the transparent black allocate() set
    if (color)
      cudaMemcpy(d_color_in, color, numPixels * colorSize, cudaMemcpyDefault);

    auto ngx = dc::divRoundUp(size.x, 16);
    auto ngy = dc::divRoundUp(size.y, 16);

    writeFrags<<<dim3(ngx, ngy), dim3(16, 16)>>>
      (m_deepComp->prepare(),
       size.x,
       size.y,
       d_depth,
       d_color_in,
       m_bufferColorType);
    // CUDA errors so far surface here, before the ranks agree to composite
    CUDA_SYNC_CHECK();
  }

  void CUDAFrameWrapper::composite()
  {
    DeviceScope gpu(m_gpu);
    const auto &size = m_bufferSize;
    const size_t numPixels = size_t(size.x) * size.y;
    const size_t colorSize = colorPixelSize(m_bufferColorType);

    auto format = dc::ColorFormat::RGBA8;
    if (m_bufferColorType == ANARI_FLOAT32_VEC4)
      format = dc::ColorFormat::RGBA32F;
    else if (m_bufferColorType == ANARI_UFIXED8_RGBA_SRGB)
      format = dc::ColorFormat::RGBA8_SRGB;
    const bool root = m_rank == 0;
    m_deepComp->finish(root ? d_color_out : nullptr,
                       root ? d_depth_out : nullptr,
                       format);
    // only rank 0 receives the composited image; elsewhere the output
    // buffers hold nothing worth copying
    if (root) {
      cudaMemcpy(m_color, d_color_out, numPixels * colorSize,
                 cudaMemcpyDefault);
      cudaMemcpy(m_depth, d_depth_out, numPixels * sizeof(float),
                 cudaMemcpyDefault);
    }

    CUDA_SYNC_CHECK();
  }

  void CUDAFrameWrapper::createCompositor(MPI_Comm comm)
  {
    // The first compositor picks a GPU by this rank's place among the ranks
    // of 'comm' on its node; later ones (after an mpiCommunicator change)
    // keep that GPU instead of moving the frame to another one. The only
    // collective here comes first: whatever fails after it, on some ranks
    // only, setCommunicator() has all ranks agree on.
    int localRank = 0;
    if (m_gpu < 0) {
      MPI_Comm node = MPI_COMM_NULL;
      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
      MPI_Comm_rank(node, &localRank);
      MPI_Comm_free(&node);
    }

    m_deepComp = std::make_unique<dc::Compositor>(1, comm);
    if (m_gpu < 0) {
      int numGPUs = 0;
      if (cudaGetDeviceCount(&numGPUs) != cudaSuccess || numGPUs == 0)
        throw std::runtime_error("cannot composite: no CUDA device");
      m_gpu = localRank % numGPUs;
      cudaDeviceProp prop{};
      cudaGetDeviceProperties(&prop, m_gpu);
      const std::string msg = "compositing on GPU " + std::to_string(m_gpu)
          + " (" + prop.name + ")";
      m_reportStatus(ANARI_SEVERITY_INFO, msg.c_str());
    }
    // the compositor makes it current around its own CUDA work, and
    // restores the current device afterwards
    m_deepComp->affinitizedGPU = m_gpu;
  }

  void CUDAFrameWrapper::destroyCompositor()
  {
    DeviceScope gpu(m_gpu);
    m_deepComp.reset();
  }

  void CUDAFrameWrapper::cleanup()
  {
    // nothing is allocated before the first compositor picked a GPU
    if (m_gpu < 0)
      return;
    DeviceScope gpu(m_gpu);
    if (d_depth) 
      cudaFree(d_depth);
    if (d_color_in)
      cudaFree(d_color_in);
    if (d_color_out)
      cudaFree(d_color_out);
    if (d_depth_out)
      cudaFree(d_depth_out);

    CUDA_SYNC_CHECK();

    d_depth = nullptr;
    d_depth_out = nullptr;
    d_color_in = nullptr;
    d_color_out = nullptr;
  }

} // namespace ptc
