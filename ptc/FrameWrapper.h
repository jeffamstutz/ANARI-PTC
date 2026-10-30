// ======================================================================== //
// Copyright 2024 Ingo Wald                                                 //
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

// anari
#include <anari/anari.h>
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/type_utility.h>
// helium
#include <helium/utility/IntrusivePtr.h>
// std
#include <mpi.h>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace ptc {

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif



  using FrameWrapperNotificationHandler = std::function<void(const void *)>;
  using FrameWrapperStatusHandler =
      std::function<void(ANARIStatusSeverity, const char *)>;

  struct FrameWrapper : public helium::RefCounted
  {
    FrameWrapper(ANARIDevice d,
                 ANARIFrame f,
                 FrameWrapperNotificationHandler onObjectDestroy,
                 FrameWrapperStatusHandler reportStatus,
                 MPI_Comm = MPI_COMM_WORLD);
    virtual ~FrameWrapper() override;

    ANARIFrame handle() const;

    // Forward generic ANARI API calls to underlying ANARIFrame /////////////////
    void setParameter(const char *name, ANARIDataType type, const void *mem);
    void unsetParameter(const char *name);
    void unsetAllParameters();
    void *mapParameterArray1D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t *elementStride);
    void *mapParameterArray2D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t numElements2,
                              uint64_t *elementStride);
    void *mapParameterArray3D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t numElements2,
                              uint64_t numElements3,
                              uint64_t *elementStride);
    void unmapParameterArray(const char *name);
    void commitParameters();
    void release();
    void retain();
    int getProperty(const char *name,
                    ANARIDataType type,
                    void *mem,
                    uint64_t size,
                    ANARIWaitMask mask);
    const void *frameBufferMap(const char *channel,
                               uint32_t *width,
                               uint32_t *height,
                               ANARIDataType *pixelType);
    void frameBufferUnmap(const char *channel);
    void renderFrame();
    int frameReady(ANARIWaitMask);
    void discardFrame();
    /////////////////////////////////////////////////////////////////////////////

  protected:
    // The frame parameters ptc tracks, with the defaults the wrapped frame
    // uses when they are unset or set with the wrong type.
    struct TrackedParameters
    {
      anari::math::uint2 size{0u, 0u};
      ANARIDataType colorType{ANARI_UNKNOWN};
      // Whether the *app* enabled channel.depth, which decides whether the
      // composited depth may be mapped. The wrapped frame always renders
      // depth, since compositing needs it.
      bool depthEnabled{false};
    };

    /*! (re)allocates the buffers if the committed size or color type
        differs from what they were allocated for */
    void updateBuffers();
    /*! frees all buffers; afterwards they are empty, for no color type */
    void freeBuffers();
    /*! allocates the subclass's resources for m_bufferSize and
        m_bufferColorType; only called with a non-empty size */
    virtual void allocate() = 0;
    /*! composites the wrapped frame's channels into m_color and m_depth;
        only called with a non-empty size */
    virtual void composite() = 0;
    /*! frees what allocate() allocated */
    virtual void cleanup() = 0;

    FrameWrapperNotificationHandler m_onObjectDestroy;
    FrameWrapperStatusHandler m_reportStatus;

    int m_rank{-1};

    ANARIDevice m_device{nullptr};
    ANARIFrame m_frame{nullptr};

    // Set/unset write m_pending; commitParameters() copies it to m_committed,
    // as the wrapped frame does with its own parameters.
    TrackedParameters m_pending;
    TrackedParameters m_committed;

    // What the buffers below were allocated for (from m_committed, at the
    // last render). With no color type the compositor still runs, on
    // transparent black input, so depth composites.
    anari::math::uint2 m_bufferSize{0u, 0u};
    ANARIDataType m_bufferColorType{ANARI_UNKNOWN};
    // m_committed.depthEnabled at the last render: whether its composited
    // depth may be mapped
    bool m_renderedDepthEnabled{false};

    float *m_depth = nullptr;
    void  *m_color = nullptr;
    
    uint32_t *d_color_in{nullptr};
    uint32_t *d_color_out{nullptr};
    float *d_depth{nullptr};
  };


  // Helper functions ///////////////////////////////////////////////////////////

  /*! bytes per pixel of the composited color buffer; without a color type
      (ANARI_UNKNOWN) the compositor still works on 32-bit RGBA */
  inline size_t colorPixelSize(ANARIDataType colorType)
  {
    return colorType == ANARI_FLOAT32_VEC4 ? 4 * sizeof(float)
                                           : sizeof(uint32_t);
  }

  template <typename T>
  T bit_cast(ANARIDataType type, const void *mem)
  {
    if (sizeof(T) != anari::sizeOf(type))
      throw std::runtime_error("T size mismatch");
    T retval;
    memcpy(&retval, mem, sizeof(T));
    return retval;
  }



} // namespace ptc
