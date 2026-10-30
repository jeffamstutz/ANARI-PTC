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

#include "FrameWrapper.h"
// std
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

namespace ptc {

// FrameWrapper definitions ///////////////////////////////////////////////////

FrameWrapper::FrameWrapper(ANARIDevice ptcDevice,
    ANARIDevice d,
    ANARIFrame f,
    FrameWrapperNotificationHandler onObjectDestroy,
    FrameWrapperStatusHandler reportStatus,
    MPI_Comm comm)
    : m_onObjectDestroy(onObjectDestroy),
      m_reportStatus(reportStatus),
      m_ptcDevice(ptcDevice),
      m_device(d),
      m_frame(f)
{
  MPI_Comm_rank(comm, &m_rank);
  anariRetain(m_device, m_device);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

FrameWrapper::~FrameWrapper()
{
  anariRelease(m_device, m_frame);
  anariRelease(m_device, m_device);
  if (m_color) free(m_color);
  if (m_depth) free(m_depth);
  m_onObjectDestroy(this);
}

ANARIFrame FrameWrapper::handle() const
{
  return m_frame;
}

void FrameWrapper::setParameter(
    const char *_name, ANARIDataType type, const void *mem)
{
  // A value of the wrong type reads as unset, as it does for the wrapped
  // frame (helium's getParam).
  std::string_view name = _name;
  if (name == "size") {
    m_pending.size = type == ANARI_UINT32_VEC2
        ? bit_cast<anari::math::uint2>(type, mem)
        : TrackedParameters{}.size;
  } else if (name == "channel.color") {
    m_pending.colorType = type == ANARI_DATA_TYPE
        ? bit_cast<ANARIDataType>(type, mem)
        : TrackedParameters{}.colorType;
  } else if (name == "channel.depth") {
    // Never forwarded: the wrapped frame keeps rendering FLOAT32 depth for
    // compositing. Only FLOAT32 is a valid depth type (as in helide).
    m_pending.depthEnabled = type == ANARI_DATA_TYPE
        && bit_cast<ANARIDataType>(type, mem) == ANARI_FLOAT32;
    return;
  } else if (name == "frameCompletionCallback") {
    // Never forwarded: renderFrame() calls it after compositing.
    m_pending.completionCallback = type == ANARI_FRAME_COMPLETION_CALLBACK
        ? bit_cast<ANARIFrameCompletionCallback>(type, mem)
        : TrackedParameters{}.completionCallback;
    return;
  } else if (name == "frameCompletionCallbackUserData") {
    // ANARI_VOID_POINTER parameters pass the pointer value itself as `mem`.
    m_pending.completionCallbackUserData = type == ANARI_VOID_POINTER
        ? mem
        : TrackedParameters{}.completionCallbackUserData;
    return;
  }

  anariSetParameter(m_device, m_frame, _name, type, mem);
}

void FrameWrapper::unsetParameter(const char *_name)
{
  std::string_view name = _name;
  if (name == "size")
    m_pending.size = TrackedParameters{}.size;
  else if (name == "channel.color")
    m_pending.colorType = TrackedParameters{}.colorType;
  else if (name == "channel.depth") {
    m_pending.depthEnabled = false; // see setParameter()
    return;
  } else if (name == "frameCompletionCallback") {
    m_pending.completionCallback = TrackedParameters{}.completionCallback;
    return;
  } else if (name == "frameCompletionCallbackUserData") {
    m_pending.completionCallbackUserData =
        TrackedParameters{}.completionCallbackUserData;
    return;
  }
  anariUnsetParameter(m_device, m_frame, _name);
}

void FrameWrapper::unsetAllParameters()
{
  m_pending = {};
  anariUnsetAllParameters(m_device, m_frame);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

void *FrameWrapper::mapParameterArray1D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  return anariMapParameterArray1D(
      m_device, m_frame, name, dataType, numElements1, elementStride);
}

void *FrameWrapper::mapParameterArray2D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  return anariMapParameterArray2D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      elementStride);
}

void *FrameWrapper::mapParameterArray3D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  return anariMapParameterArray3D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      numElements3,
      elementStride);
}

void FrameWrapper::unmapParameterArray(const char *name)
{
  anariUnmapParameterArray(m_device, m_frame, name);
}

void FrameWrapper::commitParameters()
{
  m_committed = m_pending;
  anariCommitParameters(m_device, m_frame);
}

void FrameWrapper::release()
{
  refDec(helium::RefType::PUBLIC);
}

void FrameWrapper::retain()
{
  refInc(helium::RefType::PUBLIC);
}

int FrameWrapper::getProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  return anariGetProperty(m_device, m_frame, name, type, mem, size, mask);
}

const void *FrameWrapper::frameBufferMap(const char *_channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  // Report the buffers as of the last render: that is what they hold.
  *width = m_bufferSize.x;
  *height = m_bufferSize.y;

  std::string_view channel = _channel;

  if (channel == "channel.color") {
    if (m_bufferColorType != ANARI_UNKNOWN && m_color) {
      *pixelType = m_bufferColorType;
      return m_color;
    }
    m_reportStatus(ANARI_SEVERITY_WARNING,
        m_bufferColorType != ANARI_UNKNOWN
            ? "cannot map 'channel.color': the frame has not been rendered "
              "with a non-empty 'size'"
            : "cannot map 'channel.color': the frame was rendered without "
              "a 'channel.color' parameter");
  } else if (channel == "channel.depth") {
    if (m_renderedDepthEnabled && m_depth) {
      *pixelType = ANARI_FLOAT32;
      return m_depth;
    }
    m_reportStatus(ANARI_SEVERITY_WARNING,
        m_renderedDepthEnabled
            ? "cannot map 'channel.depth': the frame has not been rendered "
              "with a non-empty 'size'"
            : "cannot map 'channel.depth': the frame was rendered without "
              "its 'channel.depth' parameter set to ANARI_FLOAT32");
  } else
    std::cout << "unsupported mapping channel " << _channel << std::endl;

  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void FrameWrapper::frameBufferUnmap(const char *channel)
{
  // no-op
}

int FrameWrapper::frameReady(ANARIWaitMask m)
{
  return anariFrameReady(m_device, m_frame, m);
}

void FrameWrapper::discardFrame()
{
  anariDiscardFrame(m_device, m_frame);
}

void FrameWrapper::updateBuffers()
{
  const auto size = m_committed.size;
  const auto colorType = m_committed.colorType;
  if (size == m_bufferSize && colorType == m_bufferColorType)
    return;

  freeBuffers();
  if (!isCompositableColorType(colorType)) {
    // the same on every rank (all see the same commits), so no rank is left
    // waiting in the compositor's collectives
    throw std::runtime_error(std::string("cannot composite 'channel.color' "
                                         "of type ")
        + anari::toString(colorType)
        + ": supported are UFIXED8_VEC4, UFIXED8_RGBA_SRGB and FLOAT32_VEC4");
  }
  m_bufferSize = size;
  m_bufferColorType = colorType;

  if (size.x == 0 || size.y == 0)
    return;

  const size_t numPixels = size_t(size.x) * size.y;
  m_color = malloc(numPixels * colorPixelSize(colorType));
  m_depth = (float *)malloc(numPixels * sizeof(float));
  try {
    allocate();
  } catch (...) {
    // don't leave buffers that look ready to composite into
    freeBuffers();
    throw;
  }
}

void FrameWrapper::freeBuffers()
{
  cleanup();
  free(m_color);
  free(m_depth);
  m_color = nullptr;
  m_depth = nullptr;
  m_bufferSize = {0u, 0u};
  m_bufferColorType = ANARI_UNKNOWN;
}

void FrameWrapper::renderFrame()
{
  updateBuffers();
  m_renderedDepthEnabled = m_committed.depthEnabled;
  anariRenderFrame(m_device, m_frame);
  anariFrameReady(m_device, m_frame, ANARI_WAIT);
  // Every rank sees the same commits, so all of them skip or composite.
  if (m_bufferSize.x != 0 && m_bufferSize.y != 0)
    composite();
  // The composited frame is ready (rendering is synchronous), so the
  // continuation has run before any anariFrameReady() returns.
  if (auto callback = m_committed.completionCallback) {
    callback(m_committed.completionCallbackUserData,
        m_ptcDevice,
        (ANARIFrame)this);
  }
}

} // namespace ptc
