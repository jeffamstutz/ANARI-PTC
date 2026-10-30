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

#include "CPUFrameWrapper.h"
// std
#include <cstring>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

namespace ptc {

  // FrameWrapper definitions ///////////////////////////////////////////////////

  CPUFrameWrapper
  ::CPUFrameWrapper(ANARIDevice ptcDevice,
                     ANARIDevice d,
                     ANARIFrame f,
                     FrameWrapperNotificationHandler onObjectDestroy,
                     FrameWrapperStatusHandler reportStatus,
                     MPI_Comm comm)
    : FrameWrapper(ptcDevice,d,f,onObjectDestroy,reportStatus,comm),
      compositor(comm)
  {
  }

  CPUFrameWrapper
  ::~CPUFrameWrapper()
  {
    CPUFrameWrapper::cleanup();
  }
  
  void CPUFrameWrapper::allocate()
  {
    if (m_bufferColorType == ANARI_FLOAT32_VEC4) {
      throw std::runtime_error
        ("support for FLOAT32_VEC4 color channel not implemented");
    }

    compositor.resize(m_bufferSize.x, m_bufferSize.y);

    if (m_bufferColorType == ANARI_UNKNOWN)
      m_noColor.assign(size_t(m_bufferSize.x) * m_bufferSize.y, 0u);
  }

  void CPUFrameWrapper::composite()
  {
    const bool hasColor = m_bufferColorType != ANARI_UNKNOWN;
    anari::math::uint2 size;
    ANARIDataType ptType = ANARI_UNKNOWN;
    const float *depth
      = (const float *)anariMapFrame(m_device, m_frame, "channel.depth",
                                     &size.x, &size.y, &ptType);
    const uint32_t *color = hasColor
      ? (const uint32_t *)anariMapFrame(m_device, m_frame, "channel.color",
                                        &size.x, &size.y, &ptType)
      : m_noColor.data();

    compositor.run((uint32_t *)m_color, m_depth, color, depth);

    anariUnmapFrame(m_device, m_frame, "channel.depth");
    if (hasColor)
      anariUnmapFrame(m_device, m_frame, "channel.color");
  }

  void CPUFrameWrapper::cleanup()
  {
    m_noColor = {};
  }

} // namespace ptc
