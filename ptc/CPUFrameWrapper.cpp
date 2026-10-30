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
                     FrameWrapperStatusHandler reportStatus)
    : FrameWrapper(ptcDevice,d,f,onObjectDestroy,reportStatus)
  {
  }

  CPUFrameWrapper
  ::~CPUFrameWrapper()
  {
    CPUFrameWrapper::cleanup();
  }
  
  void CPUFrameWrapper::allocate()
  {
    auto format = CPUCompositor::ColorFormat::RGBA8;
    if (m_bufferColorType == ANARI_FLOAT32_VEC4)
      format = CPUCompositor::ColorFormat::RGBA32F;
    else if (m_bufferColorType == ANARI_UFIXED8_RGBA_SRGB)
      format = CPUCompositor::ColorFormat::RGBA8_SRGB;
    compositor->resize(m_bufferSize.x, m_bufferSize.y, format);

    if (m_bufferColorType == ANARI_UNKNOWN)
      m_noColor.assign(size_t(m_bufferSize.x) * m_bufferSize.y, 0u);
  }

  void CPUFrameWrapper::stage(const void *color, const float *depth)
  {
    // the compositor reads the wrapped frame's channels in place
    m_inColor = color ? color : m_noColor.data();
    m_inDepth = depth;
  }

  void CPUFrameWrapper::composite()
  {
    compositor->run(m_color, m_depth, m_inColor, m_inDepth);
    m_inColor = nullptr;
    m_inDepth = nullptr;
  }

  void CPUFrameWrapper::cleanup()
  {
    m_noColor = {};
    m_inColor = nullptr;
    m_inDepth = nullptr;
  }

  void CPUFrameWrapper::createCompositor(MPI_Comm comm)
  {
    compositor = std::make_unique<CPUCompositor>(comm);
  }

  void CPUFrameWrapper::destroyCompositor()
  {
    compositor.reset();
  }

} // namespace ptc
