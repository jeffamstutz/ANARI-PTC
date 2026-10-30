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

#pragma once

#include "FrameWrapper.h"
// mpi
#include "CPUCompositor.h"
// std
#include <memory>

namespace ptc {

  struct CPUFrameWrapper : public FrameWrapper
  {
    CPUFrameWrapper(ANARIDevice ptcDevice,
                    ANARIDevice d,
                    ANARIFrame f,
                    FrameWrapperNotificationHandler onObjectDestroy,
                    FrameWrapperStatusHandler reportStatus);
    virtual ~CPUFrameWrapper();
    
    // void renderFrame() override;
    void allocate() override;
    void stage(const void *color, const float *depth) override;
    void composite() override;
    void cleanup() override;
    void createCompositor(MPI_Comm comm) override;
    void destroyCompositor() override;

    std::unique_ptr<CPUCompositor> compositor;
    /*! compositor color input when the frame has no channel.color */
    std::vector<uint32_t> m_noColor;
    /*! what stage() staged: the compositor's input */
    const void *m_inColor{nullptr};
    const float *m_inDepth{nullptr};
  };

}
