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

#include "FrameWrapper.h"
// mpi
#include "deepCompositing.h"

namespace ptc {

  struct CUDAFrameWrapper : public FrameWrapper
  {
    CUDAFrameWrapper(ANARIDevice ptcDevice,
                     ANARIDevice d,
                     ANARIFrame f,
                     FrameWrapperNotificationHandler onObjectDestroy,
                     FrameWrapperStatusHandler reportStatus,
                     MPI_Comm = MPI_COMM_WORLD);
    virtual ~CUDAFrameWrapper();
    
    // void renderFrame() override;
    void allocate() override;
    void composite() override;
    void cleanup() override;
    
    dc::Compositor m_deepComp;
    /*! composited depth (rank 0), before it is copied to m_depth */
    float *d_depth_out{nullptr};
  };

}
