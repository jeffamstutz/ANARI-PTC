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

// ptc
#include "CPUCompositor.h"
// anari
#include <anari/anari.h>
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/type_utility.h>
// helium
#include <helium/utility/IntrusivePtr.h>
// std
#include <mpi.h>
#include <array>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ptc {

  /*! Frame channels ptc composites by taking the value of each pixel's
      nearest fragment (equal depth: the lower rank's), since ids don't
      blend. Each is UINT32 and requested from the wrapped frame only when
      the app enables it. */
  inline constexpr const char *ID_CHANNELS[] = {
      "channel.primitiveId", "channel.objectId", "channel.instanceId"};
  inline constexpr size_t NUM_ID_CHANNELS =
      sizeof(ID_CHANNELS) / sizeof(ID_CHANNELS[0]);

  using FrameWrapperNotificationHandler = std::function<void(const void *)>;
  using FrameWrapperStatusHandler = std::function<void(
      ANARIStatusSeverity, ANARIStatusCode, const char *)>;

  /*! the status code of a message without a more specific one */
  inline ANARIStatusCode defaultStatusCode(ANARIStatusSeverity severity)
  {
    return severity <= ANARI_SEVERITY_ERROR ? ANARI_STATUS_UNKNOWN_ERROR
                                            : ANARI_STATUS_NO_ERROR;
  }

  /*! an error PTCDevice reports with this status code, instead of
      ANARI_STATUS_UNKNOWN_ERROR */
  struct StatusError : public std::runtime_error
  {
    StatusError(ANARIStatusCode code, const std::string &what)
        : std::runtime_error(what), code(code)
    {}
    ANARIStatusCode code;
  };

  struct FrameWrapper : public helium::RefCounted
  {
    /*! ptcDevice is the device handle the app holds (passed to its frame
        completion callback); d and f are the wrapped device and frame */
    FrameWrapper(ANARIDevice ptcDevice,
                 ANARIDevice d,
                 ANARIFrame f,
                 FrameWrapperNotificationHandler onObjectDestroy,
                 FrameWrapperStatusHandler reportStatus);
    ~FrameWrapper() override;

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
    /*! renders and composites; the frame completion callback is left to
        completeFrame(), which runs even if this throws */
    void renderFrame();
    int frameReady(ANARIWaitMask);
    void discardFrame();
    /////////////////////////////////////////////////////////////////////////////

    /*! calls the app's frame completion callback, if set; after every
        render, failed or not, once that render's error is reported */
    void completeFrame();
    /*! throws if called from inside this frame's completion callback;
        'what' is the refused operation, e.g. "render". commitParameters()
        checks itself; renderFrame()'s caller checks, so that it can skip
        completeFrame() then */
    void refuseFromCallback(const char *what) const;

  private:
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
      // Whether the app enabled each of ID_CHANNELS (as ANARI_UINT32). Unlike
      // depth, they are forwarded: the wrapped frame renders them only then.
      std::array<bool, NUM_ID_CHANNELS> idEnabled{};
      // The app's frame completion callback. Never forwarded: the wrapped
      // frame completes before compositing, and would pass its own handles.
      ANARIFrameCompletionCallback completionCallback{nullptr};
      const void *completionCallbackUserData{nullptr};
    };

    /*! (re)allocates the buffers and resizes the compositor if the
        committed size or color type differs from what they were allocated
        for. Not collective: renderFrame() has the ranks agree afterwards. */
    void updateBuffers();
    /*! frees all buffers; afterwards they are empty, for no color type */
    void freeBuffers();
    /*! the local half of compositeIds(): gathers this rank's record of
        each pixel's depth, rank and enabled ID_CHANNELS into m_idRecords */
    void stageIds(const float *depth);
    /*! composites the enabled ID_CHANNELS into m_ids: one reduction over
        m_comm keeps each pixel's nearest record; collective over m_comm */
    void compositeIds();
    /*! whether 'ok' is true on every rank of 'comm'; collective over
        'comm'. Ranks call it before a collective step that may have failed
        on some of them only, so that all of them skip that step (and
        report an error) instead of the others hanging in it. */
    static bool agree(MPI_Comm comm, bool ok);
    /*! maps a channel of the wrapped frame; throws unless it is 'type' and
        m_bufferSize */
    const void *mapWrapped(const char *channel, ANARIDataType type);
    /*! makes the compositor composite over 'comm' (a copy of it), if it
        doesn't already composite over the same ranks; collective over
        'comm' and over the communicator it replaces. MPI_COMM_NULL only
        tears down the current compositor. */
    void setCommunicator(MPI_Comm comm);

    /*! reports through the status callback, with ANARI_STATUS_NO_ERROR
        below ERROR and ANARI_STATUS_UNKNOWN_ERROR from ERROR up */
    void reportStatus(
        ANARIStatusSeverity severity, const std::string &msg) const;
    /*! makes the next commit warn that parameter 'name' was set as 'type'
        ('mem': its value), not as 'expected', and is ignored ('effect') */
    void warnAtCommit(std::string_view name,
        ANARIDataType type,
        const void *mem,
        const char *expected,
        const char *effect);

    FrameWrapperNotificationHandler m_onObjectDestroy;
    FrameWrapperStatusHandler m_reportStatus;

    // What the next commit reports about parameters set since the last one
    // (e.g. a wrong type), by parameter name: setting or unsetting the
    // parameter again replaces or drops it.
    struct Diagnostic
    {
      ANARIStatusSeverity severity;
      ANARIStatusCode code;
      std::string message;
    };
    std::map<std::string, Diagnostic, std::less<>> m_pendingDiagnostics;

    // whether the last render failed: nothing may be mapped until one
    // succeeds
    bool m_lastRenderFailed{false};
    // whether completeFrame() is calling the app's callback right now
    bool m_inCompletionCallback{false};

    // 'mpiCommunicator' (KHR_DATA_PARALLEL_MPI). The app passes a pointer to
    // an MPI_Comm as the ANARI_VOID_POINTER value, which works whether
    // MPI_Comm is an int (MPICH) or a pointer (Open MPI). It is read at the
    // next commit, so it only has to stay valid until that commit returns.
    // nullptr: unset, i.e. MPI_COMM_WORLD.
    const MPI_Comm *m_pendingComm{nullptr};
    // whether mpiCommunicator was set or unset since the last commit (true
    // at first, so the first commit creates the compositor)
    bool m_pendingCommChanged{true};
    // composites color and depth over m_comm; null while m_comm is
    // MPI_COMM_NULL
    std::unique_ptr<CPUCompositor> m_compositor;
    // The compositor's communicator: our MPI_Comm_dup of the committed one
    // (so the app may free its own), MPI_COMM_NULL before the first commit.
    MPI_Comm m_comm{MPI_COMM_NULL};
    // this rank in m_comm; rank 0 receives the composited image
    int m_rank{-1};
    // the compositor changed, so buffers must be reallocated at the next render
    bool m_reallocate{false};

    ANARIDevice m_ptcDevice{nullptr};
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
    // m_committed.idEnabled at the last render, likewise
    std::array<bool, NUM_ID_CHANNELS> m_renderedIdEnabled{};
    // composited ID_CHANNELS; empty unless enabled at the last render
    std::array<std::vector<uint32_t>, NUM_ID_CHANNELS> m_ids;
    // what stageIds() staged for compositeIds(): the enabled ID_CHANNELS
    // (indices) and one record per pixel
    std::vector<size_t> m_idChannels;
    std::vector<uint32_t> m_idRecords;

    // composited color and depth (rank 0); empty without a non-empty size
    float *m_depth = nullptr;
    void  *m_color = nullptr;
    // compositor color input when the frame has no channel.color
    std::vector<uint32_t> m_noColor;
  };


  // Helper functions ///////////////////////////////////////////////////////////

  /*! bytes per pixel of the composited color buffer; without a color type
      (ANARI_UNKNOWN) the compositor still works on 32-bit RGBA */
  inline size_t colorPixelSize(ANARIDataType colorType)
  {
    return colorType == ANARI_FLOAT32_VEC4 ? 4 * sizeof(float)
                                           : sizeof(uint32_t);
  }

  /*! whether the compositor handles this color type; ANARI_UNKNOWN (no
      color) counts, the compositor then runs on transparent black */
  inline bool isCompositableColorType(ANARIDataType colorType)
  {
    return colorType == ANARI_UNKNOWN || colorType == ANARI_UFIXED8_VEC4
        || colorType == ANARI_UFIXED8_RGBA_SRGB
        || colorType == ANARI_FLOAT32_VEC4;
  }

  /*! throws if T's size does not match the type's; callers check the type
      first, and PTCDevice turns any exception into an ERROR status */
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
