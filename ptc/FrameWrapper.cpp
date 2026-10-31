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
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

namespace ptc {

// Helper functions ///////////////////////////////////////////////////////////

namespace {

/*! index of 'name' in ID_CHANNELS, or -1 */
int idChannelIndex(std::string_view name)
{
  for (size_t i = 0; i < NUM_ID_CHANNELS; i++) {
    if (name == ID_CHANNELS[i])
      return int(i);
  }
  return -1;
}

// compositeIds() reduces one record of 32-bit words per pixel: the depth
// (as float bits), the rank, then the id of each enabled channel
constexpr size_t RECORD_DEPTH = 0;
constexpr size_t RECORD_RANK = 1;
constexpr size_t RECORD_FIRST_ID = 2;

float recordDepth(const uint32_t *record)
{
  float depth;
  std::memcpy(&depth, &record[RECORD_DEPTH], sizeof(depth));
  // NaN never wins over a hit
  return std::isnan(depth) ? INFINITY : depth;
}

/*! MPI_Op: keeps the nearer of each pair of records, the lower rank's at
    equal depth (as the compositors order equal-depth fragments), which
    makes it commutative */
void nearestRecord(void *in, void *inout, int *len, MPI_Datatype *type)
{
  int bytes = 0;
  MPI_Type_size(*type, &bytes);
  const size_t words = size_t(bytes) / sizeof(uint32_t);
  auto *a = (const uint32_t *)in;
  auto *b = (uint32_t *)inout;
  for (int i = 0; i < *len; i++, a += words, b += words) {
    const float da = recordDepth(a);
    const float db = recordDepth(b);
    if (da < db || (da == db && a[RECORD_RANK] < b[RECORD_RANK]))
      std::memcpy(b, a, bytes);
  }
}

/*! 'type', and the type it names if it is ANARI_DATA_TYPE, for messages */
std::string describeType(ANARIDataType type, const void *mem)
{
  std::string result = anari::toString(type);
  if (type == ANARI_DATA_TYPE && mem)
    result += std::string(" ") + anari::toString(*(const ANARIDataType *)mem);
  return result;
}

/*! whether a channel parameter set as 'type' (value 'mem') is an
    ANARI_DATA_TYPE naming 'want' (then 'on' says so) or ANARI_UNKNOWN */
bool isChannelType(
    ANARIDataType type, const void *mem, ANARIDataType want, bool &on)
{
  on = false;
  if (type != ANARI_DATA_TYPE)
    return false;
  const auto value = bit_cast<ANARIDataType>(type, mem);
  on = value == want;
  return on || value == ANARI_UNKNOWN;
}

} // namespace

// FrameWrapper definitions ///////////////////////////////////////////////////

FrameWrapper::FrameWrapper(ANARIDevice ptcDevice,
    ANARIDevice d,
    ANARIFrame f,
    FrameWrapperNotificationHandler onObjectDestroy,
    FrameWrapperStatusHandler reportStatus)
    : m_onObjectDestroy(onObjectDestroy),
      m_reportStatus(reportStatus),
      m_ptcDevice(ptcDevice),
      m_device(d),
      m_frame(f)
{
  anariRetain(m_device, m_device);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

FrameWrapper::~FrameWrapper()
{
  anariRelease(m_device, m_frame);
  anariRelease(m_device, m_device);
  if (m_color) free(m_color);
  if (m_depth) free(m_depth);
  // the subclass has destroyed its compositor by now
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (m_comm != MPI_COMM_NULL && !finalized)
    MPI_Comm_free(&m_comm);
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
  // frame (helium's getParam); the next commit warns about it.
  std::string_view name = _name;
  if (auto d = m_pendingDiagnostics.find(name); d != m_pendingDiagnostics.end())
    m_pendingDiagnostics.erase(d);

  if (name == "size") {
    const bool ok = type == ANARI_UINT32_VEC2;
    if (!ok) {
      warnAtCommit(
          name, type, mem, "ANARI_UINT32_VEC2", "treated as unset (0x0)");
    }
    m_pending.size =
        ok ? bit_cast<anari::math::uint2>(type, mem) : TrackedParameters{}.size;
  } else if (name == "channel.color") {
    if (type != ANARI_DATA_TYPE) {
      warnAtCommit(name, type, mem, "an ANARI_DATA_TYPE", "treated as unset");
      m_pending.colorType = TrackedParameters{}.colorType;
    } else {
      m_pending.colorType = bit_cast<ANARIDataType>(type, mem);
      // kept, so that the frame fails to render on this rank, and the
      // ranks agree to fail together
      if (!isCompositableColorType(m_pending.colorType)) {
        m_pendingDiagnostics[std::string(name)] = {ANARI_SEVERITY_ERROR,
            ANARI_STATUS_INVALID_ARGUMENT,
            std::string("cannot composite 'channel.color' of type ")
                + anari::toString(m_pending.colorType)
                + " (supported are UFIXED8_VEC4, UFIXED8_RGBA_SRGB and "
                  "FLOAT32_VEC4): the frame cannot be rendered until it is "
                  "set to one of them or unset"};
      }
    }
  } else if (name == "channel.depth") {
    // Never forwarded: the wrapped frame keeps rendering FLOAT32 depth for
    // compositing. Only FLOAT32 is a valid depth type (as in helide).
    bool on = false;
    if (!isChannelType(type, mem, ANARI_FLOAT32, on)) {
      warnAtCommit(name,
          type,
          mem,
          "ANARI_DATA_TYPE ANARI_FLOAT32",
          "depth is not composited");
    }
    m_pending.depthEnabled = on;
    return;
  } else if (name == "frameCompletionCallback") {
    // Never forwarded: renderFrame() calls it after compositing.
    const bool ok = type == ANARI_FRAME_COMPLETION_CALLBACK;
    if (!ok) {
      warnAtCommit(name,
          type,
          mem,
          "ANARI_FRAME_COMPLETION_CALLBACK",
          "treated as unset");
    }
    m_pending.completionCallback = ok
        ? bit_cast<ANARIFrameCompletionCallback>(type, mem)
        : TrackedParameters{}.completionCallback;
    return;
  } else if (name == "frameCompletionCallbackUserData") {
    // ANARI_VOID_POINTER parameters pass the pointer value itself as `mem`.
    const bool ok = type == ANARI_VOID_POINTER;
    if (!ok) {
      warnAtCommit(name,
          type,
          mem,
          "ANARI_VOID_POINTER",
          "treated as unset (null)");
    }
    m_pending.completionCallbackUserData =
        ok ? mem : TrackedParameters{}.completionCallbackUserData;
    return;
  } else if (name == "mpiCommunicator") {
    // Never forwarded: ptc implements KHR_DATA_PARALLEL_MPI. The value is a
    // pointer to the app's MPI_Comm, dereferenced at the next commit; a null
    // pointer or a wrong type reads as unset (MPI_COMM_WORLD).
    if (type != ANARI_VOID_POINTER) {
      // falling back to MPI_COMM_WORLD on some ranks only would hang
      warnAtCommit(name,
          type,
          mem,
          "ANARI_VOID_POINTER (a pointer to an MPI_Comm)",
          "MPI_COMM_WORLD is used");
    }
    m_pendingComm =
        type == ANARI_VOID_POINTER ? (const MPI_Comm *)mem : nullptr;
    m_pendingCommChanged = true;
    return;
  } else if (const int i = idChannelIndex(name); i >= 0) {
    // forwarded below: the wrapped frame renders the ids ptc composites
    bool on = false;
    if (!isChannelType(type, mem, ANARI_UINT32, on)) {
      warnAtCommit(name,
          type,
          mem,
          "ANARI_DATA_TYPE ANARI_UINT32",
          "the channel is not composited");
    }
    m_pending.idEnabled[i] = on;
  }

  anariSetParameter(m_device, m_frame, _name, type, mem);
}

void FrameWrapper::warnAtCommit(std::string_view name,
    ANARIDataType type,
    const void *mem,
    const char *expected,
    const char *effect)
{
  m_pendingDiagnostics[std::string(name)] = {ANARI_SEVERITY_WARNING,
      ANARI_STATUS_INVALID_ARGUMENT,
      "frame parameter '" + std::string(name) + "' was set as "
          + describeType(type, mem) + ", but must be " + expected + "; "
          + effect};
}

void FrameWrapper::reportStatus(
    ANARIStatusSeverity severity, const std::string &msg) const
{
  m_reportStatus(severity, defaultStatusCode(severity), msg.c_str());
}

void FrameWrapper::unsetParameter(const char *_name)
{
  std::string_view name = _name;
  if (auto d = m_pendingDiagnostics.find(name); d != m_pendingDiagnostics.end())
    m_pendingDiagnostics.erase(d);
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
  } else if (name == "mpiCommunicator") {
    m_pendingComm = nullptr;
    m_pendingCommChanged = true;
    return;
  } else if (const int i = idChannelIndex(name); i >= 0)
    m_pending.idEnabled[i] = false;
  anariUnsetParameter(m_device, m_frame, _name);
}

void FrameWrapper::unsetAllParameters()
{
  m_pending = {};
  m_pendingDiagnostics.clear();
  m_pendingComm = nullptr;
  m_pendingCommChanged = true;
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
  refuseFromCallback("commit");
  // what was set wrong since the last commit, reported once
  for (const auto &[name, d] : m_pendingDiagnostics)
    m_reportStatus(d.severity, d.code, d.message.c_str());
  m_pendingDiagnostics.clear();

  m_committed = m_pending;
  anariCommitParameters(m_device, m_frame);

  if (!m_pendingCommChanged)
    return;
  // The app's MPI_Comm only has to stay valid until this commit returns.
  const MPI_Comm comm = m_pendingComm ? *m_pendingComm : MPI_COMM_WORLD;
  m_pendingComm = nullptr;
  m_pendingCommChanged = false;
  setCommunicator(comm);
  if (comm == MPI_COMM_NULL) {
    // e.g. a rank left out of an MPI_Comm_split; it still took part in
    // tearing down the old communicator above, with the ranks that stayed
    throw std::runtime_error(
        "'mpiCommunicator' points to MPI_COMM_NULL: the frame cannot be "
        "rendered until another communicator is committed");
  }
}

void FrameWrapper::setCommunicator(MPI_Comm comm)
{
  if (m_comm != MPI_COMM_NULL) {
    int result = MPI_UNEQUAL;
    if (comm != MPI_COMM_NULL)
      MPI_Comm_compare(m_comm, comm, &result);
    // same ranks in the same order: the compositor works as it is
    if (result == MPI_IDENT || result == MPI_CONGRUENT)
      return;
    // every rank of the old communicator commits a new one in lockstep, so
    // they all get here and free it together
    cleanup();
    destroyCompositor();
    MPI_Comm_free(&m_comm);
    m_rank = -1;
    // the buffers stay mappable until the next render reallocates them
    m_reallocate = true;
  }
  if (comm == MPI_COMM_NULL)
    return;

  MPI_Comm dup = MPI_COMM_NULL;
  MPI_Comm_dup(comm, &dup);
  std::exception_ptr failure;
  try {
    createCompositor(dup);
  } catch (...) {
    failure = std::current_exception();
  }
  // If some ranks have no compositor, none may use this communicator: the
  // others would wait for them in every render.
  if (!agree(dup, !failure)) {
    destroyCompositor();
    MPI_Comm_free(&dup);
    if (failure)
      std::rethrow_exception(failure);
    throw std::runtime_error(
        "another rank failed to create its compositor for this "
        "'mpiCommunicator': the frame cannot be rendered until a "
        "communicator is committed again");
  }
  m_comm = dup;
  MPI_Comm_rank(m_comm, &m_rank);
}

bool FrameWrapper::agree(MPI_Comm comm, bool ok)
{
  int mine = ok ? 1 : 0;
  int all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
  return all != 0;
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

  if (m_lastRenderFailed) {
    // e.g. channel.color rejected at commit: say so, not that it is unset
    reportStatus(ANARI_SEVERITY_WARNING,
        "cannot map '" + std::string(channel)
            + "': the frame's last render failed");
  } else if (channel == "channel.color") {
    if (m_bufferColorType != ANARI_UNKNOWN && m_color) {
      *pixelType = m_bufferColorType;
      return m_color;
    }
    reportStatus(ANARI_SEVERITY_WARNING,
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
    reportStatus(ANARI_SEVERITY_WARNING,
        m_renderedDepthEnabled
            ? "cannot map 'channel.depth': the frame has not been rendered "
              "with a non-empty 'size'"
            : "cannot map 'channel.depth': the frame was rendered without "
              "its 'channel.depth' parameter set to ANARI_FLOAT32");
  } else if (const int i = idChannelIndex(channel); i >= 0) {
    if (m_renderedIdEnabled[i] && !m_ids[i].empty()) {
      *pixelType = ANARI_UINT32;
      return m_ids[i].data();
    }
    const std::string msg = "cannot map '" + std::string(channel) + "': "
        + (m_renderedIdEnabled[i]
                ? "the frame has not been rendered with a non-empty 'size'"
                : "the frame was rendered without its '" + std::string(channel)
                    + "' parameter set to ANARI_UINT32");
    reportStatus(ANARI_SEVERITY_WARNING, msg);
  } else {
    // e.g. channel.normal and channel.albedo: they can't be blended, and
    // ptc doesn't composite them
    const std::string msg = "cannot map '" + std::string(channel)
        + "': ptc composites only channel.color, channel.depth, "
          "channel.primitiveId, channel.objectId and channel.instanceId";
    reportStatus(ANARI_SEVERITY_WARNING, msg);
  }

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
  if (size == m_bufferSize && colorType == m_bufferColorType && !m_reallocate)
    return;

  freeBuffers();
  m_reallocate = false;
  if (!isCompositableColorType(colorType)) {
    // rejected at commit already; renderFrame() has every rank skip
    // compositing if any rank throws
    throw std::runtime_error(std::string("cannot render: 'channel.color' "
                                         "of type ")
        + anari::toString(colorType)
        + " was rejected at commit (supported are UFIXED8_VEC4, "
          "UFIXED8_RGBA_SRGB and FLOAT32_VEC4)");
  }
  if (m_comm == MPI_COMM_NULL && size.x != 0 && size.y != 0) {
    throw std::runtime_error("cannot composite: the frame has no valid "
                             "'mpiCommunicator' committed");
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
  for (auto &ids : m_ids)
    ids = {};
  m_idChannels.clear();
  m_idRecords = {};
  m_bufferSize = {0u, 0u};
  m_bufferColorType = ANARI_UNKNOWN;
}

void FrameWrapper::renderFrame()
{
  // until this render succeeds
  m_lastRenderFailed = true;

  // Every rank sees the same commits, so all of them skip or composite.
  const auto size = m_committed.size;
  const bool compositing =
      m_comm != MPI_COMM_NULL && size.x != 0 && size.y != 0;

  // Everything up to compositing is local to this rank, and may fail on it
  // alone (an unsupported color type, out of memory, a CUDA error).
  std::exception_ptr failure;
  const void *color = nullptr;
  const float *depth = nullptr;
  try {
    updateBuffers();
    m_renderedDepthEnabled = m_committed.depthEnabled;
    m_renderedIdEnabled = m_committed.idEnabled;
    anariRenderFrame(m_device, m_frame);
    anariFrameReady(m_device, m_frame, ANARI_WAIT);
    if (compositing) {
      depth = (const float *)mapWrapped("channel.depth", ANARI_FLOAT32);
      if (m_bufferColorType != ANARI_UNKNOWN)
        color = mapWrapped("channel.color", m_bufferColorType);
      stage(color, depth);
      stageIds(depth);
    }
  } catch (...) {
    if (!compositing)
      throw;
    failure = std::current_exception();
  }

  if (compositing) {
    auto unmap = [&] {
      if (depth)
        anariUnmapFrame(m_device, m_frame, "channel.depth");
      if (color)
        anariUnmapFrame(m_device, m_frame, "channel.color");
    };
    // Before the first collective, so that if any rank failed, all of them
    // skip compositing instead of the others hanging in it.
    const bool allOk = agree(m_comm, !failure);
    try {
      if (allOk) {
        composite();
        compositeIds();
      }
    } catch (...) {
      unmap();
      throw;
    }
    unmap();
    if (!allOk) {
      // nothing was composited, so there is nothing to map
      freeBuffers();
      if (failure)
        std::rethrow_exception(failure);
      throw std::runtime_error("another rank failed to render or prepare "
                               "this frame for compositing, so no rank "
                               "composited it");
    }
  }
  m_lastRenderFailed = false;
}

void FrameWrapper::completeFrame()
{
  // The composited frame is ready (rendering is synchronous), so the
  // continuation has run before any anariFrameReady() returns.
  auto callback = m_committed.completionCallback;
  if (!callback)
    return;
  // the app may release its last reference to the frame in there
  helium::IntrusivePtr<FrameWrapper> self(this);
  struct InCallback
  {
    bool &flag;
    InCallback(bool &f) : flag(f) { flag = true; }
    ~InCallback() { flag = false; }
  } inCallback{m_inCompletionCallback};
  callback(
      m_committed.completionCallbackUserData, m_ptcDevice, (ANARIFrame)this);
}

void FrameWrapper::refuseFromCallback(const char *what) const
{
  if (m_inCompletionCallback) {
    throw StatusError(ANARI_STATUS_INVALID_OPERATION,
        std::string("cannot ") + what
            + " a frame from inside its own frame completion callback");
  }
}

const void *FrameWrapper::mapWrapped(const char *channel, ANARIDataType type)
{
  uint32_t width = 0, height = 0;
  ANARIDataType mappedType = ANARI_UNKNOWN;
  const void *data =
      anariMapFrame(m_device, m_frame, channel, &width, &height, &mappedType);
  if (data && mappedType == type && width == m_bufferSize.x
      && height == m_bufferSize.y)
    return data;
  if (data)
    anariUnmapFrame(m_device, m_frame, channel);
  throw std::runtime_error(std::string("the wrapped frame did not render '")
      + channel + "' as " + anari::toString(type) + " of the frame's size");
}

void FrameWrapper::stageIds(const float *depth)
{
  m_idChannels.clear();
  m_idRecords = {};
  for (size_t i = 0; i < NUM_ID_CHANNELS; i++) {
    if (m_renderedIdEnabled[i])
      m_idChannels.push_back(i);
    else
      m_ids[i] = {};
  }
  const auto &channels = m_idChannels;
  if (channels.empty())
    return;

  // the same on every rank
  const size_t numPixels = size_t(m_bufferSize.x) * m_bufferSize.y;
  if (numPixels > size_t(INT_MAX)) {
    throw std::runtime_error(
        "cannot composite id channels: the frame has too many pixels");
  }
  const size_t words = RECORD_FIRST_ID + channels.size();
  auto &records = m_idRecords;
  records.resize(numPixels * words);

  for (size_t p = 0; p < numPixels; p++) {
    uint32_t *record = records.data() + p * words;
    std::memcpy(&record[RECORD_DEPTH], &depth[p], sizeof(float));
    record[RECORD_RANK] = uint32_t(m_rank);
  }

  uint32_t width = 0, height = 0;
  ANARIDataType type = ANARI_UNKNOWN;
  for (size_t c = 0; c < channels.size(); c++) {
    const char *name = ID_CHANNELS[channels[c]];
    auto *ids = (const uint32_t *)anariMapFrame(
        m_device, m_frame, name, &width, &height, &type);
    // still take part in the reduction, with no hits ('~0u')
    const bool valid = ids && type == ANARI_UINT32
        && size_t(width) * height == numPixels;
    if (!valid) {
      const std::string msg = std::string("the wrapped frame did not render '")
          + name + "' as ANARI_UINT32; compositing it as no hits";
      reportStatus(ANARI_SEVERITY_WARNING, msg);
    }
    for (size_t p = 0; p < numPixels; p++)
      records[p * words + RECORD_FIRST_ID + c] = valid ? ids[p] : ~0u;
    anariUnmapFrame(m_device, m_frame, name);
  }
}

void FrameWrapper::compositeIds()
{
  const auto &channels = m_idChannels;
  if (channels.empty())
    return;
  const size_t numPixels = size_t(m_bufferSize.x) * m_bufferSize.y;
  const size_t words = RECORD_FIRST_ID + channels.size();
  auto &records = m_idRecords;

  MPI_Datatype recordType;
  MPI_Type_contiguous(int(words), MPI_UINT32_T, &recordType);
  MPI_Type_commit(&recordType);
  MPI_Op op;
  MPI_Op_create(&nearestRecord, 1, &op);
  // on rank 0, in place: 'records' becomes the nearest record of every rank
  MPI_Reduce(m_rank == 0 ? MPI_IN_PLACE : records.data(),
      m_rank == 0 ? records.data() : nullptr,
      int(numPixels),
      recordType,
      op,
      0,
      m_comm);
  MPI_Op_free(&op);
  MPI_Type_free(&recordType);

  // on other ranks, this rank's own ids (the contents are undefined there)
  for (size_t c = 0; c < channels.size(); c++) {
    auto &ids = m_ids[channels[c]];
    ids.resize(numPixels);
    for (size_t p = 0; p < numPixels; p++)
      ids[p] = records[p * words + RECORD_FIRST_ID + c];
  }
  records = {};
}

} // namespace ptc
