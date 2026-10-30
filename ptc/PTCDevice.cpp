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

#include "PTCDevice.h"
// helium
#include <helium/BaseGlobalDeviceState.h>
#include <helium/BaseObject.h>
// std
#include <algorithm>
#include <cstring>
#include <memory>
#if PTC_CUDA
#include "CUDAFrameWrapper.h"
#else
#include "CPUFrameWrapper.h"
#endif
#include "WrappedLibrary.h"

namespace ptc {

int PTCDevice::s_numMpiPTCDevices = 0;
bool PTCDevice::s_mpiInitializedPrivately = false;

// Helper functions ///////////////////////////////////////////////////////////

template <typename T, typename U>
static bool sameAddress(const T *p1, const U *p2)
{
  return (const void *)p1 == (const void *)p2;
}

static FrameWrapper *asFrameWrapper(ANARIObject o)
{
  return reinterpret_cast<FrameWrapper *>(o);
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D PTCDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  return guarded("anariNewArray1D", ANARIArray1D(nullptr), [&] {
    initDevice();
    return m_ptd->newArray1D(appMemory, deleter, userData, type, numItems);
  });
}

ANARIArray2D PTCDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  return guarded("anariNewArray2D", ANARIArray2D(nullptr), [&] {
    initDevice();
    return m_ptd->newArray2D(
        appMemory, deleter, userData, type, numItems1, numItems2);
  });
}

ANARIArray3D PTCDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  return guarded("anariNewArray3D", ANARIArray3D(nullptr), [&] {
    initDevice();
    return m_ptd->newArray3D(
        appMemory, deleter, userData, type, numItems1, numItems2, numItems3);
  });
}

ANARICamera PTCDevice::newCamera(const char *subtype)
{
  return guarded("anariNewCamera", ANARICamera(nullptr), [&] {
    initDevice();
    return m_ptd->newCamera(subtype);
  });
}

ANARIFrame PTCDevice::newFrame()
{
#if PTC_CUDA
  typedef CUDAFrameWrapper FrameWrapperT;
#else
  typedef CPUFrameWrapper FrameWrapperT;
#endif
  return guarded("anariNewFrame", ANARIFrame(nullptr), [&] {
    initDevice();
    auto *fw = new FrameWrapperT(this_device(),
        (ANARIDevice)m_ptd,
        m_ptd->newFrame(),
        [=](const void *f) { this->removeFrameWrapper((FrameWrapper *)f); },
        [=](ANARIStatusSeverity severity, const char *msg) {
          this->reportMessage(severity, "%s", msg);
        });
    m_frameWrappers.push_back(fw);
    return (ANARIFrame)fw;
  });
}

ANARIGeometry PTCDevice::newGeometry(const char *subtype)
{
  return guarded("anariNewGeometry", ANARIGeometry(nullptr), [&] {
    initDevice();
    return m_ptd->newGeometry(subtype);
  });
}

ANARIGroup PTCDevice::newGroup()
{
  return guarded("anariNewGroup", ANARIGroup(nullptr), [&] {
    initDevice();
    return m_ptd->newGroup();
  });
}

ANARIInstance PTCDevice::newInstance(const char *subtype)
{
  return guarded("anariNewInstance", ANARIInstance(nullptr), [&] {
    initDevice();
    return m_ptd->newInstance(subtype);
  });
}

ANARILight PTCDevice::newLight(const char *subtype)
{
  return guarded("anariNewLight", ANARILight(nullptr), [&] {
    initDevice();
    return m_ptd->newLight(subtype);
  });
}

ANARIMaterial PTCDevice::newMaterial(const char *subtype)
{
  return guarded("anariNewMaterial", ANARIMaterial(nullptr), [&] {
    initDevice();
    return m_ptd->newMaterial(subtype);
  });
}

ANARIRenderer PTCDevice::newRenderer(const char *subtype)
{
  return guarded("anariNewRenderer", ANARIRenderer(nullptr), [&] {
    initDevice();
    return m_ptd->newRenderer(subtype);
  });
}

ANARISampler PTCDevice::newSampler(const char *subtype)
{
  return guarded("anariNewSampler", ANARISampler(nullptr), [&] {
    initDevice();
    return m_ptd->newSampler(subtype);
  });
}

ANARISpatialField PTCDevice::newSpatialField(const char *subtype)
{
  return guarded("anariNewSpatialField", ANARISpatialField(nullptr), [&] {
    initDevice();
    return m_ptd->newSpatialField(subtype);
  });
}

ANARISurface PTCDevice::newSurface()
{
  return guarded("anariNewSurface", ANARISurface(nullptr), [&] {
    initDevice();
    return m_ptd->newSurface();
  });
}

ANARIVolume PTCDevice::newVolume(const char *subtype)
{
  return guarded("anariNewVolume", ANARIVolume(nullptr), [&] {
    initDevice();
    return m_ptd->newVolume(subtype);
  });
}

ANARIWorld PTCDevice::newWorld()
{
  return guarded("anariNewWorld", ANARIWorld(nullptr), [&] {
    initDevice();
    return m_ptd->newWorld();
  });
}

// Object + Parameter Lifetime Management /////////////////////////////////////

void PTCDevice::setParameter(
    ANARIObject object, const char *name, ANARIDataType type, const void *mem)
{
  guarded("anariSetParameter", [&] {
    if (handleIsDevice(object))
      this->setParam(name, type, mem);
    else if (isFrameHandle(object))
      asFrameWrapper(object)->setParameter(name, type, mem);
    else
      m_ptd->setParameter(object, name, type, mem);
  });
}

void PTCDevice::unsetParameter(ANARIObject object, const char *name)
{
  guarded("anariUnsetParameter", [&] {
    if (handleIsDevice(object))
      this->removeParam(name);
    else if (isFrameHandle(object))
      asFrameWrapper(object)->unsetParameter(name);
    else
      m_ptd->unsetParameter(object, name);
  });
}

void PTCDevice::unsetAllParameters(ANARIObject object)
{
  guarded("anariUnsetAllParameters", [&] {
    if (handleIsDevice(object))
      this->removeAllParams();
    else if (isFrameHandle(object))
      asFrameWrapper(object)->unsetAllParameters();
    else
      m_ptd->unsetAllParameters(object);
  });
}

void *PTCDevice::mapParameterArray1D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  return guarded("anariMapParameterArray1D", (void *)nullptr, [&] {
    if (isFrameHandle(object)) {
      return asFrameWrapper(object)->mapParameterArray1D(
          name, dataType, numElements1, elementStride);
    } else {
      return m_ptd->mapParameterArray1D(
          object, name, dataType, numElements1, elementStride);
    }
  });
}

void *PTCDevice::mapParameterArray2D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  return guarded("anariMapParameterArray2D", (void *)nullptr, [&] {
    if (isFrameHandle(object)) {
      return asFrameWrapper(object)->mapParameterArray2D(
          name, dataType, numElements1, numElements2, elementStride);
    } else {
      return m_ptd->mapParameterArray2D(
          object, name, dataType, numElements1, numElements2, elementStride);
    }
  });
}

void *PTCDevice::mapParameterArray3D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  return guarded("anariMapParameterArray3D", (void *)nullptr, [&] {
    if (isFrameHandle(object)) {
      return asFrameWrapper(object)->mapParameterArray3D(name,
          dataType,
          numElements1,
          numElements2,
          numElements3,
          elementStride);
    } else {
      return m_ptd->mapParameterArray3D(object,
          name,
          dataType,
          numElements1,
          numElements2,
          numElements3,
          elementStride);
    }
  });
}

void PTCDevice::unmapParameterArray(ANARIObject object, const char *name)
{
  guarded("anariUnmapParameterArray", [&] {
    if (isFrameHandle(object))
      asFrameWrapper(object)->unmapParameterArray(name);
    else
      m_ptd->unmapParameterArray(object, name);
  });
}

void PTCDevice::commitParameters(ANARIObject object)
{
  guarded("anariCommitParameters", [&] {
    if (handleIsDevice(object))
      deviceCommitParameters();
    else if (isFrameHandle(object))
      asFrameWrapper(object)->commitParameters();
    else
      m_ptd->commitParameters(object);
  });
}

void PTCDevice::release(ANARIObject o)
{
  guarded("anariRelease", [&] {
    if (o == nullptr)
      return;

    if (handleIsDevice(o)) {
      if (--m_refCount == 0)
        delete this;
      return;
    } else if (isFrameHandle(o))
      asFrameWrapper(o)->release();
    else
      m_ptd->release(o);
  });
}

void PTCDevice::retain(ANARIObject o)
{
  guarded("anariRetain", [&] {
    if (handleIsDevice(o))
      m_refCount++;
    else if (isFrameHandle(o))
      asFrameWrapper(o)->retain();
    else
      m_ptd->retain(o);
  });
}

void *PTCDevice::mapArray(ANARIArray a)
{
  return guarded("anariMapArray", (void *)nullptr, [&] {
    return m_ptd->mapArray(a);
  });
}

void PTCDevice::unmapArray(ANARIArray a)
{
  guarded("anariUnmapArray", [&] {
    m_ptd->unmapArray(a);
  });
}

int PTCDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  return guarded("anariGetProperty", 0, [&] {
    if (handleIsDevice(object))
      return deviceGetProperty(name, type, mem, size);
    else if (isFrameHandle(object))
      return asFrameWrapper(object)->getProperty(name, type, mem, size, mask);
    else
      return m_ptd->getProperty(object, name, type, mem, size, mask);
  });
}

// Query functions ////////////////////////////////////////////////////////////

const char **PTCDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return guarded("anariGetObjectSubtypes", (const char **)nullptr, [&] {
    initDevice();
    return m_ptd->getObjectSubtypes(objectType);
  });
}

const void *PTCDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return guarded("anariGetObjectInfo", (const void *)nullptr, [&]() -> const void * {
    initDevice();
    auto *info =
        m_ptd->getObjectInfo(objectType, objectSubtype, infoName, infoType);
    if (objectType == ANARI_FRAME && infoName
        && std::strcmp(infoName, "parameter") == 0
        && infoType == ANARI_PARAMETER_LIST) {
      return m_frameParameterLists.withMpiCommunicator(
          objectSubtype, (const ANARIParameter *)info);
    }
    return info;
  });
}

const void *PTCDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return guarded("anariGetParameterInfo", (const void *)nullptr, [&] {
    if (isMpiCommunicatorParameter(objectType, parameterName, parameterType))
      return mpiCommunicatorInfo(infoName, infoType);

    initDevice();
    return m_ptd->getParameterInfo(objectType,
        objectSubtype,
        parameterName,
        parameterType,
        infoName,
        infoType);
  });
}

// FrameBuffer Manipulation ///////////////////////////////////////////////////

const void *PTCDevice::frameBufferMap(ANARIFrame f,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  return guarded("anariMapFrame", (const void *)nullptr, [&] {
    return asFrameWrapper(f)->frameBufferMap(channel, width, height, pixelType);
  });
}

void PTCDevice::frameBufferUnmap(ANARIFrame f, const char *channel)
{
  guarded("anariUnmapFrame", [&] {
    asFrameWrapper(f)->frameBufferUnmap(channel);
  });
}

// Frame Rendering ////////////////////////////////////////////////////////////

void PTCDevice::renderFrame(ANARIFrame f)
{
  guarded("anariRenderFrame", [&] {
    asFrameWrapper(f)->renderFrame();
  });
}

int PTCDevice::frameReady(ANARIFrame f, ANARIWaitMask m)
{
  return guarded("anariFrameReady", 0, [&] {
    return asFrameWrapper(f)->frameReady(m);
  });
}

void PTCDevice::discardFrame(ANARIFrame f)
{
  guarded("anariDiscardFrame", [&] {
    asFrameWrapper(f)->discardFrame();
  });
}

// Other PTCDevice definitions ////////////////////////////////////////////////

void PTCDevice::reportException(const char *entryPoint) const noexcept
{
  try {
    try {
      throw;
    } catch (const std::exception &e) {
      reportMessage(ANARI_SEVERITY_ERROR, "%s: %s", entryPoint, e.what());
    } catch (...) {
      reportMessage(
          ANARI_SEVERITY_ERROR, "%s: unknown exception", entryPoint);
    }
  } catch (...) {
    // reporting failed as well (out of memory, or a throwing status
    // callback); there is nothing left to tell the app with
  }
}

PTCDevice::PTCDevice(ANARIStatusCallback defaultCallback, const void *userPtr)
{
  anari::DeviceImpl::m_defaultStatusCB = defaultCallback;
  anari::DeviceImpl::m_defaultStatusCBUserPtr = userPtr;
}

PTCDevice::PTCDevice(ANARILibrary l) : DeviceImpl(l) {}

PTCDevice::~PTCDevice()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying PTC device (%p)", this);

  if (m_ptd)
    m_ptd->release(m_ptd->this_device());

  if (s_mpiInitializedPrivately)
    s_numMpiPTCDevices--;
  if (s_numMpiPTCDevices == 0 && s_mpiInitializedPrivately) {
    reportMessage(ANARI_SEVERITY_DEBUG, "finalizing MPI");
    MPI_Finalize();
  }
}

void PTCDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing PTC device (%p)", this);

  bool usedFallback = false;
  ANARILibrary lib = loadWrappedLibrary(
      defaultStatusCallback(), defaultStatusCallbackUserPtr(), &usedFallback);
  if (usedFallback)
    reportMessage(ANARI_SEVERITY_WARNING, "falling back to 'helide'");

  if (!lib) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to load passthrough lib");
    return;
  }

  m_ptd = (anari::DeviceImpl *)anariNewDevice(lib, "default");

  if (!m_ptd)
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to create device");

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (s_numMpiPTCDevices == 0 && !mpiInitialized) {
    reportMessage(ANARI_SEVERITY_DEBUG, "initializing MPI");
    MPI_Init(nullptr, nullptr);
    s_mpiInitializedPrivately = true;
    s_numMpiPTCDevices++;
  } else {
    reportMessage(ANARI_SEVERITY_DEBUG, "MPI already initialized");
  }

  m_initialized = true;
}

void PTCDevice::deviceCommitParameters()
{
  initDevice();
  auto handle = (ANARIDevice)m_ptd;
  m_ptd->unsetAllParameters(handle);
  for (auto param = params_begin(); param != params_end(); ++param) {
    m_ptd->setParameter(handle,
        param->first.c_str(),
        param->second.type(),
        param->second.data());
  }
  m_ptd->commitParameters(handle);
}

// Under KHR_DATA_PARALLEL_MPI anariGetProperty is collective, but device
// properties are the same on every rank, so no communication is needed.
int PTCDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  initDevice();

  if (name && std::strcmp(name, "extension") == 0
      && type == ANARI_STRING_LIST) {
    std::lock_guard<std::mutex> lock(m_extensionsMutex);
    if (!m_extensions) {
      const char *const *wrapped = nullptr;
      if (m_ptd) {
        m_ptd->getProperty(m_ptd->this_device(),
            "extension",
            ANARI_STRING_LIST,
            &wrapped,
            sizeof(wrapped),
            ANARI_WAIT);
      }
      m_extensions = std::make_unique<ExtensionList>(wrapped);
    }
    helium::writeToVoidP(mem, m_extensions->data());
    return 1;
  }

  // Everything else (version, anariVersion.*, geometryMaxIndex, ...) is
  // whatever the wrapped device reports.
  if (!m_ptd)
    return 0;
  return m_ptd->getProperty(
      m_ptd->this_device(), name, type, mem, size, ANARI_WAIT);
}

bool PTCDevice::isFrameHandle(ANARIObject o)
{
  for (auto *f : m_frameWrappers) {
    if (sameAddress(f, o))
      return true;
  }
  return false;
}

void PTCDevice::removeFrameWrapper(FrameWrapper *fw)
{
  m_frameWrappers.erase(std::remove_if(m_frameWrappers.begin(),
                            m_frameWrappers.end(),
                            [&](auto *f) { return sameAddress(f, fw); }),
      m_frameWrappers.end());
}

} // namespace ptc
