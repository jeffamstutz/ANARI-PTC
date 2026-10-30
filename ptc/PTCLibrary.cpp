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

#include "MpiExtension.h"
#include "PTCDevice.h"
#include "anari/backend/LibraryImpl.h"
#include "anari_library_ptc_export.h"
// std
#include <memory>

namespace ptc {

struct PTCLibrary : public anari::LibraryImpl
{
  PTCLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;

 private:
  std::unique_ptr<ExtensionList> m_extensions;
};

// Definitions ////////////////////////////////////////////////////////////////

PTCLibrary::PTCLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice PTCLibrary::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new PTCDevice(this_library());
}

// The wrapped library's extensions for the device ptc wraps ('default'), plus
// KHR_DATA_PARALLEL_MPI. Every ptc device subtype is the same device.
const char **PTCLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  if (!m_extensions) {
    const char *const *wrapped = nullptr;
    ANARILibrary lib =
        loadWrappedLibrary(defaultStatusCB(), defaultStatusCBUserPtr());
    if (lib)
      wrapped = anariGetDeviceExtensions(lib, "default");
    m_extensions = std::make_unique<ExtensionList>(wrapped); // copies names
    if (lib)
      anariUnloadLibrary(lib);
  }
  return m_extensions->data();
}

} // namespace ptc

// Define library entrypoint //////////////////////////////////////////////////

extern "C" PTC_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    ptc, handle, scb, scbPtr)
{
  return (ANARILibrary) new ptc::PTCLibrary(handle, scb, scbPtr);
}
