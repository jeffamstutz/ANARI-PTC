// SPDX-License-Identifier: Apache-2.0

#pragma once

// What ptc adds on top of the wrapped device to report KHR_DATA_PARALLEL_MPI:
// the extension name in extension lists, and query info for the extension's
// one parameter (ANARI_FRAME 'mpiCommunicator').

// anari
#include <anari/anari.h>
// std
#include <map>
#include <string>
#include <vector>

namespace ptc {

// KHR_DATA_PARALLEL_MPI as spelled in device extension lists (the spelling
// anari_extension_utility.h matches)
constexpr const char *MPI_EXTENSION_NAME = "ANARI_KHR_DATA_PARALLEL_MPI";

// A wrapped extension list plus KHR_DATA_PARALLEL_MPI, without duplicates.
// Strings are copied, so the source list may go away afterwards.
struct ExtensionList
{
  explicit ExtensionList(const char *const *wrapped);

  // Null-terminated list, valid for the lifetime of this object
  const char **data();

 private:
  std::vector<std::string> m_names;
  std::vector<const char *> m_list;
};

// Load the library to wrap: $PTC_ANARI_LIBRARY, or 'helide' if that is unset
// or fails to load. 'usedFallback' tells whether 'helide' was the fallback.
ANARILibrary loadWrappedLibrary(ANARIStatusCallback statusCB,
    const void *statusCBUserPtr,
    bool *usedFallback = nullptr);

// Whether this is the (object, parameter) KHR_DATA_PARALLEL_MPI defines
bool isMpiCommunicatorParameter(
    ANARIDataType objectType, const char *name, ANARIDataType type);

// anariGetParameterInfo() result for ANARI_FRAME 'mpiCommunicator'
const void *mpiCommunicatorInfo(const char *infoName, ANARIDataType infoType);

// A wrapped device's frame parameter lists (anariGetObjectInfo 'parameter')
// with 'mpiCommunicator' added, cached per frame subtype
struct FrameParameterLists
{
  const ANARIParameter *get(
      const char *subtype, const ANARIParameter *wrapped);

 private:
  std::map<std::string, std::vector<ANARIParameter>> m_lists;
};

} // namespace ptc
