// SPDX-License-Identifier: Apache-2.0

#include "WrappedLibrary.h"
// std
#include <cstdlib>

namespace ptc {

ANARILibrary loadWrappedLibrary(ANARIStatusCallback statusCB,
    const void *statusCBUserPtr,
    bool *usedFallback)
{
  ANARILibrary lib = nullptr;
  if (const char *libName = getenv("PTC_ANARI_LIBRARY"))
    lib = anariLoadLibrary(libName, statusCB, statusCBUserPtr);

  if (usedFallback)
    *usedFallback = !lib;
  if (!lib)
    lib = anariLoadLibrary("helide", statusCB, statusCBUserPtr);
  return lib;
}

} // namespace ptc
