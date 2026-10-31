// SPDX-License-Identifier: Apache-2.0

#include "WrappedLibrary.h"
// std
#include <cstdlib>

namespace ptc {

static const char *DEFAULT_LIBRARY = "helide";

WrappedLibrary loadWrappedLibrary(const std::string &fromParameter,
    ANARIStatusCallback statusCB,
    const void *statusCBUserPtr)
{
  WrappedLibrary w;
  auto load = [&](const std::string &name) {
    w.name = name;
    w.library = anariLoadLibrary(name.c_str(), statusCB, statusCBUserPtr);
    return w.library != nullptr;
  };

  if (!fromParameter.empty()) {
    if (!load(fromParameter)) {
      w.failure = "failed to load the wrapped ANARI library '" + fromParameter
          + "' (device parameter 'wrappedLibrary')";
    }
    return w;
  }

  const char *fromEnv = getenv("PTC_ANARI_LIBRARY");
  if (fromEnv && *fromEnv) {
    if (load(fromEnv))
      return w;
    w.fallbackWarning = std::string("failed to load the wrapped ANARI library '")
        + fromEnv + "' (PTC_ANARI_LIBRARY); falling back to '"
        + DEFAULT_LIBRARY + "'";
  }

  if (!load(DEFAULT_LIBRARY)) {
    w.failure = std::string("failed to load the wrapped ANARI library '")
        + DEFAULT_LIBRARY + "'"
        + (fromEnv && *fromEnv ? " (fallback for PTC_ANARI_LIBRARY)"
                               : " (the default)");
  }
  return w;
}

} // namespace ptc
