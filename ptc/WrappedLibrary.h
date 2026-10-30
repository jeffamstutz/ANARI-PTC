// SPDX-License-Identifier: Apache-2.0

#pragma once

// anari
#include <anari/anari.h>

namespace ptc {

// Load the library to wrap: $PTC_ANARI_LIBRARY, or 'helide' if that is unset
// or fails to load. 'usedFallback' tells whether 'helide' was the fallback.
ANARILibrary loadWrappedLibrary(ANARIStatusCallback statusCB,
    const void *statusCBUserPtr,
    bool *usedFallback = nullptr);

} // namespace ptc
