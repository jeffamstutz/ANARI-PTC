# Follow-ups from 12 (wrapped-device selection and load failure)

Commits reviewed: 9debfc4, 46fae0a. All eight checkboxes are met:
- **Device parameter.** `wrappedLibrary` (`ANARI_STRING`, a name as for
  `anariLoadLibrary`) chooses the wrapped library. The README documents it,
  including its precedence over `PTC_ANARI_LIBRARY`.
- **Fallback reported.** A `PTC_ANARI_LIBRARY` that fails to load falls back
  to helide with a WARNING, on the device path and on the library-level
  `getDeviceExtensions` path. The README says the library-level list
  reflects only `PTC_ANARI_LIBRARY` or helide.
- **No crash on a missing library.** Every `anariNew*` returns null and
  reports ERROR. The other entry points report ERROR on null handles. Frame
  entry points refuse handles that aren't this device's frames.
- **Load failure is FATAL_ERROR**, reported once per device.
- **`version` is ptc's own.** `anariVersion.*` is still forwarded.
- **`KHR_FRAME_COMPLETION_CALLBACK` is always listed**, at device and
  library level.
- **Tests.** `wrapped_library_parameter`, `ptc_wrapped_missing` and
  `ptc_wrapped_fallback` run on 1 to 3 ranks. The smoke conformance check
  covers `version` and the extension lists.

Both trees (CUDA and CPU) pass 130/130.

Nothing here blocks 13 or 14. Items 1 and 2 shape the wrapped-device
section of 16's README and the SDK PR, so decide them before 16. 16 is
already gated on 15, which needs you.

## 1. Precedence is the opposite of the ANARI debug device's

severity: non-blocking (decide before 16)

The implementer made the parameter win: `wrappedLibrary` first, then
`PTC_ANARI_LIBRARY`, then helide. The ticket only required the env var to
override the default, so this is consistent with it. But, per the
implementer, the SDK's debug device does it the other way round:
`ANARI_DEBUG_WRAPPED_LIBRARY` beats the `wrappedDevice` parameter. With env
first, a user can redirect an unmodified app from the environment. With the
parameter first, the app's choice is final. In the SDK (ticket 16), ptc
would sit next to the debug device and be inconsistent with it.

The names were also chosen alone. `wrappedLibrary` is a string, as the
ticket suggested. The debug device's `wrappedDevice` takes an
`ANARI_DEVICE` handle.

**Confirm** parameter-over-env and the names `wrappedLibrary` and
`wrappedDeviceSubtype`. Or switch to env-over-parameter to match the debug
device.

## 2. Fallback and severity policy differs by source

severity: non-blocking (decide before 16)

The implementer chose this policy alone:

| Source of the library name | If it fails to load |
|---|---|
| `wrappedLibrary` parameter | FATAL_ERROR, no fallback |
| `PTC_ANARI_LIBRARY` | WARNING, then helide (FATAL_ERROR if helide also fails) |
| none (default) | helide silently. The old code always warned "falling back to 'helide'" |
| library-level `getDeviceExtensions` | WARNING only, never FATAL. The list holds only ptc's own extensions |

The ticket left this open ("reported clearly, or removed, if a parameter
makes it unnecessary"). One alternative: treat a failed
`PTC_ANARI_LIBRARY` like a failed parameter, as FATAL with no fallback. A
user who sets the env var explicitly likely doesn't want helide to render
silently in its place, and a WARNING is easy to miss on many ranks.

**Confirm** the table, or make the env failure FATAL too, which removes the
helide fallback except as the unset default.

## 3. Any device commit locks the choice, not only object creation

severity: non-blocking

`deviceCommitParameters()` reads the two parameters, then calls
`initDevice()` on every device commit. So the *first* device commit
chooses the wrapped device. Suppose an app commits device parameters once,
for example to set a status callback, and later sets `wrappedLibrary` and
commits again. It gets env/helide plus a WARNING. The README says the
parameters must be "committed before the first object is created or the
device is otherwise queried". It doesn't say that an earlier device commit
also counts.

A device commit also now runs ptc's private `MPI_Init` (followups/11 item 1,
followups/03 item 2), as it already did before 12.

**Confirm** "the first device commit decides". If so, add it to the README
(and 16's). The alternative is to defer loading until the first object
creation or query, and have commits only record the names.

## 4. Scope extensions

severity: non-blocking

- **`wrappedDeviceSubtype`.** A second parameter, `ANARI_STRING` with
  default `"default"`, added because the ticket says "library/subtype". The
  branch where the library loads but `anariNewDevice(lib, subtype)` returns
  null is untested. helide and sink ignore the subtype, so no available
  library triggers it.
- **Status-code fix in `reportMessage`.** The device's comparison was
  inverted. ERROR and FATAL went out with `ANARI_STATUS_NO_ERROR`, and
  INFO and DEBUG with `ANARI_STATUS_UNKNOWN_ERROR`. Now ERROR and above carry
  `UNKNOWN_ERROR`, and lower severities `NO_ERROR`. This changes the status
  code of every device message. It is a correct fix, but it lands in 14's
  area. 14 will need a way to pass a specific code (`INVALID_ARGUMENT`)
  through `reportMessage`, since the code is now derived from the severity.
  Add a note to 14.
- **Frame-handle validation.** Map, unmap, render, `frameReady` and
  discard now refuse a handle that isn't one of this device's frames,
  reporting ERROR. Before, they cast it blindly. This is harmless, and
  listed only so you know.
- **Unloading.** The wrapped library is unloaded if device creation fails.
  It is still never unloaded on success, as before.

**Confirm** keeping `wrappedDeviceSubtype`, and add the `reportMessage`
status-code note to 14.

## 5. `version` encoding

severity: non-blocking (16)

`version` is `major*1000 + minor*100 + patch`, copied from helide. It
collides once minor or patch reaches 10. For example, 1.10.0 and 2.0.0 both
give 2000. The implementer also added `version.major`, `.minor`, `.patch`,
`.name` and `.name.size` from CMake's `ANARI_PTC_VERSION`, currently 1.1.0.
Once ptc is in the SDK (16), the natural source is the SDK's own version.

**Confirm** the encoding and the extra properties, or pick another scheme.
Also decide which version ptc reports after the port.

## 6. Coverage gaps

severity: non-blocking

- **Selection is barely tested.** The parameter test selects only `helide`
  over a missing `PTC_ANARI_LIBRARY`, so it proves that the parameter beats
  a failing env var. It doesn't prove that a different library gets
  wrapped. Selecting `sink` (or the debug device) by parameter, and checking
  something that differs from helide, would cover it.
- **Device-creation failure** is untested (item 4).

## 7. Asymmetric load failure across ranks hangs

severity: non-blocking (16 README)

Suppose the wrapped library loads on some ranks only, for example through
a per-node path difference. The failing ranks get null frames, while the
others enter the commit-time agreement or compositing collectives and hang.
This is the same class as followups/08 item 3 and followups/11 item 5, and
nothing agrees on load success. Either add it to 16's README list of
accepted limits, or have `initDevice()` agree on load success. That would
be a collective at first use, which has its own risk if ranks reach first
use at different calls.

## 8. Left open from earlier follow-ups

severity: non-blocking

- **followups/10 item 5** (normal/albedo extensions still advertised and
  their parameters forwarded) was assigned to "12, or 16". 12 did not
  touch it, so it now falls to 16 or needs a decision.
- **followups/11 item 1.** A device whose wrapped library failed to load
  still holds ptc's private MPI until it is released. This is expected
  under that item's contract.

## 9. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `12-wrapped-device-selection.md` were not
  updated. Earlier tickets had the same gap.
- `tests/ptcWrapped.cpp` is a third file duplicating the harness helpers
  (see followups/11 item 6). Share them before 16.
