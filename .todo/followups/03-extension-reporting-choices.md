# Follow-ups from 03 (Device reports KHR_DATA_PARALLEL_MPI)

Commits reviewed: 26a3a20, 3bac8c8. The commits cover all six checkboxes:
- the merged `extension` list, deduplicated;
- `version`, `anariVersion.*` and `geometryMaxIndex`, forwarded;
- the library extension list;
- frame `mpiCommunicator` parameter info;
- `anariGetDeviceExtensionStruct`;
- the extension check on every rank.

`ptcSmoke` checks each of these on every rank. Nothing here blocks a queued
ticket. The items below are choices the implementer made alone, for you to
confirm.

## 1. Library-level extension query loads and unloads the wrapped library

severity: non-blocking (fold into ticket 12)

`PTCLibrary::getDeviceExtensions` now does the following:
1. calls `loadWrappedLibrary()`, which reads `$PTC_ANARI_LIBRARY` and falls
   back to helide;
2. reads that library's `"default"` device extensions;
3. calls `anariUnloadLibrary`;
4. caches the result for the life of the library object.

This has three consequences:
- **No reach to the ticket-12 parameter.** Ticket 12 adds a device parameter
  that selects the wrapped library. That parameter cannot affect this query,
  which has no device. So the library-level list will reflect only the env
  var or helide. The device-level list can differ from it once 12 lands.
  Ticket 12 should state whether that is acceptable, or whether the
  library-level list should report only `KHR_DATA_PARALLEL_MPI`.
- **Silent fallback.** The fallback to helide here is silent. The device
  path warns. Ticket 12's "fallback reported clearly" item should cover both
  paths.
- **Risky unload.** Unloading a wrapped library at query time is a
  dlopen/dlclose cycle that did not happen before. 01 hit a TBB-workers-
  outlive-dlclose crash. A wrapped device that starts threads at load time
  could hit the same class of problem here. Helide is fine today.

## 2. Device properties now trigger `initDevice()`, and so `MPI_Init`

severity: non-blocking (ticket 11 owns MPI lifetime)

`deviceGetProperty` now calls `initDevice()`. That loads the wrapped device
and, if the app has not initialized MPI, calls `MPI_Init` privately. This
follows the existing pattern: almost every other API entry point already
calls `initDevice()`. So it does not change the lifetime model, only how
early that model kicks in. One effect: an app that queries `extension` to
decide *whether* to use ptc now gets ptc's private `MPI_Init` as a side
effect. If it then releases the device, the existing finalize logic runs.
Ticket 11 should include "query-then-release" in its lifetime cases.

## 3. `anariGetProperty` is collective but does no communication

severity: non-blocking

The spec marks `anariGetProperty` collective under this extension. The
implementation assumes every rank reports identical properties, so it does
no MPI call. Two things could break that assumption:
- ranks wrapping different libraries or GPUs, which could give different
  `geometryMaxIndex` values;
- ticket 07's per-frame communicators.

Confirm the assumption, or ask for a reduction (for example, min of
`geometryMaxIndex`) before the ticket-16 README documents the semantics.

## 4. Identity properties are the wrapped device's

severity: non-blocking

`version`, `anariVersion.*` and every other non-`extension` property are
forwarded untouched. So ptc reports helide's `version`, and inherits the
wrapped device's ANARI version even if that is below 1.1. Decide whether
ptc should report its own `version` (and possibly a fixed `anariVersion`)
before the SDK port (16).

## 5. Spelling and param-info details

severity: non-blocking

- The extension list uses `ANARI_KHR_DATA_PARALLEL_MPI`, matching helide
  and `anari_extension_utility.h`.
- `sourceExtension` (STRING) is `KHR_DATA_PARALLEL_MPI`.
- The INT32 `sourceExtension` index query returns null.
- `description` text was written by the implementer.

These are consistent with the SDK. Listed only so you know they were
chosen.

## 6. Harness: `--require-mpi-extension` removed

severity: non-blocking

The extension check is now hard on every rank, and the XFAIL path is gone.
This resolves item 2 of followups/02. The CPU float32 tests still pass on a
`PASS_REGULAR_EXPRESSION` that ignores the exit code, so on those tests the
new conformance check is bypassed. That stays with ticket 08.

## 7. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `03-report-extension-and-properties.md`
  were not updated.
- On wrapped-library load failure (`m_ptd == nullptr`), `extension` still
  returns a list holding only the MPI extension. Graceful handling belongs
  to ticket 12.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** keep the library-level list and document it. Ticket 12 has a checkbox to warn on fallback on both paths and to document the scope.
- **Item 2:** the "query then release" test is added to ticket 11.
- **Item 3:** documented as an assumption (homogeneous ranks) in 16's README list.
- **Item 4:** ptc reports its own `version` and forwards `anariVersion.*`. The checkbox is on ticket 12.
- **Item 5:** confirmed as is.
- **Item 6:** resolved by 08.
- **Item 7:** covered by ticket 12.
