# ANARI_PTC (Pass-Through Compositing) Device

ANARI (https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html) is
a Khonos Group-maintained API for cross-platform rendering.  The repo
provides an implementation that allows ANARI to be used for
compositing-based data-parallel rendering, where multiple different
MPI ranks each render their local data, in a ANARI-constent way. 

For more details, plealse see

```
Standardized Data-Parallel Rendering Using ANARI
Ingo Wald, Stefan Zellmann, Jefferson Amstutz, Qi Wu, Kevin Griffin, Milan Jaros, Stefan Wesner
IEEE Symposium on Large Data Analysis and Visualization (LDAV), 2024
```
(free PDF version available here https://arxiv.org/abs/2407.00179)

# Dependencies

This project requires
- a built and `make install`ed `ANARI_SDK`
  (https://github.com/KhronosGroup/ANARI-SDK)
- cmake, c compiler, etc (which you already need for the ANARI SDK,
  anyway)
- oneTBB 2021.6 or newer
- MPI development packages (any MPI build will do)

Compositing runs on the CPU only: ptc composites the wrapped frame's
channels in host memory, whichever device it wraps.

# Building 

- Make sure to build and install the ANARI_SDK
- `mkdir buildDir`
- `cd buildDir`
- `cmake ..`

# Testing

Configure with `-DPTC_BUILD_TESTING=ON`, build, then run `ctest` in the
build directory. This runs an MPI smoke test (`tests/ptcSmoke.cpp`) on 1, 2
and 3 ranks with `helide` as the wrapped device. The tests are skipped (with
a configure-time warning) if `mpiexec` or `helide` cannot be found. With
Open MPI's `mpiexec`, the multi-rank tests pass `--oversubscribe`, so they
also run on machines with fewer cores than ranks.

# Installing

- just regular cmake `make install` ...

# Using ANARI_PTC

Assuming you have a data-parallel application that already has a ANARI
rendering path, just use this with `ANARI_LIBRARY` set to `ptc`, and choose
the device ptc wraps as below.

## Choosing the wrapped device

ptc renders each rank's data with another ANARI device (the *wrapped*
device) and composites the results. Which library it wraps is decided
when the device first needs it, in this order:

1. The device parameter `wrappedLibrary` (`ANARI_STRING`, a library name
   as for `anariLoadLibrary`), if the app set it. There is no fallback:
   if it fails to load, that is a fatal error.
2. The environment variable `PTC_ANARI_LIBRARY`, if set. If it fails to
   load, ptc warns and falls back to `helide`.
3. `helide`.

The device parameter `wrappedDeviceSubtype` (`ANARI_STRING`, default
`"default"`) selects the device subtype created from that library.

```cpp
anariSetParameter(device, device, "wrappedLibrary", ANARI_STRING, "visrtx");
anariCommitParameters(device, device);
```

Both parameters take effect only if they are committed before the first
object is created or the device is otherwise queried (`anariGetProperty`,
`anariGetObjectSubtypes`, ...): that is when ptc loads the wrapped device.
Changing them later is ignored with a warning. They are not forwarded to
the wrapped device; every other device parameter is.

If the wrapped library or device fails to load, ptc reports an
`ANARI_SEVERITY_FATAL_ERROR` once. After that, object creation returns
null, and every call that needs the wrapped device reports an error and
does nothing. ptc's own device properties (`extension`, `version`) still
answer.

The device property `version` (and `version.major`/`.minor`/`.patch`/
`.name`) is ptc's own. `anariVersion.*`, `geometryMaxIndex` and other
properties are the wrapped device's. The `extension` list is the wrapped
device's, plus `KHR_DATA_PARALLEL_MPI` and `KHR_FRAME_COMPLETION_CALLBACK`,
which ptc implements itself.

The library-level list from `anariGetDeviceExtensions(library, ...)` has
no device to read `wrappedLibrary` from, so it reflects only
`PTC_ANARI_LIBRARY` or `helide` (with the same warning on fallback), plus
ptc's own extensions. When a device wraps a library chosen by parameter,
query its `extension` property instead.

## Compositing over a communicator other than `MPI_COMM_WORLD`

ptc implements `KHR_DATA_PARALLEL_MPI`. By default a frame composites
over `MPI_COMM_WORLD`, and world rank 0 receives the image. To composite
over another communicator, set the frame's `mpiCommunicator` parameter
(type `ANARI_VOID_POINTER`) to a *pointer to* an `MPI_Comm`:

```cpp
MPI_Comm comm = ...; // e.g. from MPI_Comm_split
anariSetParameter(device, frame, "mpiCommunicator", ANARI_VOID_POINTER, &comm);
anariCommitParameters(device, frame);
```

Passing a pointer works the same whether `MPI_Comm` is an `int` (MPICH)
or a pointer (Open MPI). ptc reads it during the frame's commit, so the
`MPI_Comm` variable only has to stay valid until `anariCommitParameters`
returns. ptc keeps its own duplicate of the communicator, so the app may
free its communicator after the commit.

- Rank 0 *of that communicator* receives the composited image. On the
  other ranks the frame can still be mapped, but its contents are
  undefined.
- Committing a frame with a new communicator is collective: every rank of
  the new communicator, and of the one it replaces, must commit the frame
  together. A communicator with the same ranks in the same order as the
  current one is kept as it is.
- Unsetting `mpiCommunicator` goes back to `MPI_COMM_WORLD`.
- A pointer to `MPI_COMM_NULL` (e.g. on a rank that `MPI_Comm_split` left
  out) takes part in tearing down the frame's previous communicator, then
  reports an error; the frame cannot be rendered until another
  communicator is committed.
- Releasing a frame is collective over its communicator: every rank of it
  must release the frame.
- If rendering or preparing a frame fails on some ranks only (e.g. out of
  memory, or an unsupported color type), the ranks agree on that before
  compositing: every rank skips compositing and reports an error, instead
  of the others hanging. Committing a communicator agrees the same way on
  whether every rank created its compositor. A failure *inside* the
  compositor's communication can still leave the other ranks waiting.

## Diagnostics

ptc prints nothing: everything it has to say goes through the ANARI status
callback.

- A frame parameter ptc tracks (`size`, `channel.color`, `channel.depth`,
  the id channels, `mpiCommunicator`, `frameCompletionCallback` and its
  user data) set with the wrong type is ignored, as in helide, and the next
  commit warns about it once.
- A `channel.color` type ptc cannot composite (anything but `UFIXED8_VEC4`,
  `UFIXED8_RGBA_SRGB` and `FLOAT32_VEC4`) is rejected at commit with an
  ERROR and `ANARI_STATUS_INVALID_ARGUMENT`, once, at the commit after it
  was set. The frame then fails to render, with an ERROR each time, on
  every rank of its communicator, until a supported type is set or
  `channel.color` is unset.
- After a failed render, mapping any channel is refused with a warning that
  says so. The frame completion callback still runs, after the render's
  ERROR was reported.
- Rendering or committing a frame from inside its own frame completion
  callback is refused with an ERROR.

## MPI initialization and device lifetime

If the app has initialized MPI before a ptc device first needs it, ptc
uses it and never finalizes it. Otherwise ptc calls `MPI_Init` itself, and
keeps MPI initialized until the last ptc device is released, which then
calls `MPI_Finalize`. MPI cannot be initialized again after that, so an app
that creates ptc devices one after another should initialize MPI itself.

Releasing a device is collective: every rank must release its devices in
the same order. Release a device's frames before the device.

# Contributors

Though the project lives under my github namespace (and all files bear
my copyright headers), much of the latest version of the code is
actually entirely credit-due to Jeff Amstutz - who always takes
whatever duct-tape mess i've once written, burns it in a bonfire, and
rewrites it in a much cleaner and more reliable way.

# Related Projects

- ANARI-SDK (https://github.com/KhronosGroup/ANARI-SDK)
- barney (https://github.com/ingowald/barney)
- pynari (https://github.com/ingowald/pynari)
- VisRTX (https://github.com/NVIDIA/VisRTX)

# Version History

## v1.0.0

- first working version

## v1.1.0

- added CI build
- fixed various config issues for different cmake/linux/gcc versions
- now properly builds even on systems that do not have cuda installed
