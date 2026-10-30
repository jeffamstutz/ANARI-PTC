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
- when buiding the *optional* (GPU-compositing version): `CUDA` (if CUDA
  is disabled or cannot be found the project will automatically fall
  back to CPU compositing)
- when building the CPU version: `tbb`
- MPI development packages. When built with CUDA support you'll need a
  CUDA-aware MPI build; for the cpu-only version any MPI build should
  do.

# Building 

- Make sure to build and install the ANARI_SDK
- `mkdir buildDir`
- `cd buildDir`
- `cmake ..` (for default build)
or
- `cmake -DPTC_DISABLE_CUDA .. (for explicitly cpu-only build

# Testing

Configure with `-DPTC_BUILD_TESTING=ON`, build, then run `ctest` in the
build directory. This runs an MPI smoke test (`tests/ptcSmoke.cpp`) on 1, 2
and 3 ranks with `helide` as the wrapped device. The tests are skipped (with
a configure-time warning) if `mpiexec` or `helide` cannot be found.

# Installing

- just regular cmake `make install` ...

# Using ANARI_PTC

Assuming you have a data-parallel application that already has a ANARI
rendering path, just use this with `ANARI_LIBRARY` set to `ptc`

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
