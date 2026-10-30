// SPDX-License-Identifier: Apache-2.0
//
// Multi-rank smoke test for the ptc device. Rank r renders one sphere at a
// distinct x position with a distinct color through ptc (wrapping helide);
// rank 0 checks that every sphere shows up in the composited image.
//
// Usage: ptcSmoke [--color ufixed8|float32] [--depth]
//                 [--require-mpi-extension] [--invert-check]
//                 [--expect-library <path>]
//
//   --color                  frame color channel type (default ufixed8)
//   --depth                  also request and check channel.depth
//   --require-mpi-extension  fail if the device does not report
//                            KHR_DATA_PARALLEL_MPI (otherwise only reported
//                            as XFAIL; ticket 03 makes this the default)
//   --invert-check           expect the wrong colors, so the pixel check must
//                            fail; used to test the harness's failure path
//   --expect-library         fail unless the loaded libanari_library_ptc is
//                            this file (guards against picking up an
//                            installed copy instead of the one under test)
//
// Every rank exits non-zero if any rank recorded a failure.

#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
#include <mpi.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

using namespace anari::math;

namespace {

struct Options
{
  ANARIDataType colorType{ANARI_UFIXED8_RGBA_SRGB};
  bool depth{false};
  bool requireMpiExtension{false};
  bool invertCheck{false};
  std::string expectLibrary;
};

const float3 kColors[] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
const uint2 kImageSize(128, 64);
const float kCameraHeight = 1.5f;

void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity sev,
    ANARIStatusCode,
    const char *msg)
{
  if (sev <= ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[anari] %s\n", msg);
}

bool parseArgs(int argc, char **argv, Options &opts)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--color" && i + 1 < argc) {
      const std::string type = argv[++i];
      if (type == "ufixed8")
        opts.colorType = ANARI_UFIXED8_RGBA_SRGB;
      else if (type == "float32")
        opts.colorType = ANARI_FLOAT32_VEC4;
      else {
        fprintf(stderr, "unknown color type '%s'\n", type.c_str());
        return false;
      }
    } else if (arg == "--depth")
      opts.depth = true;
    else if (arg == "--require-mpi-extension")
      opts.requireMpiExtension = true;
    else if (arg == "--invert-check")
      opts.invertCheck = true;
    else if (arg == "--expect-library" && i + 1 < argc)
      opts.expectLibrary = argv[++i];
    else {
      fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
      return false;
    }
  }
  return true;
}

// x position of rank r's sphere, in world space
float sphereX(int r, int size)
{
  return size == 1 ? 0.f : -1.f + 2.f * r / (size - 1);
}

// pixel column of a world-space x under the orthographic camera
int pixelX(float x)
{
  const float aspect = float(kImageSize.x) / kImageSize.y;
  return int((x / (kCameraHeight * aspect) + 0.5f) * kImageSize.x);
}

std::string canonicalPath(const std::string &path)
{
  char buf[PATH_MAX];
  return realpath(path.c_str(), buf) ? std::string(buf) : path;
}

// Path of the libanari_library_ptc mapped into this process, or "" if none
// (or not on Linux).
std::string loadedPtcLibrary()
{
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    auto pos = line.find('/');
    if (pos != std::string::npos
        && line.find("libanari_library_ptc", pos) != std::string::npos)
      return canonicalPath(line.substr(pos));
  }
  return "";
}

bool deviceHasMpiExtension(anari::Device d)
{
  const char *const *exts = nullptr;
  if (!anariGetProperty(d,
          d,
          "extension",
          ANARI_STRING_LIST,
          &exts,
          sizeof(exts),
          ANARI_WAIT)
      || !exts)
    return false;
  for (auto e = exts; *e; e++) {
    if (std::string(*e) == "KHR_DATA_PARALLEL_MPI")
      return true;
  }
  return false;
}

anari::World buildWorld(anari::Device d, int rank, int size)
{
  float3 pos(sphereX(rank, size), 0.f, 0.f);
  float3 col = kColors[rank % 4];
  auto geom = anari::newObject<anari::Geometry>(d, "sphere");
  anari::setParameterArray1D(d, geom, "vertex.position", &pos, 1);
  anari::setParameter(d, geom, "radius", 0.3f);
  anari::commitParameters(d, geom);
  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", col);
  anari::commitParameters(d, mat);
  auto surf = anari::newObject<anari::Surface>(d);
  anari::setAndReleaseParameter(d, surf, "geometry", geom);
  anari::setAndReleaseParameter(d, surf, "material", mat);
  anari::commitParameters(d, surf);
  auto world = anari::newObject<anari::World>(d);
  anari::setParameterArray1D(d, world, "surface", &surf, 1);
  anari::release(d, surf);
  auto light = anari::newObject<anari::Light>(d, "directional");
  anari::setParameter(d, light, "direction", float3(0, 0, -1));
  anari::commitParameters(d, light);
  anari::setParameterArray1D(d, world, "light", &light, 1);
  anari::release(d, light);
  anari::commitParameters(d, world);
  return world;
}

anari::Frame buildFrame(anari::Device d, const Options &opts, anari::World w)
{
  auto cam = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, cam, "position", float3(0, 0, 5));
  anari::setParameter(d, cam, "direction", float3(0, 0, -1));
  anari::setParameter(d, cam, "up", float3(0, 1, 0));
  anari::setParameter(d, cam, "height", kCameraHeight);
  anari::setParameter(
      d, cam, "aspect", float(kImageSize.x) / kImageSize.y);
  anari::commitParameters(d, cam);

  auto ren = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, ren, "background", float4(0, 0, 0, 0));
  anari::setParameter(d, ren, "ambientRadiance", 1.f);
  anari::commitParameters(d, ren);

  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", kImageSize);
  anari::setParameter(d, frame, "channel.color", opts.colorType);
  if (opts.depth)
    anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  anari::setAndReleaseParameter(d, frame, "world", w);
  anari::setAndReleaseParameter(d, frame, "camera", cam);
  anari::setAndReleaseParameter(d, frame, "renderer", ren);
  anari::commitParameters(d, frame);
  return frame;
}

// Rank 0 only: probe the pixel at the center of each rank's sphere.
int checkColor(anari::Device d, anari::Frame frame, const Options &opts, int size)
{
  int failures = 0;
  auto fb = anari::map<void>(d, frame, "channel.color");
  if (!fb.data || fb.width != kImageSize.x || fb.height != kImageSize.y
      || fb.pixelType != opts.colorType) {
    printf("FAIL: bad color map (data=%p %ux%u type=%d)\n",
        fb.data,
        fb.width,
        fb.height,
        int(fb.pixelType));
    failures++;
  } else {
    for (int r = 0; r < size; r++) {
      const int px = pixelX(sphereX(r, size));
      const int py = kImageSize.y / 2;
      const size_t idx = size_t(py) * kImageSize.x + px;
      float c[4];
      if (opts.colorType == ANARI_FLOAT32_VEC4) {
        memcpy(c, (const float *)fb.data + 4 * idx, sizeof(c));
      } else {
        uint32_t v = ((const uint32_t *)fb.data)[idx];
        for (int i = 0; i < 4; i++)
          c[i] = ((v >> (8 * i)) & 0xff) / 255.f;
      }
      float3 want = kColors[r % 4];
      if (opts.invertCheck)
        want = float3(1.f) - want;
      bool ok = c[3] > 0.9f;
      for (int i = 0; i < 3; i++)
        ok &= (want[i] > 0.5f) == (c[i] > 0.1f);
      printf("%s: rank %d sphere @ px %d -> (%.2f %.2f %.2f %.2f)\n",
          ok ? "PASS" : "FAIL",
          r,
          px,
          c[0],
          c[1],
          c[2],
          c[3]);
      failures += !ok;
    }
  }
  anari::unmap(d, frame, "channel.color");
  return failures;
}

// Rank 0 only: rank 0's sphere front is at z=0.3 and the camera at z=5, so
// depth there is ~4.7; the background corner must be +inf (or huge).
int checkDepth(anari::Device d, anari::Frame frame, int size)
{
  int failures = 0;
  auto db = anari::map<float>(d, frame, "channel.depth");
  if (!db.data || db.pixelType != ANARI_FLOAT32) {
    printf("FAIL: bad depth map\n");
    failures++;
  } else {
    const int px = pixelX(sphereX(0, size));
    const float hit = db.data[(kImageSize.y / 2) * kImageSize.x + px];
    const float corner = db.data[0];
    const bool ok =
        hit > 4.f && hit < 5.f && (std::isinf(corner) || corner > 1e19f);
    printf("%s: depth hit=%f corner=%g\n", ok ? "PASS" : "FAIL", hit, corner);
    failures += !ok;
  }
  anari::unmap(d, frame, "channel.depth");
  return failures;
}

} // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Options opts;
  if (!parseArgs(argc, argv, opts)) {
    MPI_Finalize();
    return 2;
  }

  int failures = 0;

  auto lib = anari::loadLibrary("ptc", statusFunc);
  if (!lib) {
    printf("FAIL: rank %d could not load the ptc library\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  auto d = anari::newDevice(lib, "default");

  if (!opts.expectLibrary.empty()) {
    const std::string loaded = loadedPtcLibrary();
    if (loaded != canonicalPath(opts.expectLibrary)) {
      printf("FAIL: rank %d loaded ptc from '%s', expected '%s'\n",
          rank,
          loaded.c_str(),
          opts.expectLibrary.c_str());
      failures++;
    }
  }

  // Spec: the device must report KHR_DATA_PARALLEL_MPI. Not implemented yet
  // (ticket 03), so by default this is an expected failure.
  if (!deviceHasMpiExtension(d)) {
    if (opts.requireMpiExtension) {
      printf("FAIL: rank %d: device does not report KHR_DATA_PARALLEL_MPI\n",
          rank);
      failures++;
    } else if (rank == 0) {
      printf("XFAIL: device does not report KHR_DATA_PARALLEL_MPI\n");
    }
  } else if (rank == 0 && !opts.requireMpiExtension) {
    printf("XPASS: device reports KHR_DATA_PARALLEL_MPI;"
           " pass --require-mpi-extension\n");
  }

  auto world = buildWorld(d, rank, size);
  auto frame = buildFrame(d, opts, world);

  anari::render(d, frame);
  anari::wait(d, frame);

  if (rank == 0) {
    failures += checkColor(d, frame, opts, size);
    if (opts.depth)
      failures += checkDepth(d, frame, size);
  }

  anari::release(d, frame);
  anari::release(d, d);
  anari::unloadLibrary(lib);

  int totalFailures = 0;
  MPI_Allreduce(&failures, &totalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    printf(totalFailures ? "SMOKE FAILED (%d)\n" : "SMOKE OK\n",
        totalFailures);
  }
  if (totalFailures)
    printf("rank %d exiting with failure\n", rank);
  fflush(stdout);
  MPI_Finalize();
  return totalFailures ? 1 : 0;
}
