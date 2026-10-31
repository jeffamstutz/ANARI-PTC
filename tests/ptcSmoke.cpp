// SPDX-License-Identifier: Apache-2.0
//
// Multi-rank smoke test for the ptc device. Rank r renders one sphere at a
// distinct x position with a distinct color through ptc (wrapping helide);
// rank 0 checks that every sphere shows up in the composited image. Every rank
// but the last also renders a hidden sphere straight behind the last rank's,
// so that pixel has hits from all ranks and the nearest one must win.
// Each rank's surface has id 100 + its world rank, and the visible sphere is
// primitive 0 of its geometry, the hidden one primitive 1. With --ids, every
// rank also renders a "tie" sphere, the last primitive, at the same place, so
// that pixel has hits at equal depth from all ranks.
//
// Every rank also checks that the device reports KHR_DATA_PARALLEL_MPI and
// KHR_FRAME_COMPLETION_CALLBACK, ptc's own version, and the required ANARI 1.1
// device properties. Ranks other than the compositing root
// map the frame too: the contents are undefined there, but it must not crash.
//
// Usage: ptcSmoke [--color ufixed8|ufixed8-linear|float32|float32-vec3|none]
//                 [--depth] [--ids]
//                 [--resize] [--first-color <type>] [--unset-all]
//                 [--uncommitted] [--callback] [--invert-check]
//                 [--fail-last-rank]
//                 [--comm <comm>] [--first-comm <comm>]
//                 [--background none|blend|opaque]
//                 [--expect-library <path>]
//                 [--wrapped-library <name>]
//
//   --color           frame color channel type (default ufixed8); 'none'
//                     leaves channel.color unset and checks that mapping
//                     color is refused; 'float32-vec3' is a type ptc cannot
//                     composite, so rendering must report an ERROR through
//                     the status callback on every rank (not throw) and
//                     mapping color must be refused
//   --depth           enable channel.depth and check the composited depth;
//                     without it, check that mapping depth is refused
//   --ids             enable channel.primitiveId, channel.objectId and
//                     channel.instanceId (UINT32) and check that each pixel
//                     holds the nearest fragment's ids; without it, check
//                     that mapping them is refused. At the tie sphere, the
//                     lowest rank's ids must win. channel.normal and
//                     channel.albedo, which ptc does not composite, must
//                     always be refused.
//   --resize          render a first frame at half the size, then resize
//   --first-color     render a first frame with this color type, then change
//                     it to --color
//   --unset-all       render a first frame, then anariUnsetAllParameters and
//                     set the final parameters again
//   --uncommitted     before the final render, set a different size and
//                     color type without committing them; they must not
//                     take effect
//   --callback        set frameCompletionCallback on the frame; every render
//                     must call it once, on every rank, with the ptc device
//                     and frame handles and the app's user data, after
//                     compositing (rank 0 checks the composited color from
//                     inside the callback). Afterwards the callback is unset
//                     and one more render must not call it.
//   --invert-check    expect the wrong colors, so the pixel check must fail;
//                     used to test the harness's failure path
//   --fail-last-rank  the last rank commits 'float32-vec3' color (which ptc
//                     cannot composite) for the final frame, the others
//                     --color: rendering fails on that rank only, yet every
//                     rank must report an ERROR through the status callback
//                     and return (none may hang in the compositor), and the
//                     color map is refused
//   --comm            communicator the frame composites over (default
//                     world): 'world' leaves mpiCommunicator unset (so
//                     MPI_COMM_WORLD); 'split' splits MPI_COMM_WORLD into
//                     its lower and upper half, 'interleaved' into its even
//                     and odd ranks (both need an even number of ranks), and
//                     sets mpiCommunicator. Each part renders its own image,
//                     with colors picked by world rank, and its rank 0
//                     checks it.
//   --first-comm      render a first frame over this communicator, then
//                     change it to --comm (going back to 'world' unsets
//                     mpiCommunicator)
//   --background      renderer background of each rank (default none:
//                     transparent black). 'blend' gives every rank a
//                     different semi-transparent background, 'opaque' a
//                     different opaque one; both pick it by world rank. The
//                     background corner is a miss on every rank, so ptc
//                     composites the backgrounds there in rank order (equal
//                     depth ties go to the lower rank) with premultiplied
//                     "over"; rank 0 checks it against that blend computed
//                     here in linear float, encoded like the color channel.
//                     With 'opaque' only rank 0's background shows, so it
//                     must round-trip within 1 LSB.
//   --expect-library  fail unless the loaded libanari_library_ptc is this
//                     file (guards against picking up an installed copy
//                     instead of the one under test)
//   --wrapped-library set the device parameter 'wrappedLibrary' to this and
//                     commit the device before anything else; that must
//                     load it without any warning (the tests set
//                     PTC_ANARI_LIBRARY to a library that does not exist, so
//                     the parameter has to win over it)
//
// With --resize, --first-color, --first-comm or --unset-all, a first frame is
// rendered and its color map is checked, then the frame's parameters change and
// the full checks run on the second frame. The first frame also has the
// opposite channel.depth and id channel settings, so those are enabled or
// disabled between frames too.
// Without --unset-all, parameters that go away are unset one by one.
//
// Built against the CUDA ptc (PTC_TEST_CUDA), every rank first makes a GPU
// current that differs from the one ptc picks for it, when there is more
// than one, and checks that ptc leaves it current: after creating, committing
// and rendering the frame, and from inside the frame completion callback.
//
// Every rank exits non-zero if any rank recorded a failure.

// anariGetDeviceExtensionStruct() is implemented in this translation unit
#define ANARI_EXTENSION_UTILITY_IMPL

#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
#include <mpi.h>
#ifdef PTC_TEST_CUDA
#include "cudaDevice.h"
#endif
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace anari::math;

namespace {

// Renderer background of every rank (--background)
enum class Background
{
  None, // transparent black
  Blend, // semi-transparent, different on each rank
  Opaque // opaque, different on each rank
};

// How the ranks are split into the communicators frames composite over
enum class CommSplit
{
  World, // no split: mpiCommunicator unset
  Halves, // lower and upper half of MPI_COMM_WORLD
  Interleaved // even and odd ranks of MPI_COMM_WORLD
};

// The communicator a frame composites over, and this rank's place in it.
struct Layout
{
  CommSplit split{CommSplit::World};
  MPI_Comm comm{MPI_COMM_WORLD};
  int rank{0};
  int size{1};
  // world rank of each rank of `comm`; it picks that rank's color, so each
  // part of a split renders different ones
  std::vector<int> worldRanks;
};

// The frame parameters ptc tracks. color == ANARI_UNKNOWN means unset.
struct FrameState
{
  uint2 size;
  ANARIDataType color{ANARI_UNKNOWN};
  bool depth{false};
  bool ids{false};
  bool callback{false};
  const Layout *layout{nullptr};
};

struct Options
{
  ANARIDataType colorType{ANARI_UFIXED8_RGBA_SRGB};
  bool depth{false};
  bool ids{false};
  bool resize{false};
  std::optional<ANARIDataType> firstColorType;
  bool unsetAll{false};
  bool uncommitted{false};
  bool callback{false};
  bool invertCheck{false};
  bool failLastRank{false};
  CommSplit comm{CommSplit::World};
  std::optional<CommSplit> firstComm;
  Background background{Background::None};
  std::string expectLibrary;
  std::string wrappedLibrary;

  bool twoFrames() const
  {
    return resize || firstColorType || firstComm || unsetAll;
  }
};

const float3 kColors[] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
const uint2 kImageSize(128, 64);
const float kCameraHeight = 1.5f;
const float kAspect = float(kImageSize.x) / kImageSize.y;
const int kNumColors = int(sizeof(kColors) / sizeof(kColors[0]));
// Every sphere has radius 0.3 and its center at z=0; the camera is at z=5
// looking down -z, so the depth at a sphere's center pixel is 4.7.
const float kSphereRadius = 0.3f;
const float kCameraZ = 5.f;
// x of the tie sphere (--ids): on the center row, between the ranks' spheres
const float kTieX = -0.5f;

// surface id (objectId) of the sphere of the rank with this world rank
uint32_t surfaceId(int worldRank)
{
  return 100u + uint32_t(worldRank);
}

// Channels ptc composites by taking the nearest fragment's value (UINT32)
const char *const kIdChannels[] = {
    "channel.primitiveId", "channel.objectId", "channel.instanceId"};
// Channels ptc does not composite: mapping them is always refused
const char *const kUnsupportedChannels[] = {"channel.normal", "channel.albedo"};

// Status messages at WARNING or worse that mention channel.depth/color
int g_depthWarnings = 0;
int g_colorWarnings = 0;
// Status messages at ERROR or worse that mention channel.color
int g_colorErrors = 0;
// Status messages at ERROR or worse
int g_errors = 0;
// Status messages at WARNING or worse
int g_warnings = 0;
// The last status message at WARNING or worse
std::string g_lastWarning;

void statusFunc(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity sev,
    ANARIStatusCode,
    const char *msg)
{
  if (sev <= ANARI_SEVERITY_WARNING) {
    fprintf(stderr, "[anari] %s\n", msg);
    g_depthWarnings += std::string(msg).find("channel.depth") != std::string::npos;
    g_colorWarnings += std::string(msg).find("channel.color") != std::string::npos;
    g_lastWarning = msg;
    g_warnings++;
  }
  if (sev <= ANARI_SEVERITY_ERROR) {
    g_colorErrors += std::string(msg).find("channel.color") != std::string::npos;
    g_errors++;
  }
}

bool parseColorType(const std::string &type, ANARIDataType &out)
{
  if (type == "ufixed8")
    out = ANARI_UFIXED8_RGBA_SRGB;
  else if (type == "ufixed8-linear")
    out = ANARI_UFIXED8_VEC4;
  else if (type == "float32")
    out = ANARI_FLOAT32_VEC4;
  else if (type == "float32-vec3")
    out = ANARI_FLOAT32_VEC3;
  else if (type == "none")
    out = ANARI_UNKNOWN;
  else {
    fprintf(stderr, "unknown color type '%s'\n", type.c_str());
    return false;
  }
  return true;
}

// Color types ptc composites (or ANARI_UNKNOWN: no color)
bool compositable(ANARIDataType type)
{
  return type == ANARI_UNKNOWN || type == ANARI_UFIXED8_VEC4
      || type == ANARI_UFIXED8_RGBA_SRGB || type == ANARI_FLOAT32_VEC4;
}

bool parseBackground(const std::string &bg, Background &out)
{
  if (bg == "none")
    out = Background::None;
  else if (bg == "blend")
    out = Background::Blend;
  else if (bg == "opaque")
    out = Background::Opaque;
  else {
    fprintf(stderr, "unknown background '%s'\n", bg.c_str());
    return false;
  }
  return true;
}

bool parseComm(const std::string &comm, CommSplit &split)
{
  if (comm == "world")
    split = CommSplit::World;
  else if (comm == "split")
    split = CommSplit::Halves;
  else if (comm == "interleaved")
    split = CommSplit::Interleaved;
  else {
    fprintf(stderr, "unknown communicator '%s'\n", comm.c_str());
    return false;
  }
  return true;
}

bool parseArgs(int argc, char **argv, Options &opts)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    const bool hasValue = i + 1 < argc;
    if ((arg == "--color" || arg == "--first-color" || arg == "--comm"
            || arg == "--first-comm" || arg == "--background"
            || arg == "--expect-library" || arg == "--wrapped-library")
        && !hasValue) {
      fprintf(stderr, "missing value for '%s'\n", arg.c_str());
      return false;
    }
    if (arg == "--color") {
      if (!parseColorType(argv[++i], opts.colorType))
        return false;
    } else if (arg == "--first-color") {
      ANARIDataType type;
      if (!parseColorType(argv[++i], type))
        return false;
      opts.firstColorType = type;
    } else if (arg == "--depth")
      opts.depth = true;
    else if (arg == "--ids")
      opts.ids = true;
    else if (arg == "--resize")
      opts.resize = true;
    else if (arg == "--unset-all")
      opts.unsetAll = true;
    else if (arg == "--uncommitted")
      opts.uncommitted = true;
    else if (arg == "--callback")
      opts.callback = true;
    else if (arg == "--invert-check")
      opts.invertCheck = true;
    else if (arg == "--fail-last-rank")
      opts.failLastRank = true;
    else if (arg == "--comm") {
      if (!parseComm(argv[++i], opts.comm))
        return false;
    } else if (arg == "--first-comm") {
      CommSplit split;
      if (!parseComm(argv[++i], split))
        return false;
      opts.firstComm = split;
    } else if (arg == "--background") {
      if (!parseBackground(argv[++i], opts.background))
        return false;
    } else if (arg == "--expect-library")
      opts.expectLibrary = argv[++i];
    else if (arg == "--wrapped-library")
      opts.wrappedLibrary = argv[++i];
    else {
      fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
      return false;
    }
  }
  return true;
}

// MPI_COMM_WORLD, or a new communicator for this rank's part of it. Every
// call splits anew, so two layouts with the same split have different but
// congruent communicators.
Layout makeLayout(CommSplit split)
{
  Layout l;
  l.split = split;
  int worldRank = 0, worldSize = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  if (split != CommSplit::World) {
    const int part = split == CommSplit::Halves ? worldRank / (worldSize / 2)
                                                : worldRank % 2;
    MPI_Comm_split(MPI_COMM_WORLD, part, worldRank, &l.comm);
  }
  MPI_Comm_rank(l.comm, &l.rank);
  MPI_Comm_size(l.comm, &l.size);
  l.worldRanks.resize(l.size);
  MPI_Allgather(
      &worldRank, 1, MPI_INT, l.worldRanks.data(), 1, MPI_INT, l.comm);
  return l;
}

void freeLayout(Layout &l)
{
  if (l.split != CommSplit::World)
    MPI_Comm_free(&l.comm);
}

// x position of rank r's sphere, in world space
float sphereX(int r, int size)
{
  return size == 1 ? 0.f : -1.f + 2.f * r / (size - 1);
}

// pixel column of a world-space x under the orthographic camera
int pixelX(float x)
{
  return int((x / (kCameraHeight * kAspect) + 0.5f) * kImageSize.x);
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

// Spelling used by the SDK's extension lists and anari_extension_utility.h
const char *const kMpiExtension = "ANARI_KHR_DATA_PARALLEL_MPI";
// ptc implements this one itself, so it lists it whatever it wraps
const char *const kCallbackExtension = "ANARI_KHR_FRAME_COMPLETION_CALLBACK";

// ptc's version (from CMake), reported as its device property 'version'
const int32_t kPtcVersion = PTC_VERSION_MAJOR * 1000 + PTC_VERSION_MINOR * 100
    + PTC_VERSION_PATCH;

// Count of failed conformance checks, printed with the rank that failed them.
struct Checker
{
  int rank;
  int failures{0};

  void expect(bool ok, const std::string &what)
  {
    if (!ok) {
      printf("FAIL: rank %d: %s\n", rank, what.c_str());
      failures++;
    }
  }
};

// Checks that a null-terminated extension list holds each extension ptc
// implements itself exactly once, and has no duplicates.
void checkOwnExtensions(Checker &c, const char *const *list, const char *source)
{
  int mpi = 0, callback = 0;
  std::set<std::string> seen;
  for (auto e = list; e && *e; e++) {
    c.expect(seen.insert(*e).second,
        std::string(source) + " lists '" + *e + "' twice");
    mpi += std::string(*e) == kMpiExtension;
    callback += std::string(*e) == kCallbackExtension;
  }
  c.expect(mpi == 1, std::string(source) + " lacks " + kMpiExtension);
  c.expect(callback == 1, std::string(source) + " lacks " + kCallbackExtension);
}

template <typename T>
bool getDeviceProperty(anari::Device d, const char *name, ANARIDataType type, T &out)
{
  return anariGetProperty(d, d, name, type, &out, sizeof(out), ANARI_WAIT);
}

// All ranks: KHR_DATA_PARALLEL_MPI is reported everywhere an application can
// look for it, and the required device properties have valid values.
// anariGetProperty is collective under KHR_DATA_PARALLEL_MPI, so every rank
// makes the same calls.
int checkDeviceConformance(anari::Library lib, anari::Device d, int rank)
{
  Checker c{rank};

  const char *const *exts = nullptr;
  c.expect(getDeviceProperty(d, "extension", ANARI_STRING_LIST, exts) && exts,
      "device property 'extension' not reported");
  checkOwnExtensions(c, exts, "device property 'extension'");

  int32_t version = -1, major = -1, minor = -1;
  c.expect(getDeviceProperty(d, "version", ANARI_INT32, version)
          && version == kPtcVersion,
      "device property 'version' is not ptc's (got " + std::to_string(version)
          + ", want " + std::to_string(kPtcVersion) + ")");
  c.expect(getDeviceProperty(d, "version.major", ANARI_INT32, major)
          && major == PTC_VERSION_MAJOR,
      "device property 'version.major' is not ptc's");
  c.expect(getDeviceProperty(d, "anariVersion.major", ANARI_INT32, major)
          && getDeviceProperty(d, "anariVersion.minor", ANARI_INT32, minor)
          && major == 1 && minor >= 1,
      "device property 'anariVersion' is not 1.1 or later (got "
          + std::to_string(major) + "." + std::to_string(minor) + ")");
  uint64_t maxIndex = 0;
  c.expect(getDeviceProperty(d, "geometryMaxIndex", ANARI_UINT64, maxIndex)
          && maxIndex > 0,
      "device property 'geometryMaxIndex' not reported");

  auto libExts = anariGetDeviceExtensions(lib, "default");
  checkOwnExtensions(c, libExts, "library extension list");

  auto extStruct = anari::extension::getDeviceExtensionStruct(lib, "default");
  c.expect(extStruct.ANARI_KHR_DATA_PARALLEL_MPI,
      "anariGetDeviceExtensionStruct does not report the MPI extension");

  auto info = [&](const char *infoName, ANARIDataType infoType) {
    return anariGetParameterInfo(d,
        ANARI_FRAME,
        nullptr,
        "mpiCommunicator",
        ANARI_VOID_POINTER,
        infoName,
        infoType);
  };
  auto *required = (const int32_t *)info("required", ANARI_BOOL);
  c.expect(required && !*required,
      "parameter info 'required' missing for frame 'mpiCommunicator'");
  c.expect(info("description", ANARI_STRING),
      "parameter info 'description' missing for frame 'mpiCommunicator'");
  auto *source = (const char *)info("sourceExtension", ANARI_STRING);
  c.expect(source && std::string(source) == "KHR_DATA_PARALLEL_MPI",
      "parameter info 'sourceExtension' wrong for frame 'mpiCommunicator'");

  bool listed = false;
  auto *params = (const ANARIParameter *)anariGetObjectInfo(
      d, ANARI_FRAME, nullptr, "parameter", ANARI_PARAMETER_LIST);
  for (auto p = params; p && p->name; p++) {
    listed |= std::string(p->name) == "mpiCommunicator"
        && p->type == ANARI_VOID_POINTER;
  }
  c.expect(listed, "frame parameter list lacks 'mpiCommunicator'");

  return c.failures;
}

// Color of rank r of the layout's communicator
float3 sphereColor(const Layout &l, int r)
{
  return kColors[l.worldRanks[r] % kNumColors];
}

// Backgrounds by world rank, as premultiplied linear RGB and alpha (the
// convention ptc composites in). The values are far enough apart that
// blending the sRGB-encoded values, or treating them as straight alpha, is
// off by many LSBs.
const float4 kBlendBackgrounds[] = {{0.20f, 0.02f, 0.00f, 0.40f},
    {0.00f, 0.15f, 0.05f, 0.30f},
    {0.05f, 0.00f, 0.25f, 0.50f},
    {0.10f, 0.10f, 0.00f, 0.25f}};
// rank 0's has a dark channel, where sRGB and linear 8-bit codes differ most
const float4 kOpaqueBackgrounds[] = {{0.002f, 0.18f, 0.60f, 1.f},
    {0.50f, 0.50f, 0.00f, 1.f},
    {0.00f, 0.50f, 0.50f, 1.f},
    {0.50f, 0.00f, 0.50f, 1.f}};
const int kNumBackgrounds =
    int(sizeof(kBlendBackgrounds) / sizeof(kBlendBackgrounds[0]));

float4 background(Background bg, int worldRank)
{
  switch (bg) {
  case Background::Blend:
    return kBlendBackgrounds[worldRank % kNumBackgrounds];
  case Background::Opaque:
    return kOpaqueBackgrounds[worldRank % kNumBackgrounds];
  default:
    return float4(0.f);
  }
}

// What ptc must composite where every rank of the layout misses: the ranks'
// backgrounds front to back in rank order, with premultiplied "over"
float4 expectedBackground(Background bg, const Layout &l)
{
  float4 result(0.f);
  for (int r = 0; r < l.size; r++) {
    const float4 b = background(bg, l.worldRanks[r]);
    result = result + (1.f - result.w) * b;
  }
  return result;
}

// IEC 61966-2-1 (the curve helide uses for UFIXED8_RGBA_SRGB)
float linearToSrgb(float v)
{
  return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
}

int toUnorm8(float v)
{
  return int(std::fmin(std::fmax(v, 0.f), 1.f) * 255.f + 0.5f);
}

// This rank's spheres, placed by its rank in the layout's communicator
// Primitive index of rank r's tie sphere (--ids): after its visible sphere
// and, on all but the last rank, the hidden one
uint32_t tiePrimitive(int r, int size)
{
  return r == size - 1 ? 1u : 2u;
}

anari::World buildWorld(anari::Device d, const Layout &l, bool tie)
{
  const int rank = l.rank;
  const int size = l.size;
  // [0]: this rank's visible sphere; [1]: hidden behind the last rank's
  // sphere, at a different depth on each rank; then the tie sphere
  std::vector<float3> pos = {{sphereX(rank, size), 0.f, 0.f}};
  if (rank != size - 1)
    pos.push_back({sphereX(size - 1, size), 0.f, -1.f - 0.5f * rank});
  if (tie)
    pos.push_back({kTieX, 0.f, 0.f});
  const int numSpheres = int(pos.size());
  float3 col = sphereColor(l, rank);
  auto geom = anari::newObject<anari::Geometry>(d, "sphere");
  anari::setParameterArray1D(d, geom, "vertex.position", pos.data(), numSpheres);
  anari::setParameter(d, geom, "radius", kSphereRadius);
  anari::commitParameters(d, geom);
  auto mat = anari::newObject<anari::Material>(d, "matte");
  anari::setParameter(d, mat, "color", col);
  anari::commitParameters(d, mat);
  auto surf = anari::newObject<anari::Surface>(d);
  anari::setParameter(d, surf, "id", surfaceId(l.worldRanks[rank]));
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

// Objects the frame references; kept alive so they can be set again after
// anariUnsetAllParameters.
struct Scene
{
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;
};

Scene buildScene(anari::Device d, anari::World w, Background bg)
{
  int worldRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  auto cam = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, cam, "position", float3(0, 0, kCameraZ));
  anari::setParameter(d, cam, "direction", float3(0, 0, -1));
  anari::setParameter(d, cam, "up", float3(0, 1, 0));
  anari::setParameter(d, cam, "height", kCameraHeight);
  anari::setParameter(d, cam, "aspect", kAspect);
  anari::commitParameters(d, cam);

  auto ren = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, ren, "background", background(bg, worldRank));
  anari::setParameter(d, ren, "ambientRadiance", 1.f);
  anari::commitParameters(d, ren);

  return {w, cam, ren};
}

void releaseScene(anari::Device d, const Scene &s)
{
  anari::release(d, s.world);
  anari::release(d, s.camera);
  anari::release(d, s.renderer);
}

// Frame completion callback (--callback) /////////////////////////////////////

void onFrameComplete(const void *userData, ANARIDevice d, ANARIFrame f);

// What the callback expects, passed to it as its user data.
struct CallbackContext
{
  anari::Device device{nullptr};
  anari::Frame frame{nullptr};
  int rank{0};
  const Layout *layout{nullptr};
  const Options *opts{nullptr};
  // the committed frame state of the render in flight, for rank 0's check
  const FrameState *rendering{nullptr};
  bool finalFrame{false};
  int calls{0};
  int failures{0};
};

CallbackContext g_callback;

// The app's current CUDA device (PTC_TEST_CUDA) /////////////////////////////

// The GPU this rank makes current before loading ptc; -1 without CUDA
int g_appCudaDevice = -1;
int g_cudaDeviceFailures = 0;

// Makes a GPU current that ptc would not pick for this rank if there are
// several: ptc picks by rank among the ranks on the node, this the next one.
void selectAppCudaDevice(int rank)
{
#ifdef PTC_TEST_CUDA
  const int count = cudaDeviceCount();
  if (count == 0) {
    printf("FAIL: rank %d: no CUDA device\n", rank);
    g_cudaDeviceFailures++;
    return;
  }
  g_appCudaDevice = (rank + 1) % count;
  if (!makeCudaDeviceCurrent(g_appCudaDevice)) {
    printf("FAIL: rank %d: cannot make GPU %d current\n", rank, g_appCudaDevice);
    g_cudaDeviceFailures++;
    g_appCudaDevice = -1;
  }
#else
  (void)rank;
#endif
}

// ptc must not change the calling thread's current CUDA device.
void checkCudaDevice(const char *when)
{
#ifdef PTC_TEST_CUDA
  if (g_appCudaDevice < 0)
    return;
  const int current = currentCudaDevice();
  if (current != g_appCudaDevice) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("FAIL: rank %d: current CUDA device is %d %s, the app made %d "
           "current\n",
        rank,
        current,
        when,
        g_appCudaDevice);
    g_cudaDeviceFailures++;
  }
#else
  (void)when;
#endif
}

void renderAndWait(anari::Device d, anari::Frame frame)
{
  anari::render(d, frame);
  anari::wait(d, frame);
  checkCudaDevice("after rendering the frame");
}

// What mpiCommunicator points to. ptc may only read it until the frame's
// commit returns, so commitFrame() clobbers it afterwards.
MPI_Comm g_commParameter = MPI_COMM_NULL;

void setCommunicator(anari::Device d, anari::Frame frame, const Layout &l)
{
  g_commParameter = l.comm;
  anari::setParameter(d, frame, "mpiCommunicator", (void *)&g_commParameter);
}

void commitFrame(anari::Device d, anari::Frame frame)
{
  anari::commitParameters(d, frame);
  g_commParameter = MPI_COMM_NULL;
  checkCudaDevice("after committing the frame");
}

// Set every parameter of `state` (and the scene objects) on the frame.
void setAllParameters(
    anari::Device d, anari::Frame frame, const Scene &scene, const FrameState &state)
{
  anari::setParameter(d, frame, "world", scene.world);
  anari::setParameter(d, frame, "camera", scene.camera);
  anari::setParameter(d, frame, "renderer", scene.renderer);
  anari::setParameter(d, frame, "size", state.size);
  if (state.color != ANARI_UNKNOWN)
    anari::setParameter(d, frame, "channel.color", state.color);
  if (state.depth)
    anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  if (state.ids) {
    for (const char *channel : kIdChannels)
      anari::setParameter(d, frame, channel, ANARI_UINT32);
  }
  if (state.callback) {
    anari::setParameter(d,
        frame,
        "frameCompletionCallback",
        (ANARIFrameCompletionCallback)onFrameComplete);
    anari::setParameter(d,
        frame,
        "frameCompletionCallbackUserData",
        (void *)&g_callback);
  }
  if (state.layout->split != CommSplit::World)
    setCommunicator(d, frame, *state.layout);
}

// Move the frame from `prev` to `next`: set what changed and unset what went
// away, or with `unsetAll` start over from anariUnsetAllParameters. Commits.
void changeParameters(anari::Device d,
    anari::Frame frame,
    const Scene &scene,
    const FrameState &prev,
    const FrameState &next,
    bool unsetAll)
{
  if (unsetAll) {
    anari::unsetAllParameters(d, frame);
    setAllParameters(d, frame, scene, next);
  } else {
    if (next.size != prev.size)
      anari::setParameter(d, frame, "size", next.size);
    if (next.color == ANARI_UNKNOWN)
      anari::unsetParameter(d, frame, "channel.color");
    else if (next.color != prev.color)
      anari::setParameter(d, frame, "channel.color", next.color);
    if (!next.depth)
      anari::unsetParameter(d, frame, "channel.depth");
    else if (!prev.depth)
      anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
    for (const char *channel : kIdChannels) {
      if (!next.ids)
        anari::unsetParameter(d, frame, channel);
      else if (!prev.ids)
        anari::setParameter(d, frame, channel, ANARI_UINT32);
    }
    if (next.layout->split != CommSplit::World)
      setCommunicator(d, frame, *next.layout);
    else if (prev.layout->split != CommSplit::World)
      anari::unsetParameter(d, frame, "mpiCommunicator");
  }
  commitFrame(d, frame);
}

// Rank 0 only: the frame has no channel.color, so mapping color returns
// nothing and reports why through the status callback.
int checkColorRefused(anari::Device d, anari::Frame frame, const char *which)
{
  const int warningsBefore = g_colorWarnings;
  auto fb = anari::map<void>(d, frame, "channel.color");
  const bool ok = !fb.data && fb.width == 0 && fb.height == 0
      && fb.pixelType == ANARI_UNKNOWN && g_colorWarnings > warningsBefore;
  printf("%s: %s: unset color map refused (data=%p %ux%u type=%d, "
         "warning %s)\n",
      ok ? "PASS" : "FAIL",
      which,
      fb.data,
      fb.width,
      fb.height,
      int(fb.pixelType),
      g_colorWarnings > warningsBefore ? "reported" : "missing");
  anari::unmap(d, frame, "channel.color");
  return !ok;
}

// Rank 0 only, first of two frames: the color map has that frame's size and
// type (or is refused if it has no color).
int checkFirstFrame(anari::Device d, anari::Frame frame, const FrameState &state)
{
  if (state.color == ANARI_UNKNOWN)
    return checkColorRefused(d, frame, "first frame");
  auto fb = anari::map<void>(d, frame, "channel.color");
  const bool ok = fb.data && fb.width == state.size.x
      && fb.height == state.size.y && fb.pixelType == state.color;
  printf("%s: first frame color map (data=%p %ux%u type=%d, want %ux%u "
         "type=%d)\n",
      ok ? "PASS" : "FAIL",
      fb.data,
      fb.width,
      fb.height,
      int(fb.pixelType),
      state.size.x,
      state.size.y,
      int(state.color));
  anari::unmap(d, frame, "channel.color");
  return !ok;
}

// Rank 0 of the layout only: probe the pixel at the center of each rank's
// sphere.
int checkColor(
    anari::Device d, anari::Frame frame, const Options &opts, const Layout &l)
{
  const int size = l.size;
  if (opts.colorType == ANARI_UNKNOWN || !compositable(opts.colorType))
    return checkColorRefused(d, frame, "final frame");
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
      float3 want = sphereColor(l, r);
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

// Rank 0 of the layout only, with --background: the corner pixel, a miss on
// every rank, is the ranks' backgrounds blended in linear space. 8-bit
// results may be off by 1 LSB from rounding; float ones by about as much,
// since the CUDA compositor carries fragments in 8 bits.
int checkBackground(
    anari::Device d, anari::Frame frame, const Options &opts, const Layout &l)
{
  if (opts.background == Background::None
      || opts.colorType == ANARI_UNKNOWN || !compositable(opts.colorType))
    return 0;
  const float4 want = expectedBackground(opts.background, l);
  auto fb = anari::map<void>(d, frame, "channel.color");
  int failures = 0;
  if (!fb.data || fb.pixelType != opts.colorType)
    failures++; // reported by checkColor
  else if (opts.colorType == ANARI_FLOAT32_VEC4) {
    float c[4];
    memcpy(c, fb.data, sizeof(c));
    bool ok = true;
    for (int i = 0; i < 4; i++)
      ok &= std::fabs(c[i] - want[i]) <= 1.f / 255.f;
    printf("%s: background (%.4f %.4f %.4f %.4f), want "
           "(%.4f %.4f %.4f %.4f)\n",
        ok ? "PASS" : "FAIL",
        c[0],
        c[1],
        c[2],
        c[3],
        want.x,
        want.y,
        want.z,
        want.w);
    failures += !ok;
  } else {
    const bool srgb = opts.colorType == ANARI_UFIXED8_RGBA_SRGB;
    const uint32_t v = *(const uint32_t *)fb.data;
    int got[4], expect[4];
    bool ok = true;
    for (int i = 0; i < 4; i++) {
      got[i] = (v >> (8 * i)) & 0xff;
      expect[i] = toUnorm8(srgb && i < 3 ? linearToSrgb(want[i]) : want[i]);
      ok &= std::abs(got[i] - expect[i]) <= 1;
    }
    printf("%s: background (%d %d %d %d), want (%d %d %d %d)%s\n",
        ok ? "PASS" : "FAIL",
        got[0],
        got[1],
        got[2],
        got[3],
        expect[0],
        expect[1],
        expect[2],
        expect[3],
        srgb ? " sRGB" : "");
    failures += !ok;
  }
  anari::unmap(d, frame, "channel.color");
  return failures;
}

// Rank 0 of the layout only, with --depth: the composited depth is the
// nearest hit across all ranks. Each rank's sphere center must show that
// sphere's depth: other ranks either miss there (their +inf must not win) or,
// behind the last rank's sphere, hit farther away. The background corner,
// where no rank hits, must be +inf.
int checkDepth(anari::Device d, anari::Frame frame, int size)
{
  int failures = 0;
  auto db = anari::map<float>(d, frame, "channel.depth");
  if (!db.data || db.width != kImageSize.x || db.height != kImageSize.y
      || db.pixelType != ANARI_FLOAT32) {
    printf("FAIL: bad depth map (data=%p %ux%u type=%d)\n",
        (const void *)db.data,
        db.width,
        db.height,
        int(db.pixelType));
    failures++;
  } else {
    const float want = kCameraZ - kSphereRadius;
    for (int r = 0; r < size; r++) {
      const int px = pixelX(sphereX(r, size));
      const float hit = db.data[(kImageSize.y / 2) * kImageSize.x + px];
      const bool ok = std::fabs(hit - want) < 0.05f;
      printf("%s: rank %d sphere depth @ px %d = %f (want %f)\n",
          ok ? "PASS" : "FAIL",
          r,
          px,
          hit,
          want);
      failures += !ok;
    }
    const float corner = db.data[0];
    const bool ok = std::isinf(corner) && corner > 0.f;
    printf("%s: background depth = %g (want inf)\n", ok ? "PASS" : "FAIL", corner);
    failures += !ok;
  }
  anari::unmap(d, frame, "channel.depth");
  return failures;
}

// Rank 0 only, without --depth: the app never enabled channel.depth, so
// mapping it returns nothing and reports why through the status callback.
int checkDepthRefused(anari::Device d, anari::Frame frame)
{
  const int warningsBefore = g_depthWarnings;
  auto db = anari::map<float>(d, frame, "channel.depth");
  const bool ok = !db.data && db.width == 0 && db.height == 0
      && db.pixelType == ANARI_UNKNOWN && g_depthWarnings > warningsBefore;
  printf("%s: unrequested depth map refused (data=%p %ux%u type=%d, "
         "warning %s)\n",
      ok ? "PASS" : "FAIL",
      (const void *)db.data,
      db.width,
      db.height,
      int(db.pixelType),
      g_depthWarnings > warningsBefore ? "reported" : "missing");
  anari::unmap(d, frame, "channel.depth");
  return !ok;
}

// Rank 0 only: mapping `channel` returns nothing and reports a warning that
// names the channel through the status callback.
int checkChannelRefused(
    anari::Device d, anari::Frame frame, const char *channel, const char *why)
{
  g_lastWarning.clear();
  auto fb = anari::map<void>(d, frame, channel);
  const bool warned = g_lastWarning.find(channel) != std::string::npos;
  const bool ok = !fb.data && fb.width == 0 && fb.height == 0
      && fb.pixelType == ANARI_UNKNOWN && warned;
  printf("%s: %s %s map refused (data=%p %ux%u type=%d, warning %s)\n",
      ok ? "PASS" : "FAIL",
      why,
      channel,
      fb.data,
      fb.width,
      fb.height,
      int(fb.pixelType),
      warned ? "reported" : "missing");
  anari::unmap(d, frame, channel);
  return !ok;
}

// Rank 0 of the layout only, with --ids: each pixel's ids are the nearest
// fragment's. Each rank's sphere center is primitive 0 of that rank's surface:
// other ranks either miss there or, behind the last rank's sphere, hit their
// primitive 1 farther away. Where no rank hits, the ids are ~0u.
int checkIds(anari::Device d, anari::Frame frame, const Layout &l)
{
  int failures = 0;
  for (const char *channel : kIdChannels) {
    const std::string name = channel;
    auto fb = anari::map<uint32_t>(d, frame, channel);
    if (!fb.data || fb.width != kImageSize.x || fb.height != kImageSize.y
        || fb.pixelType != ANARI_UINT32) {
      printf("FAIL: bad %s map (data=%p %ux%u type=%d)\n",
          channel,
          (const void *)fb.data,
          fb.width,
          fb.height,
          int(fb.pixelType));
      failures++;
      anari::unmap(d, frame, channel);
      continue;
    }
    // helide gives surfaces outside an instance no instance id
    if (name != "channel.instanceId") {
      for (int r = 0; r < l.size; r++) {
        const int px = pixelX(sphereX(r, l.size));
        const uint32_t got = fb.data[(kImageSize.y / 2) * kImageSize.x + px];
        const uint32_t want = name == "channel.primitiveId"
            ? 0u
            : surfaceId(l.worldRanks[r]);
        const bool ok = got == want;
        printf("%s: rank %d sphere %s @ px %d = %u (want %u)\n",
            ok ? "PASS" : "FAIL",
            r,
            channel,
            px,
            got,
            want);
        failures += !ok;
      }
    }
    if (name != "channel.instanceId") {
      // equal depth on every rank: the lowest rank is in front
      const int px = pixelX(kTieX);
      const uint32_t got = fb.data[(kImageSize.y / 2) * kImageSize.x + px];
      const uint32_t want = name == "channel.primitiveId"
          ? tiePrimitive(0, l.size)
          : surfaceId(l.worldRanks[0]);
      const bool ok = got == want;
      printf("%s: tie %s @ px %d = %u (want rank 0's %u)\n",
          ok ? "PASS" : "FAIL",
          channel,
          px,
          got,
          want);
      failures += !ok;
    }
    const uint32_t corner = fb.data[0];
    const bool ok = corner == ~0u;
    printf("%s: background %s = %u (want %u)\n",
        ok ? "PASS" : "FAIL",
        channel,
        corner,
        ~0u);
    failures += !ok;
    anari::unmap(d, frame, channel);
  }
  return failures;
}

// Rank 0 only: channels ptc does not composite are always refused, and the
// id channels are refused unless --ids enabled them.
int checkChannelsRefused(anari::Device d, anari::Frame frame, bool ids)
{
  int failures = 0;
  for (const char *channel : kUnsupportedChannels)
    failures += checkChannelRefused(d, frame, channel, "unsupported");
  if (!ids) {
    for (const char *channel : kIdChannels)
      failures += checkChannelRefused(d, frame, channel, "unrequested");
  }
  return failures;
}

// Ranks other than the layout's rank 0: the spec leaves the mapped contents
// undefined there, but mapping and reading what is returned must not crash.
void touchNonRootMaps(anari::Device d, anari::Frame frame)
{
  std::vector<const char *> channels = {"channel.color", "channel.depth"};
  channels.insert(channels.end(), std::begin(kIdChannels), std::end(kIdChannels));
  channels.insert(channels.end(),
      std::begin(kUnsupportedChannels),
      std::end(kUnsupportedChannels));
  for (const char *channel : channels) {
    auto fb = anari::map<uint8_t>(d, frame, channel);
    volatile uint8_t sum = 0;
    if (fb.data) {
      const size_t bytes =
          size_t(fb.width) * fb.height * anari::sizeOf(fb.pixelType);
      for (size_t i = 0; i < bytes; i++)
        sum += fb.data[i];
    }
    anari::unmap(d, frame, channel);
  }
}

// Called by ptc once the composited frame is ready. The handles must be the
// ones the app holds (not the wrapped device's), and on rank 0 the composited
// color must already be there: mapping it is legal inside the callback.
void onFrameComplete(const void *userData, ANARIDevice d, ANARIFrame f)
{
  if (userData != &g_callback) {
    printf("FAIL: frame completion callback got the wrong user data\n");
    g_callback.calls++;
    g_callback.failures++;
    return;
  }
  // from here on, everything is reached through the app's user data
  auto &ctx = *(CallbackContext *)userData;
  ctx.calls++;
  checkCudaDevice("inside the frame completion callback");
  Checker c{ctx.rank};
  c.expect(d == ctx.device,
      "frame completion callback got a device handle that is not the ptc "
      "device");
  c.expect(f == ctx.frame,
      "frame completion callback got a frame handle that is not the ptc "
      "frame");
  if (d == ctx.device && f == ctx.frame && ctx.rendering
      && ctx.rendering->layout->rank == 0) {
    printf("callback %d: checking composited color\n", ctx.calls);
    c.failures += ctx.finalFrame
        ? checkColor(ctx.device, ctx.frame, *ctx.opts, *ctx.layout)
        : checkFirstFrame(ctx.device, ctx.frame, *ctx.rendering);
  }
  ctx.failures += c.failures;
}

// Every rank, with --callback: one call per render so far; then unset the
// callback, render again, and expect no further call.
int checkCallbackCalls(anari::Device d, anari::Frame frame, int rank, int renders)
{
  Checker c{rank};
  c.expect(g_callback.calls == renders,
      "frame completion callback called " + std::to_string(g_callback.calls)
          + " times for " + std::to_string(renders) + " renders");

  g_callback.rendering = nullptr;
  anari::unsetParameter(d, frame, "frameCompletionCallback");
  commitFrame(d, frame);
  const int before = g_callback.calls;
  renderAndWait(d, frame);
  c.expect(g_callback.calls == before,
      "frame completion callback called after it was unset");

  if (rank == 0) {
    printf("%s: frame completion callback called %d times for %d renders, "
           "then %d times after unset\n",
        c.failures ? "FAIL" : "PASS",
        before,
        renders,
        g_callback.calls - before);
  }
  return c.failures + g_callback.failures;
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

  // ranks other than the last keep --color, and expect the render to fail
  // because the last one failed
  if (opts.failLastRank && rank == size - 1)
    opts.colorType = ANARI_FLOAT32_VEC3;

  selectAppCudaDevice(rank);

  auto lib = anari::loadLibrary("ptc", statusFunc);
  if (!lib) {
    printf("FAIL: rank %d could not load the ptc library\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  auto d = anari::newDevice(lib, "default");

  if (!opts.wrappedLibrary.empty()) {
    // the wrapped library is loaded when the device is committed
    const int warningsBefore = g_warnings;
    anari::setParameter(d, d, "wrappedLibrary", opts.wrappedLibrary.c_str());
    anari::commitParameters(d, d);
    Checker c{rank};
    c.expect(g_warnings == warningsBefore,
        "loading wrapped library '" + opts.wrappedLibrary
            + "' from the device parameter reported: " + g_lastWarning);
    failures += c.failures;
  }

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

  failures += checkDeviceConformance(lib, d, rank);

  if ((opts.comm != CommSplit::World
          || opts.firstComm.value_or(CommSplit::World) != CommSplit::World)
      && (size < 2 || size % 2)) {
    printf("FAIL: rank %d: splitting needs an even number of ranks\n", rank);
    failures++;
    opts.comm = CommSplit::World;
    opts.firstComm.reset();
  }
  Layout finalLayout = makeLayout(opts.comm);
  Layout firstLayout = makeLayout(opts.firstComm.value_or(opts.comm));

  Scene scene = buildScene(d,
      buildWorld(d, opts.twoFrames() ? firstLayout : finalLayout, opts.ids),
      opts.background);
  const FrameState finalState{kImageSize,
      opts.colorType,
      opts.depth,
      opts.ids,
      opts.callback,
      &finalLayout};
  auto frame = anari::newObject<anari::Frame>(d);
  checkCudaDevice("after creating the frame");
  g_callback = {d, frame, rank, &finalLayout, &opts};
  int renders = 0;

  if (opts.twoFrames()) {
    const FrameState first{opts.resize ? kImageSize / 2u : kImageSize,
        opts.firstColorType.value_or(opts.colorType),
        !opts.depth,
        !opts.ids,
        opts.callback,
        &firstLayout};
    setAllParameters(d, frame, scene, first);
    commitFrame(d, frame);
    g_callback.rendering = &first;
    renderAndWait(d, frame);
    renders++;
    g_callback.rendering = nullptr; // `first` goes out of scope
    if (firstLayout.rank == 0)
      failures += checkFirstFrame(d, frame, first);
    if (firstLayout.worldRanks != finalLayout.worldRanks) {
      // spheres are placed and colored by rank in the frame's communicator
      anari::release(d, scene.world);
      scene.world = buildWorld(d, finalLayout, opts.ids);
      anari::setParameter(d, frame, "world", scene.world);
    }
    changeParameters(d, frame, scene, first, finalState, opts.unsetAll);
  } else {
    setAllParameters(d, frame, scene, finalState);
    commitFrame(d, frame);
  }

  if (opts.uncommitted) {
    // ANARI parameters take effect on commit: rendering must still use the
    // committed size and color type
    anari::setParameter(d, frame, "size", kImageSize / 2u);
    anari::setParameter(d, frame, "channel.color",
        opts.colorType == ANARI_UFIXED8_VEC4 ? ANARI_UFIXED8_RGBA_SRGB
                                             : ANARI_UFIXED8_VEC4);
  }

  g_callback.rendering = &finalState;
  g_callback.finalFrame = true;
  const int errorsBefore = g_colorErrors;
  const int anyErrorsBefore = g_errors;
  renderAndWait(d, frame);
  renders++;

  // Every rank: a render error comes through the status callback (the
  // process is still here, so nothing threw out of anariRenderFrame).
  {
    Checker c{rank};
    const bool wantError = !compositable(opts.colorType);
    const bool gotError = g_colorErrors > errorsBefore;
    if (opts.failLastRank && !wantError) {
      c.expect(g_errors > anyErrorsBefore,
          "no ERROR reported after the last rank failed to render");
    } else {
      c.expect(wantError == gotError,
          wantError ? "no ERROR about channel.color reported for an "
                      "uncompositable color type"
                    : "unexpected ERROR about channel.color reported");
    }
    if (wantError && !c.failures)
      printf("PASS: rank %d: render error reported\n", rank);
    failures += c.failures;
  }

  if (opts.failLastRank) {
    // nothing was composited, so there is nothing to map
    if (finalLayout.rank == 0)
      failures += checkColorRefused(d, frame, "after a failure on one rank");
  } else if (finalLayout.rank == 0) {
    failures += checkColor(d, frame, opts, finalLayout);
    failures += checkBackground(d, frame, opts, finalLayout);
    failures += opts.depth ? checkDepth(d, frame, finalLayout.size)
                           : checkDepthRefused(d, frame);
    if (opts.ids)
      failures += checkIds(d, frame, finalLayout);
    failures += checkChannelsRefused(d, frame, opts.ids);
  } else
    touchNonRootMaps(d, frame);

  if (opts.callback)
    failures += checkCallbackCalls(d, frame, rank, renders);

  anari::release(d, frame);
  checkCudaDevice("after releasing the frame");
  failures += g_cudaDeviceFailures;
  if (g_appCudaDevice >= 0 && !g_cudaDeviceFailures)
    printf("PASS: rank %d: GPU %d stayed current\n", rank, g_appCudaDevice);
  releaseScene(d, scene);
  anari::release(d, d);
  anari::unloadLibrary(lib);
  freeLayout(firstLayout);
  freeLayout(finalLayout);

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
