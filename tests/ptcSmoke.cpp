// SPDX-License-Identifier: Apache-2.0
//
// Multi-rank smoke test for the ptc device. Rank r renders one sphere at a
// distinct x position with a distinct color through ptc (wrapping helide);
// rank 0 checks that every sphere shows up in the composited image. Every rank
// but the last also renders a hidden sphere straight behind the last rank's,
// so that pixel has hits from all ranks and the nearest one must win.
//
// Every rank also checks that the device reports KHR_DATA_PARALLEL_MPI and the
// required ANARI 1.1 device properties.
//
// Usage: ptcSmoke [--color ufixed8|ufixed8-linear|float32|float32-vec3|none]
//                 [--depth]
//                 [--resize] [--first-color <type>] [--unset-all]
//                 [--uncommitted] [--callback] [--invert-check]
//                 [--expect-library <path>]
//
//   --color           frame color channel type (default ufixed8); 'none'
//                     leaves channel.color unset and checks that mapping
//                     color is refused; 'float32-vec3' is a type ptc cannot
//                     composite, so rendering must report an ERROR through
//                     the status callback on every rank (not throw) and
//                     mapping color must be refused
//   --depth           enable channel.depth and check the composited depth;
//                     without it, check that mapping depth is refused
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
//   --expect-library  fail unless the loaded libanari_library_ptc is this
//                     file (guards against picking up an installed copy
//                     instead of the one under test)
//
// With --resize, --first-color or --unset-all, a first frame is rendered and
// its color map is checked, then the frame's parameters change and the full
// checks run on the second frame. The first frame also has the opposite
// channel.depth setting, so depth is enabled or disabled between frames too.
// Without --unset-all, parameters that go away are unset one by one.
//
// Every rank exits non-zero if any rank recorded a failure.

// anariGetDeviceExtensionStruct() is implemented in this translation unit
#define ANARI_EXTENSION_UTILITY_IMPL

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
#include <optional>
#include <set>
#include <string>

using namespace anari::math;

namespace {

// The frame parameters ptc tracks. color == ANARI_UNKNOWN means unset.
struct FrameState
{
  uint2 size;
  ANARIDataType color{ANARI_UNKNOWN};
  bool depth{false};
  bool callback{false};
};

struct Options
{
  ANARIDataType colorType{ANARI_UFIXED8_RGBA_SRGB};
  bool depth{false};
  bool resize{false};
  std::optional<ANARIDataType> firstColorType;
  bool unsetAll{false};
  bool uncommitted{false};
  bool callback{false};
  bool invertCheck{false};
  std::string expectLibrary;

  bool twoFrames() const
  {
    return resize || firstColorType || unsetAll;
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

// Status messages at WARNING or worse that mention channel.depth/color
int g_depthWarnings = 0;
int g_colorWarnings = 0;
// Status messages at ERROR or worse that mention channel.color
int g_colorErrors = 0;

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
  }
  if (sev <= ANARI_SEVERITY_ERROR)
    g_colorErrors += std::string(msg).find("channel.color") != std::string::npos;
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

bool parseArgs(int argc, char **argv, Options &opts)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    const bool hasValue = i + 1 < argc;
    if ((arg == "--color" || arg == "--first-color"
            || arg == "--expect-library")
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
    else if (arg == "--expect-library")
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

// Number of times kMpiExtension occurs in a null-terminated list; duplicates
// of any entry are reported through `c`.
int countMpiExtension(Checker &c, const char *const *list, const char *source)
{
  int count = 0;
  std::set<std::string> seen;
  for (auto e = list; e && *e; e++) {
    c.expect(seen.insert(*e).second,
        std::string(source) + " lists '" + *e + "' twice");
    count += std::string(*e) == kMpiExtension;
  }
  return count;
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
  c.expect(countMpiExtension(c, exts, "device property 'extension'") == 1,
      "device property 'extension' lacks " + std::string(kMpiExtension));

  int32_t version = -1, major = -1, minor = -1;
  c.expect(getDeviceProperty(d, "version", ANARI_INT32, version) && version >= 0,
      "device property 'version' not reported");
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
  c.expect(countMpiExtension(c, libExts, "library extension list") == 1,
      "library extension list lacks " + std::string(kMpiExtension));

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

anari::World buildWorld(anari::Device d, int rank, int size)
{
  // [0]: this rank's visible sphere; [1]: hidden behind the last rank's
  // sphere, at a different depth on each rank
  float3 pos[2] = {{sphereX(rank, size), 0.f, 0.f},
      {sphereX(size - 1, size), 0.f, -1.f - 0.5f * rank}};
  const int numSpheres = rank == size - 1 ? 1 : 2;
  float3 col = kColors[rank % kNumColors];
  auto geom = anari::newObject<anari::Geometry>(d, "sphere");
  anari::setParameterArray1D(d, geom, "vertex.position", pos, numSpheres);
  anari::setParameter(d, geom, "radius", kSphereRadius);
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

// Objects the frame references; kept alive so they can be set again after
// anariUnsetAllParameters.
struct Scene
{
  anari::World world;
  anari::Camera camera;
  anari::Renderer renderer;
};

Scene buildScene(anari::Device d, anari::World w)
{
  auto cam = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, cam, "position", float3(0, 0, kCameraZ));
  anari::setParameter(d, cam, "direction", float3(0, 0, -1));
  anari::setParameter(d, cam, "up", float3(0, 1, 0));
  anari::setParameter(d, cam, "height", kCameraHeight);
  anari::setParameter(d, cam, "aspect", kAspect);
  anari::commitParameters(d, cam);

  auto ren = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, ren, "background", float4(0, 0, 0, 0));
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
  int size{1};
  const Options *opts{nullptr};
  // the committed frame state of the render in flight, for rank 0's check
  const FrameState *rendering{nullptr};
  bool finalFrame{false};
  int calls{0};
  int failures{0};
};

CallbackContext g_callback;

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
  }
  anari::commitParameters(d, frame);
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

// Rank 0 only: probe the pixel at the center of each rank's sphere.
int checkColor(anari::Device d, anari::Frame frame, const Options &opts, int size)
{
  if (!compositable(opts.colorType) || opts.colorType == ANARI_UNKNOWN)
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
      float3 want = kColors[r % kNumColors];
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

// Rank 0 only, with --depth: the composited depth is the nearest hit across
// all ranks. Each rank's sphere center must show that sphere's depth: other
// ranks either miss there (their +inf must not win) or, behind the last
// rank's sphere, hit farther away. The background corner, where no rank hits,
// must be +inf.
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
  Checker c{ctx.rank};
  c.expect(d == ctx.device,
      "frame completion callback got a device handle that is not the ptc "
      "device");
  c.expect(f == ctx.frame,
      "frame completion callback got a frame handle that is not the ptc "
      "frame");
  if (d == ctx.device && f == ctx.frame && ctx.rank == 0 && ctx.rendering) {
    printf("callback %d: checking composited color\n", ctx.calls);
    c.failures += ctx.finalFrame
        ? checkColor(ctx.device, ctx.frame, *ctx.opts, ctx.size)
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
  anari::commitParameters(d, frame);
  const int before = g_callback.calls;
  anari::render(d, frame);
  anari::wait(d, frame);
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

  failures += checkDeviceConformance(lib, d, rank);

  const Scene scene = buildScene(d, buildWorld(d, rank, size));
  const FrameState finalState{
      kImageSize, opts.colorType, opts.depth, opts.callback};
  auto frame = anari::newObject<anari::Frame>(d);
  g_callback = {d, frame, rank, size, &opts};
  int renders = 0;

  if (opts.twoFrames()) {
    const FrameState first{opts.resize ? kImageSize / 2u : kImageSize,
        opts.firstColorType.value_or(opts.colorType),
        !opts.depth,
        opts.callback};
    setAllParameters(d, frame, scene, first);
    anari::commitParameters(d, frame);
    g_callback.rendering = &first;
    anari::render(d, frame);
    anari::wait(d, frame);
    renders++;
    g_callback.rendering = nullptr; // `first` goes out of scope
    if (rank == 0)
      failures += checkFirstFrame(d, frame, first);
    changeParameters(d, frame, scene, first, finalState, opts.unsetAll);
  } else {
    setAllParameters(d, frame, scene, finalState);
    anari::commitParameters(d, frame);
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
  anari::render(d, frame);
  anari::wait(d, frame);
  renders++;

  // Every rank: a render error comes through the status callback (the
  // process is still here, so nothing threw out of anariRenderFrame).
  {
    Checker c{rank};
    const bool wantError = !compositable(opts.colorType);
    const bool gotError = g_colorErrors > errorsBefore;
    c.expect(wantError == gotError,
        wantError ? "no ERROR about channel.color reported for an "
                    "uncompositable color type"
                  : "unexpected ERROR about channel.color reported");
    if (wantError && !c.failures)
      printf("PASS: rank %d: render error reported\n", rank);
    failures += c.failures;
  }

  if (rank == 0) {
    failures += checkColor(d, frame, opts, size);
    failures += opts.depth ? checkDepth(d, frame, size)
                           : checkDepthRefused(d, frame);
  }

  if (opts.callback)
    failures += checkCallbackCalls(d, frame, rank, renders);

  anari::release(d, frame);
  releaseScene(d, scene);
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
