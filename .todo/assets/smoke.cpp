// Multi-rank smoke test for the ptc device: rank r renders one sphere at a
// distinct x position with a distinct color; rank 0 checks that every sphere
// shows up in the composited image and that depth is sane.
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/linalg.h>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace anari::math;

static void statusFunc(const void *, ANARIDevice, ANARIObject, ANARIDataType,
    ANARIStatusSeverity sev, ANARIStatusCode, const char *msg)
{
  if (sev <= ANARI_SEVERITY_WARNING)
    fprintf(stderr, "[anari] %s\n", msg);
}

static const float3 kColors[] = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char *colorTypeStr = argc > 1 ? argv[1] : "ufixed8";
  ANARIDataType colorType = std::string(colorTypeStr) == "float32"
      ? ANARI_FLOAT32_VEC4
      : ANARI_UFIXED8_RGBA_SRGB;
  const bool wantDepth = argc > 2 && std::string(argv[2]) == "depth";

  auto lib = anari::loadLibrary("ptc", statusFunc);
  auto d = anari::newDevice(lib, "default");

  // extension query (spec: device must report KHR_DATA_PARALLEL_MPI)
  const char *const *exts = nullptr;
  bool hasMpiExt = false;
  if (anariGetProperty(d, d, "extension", ANARI_STRING_LIST, &exts,
          sizeof(exts), ANARI_WAIT)
      && exts) {
    for (auto e = exts; *e; e++)
      hasMpiExt |= std::string(*e) == "KHR_DATA_PARALLEL_MPI";
  }

  // one sphere per rank, spread along x, all at the same depth
  float x = size == 1 ? 0.f : -1.f + 2.f * rank / (size - 1);
  float3 pos(x, 0.f, 0.f);
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

  const uint2 imgSize(128, 64);
  auto cam = anari::newObject<anari::Camera>(d, "orthographic");
  anari::setParameter(d, cam, "position", float3(0, 0, 5));
  anari::setParameter(d, cam, "direction", float3(0, 0, -1));
  anari::setParameter(d, cam, "up", float3(0, 1, 0));
  anari::setParameter(d, cam, "height", 1.5f);
  anari::setParameter(d, cam, "aspect", float(imgSize.x) / imgSize.y);
  anari::commitParameters(d, cam);

  auto ren = anari::newObject<anari::Renderer>(d, "default");
  anari::setParameter(d, ren, "background", float4(0, 0, 0, 0));
  anari::setParameter(d, ren, "ambientRadiance", 1.f);
  anari::commitParameters(d, ren);

  auto frame = anari::newObject<anari::Frame>(d);
  anari::setParameter(d, frame, "size", imgSize);
  anari::setParameter(d, frame, "channel.color", colorType);
  if (wantDepth)
    anari::setParameter(d, frame, "channel.depth", ANARI_FLOAT32);
  anari::setAndReleaseParameter(d, frame, "world", world);
  anari::setAndReleaseParameter(d, frame, "camera", cam);
  anari::setAndReleaseParameter(d, frame, "renderer", ren);
  anari::commitParameters(d, frame);

  anari::render(d, frame);
  anari::wait(d, frame);

  int failures = 0;
  if (rank == 0) {
    if (!hasMpiExt) {
      printf("FAIL: device does not report KHR_DATA_PARALLEL_MPI\n");
      failures++;
    }
    auto fb = anari::map<void>(d, frame, "channel.color");
    if (!fb.data || fb.width != imgSize.x || fb.height != imgSize.y
        || fb.pixelType != colorType) {
      printf("FAIL: bad color map (data=%p %ux%u type=%d)\n", fb.data,
          fb.width, fb.height, int(fb.pixelType));
      failures++;
    } else {
      // probe the pixel at the center of each rank's sphere
      for (int r = 0; r < size; r++) {
        float rx = size == 1 ? 0.f : -1.f + 2.f * r / (size - 1);
        float aspect = float(imgSize.x) / imgSize.y;
        int px = int((rx / (1.5f * aspect) + 0.5f) * imgSize.x);
        int py = imgSize.y / 2;
        float c[4];
        size_t idx = size_t(py) * imgSize.x + px;
        if (colorType == ANARI_FLOAT32_VEC4) {
          memcpy(c, (const float *)fb.data + 4 * idx, sizeof(c));
        } else {
          uint32_t v = ((const uint32_t *)fb.data)[idx];
          for (int i = 0; i < 4; i++)
            c[i] = ((v >> (8 * i)) & 0xff) / 255.f;
        }
        float3 want = kColors[r % 4];
        bool ok = c[3] > 0.9f;
        for (int i = 0; i < 3; i++)
          ok &= (want[i] > 0.5f) == (c[i] > 0.1f);
        printf("%s: rank %d sphere @ px %d -> (%.2f %.2f %.2f %.2f)\n",
            ok ? "PASS" : "FAIL", r, px, c[0], c[1], c[2], c[3]);
        failures += !ok;
      }
    }
    anari::unmap(d, frame, "channel.color");

    if (wantDepth) {
      auto db = anari::map<float>(d, frame, "channel.depth");
      if (!db.data || db.pixelType != ANARI_FLOAT32) {
        printf("FAIL: bad depth map\n");
        failures++;
      } else {
        // rank 0's sphere front is at z=0.3, camera at z=5 -> depth ~4.7
        float aspect = float(imgSize.x) / imgSize.y;
        float r0x = size == 1 ? 0.f : -1.f;
        int px = int((r0x / (1.5f * aspect) + 0.5f) * imgSize.x);
        float hit = db.data[(imgSize.y / 2) * imgSize.x + px];
        float corner = db.data[0];
        bool ok = hit > 4.f && hit < 5.f
            && (std::isinf(corner) || corner > 1e19f);
        printf("%s: depth hit=%f corner=%g\n", ok ? "PASS" : "FAIL", hit,
            corner);
        failures += !ok;
      }
      anari::unmap(d, frame, "channel.depth");
    }
    printf(failures ? "SMOKE FAILED (%d)\n" : "SMOKE OK\n", failures);
  }

  anari::release(d, frame);
  anari::release(d, d);
  anari::unloadLibrary(lib);
  MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return failures ? 1 : 0;
}
