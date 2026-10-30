# 03: Device reports KHR_DATA_PARALLEL_MPI and required properties

**What to build:** An application querying the ptc device sees it as a spec-conforming ANARI 1.1 device that implements `KHR_DATA_PARALLEL_MPI`. Today the device-level `anariGetProperty` returns 0 for everything and the library returns no device extensions. Required device properties should be forwarded from the wrapped device, and `KHR_DATA_PARALLEL_MPI` should be added to its extension list. The object/parameter query API should advertise the `mpiCommunicator` parameter on `ANARI_FRAME`.

Spec notes: `anariGetProperty` is a collective operation under this extension.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] Device `extension` (STRING_LIST) is the wrapped device's list plus `KHR_DATA_PARALLEL_MPI`, without duplicates
- [ ] `version`, `anariVersion.major`, `anariVersion.minor`, `geometryMaxIndex` return valid values
- [ ] The library's device-extension query includes `KHR_DATA_PARALLEL_MPI`
- [ ] `anariGetParameterInfo` for `ANARI_FRAME` / `mpiCommunicator` returns info
- [ ] `anari::extension` helpers (`anariGetDeviceExtensionStruct`) report the extension as present
- [ ] Harness checks the extension on all ranks
