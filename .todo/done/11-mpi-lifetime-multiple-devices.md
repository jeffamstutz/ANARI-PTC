# 11: MPI lifetime with multiple devices

**What to build:** When the app hasn't initialized MPI and ptc does it privately, MPI stays initialized until the *last* ptc device is released. Today only the first device is counted, so releasing any device can call `MPI_Finalize` while another device still has frames. When the app initialized MPI itself, ptc never finalizes it. Device release is collective per the spec.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** done

- [x] Every device created while MPI is privately initialized is counted
- [x] `MPI_Finalize` runs only when the last such device is destroyed
- [x] App-initialized MPI is never finalized by ptc
- [x] Harness test: two devices, release the first, keep rendering with the second
- [x] Harness test: app doesn't call `MPI_Init`; device init/teardown succeeds
- [x] Asymmetric failure doesn't hang: before compositing, ranks agree on success with a one-int `MPI_Allreduce`. If any rank failed in `allocate()`/`composite()`, every rank reports an error and skips the collectives (decided 2026-10-30, followups/08 item 3)
- [x] Harness test: query a device property (which triggers `initDevice()` and possibly a private `MPI_Init`), then release the device; teardown is clean (decided 2026-10-30, followups/03 item 2)
