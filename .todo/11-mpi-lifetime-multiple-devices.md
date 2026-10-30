# 11: MPI lifetime with multiple devices

**What to build:** When the app hasn't initialized MPI and ptc does it privately, MPI stays initialized until the *last* ptc device is released. Today only the first device is counted, so releasing any device can call `MPI_Finalize` while another device still has frames. When the app initialized MPI itself, ptc never finalizes it. Device release is collective per the spec.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] Every device created while MPI is privately initialized is counted
- [ ] `MPI_Finalize` runs only when the last such device is destroyed
- [ ] App-initialized MPI is never finalized by ptc
- [ ] Harness test: two devices, release the first, keep rendering with the second
- [ ] Harness test: app doesn't call `MPI_Init`; device init/teardown succeeds
