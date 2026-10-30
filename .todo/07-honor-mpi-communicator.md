# 07: Honor mpiCommunicator

**What to build:** An application can set `mpiCommunicator` on a frame and have that frame composite over the given communicator, with rank 0 *of that communicator* receiving the image. Today every frame binds to `MPI_COMM_WORLD` at construction. The compositor must be created (or recreated) when the parameter is committed, and its collective setup has to happen in lockstep across the ranks of the new communicator.

**Convention (decided 2026-10-30):** the spec types this parameter as `VOID_POINTER`. The app passes a *pointer to an `MPI_Comm`* (`&comm`), and ptc dereferences it at frame commit. This is portable across MPICH, where `MPI_Comm` is an int, and OpenMPI, where it is a pointer. The pointed-to `MPI_Comm` only has to stay valid until the commit returns.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [x] Convention decided (pointer to `MPI_Comm`)
- [ ] Convention documented in code/README where the parameter is read
- [ ] Default remains `MPI_COMM_WORLD` when unset
- [ ] Changing the communicator between frames works; the old compositor is torn down collectively
- [ ] Mapping on a non-root rank of the communicator follows the spec (contents undefined, no crash)
- [ ] Harness test: 4 ranks split into two 2-rank sub-communicators, each composites its own image correctly
