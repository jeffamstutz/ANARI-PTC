# 06: Frame completion callback fires after compositing

**What to build:** An application that sets `frameCompletionCallback` on a frame is called back once the *composited* image is ready, and receives the ptc device and ptc frame handles. Today the parameter is forwarded to the wrapped frame. The callback therefore fires before compositing, with the wrapped device's handles, which the app has never seen.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] The callback is intercepted, not forwarded to the wrapped frame
- [ ] It fires after compositing completes, once per rendered frame
- [ ] It receives the app's user data, the ptc device handle and the ptc frame handle
- [ ] Unsetting the callback stops further calls
- [ ] Harness verifies handles and invocation count
