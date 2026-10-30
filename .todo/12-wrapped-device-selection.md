# 12: Wrapped-device selection and load failure

**What to build:** An application can choose the wrapped ANARI library with a device parameter, set and committed before the first object is created. The debug device's `wrappedDevice` is precedent; a library-name string is likely simpler. The `PTC_ANARI_LIBRARY` env var still works and overrides the default. If the wrapped library or device fails to load, the app gets a fatal status message and API calls fail gracefully instead of dereferencing a null wrapped device.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] Device parameter for the wrapped library/subtype, documented, with precedence over the env var defined
- [ ] Falling back to helide is reported clearly (or removed, if a parameter makes it unnecessary)
- [ ] With a nonexistent wrapped library, no API call crashes; object creation returns null and errors are reported
- [ ] Harness tests: selection via parameter; failure case
- [ ] Wrapped library/device load failure is reported as `ANARI_SEVERITY_FATAL_ERROR`, not ERROR (decided 2026-10-30, followups/08 item 3)
