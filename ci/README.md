# UCXX CI scripts

The scripts in this directory are run by UCXX's GitHub Actions workflows (see
`.github/workflows/`), using the RAPIDS CI images (`rapidsai/ci-conda`,
`rapidsai/ci-wheel`, `rapidsai/citestwheel`) and the `rapids-*` helpers they provide.

## Building and testing against a non-conda/non-wheel UCX (`UCXX_NO_LIBUCX`)

By default, UCXX conda packages depend on the conda-forge `ucx` package and UCXX wheels
depend on the `libucx-cu{12,13}` wheels. External projects that want to run UCXX's CI
against their own UCX installation (for example, a UCX built from source) can set:

```bash
export UCXX_NO_LIBUCX=true
```

and run the CI scripts in this directory unmodified. When unset, behavior is unchanged.

> [!IMPORTANT]
> This is intended for **testing UCXX against a specific UCX installation** in a
> controlled environment. Packages built this way have no dependency on UCX at all and
> only work on machines where a compatible UCX is installed exactly as it was at build
> time. They must not be published.

### What the switch does

| Component | Effect of `UCXX_NO_LIBUCX=true` |
|---|---|
| `dependencies.yaml` | Every `rapids-dependency-file-generator` call in these scripts passes `no_libucx=true`, selecting the existing empty `no_libucx` matrix entries, so no `ucx` / `libucx-cu*` requirement is generated. |
| `build_wheel_libucxx.sh` | Exports `RAPIDS_MATRIX_ENTRY="cuda_suffixed=true;no_libucx=true"` so `rapids-build-backend` omits `libucx-cu*` from the wheel's build and runtime dependencies. |
| `conda/recipes/{libucxx,ucxx}` | Drop `ucx` from `host`/`run` requirements and allowlist `libucp`, `libucs`, `libuct` and `libucm` in the overlinking check (`missing_dso_allowlist`). |

The value must be exactly `true`. Any other value (`1`, `yes`, `True`, ...) is silently
treated as unset. Set it consistently for **all** jobs (builds and tests).

### Requirements on the UCX installation

- **Conda builds: UCX must be installed under prefix `/usr/local`** (UCX's own
  `./configure` default). See [Conda limitations](#conda-limitations) for why other
  locations, including `/usr`, do not work.
- **Wheel builds:** UCX must be discoverable by CMake's default search (prefixes
  `/usr/local` or `/usr` were verified for the `libucxx` wheel; see
  [Wheel limitations](#wheel-limitations)).
- **Runtime:** UCX libraries must be found by the dynamic loader without help from UCXX,
  i.e., registered with `ldconfig` (e.g., via `/etc/ld.so.conf.d/`) or present in
  `LD_LIBRARY_PATH`. Neither conda packages nor wheels record UCX's location.
- **`ucx_info` must be on `PATH`** in conda test jobs (`test_cpp.sh`, `test_python.sh`,
  `test_python_distributed.sh`), which call it unconditionally to print the UCX
  configuration. Wheel test jobs fall back to querying the version from Python.
- **For conda builds, UCX should be built against glibc 2.28 or older** (the conda
  sysroot version). See [Conda limitations](#conda-limitations).

### Conda limitations

Conda packages are built by `rattler-build` with the conda-forge compiler toolchain,
which compiles against its own sysroot rather than the host system. UCX discovery only
works by side effect of CMake defaults, which leads to the following:

1. **UCX installed with prefix `/usr` does not work.** CMake finds
   `/usr/lib*/cmake/ucx`, but always strips `/usr/include` from compile lines because it is
   an implicit include directory (`Platform/UnixPaths.cmake`). The conda compiler does not
   search the host's `/usr/include`, so the build fails with
   `fatal error: ucp/api/ucp.h: No such file or directory`.
2. **UCX installed in a prefix outside CMake's default search path (e.g., `/opt/ucx`,
   `$HOME/ucx`) is expected not to work** (untested). `find_package(ucx REQUIRED)` only
   searches CMake's system prefixes (`/usr/local`, `/usr`, `/`), and the conda compiler
   activation overwrites `CMAKE_PREFIX_PATH`. `rattler-build` runs builds in isolated
   environments that only receive variables explicitly forwarded by the recipe's `env:`
   sections, and the recipes do not forward `ucx_ROOT`/`ucx_DIR`, so there is currently
   no way to point the build at another location.
3. **No UCX version constraint.** The `ucx_min`/`ucx_max` bounds in
   `conda_build_config.yaml` are not applied, and CMake does not require a specific UCX
   version, so any UCX found is accepted at build time. This includes UCX 1.22.0, which
   regular builds exclude because it can segfault
   ([#783](https://github.com/rapidsai/ucxx/issues/783)). Only UCX 1.20.0 was tested.
4. **No UCX location is recorded.** `rattler-build` removes rpaths pointing outside the
   conda prefix (e.g., `/usr/local/lib64`), so UCX must be found by the loader at runtime
   (see [Requirements](#requirements-on-the-ucx-installation)).
5. **A conda `ucx` package can silently take precedence.** The built packages do not
   depend on or conflict with conda's `ucx`. If anything else in the environment pulls in
   `ucx`, it is installed alongside, and `libucxx.so` will likely resolve `libucp.so.0`
   from the conda environment (searched via `$ORIGIN` rpath) instead of the system
   installation (untested). Verify with `ucx_info -v` and
   `ldd $CONDA_PREFIX/lib/libucxx.so` in test logs.
6. **glibc compatibility is not checked.** Packages declare `__glibc >=2.28`, matching
   the conda sysroot. A system UCX built against a newer glibc may fail to link, or yield
   packages whose real glibc requirement is higher than declared (untested).

### Wheel limitations

Wheels are built with the CI image's system compiler (gcc-toolset), not a sysroot, so the
`/usr` include problem above does not apply.

1. **UCX discovery relies on CMake's default search path** (`/usr/local`, `/usr`).
   Other prefixes are untested; `build_wheel_libucxx.sh` overwrites `SKBUILD_CMAKE_ARGS`,
   so UCX's location cannot be passed that way, though CMake also honors a
   `CMAKE_PREFIX_PATH` environment variable.
2. **The wheels are not portable.** They carry manylinux tags but require a system UCX at
   runtime; `auditwheel repair` already excludes `libucp.so.0`, so UCX is not bundled.
3. **No UCX version constraint**, as for conda (including the excluded UCX 1.22.0).
4. With UCX at prefix `/usr`, only the `libucxx` wheel build was verified; the `ucxx` and
   `distributed-ucxx` wheel builds and wheel tests were verified with `/usr/local` only.

### Test environment notes

These are not specific to `UCXX_NO_LIBUCX`, but are easy to hit when reproducing CI
outside GitHub Actions:

- `rapidsai/ci-wheel` images enable the required compiler (gcc-toolset) via
  `/etc/profile.d/enable_devtools.sh`, which is only sourced by login shells. Without it,
  the system's GCC is used and the C++ build fails.
- Scripts download artifacts from earlier jobs with `rapids-download-from-github` and
  configure S3-backed sccache with `rapids-configure-sccache`; outside GitHub Actions these
  need to be replaced or stubbed.
