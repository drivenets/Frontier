# rocprofv3 ROCTX marker-format probe — jobs 21736, 21737

Real-GPU probes (not part of the collector) answering the open `b_traces.py` question: what does
`rocprofv3 --marker-trace` actually emit for ROCTX ranges, and does the collector's current
`marker_runtime.py` dlopen strategy actually produce them? Node `amd-mi355x-3`, `--gres=gpu:1`, inside
the approved image (`sha256:dece6767...`), via `docker run --entrypoint bash ... rocprofv3 --kernel-trace
--hip-runtime-trace --marker-trace -f csv -- python3 probe_rocprofv3_format.py <mode>`. Workload: 3×
`torch.randn` (bf16, 256×256) then 3× `a @ b` inside one push/pop range.

## Finding 1 (job 21736): `marker_runtime.py`'s current dlopen target produces zero marker events

`ctypes.CDLL("libroctx64.so")` (the library `marker_runtime.MarkerBuffer` dlopens today) returns valid
function pointers and `roctxRangePushA`/`roctxRangePop` both return `0` (success) — but rocprofv3 never
logs `MARKER (ROCTx) ... initialized` and never writes a marker trace file, even though `--marker-trace`
was passed. Only `kernel_trace.csv`, `hip_api_trace.csv`, `agent_info.csv` were produced (job 21736,
confirmed again for job 21737's `local` and `global` sub-runs with `RTLD_LOCAL` and `RTLD_GLOBAL`
respectively — both silently produce nothing).

## Finding 2 (job 21737, `sdk` mode): `librocprofiler-sdk-roctx.so` is the library rocprofv3 instruments

Dlopening `librocprofiler-sdk-roctx.so` directly (same `roctxRangePushA`/`roctxRangePop` symbols, ABI
drop-in) makes rocprofv3 log `MARKER (ROCTx) version 1.1.0 initialized` and emit
`<n>_marker_api_trace.csv`. Both `.so` files are present in the image (confirmed in the prior cluster
inspection's `tools.txt`); only the `rocprofiler-sdk` one is the one the tool actually hooks in this
toolchain (ROCm 7.2.3 / rocprofiler-sdk 1.1.0).

## Marker CSV schema (`<job>_marker_api_trace.csv`)

Same 7 columns as `hip_api_trace.csv`: `Domain,Function,Process_Id,Thread_Id,Correlation_Id,
Start_Timestamp,End_Timestamp`. One complete push/pop range is **one row** — `Domain` is
`MARKER_CORE_RANGE_API`, `Function` carries the exact label text passed to `roctxRangePushA`
(confirmed verbatim: `"B|scope|probe_sdk|0|test"`), `Start_Timestamp`/`End_Timestamp` are the push/pop
instants. This is **not** a discrete push-event/pop-event stream needing stack-based pairing (that is
what `b_traces.pair_markers` was written for, apparently anticipating a different dialect) — each range
arrives pre-paired. `Correlation_Id` is a single globally-unique counter shared across every domain in
the trace (HIP API, kernel dispatch, and marker calls all draw from the same sequence) and is suitable
as `marker_id`.

## Attribution chain verified against real data

Parsed all three CSVs with Python's `csv` module (naive `awk -F','` breaks — kernel names contain
literal unescaped commas inside quoted fields). Result, exactly matching `b_extract.py`'s designed
attribution chain:

- 6 `hipLaunchKernel`/`hipExtModuleLaunchKernel` calls total in the whole trace (3 `torch.randn` RNG
  kernel launches + 3 real `a @ b` GEMM launches).
- Exactly the 3 GEMM launches (`hipExtModuleLaunchKernel`, correlation IDs 5173/5193/5213) have their
  **host-side** `[Start_Timestamp, End_Timestamp]` fully contained inside the marker row's
  `[1003815454899983, 1003817417630766]` window; the 3 pre-marker `randn` launches do not.
- `Correlation_Id` joins `hip_api_trace.csv` → `kernel_trace.csv` exactly as `b_traces.rocprof_csv`
  already assumes: correlation 5173/5193/5213 each match exactly one kernel dispatch row, each named
  `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x128_MI1...` — a hipBLASLt/Tensile GEMM kernel,
  consistent with E5's `tgemm.mm`/hipBLASLt GEMM path.

This is real, demonstrated eager attribution end-to-end on this exact toolchain (dispatch → correlation
→ host launch → marker-range containment), not a synthetic fixture. It does not touch graph replay
(still open) and uses a trivial non-SGLang workload (one GEMM, not a real forward).

## Fixes required in the existing collector (made immediately after this probe, same session)

1. `marker_runtime.py`: `ctypes.CDLL("libroctx64.so")` → `ctypes.CDLL("librocprofiler-sdk-roctx.so")`.
2. `b_traces.py`: add a `rocprof_marker_csv()` adapter that reads `*_marker_api_trace.csv` rows directly
   into the `ranges` record shape (one row = one already-paired range), instead of routing through
   `pair_markers()`, which assumes a push/pop event-stream dialect this tool does not emit in CSV mode.

## What this does not establish

- Graph-replay marker/correlation identity (untested here; still an open item for REPORT.md §10.3 /
  `IMPLEMENTATION_STATUS.md` blocker 4).
- Any SGLang call-site behavior — this is a standalone GEMM, not a served forward.
- Multi-threaded marker/thread-ID semantics (this probe's process had one active thread).
- rocpd-format marker/graph schema (still unexamined; CSV only).

## Paths

- Remote: `/opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/evidence/rocprofv3-format-probe-21736/` and `.../rocprofv3-format-probe2-21737/`
- Local (this directory and its sibling `../rocprofv3-format-probe-21736/`)
- Probe script: `dataset_b/probe_rocprofv3_format.py` (not part of the collector; a one-off diagnostic, kept for reproducibility)
