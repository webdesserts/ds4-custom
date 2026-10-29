# Suffix Cache Reuse (SCR) Integration in ds4-custom

`ds4-custom` integrates Suffix Cache Reuse (SCR) on top of upstream [`antirez/ds4`](https://github.com/antirez/ds4) (`upstream/main` at `ee7e8405`), pairing native Metal KV/SWA archive execution with the standalone Rust planner **[`webdesserts/cache-reuse-utils`](https://github.com/webdesserts/cache-reuse-utils)** (`v0.2.0`, commit `0a290dce62fbcbb7c434fa60057d246934865645`).

Whereas standard prefix caching can only reuse tokens up to the first point where a prompt diverges, SCR reuses unchanged suffix and interior spans across mid-prompt edits (such as updating a pinned working memory block or folding a file section in the middle of an agent's context).

---

## Build & Host Verification

```sh
make -j 2 ds4-server ds4-bench      # builds adapter/cache-reuse (cargo -j 2) first
make -j 2 check-scr-host            # runs all 22 strict-host fixtures; never links Metal or loads a model
sh adapter/tests/run-ffi-test.sh    # standalone C ABI smoke test for cache-reuse-utils-ffi
sh tools/verify-pins.sh             # verifies Cargo.toml, Cargo.lock, and vendored C header pin
```

`make -j 2 check-scr-host` (`tests/run-scr-host.sh`) compiles and runs all 22 host-only SCR and server contract fixtures—including `tests/test_scr_actual_planner.c`, `tests/test_scr_encoder_residency.c`, and `tests/test_scr_checkpoint_pool.c`—against `tests/scr_host_storage.c` and `tests/scr_host_forbidden.c`, ensuring no GPU framework is linked and no model is loaded during automated unit verification.

---

## Architecture & Planner Boundary

### 1. Rust Planner & C ABI (`cache-reuse-utils` `v0.2.0`)
- Consumed through `adapter/cache-reuse/`, a thin `staticlib` crate pinned to `rev = "0a290dce62fbcbb7c434fa60057d246934865645"` that builds `libds4_cache_reuse.a`.
- `adapter/include/cache_reuse_utils.h` is the `cbindgen`-generated C header for that revision, verified by `tools/verify-pins.sh`.
- Candidate discovery, multi-range placement search (`K` in `1..=32`), zero-allocation pad materialization (`cache_reuse_materialize`), and logical/physical coordinate projection (`cache_reuse_project_logical`, `cache_reuse_logical_boundary`, `cache_reuse_common_prefix`) are delegated to the Rust crate across the C ABI.
- Target coordinate maps (`ds41_scr_coordinates`) store compact `uint32_t *pads` physical positions (4 bytes per pad). Historical pad tokens already present in the source archive are treated as ordinary token content; target coordinates record only new pads inserted by the current plan.

### 2. Native Metal Execution & Capture Pipeline
- **2,048-Token Capture Arena & Batched Refresh:** Pre-RoPE SWA raw capture arenas hold `2,048` tokens (`160 MiB`), allowing both first-turn `DS4_SCR_REFRESH` and rebase gap/suffix prefills to run in `2,048`-token layer-major chunks without intermediate output-head evaluations.
- **Encoder Residency Across Capture Flushes:** For spans of `>= 16,384` tokens, `ds41_encoder_acquire` locks the 20 encoder layers' experts into host memory once and holds residency across `2,048`-token raw arena flushes until fewer than `768` tokens remain.
- **Pipelined Range-Window Read-Ahead:** During multi-range rebase staging (`selected > 1`), the engine prefetches range `i + 1`'s SWA raw window from disk while range `i` executes its GPU RoPE/scatter kernels.

### 3. Persistent SCR Checkpoint Pool (`--scr --kv-disk-dir DIR`)
- When `--scr` is combined with `--kv-disk-dir DIR`, the server maintains a persistent pool of up to 16 sealed SCR archives on disk (bounded by `--kv-disk-space-mb`, defaulting to 32 GiB when `0` or unset).
- Each pool entry is stored under a 64-bit content hash as three atomically renamed files:
  - `<id>.scr.meta`: header (`DS4SCR01`), context/archive geometry, generation and coordinate identity, timestamps, token array, and `uint32_t` pad positions.
  - `<id>.scr.owners`: sealed compressed KV and index owner rows (`6,400` bytes per token).
  - `<id>.scr.raw`: canonical pre-RoPE SWA records, created via an zero-copy APFS `fclonefileat` clone when `TMPDIR` and `--kv-disk-dir` share a volume (with a chunked copy fallback).
- On startup, the server loads a lightweight metadata-only catalog in memory without touching the GPU.
- Before any planned SCR request that does not simply extend the live transcript, the server scores the hot GPU archive alongside all catalog entries using `cache_reuse_plan` (`covered_tokens`). If a pooled checkpoint covers more tokens than the hot archive (for example, on the first turn after a server restart or when switching between conversation branches), the server saves the current hot archive and restores the winning checkpoint before executing the multi-range splice.

---

## Server CLI & Per-Request Reuse Mode

- **`--scr` Production Stack:**
  - Passing `--scr` (or a nonzero `--scr-tail-tokens C`) enables the full SCR stack: logical coordinates, batched capture, planner fresh-gap space padding, archive capacity defaulting to `--ctx`, `--scr-max-ranges` defaulting to `32`, and `--scr-planner-budget-ms` defaulting to `5000` (`0..60000`).
  - Optional tuning flags: `--scr-tail-tokens C` (`128..1048576`, 8-token aligned, `<= ctx`), `--scr-max-ranges K` (`1..32`), `--scr-planner-budget-ms MS` (`0..60000`), `--kv-disk-dir DIR`, and `--kv-disk-space-mb MB`.
- **Per-Request `"reuse"` Mode (`"scr" | "exact"`):**
  - `/v1/chat/completions` accepts an optional `"reuse": "scr" | "exact"` field.
  - Under `--scr`, `"scr"` is the default and runs the multi-range planner and checkpoint pool.
  - `"exact"` bypasses the SCR planner and pool for that request, reusing only an exact live prefix so callers can perform clean A/B comparisons on the same server instance.

---

## Reuse Telemetry (`ds4.reuse.v1`)

Each request under `--scr` emits one compact `ds4.reuse.v1` record (written to `$DS4_SCR_METADATA_DIR/reuse.v1.jsonl` when configured, and streamed in-band before `[DONE]` when `"stream_options": {"include_ds4_scr": true}` is set):

```json
{"v":1,"schema":"ds4.reuse.v1","request":"chatcmpl-…","slot":0,"seq":2,
 "tokens":{"prompt":5956,"prepared":5957,"reused":5588,"live_prefix":0,"fresh":369,"pad":1,"discovered_overlap":5588,"generated":16},
 "ms":{"queue":0.054,"plan":0.079,"stage":12.604,"install":1.003,"prefill":1339.071,"capture":40.541,"head":1.819,"first_token":2038.647,"decode":597.769,"total":2636.42},
 "plan":{"status":"complete","k":32,"range_count":1,"ranges":[{"src":346,"dst":360,"n":5588}],"source":{"kind":"archive","generation":3,"tokens":5952}},
 "archive":{"capacity":32768,"occupancy":5968,"action":"rebased","device_mib":794.8,"pool_restored":false,"pool_entries":1},
 "decode":{"steps":16,"tps":26.77},
 "error":null}
```

- `plan.ranges` contains the exact `0..K` selected ranges without fixed-slot padding.
- `tokens.discovered_overlap` reports the longest match discovered before range selection.
- `archive.pool_restored` and `archive.pool_entries` appear when `--kv-disk-dir` is active, reporting whether the source archive was restored from the disk checkpoint pool and the catalog entry count at emission time.

---

## Metal Benchmark on Apple M5 Ultra (256 GiB)

Measured with `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` (`--backend metal --ctx 32768 --scr --kv-disk-dir`) on a ~5,950-token structured agent prompt where Turn 2 edits a mid-prompt working-memory section (~245 tokens into the prompt) while keeping the trailing ~5,588 tokens of file sections unchanged:

| Scenario | Reuse Mode | Prompt Tokens | Reused | Fresh | `ms.plan` | `ms.stage` | `ms.prefill` | `ms.first_token` | `ms.total` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold capture (`refresh`) | `scr` | 5,943 | 0 | 5,943 | — | — | 10,056.5 ms | 10,120.0 ms | 10,720.2 ms |
| Cold baseline (no capture) | `exact` | 5,943 | 0 | 5,943 | — | — | 10,119.1 ms | 10,119.4 ms | 10,718.7 ms |
| Mid-prompt edit (prefix-only) | `exact` | 5,956 | 0 | 5,956 | — | — | **9,395.0 ms** | **9,395.2 ms** | **9,991.9 ms** |
| Mid-prompt edit (1-range splice) | `scr` | 5,956 | **5,588** | **369** | **0.079 ms** | **12.6 ms** | **1,339.1 ms (7.0×)** | **2,038.6 ms (4.6×)** | **2,636.4 ms (3.8×)** |
| First turn after server restart (`pool_restored: true`, 2 ranges) | `scr` | 5,927 | **5,826** | **101** | **0.087 ms** | **16.7 ms** | **943.9 ms (10.7×)** | **1,210.2 ms (8.4×)** | **1,776.8 ms (6.0×)** |

---

## Source Layout

- `ds4_scr.h` (included from `ds4.h`): public SCR session API, checkpoint pool declarations, Engram/archive observation counters, and planner budget parser.
- `ds4_scr_engine.inc`: native SCR engine implementation included by section macro in `ds4.c` (*Archive & Raw Store*, *Coordinates & Prefix*, *Rebase Transaction*, *Multi-Range Execution & Sync*, *Planner Adapter & Logical Policy*, *Checkpoint Pool*, and *Session Guards*).
- `ds4_server_scr.h`: server integration included by section macro in `ds4_server.c` (*Types*, *Guards*, *Runtime*, *Transport*, and *Configuration*).
- `adapter/`: Rust `staticlib` wrapper crate, `cbindgen` header `adapter/include/cache_reuse_utils.h`, and C ABI smoke tests.
