# ds4-custom recipes

How ds4-custom got from 26.6 to ~60 tok/s single-slot decode (60.7 before upstream's FP4 accuracy fix, 59.7 after) with no accuracy loss, ~95 tok/s on code-edit replies, four SCR + DSpark slots in less memory, and much faster resumes of edited conversations. Each recipe records what changed, why, which resource was going unused, **and what it costs**. Like tuning parts in Armored Core, each change trades one resource for another, so every recipe carries a **Costs** line and the "Parts that interact" section records which changes reinforce or constrain each other. Costs come from the tradeoff census, which measures each kill switch for GPU memory, process footprint, CPU time per token, GPU busy time per token, energy (`macmon`: watts, joules per token, GPU clock, temperature), wasted work, prefill and time to first token, disk I/O and complexity. Costs still marked *(expected)* are reasoned from the design.

Every recipe below follows the same contract: **exact** (greedy token ids are identical, and each new kernel has a bitwise unit test against the old path, usually with a mutant that must fail), **switchable until proven** (each part lands behind a kill switch; once the census proves it, the switch and the old path are deleted, see Complexity), and **measured** (a real-model A/B in an unload window on the M5 Ultra, 256 GiB, DeepSeek V4.1 Flash 1017 IQ2_XXS/Q2_K, DSpark drafter, ctx 262144). Exceptions are named where they apply: upstream's FP4 accuracy fix (19) and SCR archive reuse (10) change outputs by design.

---

## Where the time was going

At the start (`85ba418`), single-slot decode ran at **26.6 tok/s**, measured on the CLI prompt with greedy decoding. One speculative cycle drafts up to 5 tokens with the DSpark drafter, then verifies them in one batched target step. The first stage-labelled profile of a 6-row verify showed:

| resource | state |
|---|---|
| GPU | **idle 38% of every verify** (~47 ms of ~125 ms): waiting on SSD reads and on a forced mid-step drain |
| SSD (Engram tables, 188 GiB on disk) | read synchronously on the CPU before any GPU work, ~3.7 ms per row |
| CPU encode thread | encoding layers 14–40 while the GPU sat idle behind a commit-and-wait |
| Memory bandwidth (1135 GB/s measured peak) | dense matvecs re-streamed weights once per verify row |
| GPU dispatch | thousands of tiny per-row dispatches (rope, copies, quantize) per verify |

The plausible theory, "dense weights re-streamed per row cost ~78 ms", turned out wrong. The kernels were already partly cache-served, and the real loss was idle GPU time. **Profile before optimizing.**

## 0. Measurement tooling

- **Stage timeline** (`92e6a28`): `DS4_METAL_ENCODER_TIMELINE=<file>` records every GPU encoder with start, end, gap, grid, stage label and kernel name, plus `S` lines for CPU stage starts. `timeline_report.py` summarizes GPU busy and idle time per verify width. Timeline mode inflates wall time, so take tok/s from a separate clean run.
- **Cycle accounting** (`96af7b6`, `62a14da`): `DS4_DSPARK_SPEC_LOG=1` logs both speculative and single-token cycles, and `DS4_DSPARK_TRACE=<file>` writes one JSON line per cycle (drafts, confidences, target argmax, timings). Check that the parts sum to wall time. The first estimate counted 77 cycles per 256 tokens; there were really 124, because the log hid the single-token cycles.
- **Bench** (`95aae3c`, `ce8a87d`): `ds4-bench --show-output` prints token ids for parity diffs; `--rendered-prompt-file` / `--ctx-prompt` run chat-templated prompts.
- **Profiling windows:** the real model cannot share GPU memory with the loaded live server; side-by-side runs hit `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Unload through llama-swap, run with the default `/tmp/ds4.lock`, then reload with a fixed greedy health request whose answer must match the baseline exactly.

## 1. Exact multi-row decode kernels (`9907941`, `ee078a1`)

- **Unused resource:** memory bandwidth and cache. Q8_0 and F16 matrix-vector kernels dispatched one threadgroup row per token, so a 6-row verify streamed each dense matrix 6 times.
- **Change:** row-tile kernels (Q8_0 tiles 2–8, F16 float4 tiles) load each weight block once and run the unchanged per-row reduction for each row. A grid planner (`DS4_METAL_DECODE_ROWS_MODE=auto|rows|interleave|off`) picks tiles by threadgroup count, so narrow shapes keep their occupancy.
- **Exactness:** 1,701 `memcmp` comparisons in `--decode-rows`. F16 equality depends on compiler behaviour under fast math (hoisting weights into registers broke it), so the tests are the guard.
- **Effect:** 26.6 → 34.8 tok/s; 6-row verify 116 → 98 ms.
- **Costs:** no memory or CPU cost *(expected)*. More kernel variants to maintain, and the F16 path's exactness is fragile under compiler changes. The `auto` grid plan was tuned on this machine, so other GPUs may need retuning.

## 2. Keep the GPU busy: background Engram reads + streaming command buffers (`cfa0a93`, `96af7b6`)

- **Unused resources:** GPU time (38% idle) and SSD/CPU parallelism.
  - Verify read both Engram tables for every row before encoding layer 0, although layer 1 needs only table 0 and layer 14 only table 1.
  - A layer-13 drain existed only because table 1's upload reused table 0's buffer.
  - Each step was one or two commit-and-wait command buffers, so the CPU encoded while the GPU idled.
- **Change:**
  - Engram reads run on background workers (`ds4_engram_async`) and are joined just before their layer.
  - Table 1 gets its own upload rows, so the drain is gone.
  - Every layer is committed without waiting (`ds4_gpu_flush_commands`), so the GPU runs layer *k* while the CPU encodes *k+1*.
  - The same applies to single-token decode, which DSpark uses for ~38% of cycles when confidence prunes the draft.
- **Exactness:** same bytes, kernels and order; only scheduling changed.
- **Effect:** GPU idle per verify ~47 → ~1 ms; 34.8 → 41.8 tok/s, then 42.0 → 45.9 tok/s with decode overlap; single-token decode 36.7 → 26.0 ms.
- **Switches:** graduated Nov 1 (old paths deleted); the flush interval is now fixed.
- **Costs** *(expected)*:
  - Background Engram threads use more CPU, but total CPU time per token should fall because the waits overlap.
  - The extra upload rows for table 1 cost a small amount of memory.
  - ~40 command buffers per step instead of 1–2 adds submission overhead, which is why flushing every 2+ layers measured worse.
  - The GPU stays busy, so power draw while generating rises, while energy per token should fall.

## 3. Fewer, larger dispatches with unchanged per-row arithmetic (`8d091e7`, `720b5cf`, `cd5d843`, `f83c8f8`)

- **Unused resource:** GPU dispatch capacity. Per-row rope, FP8, BF16 and copy steps meant ~800 rope and ~700 copy dispatches per verify. Several projections re-read their weights per row.
- **Change:**
  - One dispatch over all rows for rope/FP8/BF16/inverse rope. Window writes, indexer, gather and attention stay in row order; a test proves that hoisting the window writes *would* change results.
  - Multi-row tiles for `attn_out_low` and the drafter's F16 low projection.
  - Multi-row F32 matvecs for routers and drafter mixers.
- **Switches:** graduated Nov 1 (old paths deleted).
- **Costs:** close to free *(expected)*. A new invariant: anything that reads or writes the causal window must stay in row order. Merged scratch tensors are sized for the maximum row count.

## 4. Drafter: stop waiting on itself (`85bb6ad`)

- **Unused resource:** GPU time during drafting. The drafter encoded its three stages into one command buffer and the GPU idled while the CPU encoded.
- **Change:** commit without waiting, as in verify.
- **Effect:** draft 7.7 → 4.8 ms per cycle, together with item 3's drafter changes.
- **Switch:** graduated Nov 1.
- **Costs:** a few more command buffers per cycle *(expected)*.

## 5. Fuse BF16 rounding into the producing kernel (`4b7497d`, `85fca43`, `1f6b3dc`, `bc8e059`, `571318d`, `1524142`)

- **Unused resource:** dispatches and memory round-trips. The model rounds many intermediates to BF16, and each rounding was a separate tiny kernel that re-read and re-wrote the tensor: ~900 per verify, ~1.7 ms per cycle in total.
- **Change:** rounding happens in the writer of the Q8_0/F16/F32 matvecs, the RMS norm, HC weighted-sum/expand, swiglu and the FFN add.
- **Exactness:** each kernel has a bitwise test, and a mutant that drops the rounding fails it.
- **Effect:** about +3% aggregate across the two chunks.
- **Switches:** graduated Nov 1. The single-row Q8 writer measured 0.0% and was deleted; the others stay always on (the norm and F16/F32 writer are small alone but ~0.7% together).
- **Costs:** no memory cost; less memory traffic. More pipeline variants mean slightly longer Metal compile and load time *(expected)*. Each fused writer must keep the rounding point exactly where the model applies it.

## 6. Speculation policy from traces (`9f1f59f`; crate `dspark-policy-sim`)

- **Unused resource:** verify capacity. The fixed confidence threshold (0.6) often discarded a finished draft and paid ~26 ms for one token, when a 2-row verify costs ~33 ms and usually yields two.
- **Change:**
  - The stateless crate `dspark-policy-sim` replays full-cap traces against a cost model fitted from the traces.
  - Its prediction of the old policy came within 2.3% of measurement.
  - The winning rule keeps drafting while P(all drafts so far accepted) beats the next row's marginal verify cost, valued at 65 tok/s.
- **Exactness:** under greedy decoding any stopping rule yields identical tokens.
- **Effect:** +1.6% aggregate, +2.7% on agentic prompts.
- **Switch:** `DS4_DSPARK_GAIN=0`.
- **Costs:** **more wasted GPU work**. It verifies more draft rows that end up rejected, trading energy per token for latency. The census should count rejected rows per accepted token. Its value of 65 tok/s is tuned for this machine and drafter.

## 7. One scratch workspace for all sessions (`7dcca42`, `ed5a0f0`)

- **Unused resource:** memory. Each V4.1 session at ctx 262144 allocated 10.37 GiB of graph memory, 8.79 GiB of which is dead between engine calls (prefill batch rows, call-local scratch).
- **Change:**
  - With `--batched-session`, one engine-owned workspace is lent to whichever session is executing; sessions keep only their 1.58 GiB of persistent state.
  - An ownership guard turns any overlapping use into a hard error.
  - The deferred-decoder carry is reserved for its prefix call.
- **Exactness:** 20 checkpoints (tokens plus a bitwise logits hash) across 4 interleaved sessions match unshared and solo runs.
- **Effect:** 4 sessions need 15.1 GiB instead of 41.5 GiB of graph memory.
- **Switch:** graduated Nov 1 (always on with `--batched-session`).
- **Costs:** **serialization**. Only one session can execute an engine call at a time, so concurrent slots cannot overlap prefill and decode. Any future concurrency work (prefill fairness, M3 fused decode) must schedule around the guard or add a second workspace, buying concurrency back with ~8.8 GiB.

## 8. Four slots with SCR + DSpark (`7ec4e08`, `2fcbfe8`, `ef6e6b3`, `2011a74`, `fcd9c59`)

- **Unused resource:** the server could keep only one conversation warm, so every other caller evicted it.
- **Change:**
  - `--batched-session 4` works with `--scr --dspark`.
  - The checkpoint pool keeps a saved-state record per live archive and serializes pool calls.
  - Each generation runs the single-slot DSpark cycle in fair executor turns. Fused batch decode refuses SCR sessions today; making it SCR-aware is future work.
  - A Rust slot router (`adapter/cache-reuse/src/slot_router.rs`, C ABI `ds4_slot_route`, ~9 ns per decision) sends continuations back to the slot holding their conversation and places new callers on empty or least-valuable slots.
- **Effect:**
  - Outputs are identical to the single-slot server.
  - A continuation reused 10,896 of 11,152 tokens, first token in 1.4 s.
  - Total throughput: ~64.6 tok/s with 2 active, ~61 tok/s with 4.
- **Known limit:** a long SCR prefill blocks other slots (12 s at 7.9k tokens).
- **Router:** the legacy route was deleted Nov 1; recipe 21 added archive coverage.
- **Costs** (measured):
  - Each sealed SCR archive adds ~4.24 GB of memory: 166.9 GiB planned at 4 slots, ~193 GiB projected with 4 archives, out of 222.7 GiB.
  - With k active slots, each slot gets about 1/k of the decode throughput.
  - Fairness stalls behind long prefills.
  - Sampling waits on the inference lock.
  - Pool archive writes reach ~3.7 GB per save to the kv disk directory (SSD wear and space; capped by `--kv-disk-space-mb`).

## 9. Measure expert overlap before writing a grouping kernel (crate `moe-route-planner`)

- **Possibly unused resource:** memory bandwidth in routed MoE. Each verify row streams its 6 experts separately, so when rows pick the same expert its weights are read again. Routed MoE runs at 39–46% of peak bandwidth.
- **Change:** a stateless Rust crate that reads the server's routing trace (`DS4_V41_ROUTE_TRACE`, JSONL v1). It reports how often rows and slots share experts, then prices grouping strategies with a cost model fitted from the GPU timeline: gate/up 31.8 µs + 8.22 µs per (row, expert) pair, down 22.1 µs + 5.58 µs.
- **Why first:** with uniform routing, grouping saves ~1%; with correlated routing, ~20% (and ~39% across 4 slots). Only real traces can tell which case applies, and the kernel is worth writing only in the second.
- **Caveats:** at peak bandwidth an expert read already takes ~65% of the measured per-pair cost, so compute fractions above ~0.35 are non-physical. The down kernel's falling per-row cost suggests the cache already serves some repeats, so the predictions lean high.

## 10. Short follow-ups from the SCR archive (`c934bec`, live since Oct 31 22:15)

- **Unused resource:** sealed SCR archives. A request diverging within 128 tokens of the live transcript (regenerate, edit last turn) re-prefilled the whole conversation (~10.5 s at 7–11k tokens) even though an archive covered almost all of it.
- **Change:** such requests go through the archive planner, taking 0.7–1.1 s and reusing 7.0k–10.9k tokens.
- **Accuracy:** archive reuse is not fresh-prefill equivalent, so greedy output differs. It is the same class as the existing longer-tail reuse. Live since Oct 31 22:15: Michael decided that reusing the cache is the point of the cache. The opt-out switch was deleted Nov 1.
- **Costs** (measured): **accuracy class** (no longer fresh-equivalent); decode right after the reuse slows to ~40 tok/s, probably because the drafter has little fresh context; a pool archive read (~2.4 s select on a 41k-token conversation).

## 11. Routing trace (`1e1e05b`)

- `DS4_V41_ROUTE_TRACE=<file>` writes one JSONL line per target step (decode1, verify, batch) with each layer's 6 expert ids and weights, plus `t0_ms` and `t_ms`.
- A GPU copy into a host-visible buffer inside the existing command buffers means no extra waits; overhead is ~0.7%, and ids are identical with the trace on and off.
- **Costs** (measured): ~0.7% tok/s, two small host-visible buffers (~60 KB), and disk at ~13 KB per step (2 MiB per 158 steps measured live), about 1 GB per hour of continuous generation. It needs rotation if left on.
- Dropped from the live config Oct 31 23:25 once the planner study was done; the code stays, off by default.
- First real result: verify rows share experts heavily (1.58× read reduction at 6 rows), which motivates recipe 12, the grouped MoE kernel.

## 12. Grouped MoE: measured no-go (crate `moe-route-planner` + micro-bench `cb7d2f4`)

- **The hypothesis:** rows sharing experts re-read the same weights, so grouping saves bandwidth (predicted ~20%).
- **Measurement:** with real selections, a 2-row layer call takes 201 µs, against 204 µs with no sharing and 193 µs with every row sharing; at 6 rows 518 vs 527 vs 505. Exact prototypes that decode shared weights once were *slower* (608–657 µs at 6 rows). The GPU cache already serves most repeats, so the per-pair cost is dominated by dequantization and compute, not DRAM.
- **Lesson:** a cost model fitted from timing cannot tell bandwidth from compute; build the micro-bench before the kernel. The planner crate earned its keep by making the go/no-go cheap.

## 13. Router selection in one dispatch (`f37eb62`)

- **Unused resource:** dispatch capacity. Each layer's top-6-of-384 selection was several small kernels (argsort and friends).
- **Change:** a single fused kernel; 48 bitwise cases and 3 mutants.
- **Effect:** +2.8% tok/s, −2.5% J/token.
- **Costs:** none in memory; two old scratch buffers are now unused. It relies on the argsort thread count and on no reassociation.
- **Switch:** graduated Nov 1; reference runs use a test flag.

## 14. Verify indexer projections in one pass (`3c13d04`)

- **Unused resource:** memory bandwidth. Index-source layers re-read their query and weight projections per verify row.
- **Change:** one multi-row pass with the exact row kernels.
- **Effect:** +0.7%, −0.9% J/token.
- **Costs:** positions below 4 are projected and unused.
- **Switch:** graduated Nov 1.

## 15. Compile-time MoE kernel geometry (`cbfdeb0`)

- **Effect:** +0.3% end to end, although the micro-bench promised up to −5.6% at 1 row.
- **Costs:** ~230 lines of duplicated kernels, plus two pipelines built lazily.
- **Verdict:** kept at Michael's call and graduated Nov 1 (switch removed; upstream's generic kernels remain as the test reference).

## 16. Engram rows read concurrently (`b03e103`)

- **Unused resource:** SSD queue depth. The Engram file is opened with `F_NOCACHE`, and each token's 24 rows were read serially (~80 µs each), leaving a ~0.85 ms GPU hole after layer 0 in every window.
- **Change:** `dispatch_apply` reads the 24 rows of each (token, table) in parallel.
- **Effect:** +3.1% on all 6 prompts.
- **Costs:** +2.4 ms CPU per token on the system dispatch pool; same bytes read; energy within noise.
- **Switch:** graduated Nov 1.

## 17. Idle keepalive (`a1c0e7c`, `3823946`)

- **Unused resource:** CPU while idle. To keep the model's GPU residency warm, the old keepalive committed a tiny GPU kernel every second. Each commit spent ~0.24 s in the driver's submit call: ~25% of a core and ~2.8 W while idle. Without a keepalive, residency lapses within 2–3 s, and the next first token is ~0.85 s slower (1.2–1.7 s instead of ~0.47 s).
- **Change:** renew the model residency set (`requestResidency`) once per second instead of submitting work. While the set is still resident, a renewal costs ~0.02 ms; once it has lapsed, renewing re-establishes it at 0.2–3.8 s of system time.
- **The first version (`a1c0e7c`) failed live.** It slept a chain of twenty 50 ms `usleep`s per tick. Under launchd (llama-swap's children) those sleeps overran, stretching the period to ~2.2 s, so residency lapsed between renewals: 14–19% of a core. Terminal-launched test servers didn't overrun, so the test window showed 0%. `3823946` waits once on an absolute 1 s deadline (`pthread_cond_timedwait_relative_np` on `CLOCK_UPTIME_RAW`), so the period holds.
- **Verified live (`3823946`):** 0.00 CPU-seconds over 90 s idle, counting kernel time; health output identical.
- **Tradeoff table** (launchd test servers; CPU is per-thread system plus user):

  | mode | idle CPU | TTFT after 5 / 30 / 60 s idle |
  |---|---|---|
  | 1 s deadline (default) | 0% (0 of 160 renewals expensive) | 0.47 / 0.47 / 0.47 s |
  | `MS=2000` | 0–11% | 0.46 / 0.47 s |
  | `MS=3000` | 9–16% | 0.93 / 0.47 s |
  | `HOLD_S=20` (stop after 20 s idle) | 0% | 0.45 s within the hold; 1.25 s after it |
  | `MODE=off` | 0% | 1.2–1.7 s |
  | `MODE=submit` (original) | ~25% | 0.47–2.3 s |
  | chained sleeps (`a1c0e7c`) | 14–19% | 0.8–3.4 s |
- **Costs:** none measured: no new buffers, the same one thread, decode untouched. The deadline clock pauses during system sleep, so one tick fires late after a wake; that's harmless.
- **Lessons:** measure per-thread system plus user time (`ps -M`), not `%cpu`, and confirm on the live launch path. Prefer deadline waits over chained sleeps for anything periodic.
- **Switches:** trimmed Nov 1 to the fixed 1 s residency renewal. `DS4_METAL_QUEUE_KEEPALIVE_TRACE=1` remains for diagnosis, and upstream's `DS4_METAL_DISABLE_QUEUE_KEEPALIVE=1` turns it off. The table above records the alternatives that were measured and removed.

## 18. Speculation policy as a part (`30844b9`; `dspark-policy-sim` `9ba062e`)

- **Change:** the draft-stop rule is now `adapter/dspark-policy` (pure Rust, C ABI). The server and the simulator call the same function.
- **Exactness:** identical to the C rule on 7.95M decisions, including one-ulp probes at each break-even point; 5 mutants killed; identical ids and verify widths live.
- **Costs:** none measurable. ds4 builds now need the Rust crate.
- **Retune at today's speed:** predicted +0.1–0.2%, measured within noise (60.68 vs 60.71), so the tables are unchanged. Single-token decode sped up alongside verify, so the trade barely moved.
- **Switches:** `DS4_DSPARK_GAIN=0` (old threshold) stays as a tuning value. The C rule was removed Nov 1 and lives on as the test's frozen reference.

## 19. Upstream merge with the exact FP4 quantizer (`fde3301`, live Nov 1 00:55)

Merged 13 upstream commits with no conflicts. Only `97e3276` (exact divisions in the V4.1 FP4 activation quantizer) changes outputs; the other 12 are output-identical when that kernel alone is swapped back. A new `--fp4-rounding` test checks 524,288 values at and next to every rounding midpoint against a CPU reference and kills three mutant kernels. **Cost:** 59.7 vs 60.8 tok/s (−1.8%), entirely from DSpark acceptance on the changed token stream (2.70 vs 2.80 per cycle on 6 prompts); GPU time per verify width is identical, and J/token is within noise. Michael accepted the speed for accuracy. A division-free rewrite won nothing and was reverted. Complexity: +1 test mode, no switches.

## 20. Graduation cleanup (`cleanup-graduate` → `bbbf218`, live Nov 1 01:20)

Applied the graduation rule: every improvement that earned its place keeps its code and loses its switch and the old path. Removed 25 switches; non-test lines went from +96/−443 down to roughly −350 net. Deleted only the single-row Q8 BF16 writer (0.0% in two censuses). The norm and F16/F32 writer (each <0.5% alone, ~0.7% together) were restored always-on. The C draft-length rule became the test's frozen reference. Router and MoE-geometry reference runs go through upstream's test-flag hook, and `ds4_gpu_test_v41_specialized_count()` proves the flag took effect. Greedy ids are identical to the merged build on 6/6 prompts. `getenv("DS4_` count: ds4.c 459 → 445, ds4_metal.m 370 → 365. **Engine admission now counts the drafter** (`97e2f90`, +8.05 GiB in the plan line, with a host test).

## 21. Route by archive coverage (`633e1d7`, live Nov 1 01:20)

**Problem:** the slot router scored only the live prefix above the shared floor (~7.1k). Ren's prompt changes at token ~920 (the WM header), so every turn went to an empty slot, then a pool restore (~2 s live) and a rebase. **Fix:** the Rust slot router takes `archive_tokens` (the slot's SCR archive coverage, computed with the pool selector's scoring); a slot owns the request if coverage ≥ floor + min_match. **Result:** test turns stay on their slot, the pool select takes 1–3 ms with no restore, and the text is identical. **Cost:** a coverage check per request (1.3–1.6 ms); 16 router tests, 7/8 mutants killed (the eighth is equivalent). The rebase import (~1.5 s) remains; see the next recipe.

## 22. Cheaper rebase import (`de95f76`, `80a921f`, live Nov 1 ~01:25 CST)

**Why:** a non-continuation rebase writes a fresh archive so the old one stays valid until commit, so every reused token's 80 KB of KV rows is copied, wherever the first break falls. Breakdown for a 37k range: owner-row copies 0.21 s (one GPU sync per 128-token chunk), reading the old file 0.70 s (usually evicted by the next turn), regather 0.24 s, write 0.20 s. **Fix:** one GPU command per range for owner rows, plus `F_RDADVISE` read-ahead of the import ranges. **Result:** import 1.51–1.54 → 0.68–0.70 s, test TTFT 2.69–2.74 → 2.12–2.23 s, output identical over 4/4 turns, RSS and footprint unchanged. **Cost:** ~+52 non-test lines; the existing strict-host splice test kills two owner-copy mutants. Read-ahead is advisory, so it has no test. **No-go:** mmapping the old store (`bf60036`, reverted in `8cc5676`) took 17.2 s per import, because strided access on an evicted file re-faulted pages.

## 23. Context-copy drafts (`d6e0e6c`, `1456c0d`, live Nov 1 ~02:45 CST)

**Unused resource:** when a reply repeats text already in the context (code edits, renames, quoted files), the next tokens are already known, yet the DSpark forward still runs. **Change:** each cycle looks for the most recent earlier occurrence of the last 11 tokens plus the next token (a 12-token key). If the 5 tokens that followed it fill the draft depth, they replace the DSpark forward for that cycle; the target still verifies every row, and verify's capture keeps the drafter current. Skipped under exact sampling and tensor parallelism. Copy cycles stay out of the DSpark acceptance stats and traces. Idea from omlx PR #4363. **Result:** code-edit replies 79–82 → ~95 tok/s (+16–20%, 307/307 and 309/309 copied tokens accepted); other prompts −0.8% to +0.5%; ids identical in 36 bench comparisons plus 3 through the server. **Cost:** no memory, buffers or threads; CPU 19.1 → 18.9 ms/token; J/token slightly better; search ~11 µs/cycle at 40k (0.27 ms at 1M, an uncapped linear scan); a partly accepted copy costs one 6-row verify (~25–30 ms). Complexity +152/−12 over 5 files (34-line search header, 48 net lines in ds4.c, 61-line host test, 6/7 mutants killed plus a case for the 7th); switches `DS4_DSPARK_CONTEXT_COPY=0` and `_MIN`. **Fragility:** decode-loop exactness rests on window runs, with no model-free test at that level.

## Measured no-gos (chunk 8)

- **Tensor-core (Metal 4 matmul) verify rows** (`fe5c621`, branch `exp-simdgroup-mma`, Nov 1): routing dense Q8_0/F16 verify rows (2–8) through upstream's tensor matmul gave −30% tok/s (42.1 vs 59.75); it adds ~22 ms flat per cycle (too few threadgroups at small batch) and cuts the per-row slope only from 8.2 to 6.8 ms. Per extra verify row (8.2 ms): dense matmuls ~1.4 ms, routed MoE ~3.4–3.5 ms, attention ~1.4 ms, norms/copies/HC ~2 ms (timeline estimates). Routed MoE is the biggest per-row lever. Ceiling ~+5% even if perfect, and inexact, so closed.

- **Verify blit copies as compute:** exact, but 60.73 vs 60.73 tok/s. The overlapping command-buffer stream already hides that latency, and the timeline overstated it.
- **Attention-rows batching:** per-call cost is 38–43 µs, mostly fixed. It only pays if it removes many dispatches; the ring-window ordering makes that large and uncertain.

## Complexity: a part stat (added Oct 31)

Every part also carries a **Complexity** cost: non-test lines, new code paths and switches, any concurrency introduced, upstream conflict surface, and test burden.

Oct 31 totals: +4,015 / −303 non-test lines, ~2,080 test lines, ~37 new `DS4_*` switches. The BF16-fusion series is ~1,000 lines; its pieces measure small individually (0–0.6%) but ~0.75% together.

**Graduation rule:** once a part's switch proves it faster, or otherwise worth keeping, delete the old path and the switch, and keep the improvement. Real tradeoffs (decode-rows mode, tuning knobs) and off-by-default diagnostics may keep a switch.

## Tradeoff census (one run per kill switch, at `cbfdeb0`)

Historical: most of these switches graduated on Nov 1 (recipe 20), after a second census on the merged build confirmed the same ranking.

| switch off | tok/s | J/token |
|---|---|---|
| VERIFY_OVERLAP | −24% | — |
| LOW_ROWS | −2.8% | +8% |
| DECODE_OVERLAP | −1.9% | — |
| F32_ROWS | −1.6% | +2% |
| VERIFY_MERGE_ROWS | −1.6% | ~0 |
| DRAFT_LOW_ROWS | −1.2% | +5% |
| DSPARK_GAIN | −1.1% | — |
| FUSED_NORM | −0.6% | ~0 |
| DRAFT_FLUSH | −0.4% | +3% |
| FUSED_BF16 | noise | — |
| SHARED_WORKSPACE | noise (a memory win) | ~0 |
| DECODE_ROWS_MODE=off | −16.8% | +23% |
| ENGRAM_ASYNC_ROWS | −3.1% | noise |
| FUSED_HC | −0.9% | ~0 (CPU +1.3 ms/token) |
| FUSED_EW | −0.5% | +1% |
| FUSED_F16 / FUSED_Q8 | −0.2% | ~0 |

The Metal peak was 166.24 GiB in every run. Most speedups also save energy: finishing sooner beats the extra GPU busy time. Generation draws 88–97 W, and the GPU stays at its full 1619 MHz clock, so bursty submission does not lower clocks.

## Parts catalog: why decisions live in crates

Michael's principle: build each approach as a self-contained part, so it can be measured alone, its effect on the whole system estimated, and it can be swapped out if it doesn't work. In practice:
- **Decisions belong in stateless Rust crates** (input → output, a bench, unit tests, an FFI entry): the slot router (`ds4_slot_route`), and the speculation policy (`adapter/dspark-policy`, the same function `dspark-policy-sim` calls). Ideally the simulator and the server call the *same* function, so the simulator's prediction is the server's behaviour.
- **Analysis lives in offline crates** that read server traces: `dspark-policy-sim` for DSpark cycles, `moe-route-planner` for routing.
- **GPU kernels can't be crates**, so their "part boundary" is a kill switch, a bitwise test against the old kernel, and a cost table.
- **Every part's stat sheet** gives its benefit and costs (memory, CPU, energy, wasted work, constraints on other parts) under the same conditions, so parts can be compared and combined.

## Energy

Energy is a first-class axis alongside speed and memory. Two kinds of measurement complement each other:
- **Daily device energy** from a future power-management system or metered plug. This is the ground truth for what the server actually costs per day, including RAM, SSD, fans and power-supply losses.
- **Per-run on-chip readings** (`macmon` or `powermetrics`: CPU, GPU and RAM power plus GPU clock speed) to attribute that energy to specific changes. These come from on-chip models, so they suit on/off comparisons.

What to track per change: joules per generated token, joules per prefilled token, and GPU clock speed while busy. Bursty submission may keep the GPU below full clock, which could be part of the bandwidth gap. For the always-on server, also track the idle floor: model loaded and wired with slots warm, versus unloaded.

**Measured (Oct 31, `macmon`):**
- Generation draws 88–97 W at ~1.5 J per token, with the GPU at its full 1619 MHz clock, so bursty submission was not holding the clock down.
- Each part's J/token change is in the census. Turning decode rows off costs +23%; low rows, +8%.
- The idle floor with the model loaded now matches unloaded (~1.6–1.7 W on a quiet machine), after the keepalive fix (recipe 17). It was ~3 W higher before, about 70 Wh a day.

## Parts that interact

- **Shared workspace (7) vs concurrency** (prefill fairness, M3 fused decode): memory savings come from forbidding overlap, so the two pull against each other. The likely balance is a yield at prefill chunk boundaries rather than a second workspace.
- **Expected-gain policy (6) vs cheaper verify** (1–5, 12): every verify speedup lowers the marginal cost of a row, which makes longer drafts worthwhile. Retune the policy's value after each kernel win.
- **Grouped MoE (12) vs batch width:** grouping pays more the more rows share a step, so it compounds with longer drafts (6) and with cross-slot fused decode (M3: predicted 39% at 4 slots).
- **4 slots (8) vs memory headroom:** SCR archives grow with each warm slot. A second workspace, larger ctx or a bigger drafter all compete for the same ~30 GiB margin.
- **Routing trace (11) vs throughput:** turned off live on Oct 31 once the planner data was collected.
- **Context-copy drafts (23) vs the speculation policy (6):** copy cycles are kept out of the policy's traces, so `dspark-policy-sim` never sees them. Retune on drafter-only cycles.
- **Router (21) and rebase import (22) vs time to first token:** together they remove the pool restore and halve the import on every Ren turn; the remaining import is the target of the background-import plan.

## Results

| step | tok/s | notes |
|---|---|---|
| `85ba418` baseline | 26.6 | CLI prompt, decode rows off |
| multi-row kernels | 34.8 | CLI |
| verify overlap | 41.8 | CLI; 49.2 ms per cycle |
| decode overlap, merged rows, low rows | 46.5–47.6 | CLI; 43–44 ms per cycle |
| 6-prompt set, expected-gain policy | 52.8 aggregate / 55.6 agentic | from here on, 6 rendered chat prompts |
| draft flush, F32 rows, fused BF16 (part 1) | 55.6 / 58.5 | |
| fused BF16 (part 2) | 56.4 / 59.3 | json_tool 60.4, ren_toolcall 59.3, ren_agent 58.7, code 59.2, cli 52.8, summarize 51.0 |
| router, index rows, MoE geometry (W1–W3) | 58.9 | same-window reference 56.7 |
| Engram rows concurrent (W4) | **60.7** / agentic 63–65 | same-window reference 58.9 |
| upstream merge with the exact FP4 quantizer (Nov 1) | 59.7 | accuracy fix; −1.8% from DSpark acceptance, same GPU cost |
| graduation cleanup (Nov 1) | ~59.6 | within noise of the merge; 25 switches removed |
| context-copy drafts (Nov 1) | 6-prompt within noise; code edits 79–82 → ~95 | 9-prompt aggregate including edits 63.2 → 64.9 |

Where it stands against the hardware: single-token decode reads ~11.1 GB per token, so its floor is ~9.8 ms against 25.3 ms measured (39% of peak). Routed-MoE kernels run at 39–46% of peak. An earlier estimate that inter-encoder gaps took ~37% of GPU time overcounted overlapping encoders: real GPU idle is only 1–3% per window. Remaining exact gains must come from doing less work per kernel or fewer dispatches, not from filling gaps.

## Pending sections

- **E2 option 4, background rebase import (plan, needs Michael):** the current request decodes from stage/install, not the new archive, so the import (~0.7 s) only feeds the *next* turn. A helper thread would write whole interior records out of order (`pwrite` at final offset plus a completion bitmap; `committed_tokens` advances over the contiguous prefix) while the GPU runs gaps, suffix and decode. Seal at request end or before the next request on that slot. Owner rows stay synchronous (~0.01 s). Expected a further −0.6 to −0.8 s TTFT per Ren turn; 1–1.5 days. Risks: seal/retire ordering, and holding both archives through decode.
- **Other candidates:** prefill fairness (yield between layers in a wide prefill) · SCR-aware fused multi-slot decode (M3) · short-follow-up planner minimum-suffix recompute (needs an accuracy decision) · sampling semantics (`--mtp-exact-sampling` matters only above temperature 0; Michael now picks temperature per agent).
