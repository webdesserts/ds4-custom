/* Suffix cache reuse (SCR) declarations for DeepSeek V4.1 Flash, in three
 * parts:
 *
 *   - Observation counters and the planner budget parser: self-contained,
 *     usable without ds4.h (the Engram reader includes only this part).
 *   - DS4_SCR_OBSERVATION_CLOCK: the monotonic clock helper, for translation
 *     units that already select POSIX clock_gettime.
 *   - The public session API, emitted once ds4.h has declared the session,
 *     token and payload types it builds on (ds4.h includes this header after
 *     those declarations). */
#ifndef DS4_SCR_H
#define DS4_SCR_H

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Numeric observations only. Counters are cumulative within one session graph;
 * a caller-owned per-call result can be merged after its readers have joined. */
typedef struct {
    bool enabled, counters_overflow, clock_failed, time_overflow;
    uint64_t calls, batch_calls, failed_calls, row_attempts, rows_completed;
    uint64_t pread_calls, read_bytes, row_read_nanoseconds, call_wall_nanoseconds;
} ds4_engram_observation;

static inline void ds4_observation_add_u64(uint64_t *sum, uint64_t value, bool *overflow) {
    if (value > UINT64_MAX - *sum) { *sum = UINT64_MAX; *overflow = true; }
    else *sum += value;
}

static inline void ds4_engram_observation_add(ds4_engram_observation *sum,
                                               const ds4_engram_observation *part) {
    if (!sum || !part || !part->enabled) return;
    sum->enabled = true;
    sum->counters_overflow |= part->counters_overflow;
    sum->clock_failed |= part->clock_failed;
    sum->time_overflow |= part->time_overflow;
#define DS4_OBS_ADD(field) ds4_observation_add_u64(&sum->field, part->field, &sum->counters_overflow)
    DS4_OBS_ADD(calls); DS4_OBS_ADD(batch_calls); DS4_OBS_ADD(failed_calls);
    DS4_OBS_ADD(row_attempts); DS4_OBS_ADD(rows_completed); DS4_OBS_ADD(pread_calls); DS4_OBS_ADD(read_bytes);
#undef DS4_OBS_ADD
    ds4_observation_add_u64(&sum->row_read_nanoseconds, part->row_read_nanoseconds, &sum->time_overflow);
    ds4_observation_add_u64(&sum->call_wall_nanoseconds, part->call_wall_nanoseconds, &sum->time_overflow);
}

/* Last checked allocation is not a fresh filesystem sample or physical-device
 * traffic. owner_id is process-local and never derived from a pointer/generation. */
typedef struct {
    bool known, raw_present, allocation_known, allocation_time_known;
    bool descriptor_present, identity_known, identity_lost, readable, writable;
    bool retirement_attempted, retirement_complete, retiring, counters_overflow, observation_time_known;
    bool raw_timing_unknown;
    uint64_t observed_nanoseconds;
    uint64_t owner_id, accounted_bytes, file_bytes, allocated_bytes, read_bytes, written_bytes;
    uint64_t allocation_checked_nanoseconds;
    uint32_t cleanup_state;
    int close_error, cleanup_error;
    double raw_readback_seconds, raw_write_seconds, raw_read_seconds, raw_gather_seconds;
} ds4_scr_owner_observation;

/* Successfully discharged owners only: add these to current active/extra
 * counters, never to last-retirement snapshots of the same discharged owners. */
typedef struct {
    bool known, counters_overflow, timing_unknown;
    uint64_t owners, read_bytes, written_bytes;
    double raw_readback_seconds, raw_write_seconds, raw_read_seconds, raw_gather_seconds;
} ds4_scr_closed_observation;

/* Bit inspection remains valid in the engine's fast-math build. */
static inline bool ds4_observation_seconds_valid(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000) &&
        (!(bits >> 63) || bits == UINT64_C(0x8000000000000000));
}

#define DS4_SCR_PLANNER_DEFAULT_BUDGET_MS 5000u
static inline int ds4_scr_parse_planner_budget(const char *text, uint32_t *out) {
    if (!text || !text[0] || !out) return 1;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end || value < 0 || value > 60000 || value > UINT32_MAX) return 1;
    *out = (uint32_t)value;
    return 0;
}
#endif /* DS4_SCR_H */

#if defined(DS4_SCR_OBSERVATION_CLOCK) && !defined(DS4_SCR_OBSERVATION_CLOCK_DONE)
#define DS4_SCR_OBSERVATION_CLOCK_DONE
#include <time.h>

/* An observer failure must never replace the model/storage operation's errno. */
static inline bool ds4_observation_clock(uint64_t *nanoseconds) {
    int saved = errno;
    struct timespec t;
    bool ok = clock_gettime(CLOCK_MONOTONIC, &t) == 0 && t.tv_sec >= 0 &&
        t.tv_nsec >= 0 && t.tv_nsec < 1000000000 &&
        (uint64_t)t.tv_sec <= (UINT64_MAX - (uint64_t)t.tv_nsec) / UINT64_C(1000000000);
    if (ok) *nanoseconds = (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
    errno = saved;
    return ok;
}
#endif

#if defined(DS4_H) && !defined(DS4_SCR_SESSION_API)
#define DS4_SCR_SESSION_API

/* Experimental canonical Flash capture, disabled by default. Exclusive session
 * use only; single-device text-only Metal. Enable before any source computation
 * (empty graph/checkpoint), with an 8-aligned capacity in [128,65536], <=context.
 * Returns 0 on success, 1 on unsupported/admission/allocation/refusal. Capture
 * copy errors propagate through normal evaluation; missing coverage only refuses
 * sealing. No operation changes live tokens, logits or checkpoint validity.
 *
 * Seal one immutable tail ending at the complete checkpoint frontier. Start/end
 * must be 8-aligned, count>=128, and exact tail token identity/coverage is checked.
 * The private capsule is tied to this session/model and survives graph reset/load;
 * mutable coverage does not. It is not a restored splice or a disk payload.
 * bytes reports the coherent total SCR device/host reservation: bank+capsule
 * at enable, plus lazy reusable staging/scratch after materialization, all admitted.
 * Reset/load invalidates staging but keeps storage and the sealed source.
 * Release/destruction drains device work and frees all owned SCR storage.
 * No mutable tensor handle escapes the session lifetime. */
/* Largest eight-aligned source below the native physical transcript ceiling.
 * Eligibility does not replace context, memory or disk admission. */
#define DS4_SCR_ARCHIVE_MAX_CAPACITY (1048576u - 8u)
/* Explicit canonical archive, disabled by default; exclusive text-only single-device
 * Metal session. Prepare reserves [start,start+capacity) on an empty graph, both
 * bounds 8-aligned, capacity in [128,DS4_SCR_ARCHIVE_MAX_CAPACITY] and within
 * context. Refuses legacy SCR
 * coexistence. Owned hot/stage/scratch/capture/I/O buffers and metadata are admitted
 * before computation; bytes excludes opaque driver overhead, file cache and RSS.
 * Seal publishes an 8-aligned count in [128,capacity], with the exact completed
 * native checkpoint at start+count, matching tokens and complete owner/all40-layer
 * raw coverage. Reserved but unpublished positions are never reusable source.
 * Stage selects an interior source slice using exact native tokens, length>=128,
 * 8-aligned source/destination/count and a nonempty later destination suffix.
 * It transforms immutable canonical originals directly at final positions, with
 * windows from the SELECTED end. Session cancellation before/during cold reads
 * refuses staging without retiring a readable immutable source. An I/O fault
 * fences unreadable source state for checked retirement. No live cache/frontier/
 * logit/checkpoint mutation, prefix readiness, installation, inference, matcher
 * or public tensor handles.
 * Reset/load clears mutable coverage/staging but retains certified source lineage.
 * Release returns failure on active commands/failed drain, retains full ownership
 * and charge, and fences reuse until successful release or terminal destruction.
 * Raw-backed archives require successful checked release/disposal before terminal
 * graph destruction; unresolved ownership must not be silently abandoned.
 * All return 0 on success, 1 on refusal/error; bytes returns 0 if absent. */
int ds4_session_scr_archive_prepare(ds4_session *s, uint32_t start, uint32_t capacity);
/* Complete bounded capture intervals before advancing beyond the 128-row raw
 * arena. Requires the actual valid headed frontier; publishes file records only,
 * not a reusable source or logits. Owning refresh drives this automatically.
 * Returns DS4_SESSION_SYNC_INTERRUPTED on cancellation. */
int ds4_session_scr_archive_flush(ds4_session *s);
int ds4_session_scr_archive_seal(ds4_session *s, const int *tokens, uint32_t count);
int ds4_session_scr_archive_stage(ds4_session *s, uint32_t source_start,
                                uint32_t destination_start, const int *tokens, uint32_t count);
uint64_t ds4_session_scr_archive_bytes(const ds4_session *s);
int ds4_session_scr_archive_release(ds4_session *s);

typedef struct {
    bool presence_known, present, sealed, lost, retiring;
    uint32_t start, end, capacity, sealed_tokens;
    uint64_t generation, source_generation, accounted_bytes;
    bool append_armed, append_incompatible, append_blocked, append_saturated, history_hole;
    bool pending_known, io_state_known, raw_readable, raw_writable;
    uint32_t certified_captured_tokens, pending_tokens, backend_written_tokens, backend_committed_tokens;
    uint64_t file_bytes, unpublished_file_bytes;
    bool extra_owner_known, extra_owner_present, extra_owner_retiring;
    uint64_t extra_owner_accounted_bytes;
    bool provenance_known, rebased;
    uint32_t published_imported_tokens, published_computed_tokens, rebase_head_end;
    uint32_t predecessor_start, predecessor_end, predecessor_capacity;
    uint64_t predecessor_generation, predecessor_coordinate_identity;
    /* Optional content-free observations at the same exclusive safe boundary.
     * Repeated cumulative values are not independent work. No raw I/O occurs. */
    bool observation_known, observation_time_known;
    uint64_t observed_nanoseconds;
    ds4_scr_owner_observation active_io, extra_io, retired_io[2];
    ds4_scr_closed_observation closed_io;
    ds4_engram_observation engram_io;
} ds4_scr_archive_snapshot;
/* Scalar host metadata only: no synchronization, tensor reads or inference.
 * Caller owns exclusive session use and must exclude server quarantine.
 * Unknown presence/provenance is not absence or zero coverage; sealed state is
 * not request eligibility. Computed coverage includes assembly and later append;
 * imported canonical history is not fresh-reference-equivalent computation. */
ds4_scr_archive_snapshot ds4_session_scr_archive_snapshot(const ds4_session *s);

typedef enum {
    DS4_SCR_RETIRE_INVALID, DS4_SCR_RETIRE_ABSENT, DS4_SCR_RETIRE_RELEASED,
    DS4_SCR_RETIRE_ALREADY_FENCED, DS4_SCR_RETIRE_FAILED_RETAINED
} ds4_scr_retire_outcome;
typedef struct {
    ds4_scr_retire_outcome outcome;
    bool attempted, charge_known;
    uint64_t charged_bytes_before, charged_bytes_after;
} ds4_scr_retire_receipt;
/* Caller owns exclusive session use. Refuses foreign/live/borrowed ownership
 * before mutation. A fenced archive is never retried; explicit release remains
 * available for separately authorized retries. Charges are logical reservations,
 * not physical reclamation. INVALID does not establish archive absence. */
ds4_scr_retire_outcome ds4_session_scr_archive_retire_once(
    ds4_session *s, ds4_scr_retire_receipt *receipt);

typedef enum {
    DS4_SCR_REFRESH_PREFLIGHT, DS4_SCR_REFRESH_RETIRE, DS4_SCR_REFRESH_PREPARE,
    DS4_SCR_REFRESH_PREFIX, DS4_SCR_REFRESH_SOURCE, DS4_SCR_REFRESH_SEAL,
    DS4_SCR_REFRESH_REMAINDER, DS4_SCR_REFRESH_COMPLETE
} ds4_scr_refresh_phase;
typedef enum {
    DS4_SCR_REFRESH_INVALID, DS4_SCR_REFRESH_UNSUPPORTED,
    DS4_SCR_REFRESH_RESOURCE, DS4_SCR_REFRESH_RELEASE,
    DS4_SCR_REFRESH_COMPUTE, DS4_SCR_REFRESH_CANCELLED, DS4_SCR_REFRESH_OK
} ds4_scr_refresh_reason;
typedef struct {
    ds4_scr_refresh_phase phase;
    ds4_scr_refresh_reason reason;
    uint32_t requested_tokens, source_start, source_end, source_count;
    uint32_t reserved_capacity, published_end, published_count;
    uint32_t normal_calls, source_chunks;
    uint32_t prefix_completed, source_completed, remainder_completed;
    uint32_t completed_heads;
    bool work_unknown, head_unknown, complete_n;
    bool target_mutated; /* False for admission-only refusal. */
    bool old_retirement_attempted, old_retired, prepared, sealed;
    bool failed_source_retirement_attempted, failed_source_retired;
    bool source_usable, source_retiring;
    uint64_t temporary_host_bytes, publication_growth_bytes;
    uint64_t prospective_archive_bytes, source_charged_bytes;
    uint64_t engine_charged_bytes_before, engine_charged_bytes_after_retirement;
    uint64_t engine_charged_bytes_final;
    double retire_seconds, prepare_seconds, prefix_seconds, source_seconds;
    double seal_seconds, remainder_seconds, complete_seconds, cleanup_seconds, total_seconds;
    int error_code;
} ds4_scr_refresh_receipt;
/* Explicit dense-source producer, not matching/policy or a fallback. Exclusive
 * nonreentrant text-only single-device Metal Flash use: caller excludes sampling,
 * save/copy_logits and cancel-callback observation throughout. Full native input
 * (including session accessor input) is owned before mutation. [start,start+count)
 * is eight-aligned, count128..DS4_SCR_ARCHIVE_MAX_CAPACITY, within N/context
 * with generation room. Public refresh reserves exactly count; private owning
 * dense/even calls may admit more capacity for continuous compatible append.
 * Retires old source deliberately: no rollback. Failed release retains full
 * charge/pointer/retiring fence and stops, never retries. Fresh prepare precedes
 * normal optimized prefix once, monotonic <=128 source appends, genuine all-owner/
 * all40 seal at M, then actual remainder once. Normal intermediate heads and valid
 * checkpoints are REAL, unlike private multi-sync; callbacks are suppressed and
 * restored, cancellation remains enabled. Normal timings include heads/capture;
 * receipt work is per-call proven full-model deltas, not display/deferred rows.
 * Success requires full N identity/frontier/head AND requested usable sealed
 * source. Receipt source_count/end describe the initial seal; published_count/end
 * describe actual final coverage. Failure invalidates the affected destination;
 * a certified sealed prefix is retained with its writer blocked, otherwise new
 * unsealed ownership is explicitly retired. No replay or hidden fresh retry.
 * Returns 0 on success, 1 on error, or DS4_SESSION_SYNC_INTERRUPTED. */
int ds4_session_scr_archive_refresh(ds4_session *s, const ds4_tokens *request,
                                  uint32_t start, uint32_t count,
                                  ds4_scr_refresh_receipt *receipt,
                                  char *err, size_t errlen);

typedef enum {
    DS4_SCR_ERROR, DS4_SCR_CANCELLED, DS4_SCR_REUSED, DS4_SCR_FALLBACK
} ds4_scr_outcome;
typedef enum {
    DS4_SCR_REASON_NONE, DS4_SCR_REASON_NO_CANDIDATE,
    DS4_SCR_REASON_UNSUPPORTED, DS4_SCR_REASON_MISMATCH,
    DS4_SCR_REASON_INVALID, DS4_SCR_REASON_RESOURCE, DS4_SCR_REASON_COMPUTE
} ds4_scr_reason;
/* Explicit 0..MAX ranges from one pinned archive, absolute native positions.
 * Count is validated before descriptor access; count0 means fresh recompute.
 * Experimental receipt layout grows with this bound; rebuild all consumers. */
#define DS4_SCR_MULTI_MAX_RANGES 32u /* == CACHE_REUSE_MAX_RANGES */
typedef struct {
    uint32_t source_start, destination_start, count;
} ds4_scr_range;
typedef enum {
    DS4_SCR_MULTI_PREFLIGHT, DS4_SCR_MULTI_PREFIX, DS4_SCR_MULTI_STAGE,
    DS4_SCR_MULTI_INSTALL, DS4_SCR_MULTI_GAP, DS4_SCR_MULTI_SUFFIX,
    DS4_SCR_MULTI_HEAD, DS4_SCR_MULTI_PUBLICATION, DS4_SCR_MULTI_FRESH
} ds4_scr_multi_phase;
typedef struct {
    ds4_scr_outcome outcome;
    ds4_scr_reason reason;
    ds4_scr_multi_phase phase;
    uint32_t requested_tokens, input_spans, normalized_spans;
    ds4_scr_range ranges[DS4_SCR_MULTI_MAX_RANGES];
    uint32_t attempted_spans, materialized_spans, installed_spans, committed_spans;
    uint32_t attempted_tokens, materialized_tokens, installed_tokens, committed_tokens;
    /* Materialized ranges whose raw window was read ahead during the previous gap. */
    uint32_t prefetched_spans;
    /* Gap i follows normalized range i; final and unused slots stay zero. */
    uint32_t prefix_completed, gap_completed[DS4_SCR_MULTI_MAX_RANGES], suffix_completed, fresh_completed;
    uint32_t discarded_tokens;
    bool partial_work_unknown, head_completed, published;
    bool target_mutated; /* Destination failure versus pre-effect refusal. */
    /* Extra host target/logits/descriptor storage plus checkpoint capacity
     * growth; archive/device reservation remains independently fully charged. */
    uint64_t temporary_host_bytes, publication_growth_bytes, archive_bytes;
    uint32_t planned_spans, planned_tokens, covered_head_end, imported_tokens;
    bool rebase_prepared, covered_head_completed, rebase_committed, pending_owner_retained;
    double rebase_prepare_seconds, import_seconds, covered_head_seconds;
    double prefix_seconds, stage_seconds, install_seconds, gap_seconds;
    double suffix_seconds, fresh_seconds, head_seconds, publication_seconds, total_seconds;
} ds4_scr_multi_receipt;
/* Exclusive, nonreentrant, single-device text-only Metal Flash operation.
 * Caller excludes sampling/save/copy_logits and cancel-callback observation.
 * Owns target/descriptors/private final logits before mutation. Public validity
 * remains false until ONE full publication; no intermediate heads/callbacks.
 * Malformed/foreign/retiring inputs and unsupported runtimes error before reset.
 * Each input must independently meet native identity/min128/eight alignment;
 * only then can source-AND-target adjacent chains compact (no fragment rescue).
 * Valid candidate refusal takes one private fresh recompute. Post-mutation
 * failures invalidate destination, retain source and known work, never retry.
 * Installed rows are not computed work; committed counts require publication.
 * No automatic source seal/refresh/release, matching, or fresh edit parity. */
int ds4_session_scr_multi_sync(ds4_session *s, const ds4_tokens *request,
                              const ds4_scr_range *ranges, uint32_t range_count,
                              ds4_scr_multi_receipt *receipt, char *err, size_t errlen);
/* Separate experimental native-island policy. No server default or fresh-edit
 * equivalence: live continuation can retain prior SCR-conditioned state.
 * Same exclusive observation/cancel-callback exclusions as multi/refresh. */
#define DS4_SCR_DENSE_COMPARISONS_MAX 16777216u
#define DS4_SCR_DENSE_RANGES_MAX 4096u
typedef enum {
    DS4_SCR_DENSE_ERROR, DS4_SCR_DENSE_CONTINUE, DS4_SCR_DENSE_FRESH,
    DS4_SCR_DENSE_REUSE, DS4_SCR_DENSE_REFRESH
} ds4_scr_dense_route;
typedef enum {
    DS4_SCR_DENSE_INVALID, DS4_SCR_DENSE_LIVE, DS4_SCR_DENSE_CUTOFF,
    DS4_SCR_DENSE_HIT, DS4_SCR_DENSE_MISSING, DS4_SCR_DENSE_MISS,
    DS4_SCR_DENSE_BUDGET, DS4_SCR_DENSE_SHORT, DS4_SCR_DENSE_CANCELLED,
    DS4_SCR_DENSE_RESOURCE, DS4_SCR_DENSE_UNSUPPORTED,
    DS4_SCR_DENSE_DIAGNOSTIC_FRESH /* 11: actual one-shot no-capture attempt. */
} ds4_scr_dense_reason;
/* The planner's selected ranges (planner coordinates: archive source, logical
 * destination); selected_ranges entries of plan are meaningful. */
typedef struct {
    uint32_t selected_ranges;
    ds4_scr_range plan[DS4_SCR_MULTI_MAX_RANGES];
} ds4_scr_dense_match_receipt;
typedef struct {
    uint32_t attempted, completed, certified, min_completed, max_completed;
} ds4_scr_suffix_counts;
typedef struct {
    ds4_scr_dense_route route;
    ds4_scr_dense_reason reason, trigger;
    uint32_t requested_tokens, configured_cap, configured_k, native_lcp, retained_tokens;
    uint32_t source_start, source_end;
    uint64_t source_generation, outer_host_bytes, peak_outer_host_bytes;
    uint32_t continuation_completed, continuation_heads;
    bool continuation_work_unknown, continuation_head_unknown, partial_valid, complete_n;
    ds4_scr_dense_match_receipt match;
    /* Inner counters/timers are actual and overlap outer total_seconds. */
    ds4_scr_multi_receipt multi;
    ds4_scr_refresh_receipt refresh;
    bool suffix_counts_known;
    ds4_scr_suffix_counts suffix_counts;
    double total_seconds;
    int error_code;
} ds4_scr_dense_policy_receipt;
/* Effective receipt authority value. Selected at startup, never by request
 * evidence; 0 means no insertion authority (no startup selection). */
enum { DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1 = 2 };
typedef enum {
    DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE = 1
} ds4_scr_padding_policy;
/* Startup only, before any checkpoint/archive/coordinate state. Resolves and
 * verifies one ordinary ASCII-space token from the loaded vocab; that token
 * replaces any logical-policy pad_token argument. Without this selection the
 * logical policy grants no artificial-token placement. */
int ds4_session_scr_set_padding_policy(ds4_session *s, ds4_scr_padding_policy policy,
                                      char *err, size_t errlen);
/* Selected fast capture body: known unequal logits, no formal quality claim.
 * Empty startup only. */
int ds4_session_scr_enable_capture_batch_chunks(ds4_session *s, char *err, size_t errlen);
/* Experimental RAM-only logical request entry. Counts/identity are immutable
 * prefill receipts; generation may subsequently advance the session map.
 * execution and physical_plan use physical coordinates, logical_match does not.
 * A false physical_known means unknown, never an admitted zero-length target.
 * No normal server route selects this until its coordinate protocol is enabled. */
typedef struct {
    ds4_scr_dense_policy_receipt execution;
    ds4_scr_dense_match_receipt logical_match;
    ds4_scr_range physical_plan[DS4_SCR_MULTI_MAX_RANGES];
    uint32_t logical_tokens, physical_tokens, new_pads, carried_pads;
    uint32_t imported_pads, useful_committed, placed_useful, planned_ranges;
    uint32_t boundary_policy; /* Effective startup insertion authority. */
    uint64_t map_identity, planning_work, planning_comparisons, peak_host_bytes;
    /* Planner result: best_useful is the best useful score before K/alignment
     * trims; covered is what the selected ranges reuse. Discovery stats are
     * the matches found and the longest one, valid when discovery_known. */
    uint32_t planner_status, planner_best_useful, planner_admissible_min;
    uint32_t planner_covered, planner_padding;
    uint32_t planner_discovered_matches, planner_longest_match;
    uint64_t planner_elapsed_ns, planner_preparation_ns, planner_discovery_ns, planner_search_ns;
    bool logical_known, physical_known, planned, published, target_mutated, planner_status_known;
    bool planner_discovery_known, planner_search_known, planner_timing_known;
} ds4_scr_logical_receipt;
int ds4_session_scr_logical_policy(ds4_session *s, const ds4_tokens *logical_request,
                                  uint32_t archive_cap, uint32_t max_ranges, int pad_token,
                                  ds4_scr_logical_receipt *receipt, char *err, size_t errlen);

/* Persistent checkpoint pool: up to DS4_SCR_POOL_MAX_ENTRIES sealed archives
 * kept in one directory as <name>.scr.meta (header, tokens, pads), .owners
 * (sealed compressed/index owner rows) and .raw (canonical SWA records), so a
 * restarted server or an alternating branch can splice from an earlier turn.
 * The catalog holds metadata only; device memory is used by the one restored
 * hot archive. The caller serializes all pool use with its session (the
 * server's inference lock). Files are written as .tmp and renamed, the meta
 * file last; terminal SWA windows are rebuilt from raw records when staged. */
#define DS4_SCR_POOL_MAX_ENTRIES 16u
#define DS4_SCR_POOL_DEFAULT_BYTES (UINT64_C(32) << 30)
typedef struct ds4_scr_pool ds4_scr_pool;
typedef struct {
    bool selected;       /* Candidates were scored for this request. */
    bool restored;       /* A pool checkpoint replaced the hot archive. */
    bool saved;          /* The hot archive was written as a new entry. */
    uint32_t entries;    /* Catalog size after the operation. */
    uint32_t candidates, hot_covered, best_covered;
    uint64_t written_bytes;
    double seconds;
} ds4_scr_pool_receipt;
/* Scans `directory` (which must exist) for complete checkpoints of this
 * context and archive capacity; other files are left alone, stale pool
 * temporaries and orphaned parts are removed. Zero budget selects
 * DS4_SCR_POOL_DEFAULT_BYTES. Returns 0 on success. */
int ds4_scr_pool_open(ds4_scr_pool **out, const char *directory, uint32_t context,
                      uint32_t capacity, uint64_t budget_bytes, char *err, size_t errlen);
void ds4_scr_pool_close(ds4_scr_pool *pool);
uint32_t ds4_scr_pool_entries(const ds4_scr_pool *pool);
/* Before the logical policy: unless the live transcript continues into the
 * request, score the hot archive and every catalog entry with the planner and,
 * when an entry covers more of the request, save the hot archive and restore
 * that entry in its place. Same exclusive-use rules as the logical policy.
 * Returns 0 when the session is ready for the policy (restored or not), 1 when
 * a failed swap left no archive (the policy then refreshes). */
int ds4_session_scr_pool_select(ds4_session *s, ds4_scr_pool *pool,
                                const ds4_tokens *logical_request, uint32_t max_ranges,
                                int pad_token, ds4_scr_pool_receipt *receipt);
/* Persist the sealed hot archive unless the pool already holds it. After a
 * request, pass include_appends=false: an archive that a refresh or rebase
 * sealed is saved, while one that only appended to a saved archive is not
 * rewritten each turn. At shutdown pass true to keep the appended tail.
 * Entries whose tokens it extends are superseded; least recently used entries
 * are evicted to stay within the entry and byte budgets. */
int ds4_session_scr_pool_save(ds4_session *s, ds4_scr_pool *pool, bool include_appends,
                              ds4_scr_pool_receipt *receipt);

#endif
