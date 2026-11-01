/* Faults at the real owning splice/syscall boundaries. Reuse the same arithmetic
 * fixture and adapters; every case owns fresh anonymous files and a fresh graph. */
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>
static ssize_t fault_pread(int, void *, size_t, off_t);
static ssize_t fault_pwrite(int, const void *, size_t, off_t);
#define pread fault_pread
#define pwrite fault_pwrite
#define main rebase_positive_main
#include "test_scr_rebase_owned.c"
#undef main
#undef pread
#undef pwrite

typedef enum {
    MISSING_LAYER, WRONG_M_HEAD, WRONG_N_HEAD, CANCEL_TAIL,
    IMPORT_SHORT_READ, IMPORT_CANCEL, IMPORT_BAD_FD, PREFETCH_READ_ERROR, WRITE_CANCEL,
    DESTINATION_CLEANUP_REFUSAL, OLD_RETIREMENT_REFUSAL, WRONG_MAP_POSITION,
    UNSAFE_PAD_CLIP, WRONG_M_CERTIFICATE, MISSING_DESTINATION,
    CANCEL_AFTER_RETIREMENT, WRONG_DESTINATION_TOKENS, WRONG_IMPORTED_RECEIPT,
    CANCEL_BEFORE_RETIREMENT, FAULT_COUNT
} fault_case;
static const char *const fault_names[] = {
    "missing computed layer", "wrong covered head", "wrong final head", "cancelled tail",
    "imported short read", "cancelled import", "corrupt old fd", "read-ahead I/O error",
    "postwrite cancellation",
    "destination cleanup refusal", "old retirement refusal", "wrong pad position",
    "unsafe clipped pad endpoint", "wrong covered certificate", "missing destination",
    "cancellation after irreversible retirement", "wrong destination tokens", "wrong imported receipt",
    "cancellation after final head before retirement"
};
static fault_case fault;
static uint32_t import_reads, events;
static bool faults_enabled;
static pthread_t owner_thread;
static ssize_t fault_pread(int fd, void *bytes, size_t count, off_t offset) {
    /* Only the range read-ahead helper reads off the owning thread. */
    if (faults_enabled && fault == PREFETCH_READ_ERROR && !pthread_equal(pthread_self(), owner_thread)) {
        ++events; errno = EIO; return -1;
    }
    ds41_gpu_graph *g = session ? &session->ds41_graph : NULL;
    bool importing = faults_enabled && g && g->archive && g->archive_pending &&
        g->archive_capture_pending && g->archive->range.valid && fd == g->archive->raw->fd;
    if (importing && (fault == IMPORT_SHORT_READ || fault == IMPORT_CANCEL || fault == IMPORT_BAD_FD)) {
        ++import_reads; ++events;
        if (fault == IMPORT_BAD_FD) {
            CHECK(!close(fd)); /* Only this case's exclusively owned anonymous fd. */
            errno = EBADF; return -1;
        }
        if (fault == IMPORT_SHORT_READ && import_reads > 1u) return 0;
        size_t short_count = count < 4096u ? count : 4096u;
        ssize_t result = pread(fd, bytes, short_count, offset);
        if (fault == IMPORT_CANCEL) cancel_now = true;
        return result;
    }
    return pread(fd, bytes, count, offset);
}
static ssize_t fault_pwrite(int fd, const void *bytes, size_t count, off_t offset) {
    ssize_t result = pwrite(fd, bytes, count, offset);
    ds41_gpu_graph *g = session ? &session->ds41_graph : NULL;
    if (faults_enabled && fault == WRITE_CANCEL && g && g->archive_pending &&
        fd == g->archive_pending->raw->fd && offset == (off_t)ds41_raw_bytes(304u) &&
        result == (ssize_t)count) { ++events; cancel_now = true; }
    return result;
}
static uint64_t digest_bytes(uint64_t digest, const unsigned char *bytes, size_t count) {
    for (size_t i = 0; i < count; ++i) digest = (digest ^ bytes[i]) * UINT64_C(1099511628211);
    return digest;
}
/* Direct test-owned reads can verify immutable bytes after a deliberate reader
 * fence. They confer no runtime readability/publication authority. */
static uint64_t immutable_digest(ds41_scr_archive *a) {
    unsigned char buffer[65536];
    uint64_t digest = UINT64_C(1469598103934665603);
    uint64_t source_bytes = ds4_gpu_tensor_bytes(a->source);
    for (uint64_t at = 0; at < source_bytes;) {
        size_t count = sizeof(buffer);
        if (count > source_bytes - at) count = source_bytes - at;
        CHECK(ds4_gpu_tensor_read(a->source, at, buffer, count));
        digest = digest_bytes(digest, buffer, count); at += count;
    }
    for (uint64_t at = 0; at < ds41_raw_bytes(512u);) {
        size_t count = sizeof(buffer);
        if (count > ds41_raw_bytes(512u) - at) count = ds41_raw_bytes(512u) - at;
        CHECK(pread(a->raw->fd, buffer, count, (off_t)at) == (ssize_t)count);
        digest = digest_bytes(digest, buffer, count); at += count;
    }
    return digest;
}
static void live_buffers(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph;
    uint64_t charge = (uint64_t)DS4_N_VOCAB * sizeof(uint32_t);
    g->token_map = malloc((size_t)charge); CHECK(g->token_map);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) g->token_map[i] = i;
    for (uint32_t o = 0; o < 4u; ++o) {
        uint32_t ratio = o < 3u ? 2u : 1u;
        uint64_t main_bytes = (uint64_t)(g->ctx / ratio) * 512u * sizeof(float);
        uint64_t index_bytes = (uint64_t)(g->ctx / ratio) * 128u * sizeof(float);
        g->compressed[o] = ds4_gpu_tensor_alloc(main_bytes);
        g->index_cache[o] = ds4_gpu_tensor_alloc(index_bytes);
        CHECK(g->compressed[o] && g->index_cache[o]); charge += main_bytes + index_bytes;
    }
    for (uint32_t l = 1u; l < 40u; ++l) {
        g->window[l] = ds4_gpu_tensor_alloc(live_charge); CHECK(g->window[l]); charge += live_charge;
    }
    g->allocation_bytes += charge; s->engine->ds41_session_bytes += charge;
}
static void run_fault(fault_case mode) {
    fault = mode; import_reads = events = 0; faults_enabled = false;
    reject_open = reject_after_open = reject_close = false; cancel_after_close_fd = -1;
    ds4_engine engine = {.backend = DS4_BACKEND_METAL};
    ds4_session s = {.engine = &engine, .ctx_size = 1024, .ds41_graph_ready = true,
        .progress = progress, .display_progress = progress, .cancel = cancel};
    session = &s;
    ds41_gpu_graph *g = &s.ds41_graph;
    g->ctx = 1024; g->tp_world = 1; g->prefill_cap = 2048; g->valid = true;
    g->table[0].fd = g->table[1].fd = -1;
    g->window[0] = ds4_gpu_tensor_alloc(live_charge); CHECK(g->window[0]);
    g->allocation_bytes = engine.ds41_session_bytes = live_charge;
    s.logits = malloc(DS4_N_VOCAB * sizeof(float)); CHECK(s.logits);
    int source[512], tokens[315];
    for (uint32_t i = 0; i < 512u; ++i) source[i] = (int)i;
    ds4_tokens initial = {.v = source, .len = 512, .cap = 512};
    controls(source, 512u);
    ds4_scr_refresh_receipt refresh; char err[256];
    CHECK(!ds4_session_scr_archive_refresh(&s, &initial, 0u, 512u, &refresh, err, sizeof(err)));
    live_buffers(&s);
    ++s.scr_coordinate_serial; /* A mapped source needs an issued nonzero identity. */
    ds41_scr_archive *old = g->archive; int old_fd = old->raw->fd;
    uint32_t old_pads[] = {30u, 300u, 439u};
    uint32_t old_count = mode == UNSAFE_PAD_CLIP ? 3u : 2u;
    CHECK(ds41_coordinates_build(&old->coordinates, source, 512u, DS4_N_VOCAB,
                                 old_pads, old_count, s.scr_coordinate_serial));
    uint64_t old_heap = old_count * sizeof(uint32_t);
    old->allocation_bytes += old_heap; g->allocation_bytes += old_heap;
    engine.ds41_session_bytes += old_heap;
    uint64_t old_charge = engine.ds41_session_bytes, digest = immutable_digest(old);
    uint64_t old_archive_charge = old->allocation_bytes, source_generation = old->source_generation;
    for (uint32_t i = 0; i < 315u; ++i) tokens[i] = (int)(6000u + i);
    memcpy(tokens + 8u, source + 18u, 128u * sizeof(int));
    memcpy(tokens + 146u, source + 276u, 168u * sizeof(int));
    ds4_tokens full = {.v = tokens, .len = 315, .cap = 315};
    uint32_t pads[] = {20u, 170u, 309u, 314u};
    if (mode != UNSAFE_PAD_CLIP) pads[2] = pads[3];
    uint32_t count = mode == UNSAFE_PAD_CLIP ? 4u : 3u;
    ds41_scr_coordinates map = {0};
    CHECK(ds41_coordinates_build(&map, tokens, 315u, DS4_N_VOCAB, pads, count,
                                 s.scr_coordinate_serial + 1u));
    size_t heap; CHECK(ds41_coordinates_heap_bytes(map.capacity, &heap));
    ds41_policy_owner owner = {.session = &s, .identity = &owner, .full = &full,
        .bytes = tokens, .length = 315, .capacity = 315, .archive_capacity = 512,
        .live_host_bytes = sizeof(tokens) + sizeof(owner) + heap, .active = true};
    s.scr_policy_active = true; s.scr_dense_owner = &owner;
    CHECK(ds41_policy_bind_coordinates(&owner, &map, heap));
    controls(tokens, 315u); faults_enabled = true;
    if (mode == MISSING_LAYER || mode == DESTINATION_CLEANUP_REFUSAL) omit_pos = 310;
    if (mode == WRONG_M_HEAD) bad_head_pos = 312;
    if (mode == WRONG_N_HEAD) bad_head_pos = 315;
    if (mode == CANCEL_TAIL) cancel_pos = 315;
    if (mode == DESTINATION_CLEANUP_REFUSAL) reject_close = true;
    ds4_scr_range ranges[] = {{18u, 8u, 128u}, {276u, 146u, 168u}};
    ds4_scr_multi_receipt r;
    uint32_t opens_before = opens;
    int rc = ds41_scr_multi_owned(&s, &full, ranges, 2u, &r, err, sizeof(err), &owner);
    bool commit_fault = mode >= OLD_RETIREMENT_REFUSAL && mode != UNSAFE_PAD_CLIP;
    if (commit_fault) {
        CHECK(!rc && r.published && r.rebase_prepared && g->archive_pending);
        ds41_scr_archive *next = g->archive_pending;
        CHECK(ds41_policy_publish_coordinates(&owner, &map));
        uint64_t both_charge = engine.ds41_session_bytes; /* Includes the independently published live map. */
        if (mode == OLD_RETIREMENT_REFUSAL) { destination_fd = old_fd; reject_close = true; }
        if (mode == WRONG_MAP_POSITION) ++s.scr_coordinates.pads[0];
        if (mode == WRONG_M_CERTIFICATE) r.covered_head_end -= 8u;
        if (mode == WRONG_DESTINATION_TOKENS) ++next->tokens[310];
        if (mode == WRONG_IMPORTED_RECEIPT) r.imported_tokens -= 2u;
        if (mode == MISSING_DESTINATION) g->archive_pending = NULL;
        if (mode == CANCEL_AFTER_RETIREMENT) cancel_after_close_fd = old_fd;
        if (mode == CANCEL_BEFORE_RETIREMENT) cancel_now = true;
        rc = ds41_session_rebase_commit(&s, &owner, &r);
        if (mode == CANCEL_AFTER_RETIREMENT) {
            CHECK(!rc && cancel_now && r.rebase_committed && g->archive == next &&
                  !g->archive_pending && s.checkpoint_valid && g->valid && next->rebased &&
                  fcntl(old_fd, F_GETFD) < 0 && errno == EBADF);
            CHECK(engine.ds41_session_bytes == both_charge - old_archive_charge);
        } else {
            CHECK(rc == (mode == CANCEL_BEFORE_RETIREMENT ? DS4_SESSION_SYNC_INTERRUPTED : 1) &&
                  !r.rebase_committed && g->archive == old &&
                  engine.ds41_session_bytes == both_charge);
            if (mode == MISSING_DESTINATION) {
                CHECK(!ds41_session_rebase_abort(&s, &r));
                CHECK(!s.checkpoint_valid && !g->valid && !g->archive_capture_pending);
                g->archive_pending = next; /* Restore test-owned pointer solely for checked cleanup. */
            }
            int aborted = ds41_session_rebase_abort(&s, &r);
            CHECK(!s.checkpoint_valid && !g->valid && !g->archive_capture_pending);
            if (mode == OLD_RETIREMENT_REFUSAL) {
                CHECK(aborted && old->retiring && next->retiring && next->retirement_refused &&
                      g->archive_pending == next && r.pending_owner_retained &&
                      engine.ds41_session_bytes == both_charge && fcntl(old_fd, F_GETFD) >= 0);
            } else CHECK(!aborted && !g->archive_pending);
        }
    } else {
        CHECK(rc && !r.published && !r.rebase_committed && g->archive == old);
        if (mode == UNSAFE_PAD_CLIP) {
            CHECK(!r.target_mutated && !calls && !heads && !r.rebase_prepared &&
                  opens == opens_before && s.checkpoint_valid && g->pos == 512u);
        } else {
            CHECK(r.target_mutated && !s.checkpoint_valid && !g->valid && !g->archive_capture_pending);
            if (mode == DESTINATION_CLEANUP_REFUSAL) {
                CHECK(r.pending_owner_retained && g->archive_pending && g->archive_pending->retiring &&
                      engine.ds41_session_bytes == old_charge + g->archive_pending->allocation_bytes &&
                      fcntl(destination_fd, F_GETFD) >= 0);
            } else CHECK(!g->archive_pending && engine.ds41_session_bytes == old_charge);
        }
        if (mode == IMPORT_SHORT_READ || mode == IMPORT_BAD_FD || mode == PREFETCH_READ_ERROR)
            CHECK(events && old->retiring && !old->raw->readable);
        /* The failed read-ahead belongs to range 2: range 1 was installed, range 2 never staged. */
        if (mode == PREFETCH_READ_ERROR)
            CHECK(r.installed_spans == 1u && r.materialized_spans == 1u && !r.prefetched_spans &&
                  r.gap_completed[0] == 10u && r.partial_work_unknown);
        if (mode == IMPORT_CANCEL || mode == WRITE_CANCEL || mode == CANCEL_TAIL)
            CHECK(rc == DS4_SESSION_SYNC_INTERRUPTED && old->raw->readable && !old->retiring);
        if (mode == IMPORT_CANCEL || mode == WRITE_CANCEL) CHECK(events);
    }
    faults_enabled = false; cancel_now = false; cancel_after_close_fd = -1;
    if (mode != CANCEL_AFTER_RETIREMENT && mode != IMPORT_BAD_FD)
        CHECK(immutable_digest(old) == digest && !memcmp(old->tokens, source, sizeof(source)) &&
              old->end == 512u && old->source_generation == source_generation);
    owner.active = false; s.scr_dense_owner = NULL; s.scr_policy_active = false;
    if (mode == DESTINATION_CLEANUP_REFUSAL) {
        int prior_calls = calls, prior_heads = heads;
        ds41_scr_archive *retained = g->archive_pending;
        uint64_t charge = engine.ds41_session_bytes;
        CHECK(ds4_session_eval(&s, tokens[0], err, sizeof(err)) == 1 &&
              strstr(err, "synchronized checkpoint") && calls == prior_calls && heads == prior_heads);
        ds4_scr_refresh_receipt refused;
        CHECK(ds4_session_scr_archive_refresh(&s, &full, 0u, 128u, &refused, err, sizeof(err)) == 1 &&
              !refused.target_mutated && !refused.normal_calls && calls == prior_calls &&
              heads == prior_heads && g->archive_pending == retained && engine.ds41_session_bytes == charge);
    }
    CHECK(!ds4_session_scr_archive_release(&s));
    ds41_session_coordinates_clear(&s); ds41_coordinates_free(&map);
    CHECK(engine.ds41_session_bytes == g->allocation_bytes);
    engine.ds41_session_bytes -= g->allocation_bytes; ds41_graph_free(g);
    CHECK(!engine.ds41_session_bytes);
    token_vec_free(&s.checkpoint); free(s.logits); session = NULL;
    printf("owning failure: %s passed\n", fault_names[mode]); fflush(stdout);
}
int main(void) {
    CHECK(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    owner_thread = pthread_self();
    rows = ds4_gpu_tensor_alloc(sizeof(values)); index_rows = ds4_gpu_tensor_alloc(sizeof(values));
    CHECK(rows && index_rows);
    for (fault_case mode = 0; mode < FAULT_COUNT; ++mode) run_fault(mode);
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(index_rows);
    CHECK(ds4_gpu_synchronize()); ds4_gpu_cleanup();
    puts("owning rebase failures: exact head/map/ownership/cancellation/retirement and immutable old bytes; arithmetic mocked, not model/scale acceptance");
    return 0;
}
