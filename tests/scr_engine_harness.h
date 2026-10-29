/* Shared model-free engine harness for SCR host fixtures: includes ds4.c with
 * simulated model rows/heads/history and records copies, installs and drains.
 * Link with tests/scr_host_storage.c + scr_host_forbidden.c only; no Metal. */
#ifndef DS4_TEST_SCR_ENGINE_HARNESS_H
#define DS4_TEST_SCR_ENGINE_HARNESS_H
#include "../ds4.h"
#include "../ds4_gpu.h"
static bool model_step(void *, const void *, const void *, int, float *);
static bool model_short(void *, const void *, const void *, const int *, uint32_t);
static bool model_prefill(void *, const void *, const void *, const int *, uint32_t,
    ds4_session_progress_fn, void *, int, ds4_session_cancel_fn, void *);
static bool model_sweep(void *, const void *, const void *, const int *, uint32_t,
    ds4_session_progress_fn, void *, int, ds4_session_cancel_fn, void *, bool, bool);
static bool model_head(void *, const void *, const void *, float *);
static void encoder_acquire(void *, const void *, const void *, uint32_t, void *, ds4_session_cancel_fn, void *);
static void encoder_release(void *, const void *, void *);
static int copy_boundary(ds4_gpu_tensor *, uint64_t, const ds4_gpu_tensor *, uint64_t, uint64_t);
static int drain_boundary(void);
#define DS41_SYNC_STEP model_step
#define DS41_SYNC_SHORT model_short
#define DS41_SYNC_PREFILL model_prefill
#define DS41_SYNC_SWEEP model_sweep
#define DS41_SYNC_HEAD model_head
#define DS41_SYNC_CAPTURE_BATCH_PREFILL model_prefill
#define DS41_SYNC_ACQUIRE encoder_acquire
#define DS41_SYNC_RELEASE encoder_release
#define ds4_gpu_tensor_copy copy_boundary
#define ds4_gpu_synchronize drain_boundary
#include "../ds4.c"
#undef ds4_gpu_tensor_copy
#undef ds4_gpu_synchronize
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static ds4_session *session;
#ifdef DS41_SCR_PLANNER_TEST_HOOKS
uint32_t ds41_scr_planner_test_variant_limit = 4096u;
#endif
static const int *target;
static bool cancel_now, nested, fail_release, fail_head, fail_drain, cancel_head, bad_head;
static int fail_pos = -1, cancel_pos = -1, fail_stage, fail_install;
static uint32_t heads, calls, installs, callbacks, copies;
static uint32_t encoder_acquires, encoder_releases, resident_calls;
static bool history_seen;
static ds4_gpu_tensor *rows, *indices;
static float values[DS41_RAW_ARENA_TOKENS * 512u], got[128u * 512u];
static float row_value(uint32_t p, uint32_t domain, uint32_t col) {
    return (float)(p % 79u + domain + col % 11u) / 128.0f;
}
static void progress(void *u, const char *p, int a, int b) {
    (void)u; (void)p; (void)a; (void)b; ++callbacks;
}
static bool cancel(void *u) { (void)u; return cancel_now; }
static void capture(ds41_gpu_graph *g, uint32_t from, uint32_t n) {
    ds41_scr_archive *a = ds41_graph_capture_archive(g);
    if (!ds41_archive_capturing(a) || from + n <= a->start || from >= a->limit) return;
    CHECK(n <= DS41_RAW_ARENA_TOKENS);
    for (uint32_t o = 0; o < 4; ++o) {
        uint32_t ratio = o < 3 ? 2u : 1u;
        uint32_t first = from / ratio * ratio;
        uint32_t count = ((from + n) / ratio * ratio - first) / ratio;
        if (!count) continue;
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < 512; ++j) values[i*512u+j] = row_value(first+i*ratio, o, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)count * 512u * 4u));
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < 128; ++j) values[i*128u+j] = row_value(first+i*ratio, o+4u, j);
        CHECK(ds4_gpu_tensor_write(indices, 0, values, (uint64_t)count * 128u * 4u));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds41_archive_capture_owner(a, o, first, count, rows, indices));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
    for (uint32_t l = 0; l < 40; ++l) {
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < 512; ++j) values[i*512u+j] = row_value(from+i, l+8u, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)n * 512u * 4u));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds41_archive_capture_window(a, l, from, n, rows));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
}
/* Nested public SCR entry points must refuse while an owner is executing. */
static void guards(void) {
    ds4_session *s = session;
    const ds41_policy_owner *o = s->scr_dense_owner;
    CHECK(o && ds41_policy_authorized(s, o->full, o) == !s->scr_multi_active);
    ds4_scr_multi_receipt m; ds4_scr_refresh_receipt f; ds4_scr_logical_receipt l; char err[256];
    CHECK(ds4_session_scr_multi_sync(s, o->full, NULL, 0, &m, err, sizeof(err)) == 1);
    CHECK(ds4_session_scr_archive_refresh(s, o->full, 0, 128, &f, err, sizeof(err)) == 1);
    CHECK(ds4_session_scr_logical_policy(s, o->full, 1024, 6, 201, &l, err, sizeof(err)) == 1);
    CHECK(s->scr_policy_active && s->scr_dense_owner == o && !s->progress && !s->display_progress);
    CHECK(!m.fresh_completed && !f.normal_calls && !l.execution.complete_n);
}
#ifndef DS41_FIXTURE_CHUNK
#define DS41_FIXTURE_CHUNK(g,n,prefill) ((void)0)
#endif
#ifndef DS41_FIXTURE_BEFORE_ROWS
#define DS41_FIXTURE_BEFORE_ROWS(g,n) false
#define DS41_FIXTURE_AFTER_ROWS(g,v,n) ((void)0)
#endif
static bool model_rows(ds41_gpu_graph *g, const int *v, uint32_t n, bool defer) {
    DS41_FIXTURE_CHUNK(g,n,false);
    if (DS41_FIXTURE_BEFORE_ROWS(g,n)) return false;
    CHECK(session->scr_policy_active && session->scr_dense_owner);
    CHECK(!session->progress && !session->display_progress);
    CHECK(!memcmp(v, target + g->pos, (size_t)n * sizeof(int)));
    if (nested) { nested = false; guards(); }
    ++calls;
    resident_calls += g->encoder_resident;
    ds41_scr_archive *a = g->archive;
    if (installs && a && g->pos == a->range.end) {
        for (uint32_t i = 0; i < 3; ++i)
            CHECK(g->history.tail[i] == (int32_t)g->token_map[target[g->pos-1u-i]]);
        history_seen = true;
    }
    if (fail_pos >= 0 && g->pos == (uint32_t)fail_pos) return false;
    capture(g, g->pos, n); g->pos += n; g->valid = !defer;
    /* Model prefill advances the rolling history with the actual target
     * tokens; the native batch-capture suffix proof reads it at the new
     * frontier. Keep the fixture's model stub faithful to that contract. */
    for (uint32_t i = 0; i < 3u && g->pos > i; ++i)
        g->history.tail[i] = (int32_t)g->token_map[target[g->pos - 1u - i]];
    DS41_FIXTURE_AFTER_ROWS(g,v,n);
    if (cancel_pos >= 0 && g->pos >= (uint32_t)cancel_pos) cancel_now = true;
    return true;
}
static bool model_step(void *g, const void *m, const void *w, int t, float *v) {
    (void)m; (void)w; CHECK(!v); return model_rows(g, &t, 1, false);
}
static bool model_short(void *g, const void *m, const void *w, const int *v, uint32_t n) {
    (void)m; (void)w; return model_rows(g, v, n, false);
}
static bool model_prefill(void *g, const void *m, const void *w, const int *v, uint32_t n,
    ds4_session_progress_fn cb, void *u, int total, ds4_session_cancel_fn c, void *cu) {
    (void)m; (void)w; (void)u; (void)total; (void)c; (void)cu; CHECK(!cb);
    DS41_FIXTURE_CHUNK(g,n,true);
    return model_rows(g, v, n, false);
}
static bool model_sweep(void *g, const void *m, const void *w, const int *v, uint32_t n,
    ds4_session_progress_fn cb, void *u, int total, ds4_session_cancel_fn c, void *cu,
    bool defer, bool pending) {
    (void)pending; (void)m; (void)w; (void)u; (void)total; (void)c; (void)cu; CHECK(!cb);
    return model_rows(g, v, n, defer);
}
static bool model_head(void *p, const void *m, const void *w, float *v) {
    (void)m; (void)w; ++heads; CHECK(!((ds41_gpu_graph *)p)->encoder_resident);
    if (fail_head) return false;
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) v[i] = (float)i;
    if (cancel_head) cancel_now = true;
    if (bad_head) --((ds41_gpu_graph *)p)->pos;
    return true;
}
/* Residency double: the production gate admits >=16384 remaining rows; the
 * counters let fixtures prove residency spans arena flushes. */
static void encoder_acquire(void *p, const void *m, const void *w, uint32_t n, void *r,
    ds4_session_cancel_fn c, void *u) {
    (void)m; (void)w; (void)r; (void)c; (void)u;
    ds41_gpu_graph *g = p;
    CHECK(!g->encoder_resident);
    if (n >= 16384u) { g->encoder_resident = true; ++encoder_acquires; }
}
static void encoder_release(void *p, const void *m, void *r) {
    (void)m; (void)r;
    ds41_gpu_graph *g = p;
    if (g->encoder_resident) ++encoder_releases;
    g->encoder_resident = false;
}
static int copy_boundary(ds4_gpu_tensor *dst, uint64_t d, const ds4_gpu_tensor *src,
                         uint64_t from, uint64_t n) {
    if (session && session->scr_multi_active) {
        ds41_gpu_graph *g = &session->ds41_graph; ds41_scr_archive *a = g->archive;
        if (a && dst == a->scratch && src == a->source &&
            (fail_stage == 1 || (fail_stage == 2 && installs == 1))) return 0;
        bool live = false;
        for (uint32_t o = 0; o < 4; ++o) live |= dst == g->compressed[o] || dst == g->index_cache[o];
        for (uint32_t l = 0; l < 40; ++l) live |= dst == g->window[l];
        if (live && (fail_install == 1 || (fail_install == 2 && installs == 1))) return 0;
        if (live) {
            CHECK(!g->valid && a && !a->range.valid);
            if (dst == g->window[39]) ++installs;
            ++copies;
            CHECK(n <= sizeof(got) || dst != g->window[39]);
        }
    }
    int ok = ds4_gpu_tensor_copy(dst, d, src, from, n);
    return ok;
}
static int drain_boundary(void) {
    if (session && session->ds41_graph.archive && session->ds41_graph.archive->retiring && fail_release)
        return 0;
    if (session && session->scr_multi_active && heads && fail_drain) return 0;
    return ds4_gpu_synchronize();
}
static void controls(const int *v) {
    target = v; cancel_now = nested = fail_release = fail_head = fail_drain = cancel_head = bad_head = false;
    fail_pos = cancel_pos = -1; fail_stage = fail_install = 0;
    heads = calls = installs = callbacks = copies = 0; history_seen = false;
    encoder_acquires = encoder_releases = resident_calls = 0;
}
/* Host-storage Flash41 session of `ctx` positions with newline-only vocab
 * text (pad 201 is a valid unpadded-policy separator). */
static void harness_session_open(ds4_engine *e, ds4_session *s, uint32_t ctx, uint32_t prefill_cap) {
    *e = (ds4_engine){.backend = DS4_BACKEND_METAL};
    *s = (ds4_session){.engine = e, .ctx_size = (int)ctx, .ds41_graph_ready = true,
        .progress = progress, .display_progress = progress, .cancel = cancel,
        .scr_planner_budget_ms = 5000u};
    session = s;
    ds41_gpu_graph *g = &s->ds41_graph;
    g->ctx = ctx; g->tp_world = 1; g->prefill_cap = prefill_cap; g->valid = true;
    g->table[0].fd = g->table[1].fd = -1;
    g->token_map = malloc(DS4_N_VOCAB * sizeof(uint32_t));
    s->logits = malloc(DS4_N_VOCAB * sizeof(float));
    s->checkpoint.v = malloc((size_t)ctx * sizeof(int)); s->checkpoint.cap = (int)ctx;
    CHECK(g->token_map && s->logits && s->checkpoint.v);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) g->token_map[i] = i ^ 73u;
    uint64_t live = 0;
    for (uint32_t o = 0; o < 4; ++o) {
        uint32_t ratio = o < 3 ? 2u : 1u;
        g->compressed[o] = ds4_gpu_tensor_alloc((uint64_t)ctx / ratio * 512u * 4u);
        g->index_cache[o] = ds4_gpu_tensor_alloc((uint64_t)ctx / ratio * 128u * 4u);
        CHECK(g->compressed[o] && g->index_cache[o]); live += (uint64_t)ctx / ratio * 640u * 4u;
    }
    for (uint32_t l = 0; l < 40; ++l) {
        g->window[l] = ds4_gpu_tensor_alloc(128u * 512u * 4u); CHECK(g->window[l]);
        live += 128u * 512u * 4u;
    }
    g->allocation_bytes = e->ds41_session_bytes = live;
    if (!rows) rows = ds4_gpu_tensor_alloc(sizeof(values));
    if (!indices) indices = ds4_gpu_tensor_alloc(DS41_RAW_ARENA_TOKENS * 128u * 4u);
    CHECK(rows && indices);
    e->vocab.n_vocab = DS4_N_VOCAB;
    e->vocab.token = calloc(DS4_N_VOCAB, sizeof(*e->vocab.token)); CHECK(e->vocab.token);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) e->vocab.token[i] = (ds4_str){.ptr = "\xc4\x8a", .len = 2};
}
static void harness_session_close(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph;
    CHECK(!ds4_session_scr_archive_release(s));
    ds41_session_coordinates_clear(s);
    for (uint32_t o = 0; o < 4; ++o) { ds4_gpu_tensor_free(g->compressed[o]); ds4_gpu_tensor_free(g->index_cache[o]); }
    for (uint32_t l = 0; l < 40; ++l) ds4_gpu_tensor_free(g->window[l]);
    free(g->token_map); free(s->logits); free(s->checkpoint.v);
    free(s->engine->vocab.token);
    session = NULL;
}
/* Prepare, capture and seal source[start, start+count) as the session archive
 * of capacity `cap`, the way a completed refresh would leave it. */
static void harness_archive(ds4_session *s, const int *source, uint32_t start,
                            uint32_t count, uint32_t cap) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds4_session_invalidate(s);
    s->checkpoint.len = 0; s->checkpoint_valid = false; g->pos = 0; g->valid = true;
    controls(source);
    CHECK(!ds4_session_scr_archive_prepare(s, start, cap));
    for (uint32_t from = start; from < start + count; from += 128u) {
        capture(g, from, 128u);
        memcpy(s->checkpoint.v, source, (size_t)(from + 128u) * sizeof(int));
        s->checkpoint.len = (int)(from + 128u); g->pos = from + 128u;
        s->checkpoint_valid = g->valid = true;
        CHECK(!ds4_session_scr_archive_flush(s));
    }
    CHECK(!ds4_session_scr_archive_seal(s, source + start, count));
}
/* Independent exhaustive oracle: enumerate all aligned endpoints and retain
 * those which cannot expand by an eight-token block. No hashes/diagonals used. */
#endif
