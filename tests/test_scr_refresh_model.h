/* SAME producer/shared scheduler and real Metal A capture/tags/seal/stage.
 * Only model, encoder, allocator, cancellation/failure boundaries are simulated.
 * Canonical row fixtures are not Transformer inference or native-scale proof. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdlib.h>
#include <sys/stat.h>
static bool model_step(void *, const void *, const void *, int, float *);
static bool model_short(void *, const void *, const void *, const int *, uint32_t);
static bool model_prefill(void *, const void *, const void *, const int *, uint32_t,
    ds4_session_progress_fn, void *, int, ds4_session_cancel_fn, void *);
static bool model_sweep(void *, const void *, const void *, const int *, uint32_t,
    ds4_session_progress_fn, void *, int, ds4_session_cancel_fn, void *, bool, bool);
static bool model_head(void *, const void *, const void *, float *);
static void encoder_acquire(void *, const void *, const void *, uint32_t, void *,
                            ds4_session_cancel_fn, void *);
static void encoder_release(void *, const void *, void *);
static int drain_boundary(void);
static int raw_stat_boundary(int, struct stat *);
static ds4_gpu_tensor *alloc_boundary(uint64_t);
#define DS41_EVAL_STEP model_step
#define DS41_SYNC_STEP model_step
#define DS41_SYNC_SHORT model_short
#define DS41_SYNC_PREFILL model_prefill
#define DS41_SYNC_SWEEP model_sweep
#define DS41_SYNC_HEAD model_head
#define DS41_SYNC_ACQUIRE encoder_acquire
#define DS41_SYNC_RELEASE encoder_release
#define ds4_gpu_synchronize drain_boundary
#define ds4_gpu_tensor_alloc alloc_boundary
#define fstat raw_stat_boundary
#include "../ds4.c"
#undef ds4_gpu_synchronize
#undef ds4_gpu_tensor_alloc
#undef fstat
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static ds4_session *session;
static const int *target;
static uint32_t target_n, next_position;
static int calls, heads, callbacks, acquisitions, sweeps, deferred, scalar;
static int fail_pos = -1, cancel_pos = -1, omit_pos = -1, bad_head_pos = -1;
static bool cancel_now, fail_old_release, fail_new_release, fail_final_drain, fail_seal_identity;
static uint32_t cancel_committed;
static bool fail_alloc, simulate_encoder, check_reentry, fail_head, eval_mode;
static const uint64_t live_charge = 128u * 512u * 4u;
static ds4_gpu_tensor *rows, *index_rows;
static float values[DS41_RAW_ARENA_TOKENS * 512u];
static uint64_t components;
static float original(uint32_t pos, uint32_t domain, uint32_t col) {
    return (float)((pos % 79u) + domain + col % 11u) / 128.0f;
}
static void progress(void *u, const char *phase, int i, int n) {
    (void)u; (void)phase; (void)i; (void)n; ++callbacks;
}
static bool cancel(void *u) {
    (void)u;
    ds41_scr_archive *a = session ? session->ds41_graph.archive : NULL;
    return cancel_now || (cancel_committed && a && a->raw &&
                          a->raw->committed_tokens >= cancel_committed);
}
static void capture(ds41_gpu_graph *g, uint32_t from, uint32_t n) {
    ds41_scr_archive *a = ds41_graph_capture_archive(g);
    if (!ds41_archive_capturing(a) || from + n <= a->start || from >= a->limit) return;
    CHECK(n <= DS41_RAW_ARENA_TOKENS);
    for (uint32_t o = 0; o < 4; ++o) {
        uint32_t ratio = o < 3 ? 2u : 1u;
        uint32_t start = from / ratio * ratio;
        uint32_t count = ((from + n) / ratio * ratio - start) / ratio;
        if (!count) continue;
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < 512; ++j)
                values[i * 512u + j] = original(start + i * ratio, o, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)count * 512u * 4u));
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < 128; ++j)
                values[i * 128u + j] = original(start + i * ratio, o + 4u, j);
        CHECK(ds4_gpu_tensor_write(index_rows, 0, values, (uint64_t)count * 128u * 4u));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds41_archive_capture_owner(a, o, start, count, rows, index_rows));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
    for (uint32_t l = 0; l < 40; ++l) {
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < 512; ++j)
                values[i * 512u + j] = original(from + i, l + 8u, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)n * 512u * 4u));
        CHECK(ds4_gpu_begin_commands());
        /* Omit a real interior row at layer17: do not fabricate coverage tags. */
        if (l == 17u && omit_pos >= (int)from && omit_pos < (int)(from + n)) {
            uint32_t before = (uint32_t)omit_pos - from;
            CHECK(ds41_archive_capture_window(a, l, from, before, rows));
            for (uint32_t i = before + 1u; i < n; ++i) {
                CHECK(ds4_gpu_tensor_copy(index_rows, 0, rows, (uint64_t)i * 512u * 4u, 512u * 4u));
                CHECK(ds41_archive_capture_window(a, l, from + i, 1, index_rows));
            }
        } else CHECK(ds41_archive_capture_window(a, l, from, n, rows));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
}
static bool model_rows(ds41_gpu_graph *g, const int *v, uint32_t n, bool defer) {
    CHECK(eval_mode ? (!session->scr_multi_active && !session->scr_policy_active) :
          (session->scr_multi_active && !session->progress && !session->display_progress));
    if (g->archive_capture_pending && g->archive_pending && g->pos > next_position &&
        g->archive_pending->captured_end == g->pos && g->archive_pending->imported_tokens)
        next_position = g->pos; /* Real graft installation advances without model work. */
    CHECK(g->pos == next_position); /* No reset/replay across any actual call. */
    CHECK(!memcmp(v, target + g->pos, (size_t)n * sizeof(int)));
    ++calls;
    if (check_reentry) {
        check_reentry = false;
        ds4_tokens t = {.v = (int *)target, .len = (int)target_n, .cap = (int)target_n};
        ds4_scr_refresh_receipt r; char err[256];
        CHECK(ds4_session_scr_archive_refresh(session, &t, 0, 128, &r, err, sizeof(err)) == 1);
        CHECK(session->scr_multi_active && !session->progress && !r.normal_calls);
    }
    if (fail_pos >= 0 && g->pos == (uint32_t)fail_pos) return false;
    capture(g, g->pos, n);
    g->pos += n; next_position += n; g->valid = true;
    if (cancel_pos >= 0 && g->pos >= (uint32_t)cancel_pos) cancel_now = true;
    if (defer) ++deferred;
    return true;
}
static bool model_step(void *p, const void *m, const void *w, int t, float *v) {
    (void)m; (void)w; ++scalar;
    if (!model_rows(p, &t, 1, false)) return false;
    if (v) {
        CHECK(eval_mode); ++heads;
        ds41_gpu_graph *g = p;
        for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) v[i] = (float)g->pos + (float)(i % 17u);
        if (bad_head_pos >= 0 && g->pos == (uint32_t)bad_head_pos) --g->pos;
    }
    return true;
}
static bool model_short(void *p, const void *m, const void *w, const int *v, uint32_t n) {
    (void)m; (void)w; return model_rows(p, v, n, false);
}
static bool model_prefill(void *p, const void *m, const void *w, const int *v, uint32_t n,
    ds4_session_progress_fn cb, void *u, int total, ds4_session_cancel_fn c, void *cu) {
    (void)m; (void)w; (void)u; (void)total; (void)c; (void)cu;
    CHECK(!cb); return model_rows(p, v, n, false);
}
static bool model_sweep(void *p, const void *m, const void *w, const int *v, uint32_t n,
    ds4_session_progress_fn cb, void *u, int total, ds4_session_cancel_fn c, void *cu,
    bool defer, bool pending) {
    (void)pending; ++sweeps;
    (void)m; (void)w; (void)u; (void)total; (void)c; (void)cu;
    CHECK(!cb); return model_rows(p, v, n, defer);
}
static bool model_head(void *p, const void *m, const void *w, float *v) {
    (void)m; (void)w; ++heads;
    ds41_gpu_graph *g = p;
    CHECK(!session->checkpoint_valid && !g->encoder_resident);
    if (fail_head) return false;
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) v[i] = (float)g->pos + (float)(i % 17u);
    if (bad_head_pos >= 0 && g->pos == (uint32_t)bad_head_pos) --g->pos;
    return true;
}
static void encoder_acquire(void *p, const void *m, const void *w, uint32_t n, void *r,
                            ds4_session_cancel_fn c, void *u) {
    (void)m; (void)w; (void)r; (void)c; (void)u; ++acquisitions;
    if (simulate_encoder && n >= 16384u) ((ds41_gpu_graph *)p)->encoder_resident = true;
}
static void encoder_release(void *p, const void *m, void *r) {
    (void)m; (void)r; ((ds41_gpu_graph *)p)->encoder_resident = false;
}
static int drain_boundary(void) {
    ds41_scr_archive *a = session ? session->ds41_graph.archive : NULL;
    if (a && a->retiring && ((fail_old_release && !calls) || (fail_new_release && calls))) {
        fail_old_release = fail_new_release = false; return 0;
    }
    if (fail_final_drain && a && a->sealed && session->checkpoint_valid &&
        session->checkpoint.len == (int)target_n) {
        fail_final_drain = false; return 0;
    }
    return ds4_gpu_synchronize();
}
static int raw_stat_boundary(int fd, struct stat *st) {
    ds41_scr_archive *a = session ? session->ds41_graph.archive : NULL;
    if (fail_seal_identity && a && a->raw && fd == a->raw->fd && !a->sealed &&
        a->raw->committed_tokens == a->cap && session->checkpoint_valid &&
        session->checkpoint.len == (int)(a->start + a->raw->committed_tokens)) {
        fail_seal_identity = false; errno = EIO; return -1;
    }
    return fstat(fd, st);
}
static ds4_gpu_tensor *alloc_boundary(uint64_t bytes) {
    if (fail_alloc) { fail_alloc = false; return NULL; }
    return ds4_gpu_tensor_alloc(bytes);
}
static void controls(const int *v, uint32_t n) {
    target = v; target_n = n; next_position = 0;
    calls = heads = callbacks = acquisitions = sweeps = deferred = scalar = 0;
    cancel_committed = 0;
    fail_pos = cancel_pos = omit_pos = bad_head_pos = -1;
    cancel_now = fail_old_release = fail_new_release = fail_final_drain = fail_seal_identity = false;
    fail_alloc = simulate_encoder = check_reentry = fail_head = eval_mode = false;
}
