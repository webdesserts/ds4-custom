/* V4.1 shared session workspace: accounting split, lending and release of the
 * call-local transients, and the ownership guard. Strict host: GPU buffers are
 * tests/scr_host_storage.c memory and the prefill body is a stub, so this is
 * ownership evidence, not inference. */
#include <stdbool.h>
#include <stdint.h>
static bool fake_sweep_ok = true;
static unsigned fake_sweeps;
#define DS41_PREFILL_BODY_RUN(g, m, w, tokens, count, progress, progress_ud, total, \
                              cancel, cancel_ud, encoder_only, resume_encoder) \
    ((void)(g), (void)(m), (void)(w), (void)(tokens), (void)(count), (void)(progress), \
     (void)(progress_ud), (void)(total), (void)(cancel), (void)(cancel_ud), \
     (void)(encoder_only), (void)(resume_encoder), ++fake_sweeps, fake_sweep_ok)
#include "../ds4.c"
#include <assert.h>

/* Size rule of ds4_metal.m (no Metal object is linked here). */
uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    const uint64_t flags = (((uint64_t)rows + ((uint64_t)source_rows + 63u) / 64u) * 4u + 255u) &
        ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u +
        (((uint64_t)source_rows + 63u) / 64u) * 64u * 128u * 2u;
}

static double gib(uint64_t b) { return (double)b / 1073741824.0; }

/* The split is exact and the engine plans the shared part once. */
static void accounting(void) {
    const uint32_t ctxs[] = {1024, 4096, 65536, 262144, 1048576};
    for (size_t i = 0; i < sizeof(ctxs) / sizeof(*ctxs); i++) {
        const uint32_t ctx = ctxs[i];
        assert(ds41_graph_session_bytes(ctx) + ds41_graph_shared_bytes(ctx) == ds41_graph_bytes(ctx));
        assert(ds41_graph_shared_bytes(ctx) > 0 && ds41_graph_session_bytes(ctx) > 0);
    }
    ds4_engine e = {.backend = DS4_BACKEND_METAL, .share_session_prefill_workspace = true};
    const uint32_t ctx = 262144;
    const uint64_t one = ds41_graph_bytes(ctx), session = ds41_graph_session_bytes(ctx),
        shared = ds41_graph_shared_bytes(ctx);
    assert(ds41_shared_workspace_enabled(&e));
    assert(ds41_planned_graph_bytes(&e, ctx, 1) == one);
    assert(ds41_planned_graph_bytes(&e, ctx, 4) == 4u * session + shared);
    e.share_session_prefill_workspace = false;
    assert(!ds41_shared_workspace_enabled(&e));
    assert(ds41_planned_graph_bytes(&e, ctx, 4) == 4u * one);
    e.share_session_prefill_workspace = true; e.tp.active = true;
    assert(!ds41_shared_workspace_enabled(&e));
    e.tp.active = false; e.backend = DS4_BACKEND_CUDA;
    assert(!ds41_shared_workspace_enabled(&e));
    printf("ctx=%u graph: per session %.3f GiB + transients %.3f GiB = %.3f GiB; "
           "4 sessions %.3f GiB shared vs %.3f GiB private\n",
           ctx, gib(session), gib(shared), gib(one), gib(4u * session + shared), gib(4u * one));
}

static void graph_shape(ds41_gpu_graph *g, uint32_t ctx) {
    memset(g, 0, sizeof(*g));
    *g = ds41_graph_shape(ctx);
    g->table[0].fd = g->table[1].fd = -1;
    g->workspace_owner = g;
    g->valid = true; g->tp_world = 1;
}

/* One session's transients become the workspace; a second session borrows
 * the same tensors; freeing sessions keeps them until the workspace goes. */
static void lending(void) {
    const uint32_t ctx = 1024;
    ds41_gpu_graph *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b)), *c = calloc(1, sizeof(*c));
    assert(a && b && c);
    graph_shape(a, ctx);
    a->allocation_bytes = ds41_graph_bytes(ctx);
    assert(ds41_graph_alloc_transients(a));
    float probe[4];
    assert(ds4_gpu_tensor_read(a->batch.block_mask, 0, probe, sizeof(probe)) &&
           probe[0] == 0.0f && probe[3] == 0.0f);
    ds41_workspace *ws = ds41_workspace_adopt(a);
    assert(ws && a->workspace == ws && ws->users == 1);
    assert(a->allocation_bytes == ds41_graph_session_bytes(ctx));
    assert(ws->bytes == ds41_graph_shared_bytes(ctx));
    assert(ds41_workspace_fits(ws, ctx) && !ds41_workspace_fits(ws, 2048) &&
           !ds41_workspace_fits(ws, 512));
    graph_shape(b, ctx);
    ds41_workspace_lend(ws, b);
    assert(ws->users == 2 && b->workspace == ws);
    assert(b->batch.q == a->batch.q && b->rows_view == a->rows_view &&
           b->rows_view[7].block_mask == a->rows_view[7].block_mask &&
           b->index_scores == a->index_scores && b->engram_prefetch == a->engram_prefetch &&
           b->raw_prefill == a->raw_prefill && b->index_packed == a->index_packed &&
           b->carry.residual == a->carry.residual);
    /* A write through one session is visible through the other: one copy. */
    const float mark = 3.5f;
    float seen = 0.0f;
    assert(ds4_gpu_tensor_write(a->rows_view[3].q, 4, &mark, sizeof(mark)) &&
           ds4_gpu_tensor_read(b->rows_view[3].q, 4, &seen, sizeof(seen)) && seen == mark);
    /* A differently shaped session keeps private transients. */
    graph_shape(c, 2048);
    assert(ds41_graph_alloc_transients(c) && c->batch.q != a->batch.q && !c->workspace);
    ds41_graph_free(c);
    ds41_graph_free(b);
    assert(ws->users == 1 && !b->workspace && !b->batch.q && !b->rows_view);
    assert(ds4_gpu_tensor_read(a->rows_view[3].q, 4, &seen, sizeof(seen)) && seen == mark);
    ds41_graph_free(a);
    assert(ws->users == 0);
    ds41_workspace_free(ws);
    free(a); free(b); free(c);
}

/* Overlapping use and use while another session's deferred decoder holds the
 * carry fail; row copies of the owner keep its identity. */
static void guard(void) {
    ds41_workspace ws = {0};
    ds41_gpu_graph *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    assert(a && b);
    graph_shape(a, 1024); graph_shape(b, 1024);
    a->workspace = b->workspace = &ws; ws.users = 2;

    assert(ds41_workspace_enter(a, "test"));
    assert(!ds41_workspace_enter(b, "test"));
    ds41_gpu_graph row = *a;
    assert(ds41_workspace_enter(&row, "nested row"));
    ds41_workspace_leave(&row);
    ds41_workspace_leave(a);
    assert(ws.depth == 0 && ds41_workspace_enter(b, "test"));
    ds41_workspace_leave(b);

    /* Encoder-only sweep: the carry stays reserved for the resume. */
    fake_sweeps = 0;
    assert(ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, true, false));
    assert(ws.carry_pending && ws.owner == a && ws.depth == 0);
    assert(!ds41_workspace_enter(b, "test"));
    assert(!ds41_graph_prefill_sweep(b, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, false, false));
    assert(!ds41_graph_step(b, NULL, NULL, 0, NULL) && !b->valid);
    b->valid = true;
    assert(!ds41_graph_prefill_sweep(b, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, false, true));
    assert(fake_sweeps == 1);
    row = *a;
    assert(ds41_workspace_enter(&row, "owner row"));
    ds41_workspace_leave(&row);
    assert(ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, false, true));
    assert(!ws.carry_pending && fake_sweeps == 2);
    assert(ds41_workspace_enter(b, "test"));
    ds41_workspace_leave(b);

    /* A failed resume releases the carry too. */
    assert(ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, true, false));
    fake_sweep_ok = false;
    assert(!ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, false, true));
    fake_sweep_ok = true;
    assert(!ws.carry_pending);
    /* A failed encoder-only sweep reserves nothing. */
    fake_sweep_ok = false;
    assert(!ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, true, false));
    fake_sweep_ok = true;
    assert(!ws.carry_pending && ds41_workspace_enter(b, "test"));
    ds41_workspace_leave(b);

    /* A resume without a reserved carry is refused before running. */
    fake_sweeps = 0;
    assert(!ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, false, true));
    assert(fake_sweeps == 0);

    /* The prefix call ending (abandon) or the owner closing releases it. */
    assert(ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, true, false));
    ds41_workspace_abandon(b);
    assert(ws.carry_pending);
    ds41_workspace_abandon(a);
    assert(!ws.carry_pending && ds41_workspace_enter(b, "test"));
    ds41_workspace_leave(b);
    assert(ds41_graph_prefill_sweep(a, NULL, NULL, NULL, 1, NULL, NULL, 0, NULL, NULL, true, false));
    ds41_workspace_detach(a);
    assert(!ws.carry_pending && ws.users == 1 && ds41_workspace_enter(b, "test"));
    ds41_workspace_leave(b);
    ds41_workspace_detach(b);
    assert(ws.users == 0);
    free(a); free(b);
}

int main(void) {
    assert(ds4_gpu_init());
    g_ds4_shape = DS4_SHAPE_FLASH41;
    accounting();
    lending();
    guard();
    ds4_gpu_cleanup();
    puts("test_v41_shared_workspace: ok");
    return 0;
}
