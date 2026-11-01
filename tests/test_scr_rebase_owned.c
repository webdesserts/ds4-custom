/* Real owning splice, storage and checked ownership boundaries. Model arithmetic
 * and fault events are simulated; this is not Transformer or scale acceptance. */
#include "../ds4.h"
#include <unistd.h>
static bool rebase_open(void *, const char *, uint32_t, uint64_t);
static int rebase_close(int);
#define DS41_REBASE_OPEN rebase_open
#define close rebase_close
#include "test_scr_refresh_model.h"
#undef close
static bool reject_open, reject_after_open, reject_close;
static uint32_t opens;
static int destination_fd = -1, cancel_after_close_fd = -1;
static bool rebase_open(void *store, const char *directory, uint32_t capacity, uint64_t quota) {
    ds41_gpu_graph *g = &session->ds41_graph;
    ds41_scr_archive *a = g->archive_pending;
    CHECK(a && a->raw == store && a != g->archive &&
          a->allocation_bytes <= g->allocation_bytes &&
          a->allocation_bytes <= session->engine->ds41_session_bytes);
    ++opens;
    if (reject_open) { reject_open = false; errno = EACCES; return false; }
    bool ok = ds41_raw_open(store, directory, capacity, quota);
    if (ok) destination_fd = a->raw->fd;
    if (ok && reject_after_open) { reject_after_open = false; errno = EIO; return false; }
    return ok;
}
static int rebase_close(int fd) {
    if (fd == destination_fd && reject_close) {
        reject_close = false; errno = EINTR; return -1;
    }
    int rc = close(fd);
    if (!rc && fd == cancel_after_close_fd) cancel_now = true;
    return rc;
}
static void forward(ds4_session *s, uint32_t count) {
    ds41_gpu_graph *g = &s->ds41_graph;
    uint32_t from = g->pos;
    CHECK(model_short(g, &s->engine->model, &s->engine->weights, target + from, count));
    for (uint32_t i = from; i < from + count; ++i) token_vec_push(&s->checkpoint, target[i]);
}
static uint32_t canonical_position(uint32_t pos) {
    if (pos >= 8u && pos < 136u) return 18u + pos - 8u;
    if (pos >= 146u && pos < 310u) return 276u + pos - 146u;
    return pos;
}
static void exact_splice(ds41_scr_archive *a) {
    for (uint32_t block = 0; block < 312u; block += 8u) {
        CHECK(ds41_raw_read(a->raw, block / 8u, 1u, a->raw_io, DS41_ARCHIVE_READ_BYTES, NULL, NULL));
        const float *raw = (const float *)a->raw_io;
        for (uint32_t layer = 0; layer < 40u; ++layer)
            for (uint32_t row = 0; row < 8u; ++row)
                for (uint32_t col = 0; col < 512u; ++col) {
                    CHECK(raw[(layer * 8u + row) * 512u + col] ==
                          original(canonical_position(block + row), layer + 8u, col));
                    ++components;
                }
    }
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t index = 0; index < 2u; ++index) {
            const uint32_t width = index ? 128u : 512u;
            uint64_t base = ds41_archive_owner(a->cap, owner) +
                (index ? (uint64_t)(a->cap / ratio) * 512u * sizeof(float) : 0u);
            for (uint32_t pos = 0; pos < 312u; pos += ratio) {
                CHECK(ds4_gpu_tensor_read(a->source, base + (uint64_t)(pos / ratio) * width * sizeof(float),
                                         values, width * sizeof(float)));
                for (uint32_t col = 0; col < width; ++col) {
                    CHECK(values[col] == original(canonical_position(pos), owner + index * 4u, col));
                    ++components;
                }
            }
        }
    }
}
/* A prefetched window is gathered layer-major by kv_archive_gather_layers;
 * its staged rings must equal the per-layer gather of a synchronous read.
 * The window end (252) is not record-aligned, so 17 records are gathered. */
static void prefetched_gather_matches(ds4_session *s, ds41_scr_archive *a) {
    const uint32_t source = 122u, destination = 16u, count = 130u;
    const int *tokens = a->tokens + source - a->start;
    const size_t ring = 128u * DS41_RAW_ROW_BYTES;
    unsigned char *scalar = malloc(40u * ring), *gathered = malloc(40u * ring);
    unsigned char *data = malloc(DS41_ARCHIVE_READ_BYTES);
    CHECK(scalar && gathered && data);
    CHECK(ds41_archive_materialize_aligned(a, s, &s->engine->model, source, destination,
                                           tokens, count, s->ds41_graph.ctx, 2u));
    for (uint32_t l = 0; l < 40u; ++l)
        CHECK(ds4_gpu_tensor_read(a->stage, ds41_archive_ring(a->cap, l), scalar + l * ring, ring));
    memset(gathered, 0, 40u * ring);
    for (uint32_t l = 0; l < 40u; ++l)
        CHECK(ds4_gpu_tensor_write(a->stage, ds41_archive_ring(a->cap, l), gathered + l * ring, ring));
    uint32_t first, records;
    CHECK(ds41_raw_window_records(ds41_archive_count(a), source - a->start + count, &first, &records) &&
          records == 17u && (source - a->start + count) % 8u);
    ds41_raw_prefetch window = {0};
    CHECK(ds41_raw_prefetch_start(&window, a->raw, first, records, data) &&
          ds41_raw_prefetch_finish(&window));
    CHECK(ds41_archive_materialize_window(a, s, &s->engine->model, source, destination,
                                          tokens, count, s->ds41_graph.ctx, 2u, &window, false));
    for (uint32_t l = 0; l < 40u; ++l)
        CHECK(ds4_gpu_tensor_read(a->stage, ds41_archive_ring(a->cap, l), gathered + l * ring, ring));
    CHECK(!memcmp(scalar, gathered, 40u * ring));
    free(scalar); free(gathered); free(data);
}
static void owned_splice(ds4_session *s, const int *old_tokens) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *old = g->archive;
    int old_fd = old->raw->fd;
    uint32_t old_pads[] = {30u, 300u};
    CHECK(ds41_coordinates_build(&old->coordinates, old_tokens, 512u, DS4_N_VOCAB,
                                 old_pads, 2u, s->scr_coordinate_serial));
    uint64_t map_heap = 2u * sizeof(uint32_t);
    old->allocation_bytes += map_heap; g->allocation_bytes += map_heap;
    s->engine->ds41_session_bytes += map_heap;
    uint64_t live = (uint64_t)DS4_N_VOCAB * sizeof(uint32_t);
    g->token_map = malloc((size_t)live); CHECK(g->token_map);
    for (uint32_t i = 0; i < DS4_N_VOCAB; ++i) g->token_map[i] = i;
    for (uint32_t o = 0; o < 4u; ++o) {
        uint32_t ratio = o < 3u ? 2u : 1u;
        uint64_t main_bytes = (uint64_t)(g->ctx / ratio) * 512u * sizeof(float);
        uint64_t index_bytes = (uint64_t)(g->ctx / ratio) * 128u * sizeof(float);
        g->compressed[o] = ds4_gpu_tensor_alloc(main_bytes);
        g->index_cache[o] = ds4_gpu_tensor_alloc(index_bytes);
        CHECK(g->compressed[o] && g->index_cache[o]); live += main_bytes + index_bytes;
    }
    for (uint32_t l = 1u; l < 40u; ++l) {
        g->window[l] = ds4_gpu_tensor_alloc(live_charge); CHECK(g->window[l]); live += live_charge;
    }
    g->allocation_bytes += live; s->engine->ds41_session_bytes += live;
    int tokens[320];
    for (uint32_t i = 0; i < 320u; ++i) tokens[i] = (int)(6000u + i);
    memcpy(tokens + 8u, old_tokens + 18u, 128u * sizeof(int));
    memcpy(tokens + 146u, old_tokens + 276u, 168u * sizeof(int));
    tokens[315] = tokens[314]; /* An equal numeric pad ID is ordinary generation. */
    ds4_tokens full = {.v = tokens, .len = 315, .cap = 320};
    uint32_t pads[] = {20u, 170u, 314u};
    ds41_scr_coordinates map = {0};
    CHECK(ds41_coordinates_build(&map, tokens, 315u, DS4_N_VOCAB, pads, 3u,
                                 s->scr_coordinate_serial + 1u));
    size_t heap; CHECK(ds41_coordinates_heap_bytes(map.capacity, &heap));
    ds41_policy_owner owner = {.session = s, .identity = &owner, .full = &full,
        .bytes = tokens, .length = 315, .capacity = 320, .archive_capacity = 512u,
        .live_host_bytes = sizeof(tokens) + sizeof(owner) + heap, .active = true};
    s->scr_policy_active = true; s->scr_dense_owner = &owner;
    CHECK(ds41_policy_bind_coordinates(&owner, &map, heap));
    ds4_scr_range plan[] = {{18u, 8u, 128u}, {276u, 146u, 168u}};
    controls(tokens, 315u);
    ds4_scr_multi_receipt receipt; char err[256];
    int rc = ds41_scr_multi_owned(s, &full, plan, 2u, &receipt, err, sizeof(err), &owner);
    if (rc) fprintf(stderr, "owning rebase rc=%d: %s\n", rc, err);
    /* The 2-row capture suffix [310,312) is one batched call; only the
     * post-cover tail [312,315) steps token by token. */
    CHECK(scalar == 3);
    CHECK(!rc && receipt.published && receipt.rebase_prepared && receipt.head_completed &&
          receipt.covered_head_completed && receipt.covered_head_end == 312u &&
          receipt.planned_tokens == 296u && receipt.imported_tokens == 292u &&
          receipt.committed_tokens == 292u && receipt.committed_spans == 2u &&
          receipt.ranges[1].count == 164u && receipt.prefix_completed == 8u &&
          receipt.gap_completed[0] == 10u && receipt.suffix_completed == 5u &&
          receipt.materialized_spans == 2u && receipt.prefetched_spans == 1u &&
          heads == 2 && s->logits[0] == 315.0f && g->archive == old);
    ds41_scr_archive *next = g->archive_pending;
    CHECK(next && next->sealed && next->end == 312u && next->captured_end == 315u &&
          next->raw_origin == 312u && next->raw->committed_tokens == 312u &&
          ds41_raw_capture_complete(next->raw_tags, 312u, 3u) &&
          next->coordinates.count == 2u && next->coordinates.capacity == 3u &&
          next->coordinates.pads[2] == 314u && next->rebase_predecessor == old &&
          ds41_archive_published_valid(old));
    exact_splice(next);
    ds41_even_receipt reconciliation = {0};
    reconciliation.execution.multi = receipt;
    reconciliation.plan = (ds41_even_plan){.range_count = 2u,
        .physical_reused = 296u, .imported_pads = 2u, .verified = true};
    memcpy(reconciliation.plan.ranges, plan, sizeof(plan));
    ds41_pad_budget budget = {.limit = DS41_SCR_PLAN_WORK_MAX, .session = s};
    uint32_t imported_pads;
    CHECK(ds41_even_executed_subset(&reconciliation, &map, &budget, &imported_pads) && imported_pads == 2u);
    CHECK(ds41_policy_publish_coordinates(&owner, &map));
    /* A prepared owning transaction cannot silently succeed if its destination
     * slot disappears. Keep the test-owned pointer for explicit checked cleanup. */
    g->archive_pending = NULL;
    CHECK(ds41_session_rebase_commit(s, &owner, &receipt) == 1);
    CHECK(g->archive == old && !receipt.rebase_committed && fcntl(old_fd, F_GETFD) >= 0);
    g->archive_pending = next;
    CHECK(!ds41_session_rebase_commit(s, &owner, &receipt));
    CHECK(receipt.rebase_committed && !g->archive_pending && g->archive == next &&
          next->rebased && !next->rebase_predecessor && next->imported_tokens == 292u &&
          next->writer_coordinate_identity == s->scr_coordinates.identity &&
          s->checkpoint_valid && g->valid && s->checkpoint.len == 315);
    CHECK(fcntl(old_fd, F_GETFD) < 0 && errno == EBADF);
    ds4_scr_multi_receipt non_rebase = {0};
    CHECK(!ds41_session_rebase_commit(s, &owner, &non_rebase));
    int probe[149];
    for (uint32_t i = 0; i < 149u; ++i) probe[i] = (int)(9000u + i);
    memcpy(probe + 16u, tokens + 124u, 132u * sizeof(int));
    ds41_scr_coordinates probe_map = {0};
    uint32_t probe_pad = 62u;
    CHECK(ds41_coordinates_build(&probe_map, probe, 149u, DS4_N_VOCAB, &probe_pad, 1u, 999u));
    CHECK(ds41_multi_pad_correspondence(&next->coordinates, &probe_map, 124u, 16u, 132u));
    prefetched_gather_matches(s, next);
    CHECK(ds41_archive_range_check_aligned(next, s, &s->engine->model,
        124u, 16u, probe + 16u, 132u, g->ctx, 2u));
    CHECK(ds41_archive_materialize_aligned(next, s, &s->engine->model,
        124u, 16u, probe + 16u, 132u, g->ctx, 2u));
    ds41_coordinates_free(&probe_map);
    owner.active = false; s->scr_dense_owner = NULL; s->scr_policy_active = false;
    controls(tokens, 320u); next_position = 315u; eval_mode = true;
    for (uint32_t i = 315u; i < 320u; ++i) CHECK(!ds4_session_eval(s, tokens[i], err, sizeof(err)));
    eval_mode = false;
    CHECK(next->end == 320u && next->captured_end == 320u &&
          next->raw->committed_tokens == 320u && next->coordinates.count == 3u &&
          next->coordinates.pads[2] == 314u &&
          next->imported_tokens == 292u && s->logits[0] == 320.0f);
    ds41_session_coordinates_clear(s);
}
int main(void) {
    CHECK(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_engine e = {.backend = DS4_BACKEND_METAL};
    ds4_session s = {.engine = &e, .ctx_size = 1024, .ds41_graph_ready = true,
        .progress = progress, .display_progress = progress, .cancel = cancel};
    session = &s;
    ds41_gpu_graph *g = &s.ds41_graph;
    g->ctx = 1024; g->tp_world = 1u; g->prefill_cap = 2048u; g->valid = true;
    g->table[0].fd = g->table[1].fd = -1;
    s.logits = malloc(DS4_N_VOCAB * sizeof(float));
    rows = ds4_gpu_tensor_alloc(sizeof(values)); index_rows = ds4_gpu_tensor_alloc(sizeof(values));
    g->window[0] = ds4_gpu_tensor_alloc(live_charge);
    CHECK(s.logits && rows && index_rows && g->window[0]);
    g->allocation_bytes = e.ds41_session_bytes = live_charge;
    int old_tokens[512], tokens[315];
    for (uint32_t i = 0; i < 512u; ++i) old_tokens[i] = (int)i;
    memcpy(tokens, old_tokens, sizeof(tokens)); tokens[307] = 777;
    ds4_tokens original_request = {.v = old_tokens, .len = 512, .cap = 512};
    controls(old_tokens, 512u);
    ds4_scr_refresh_receipt refresh; char err[256];
    CHECK(!ds4_session_scr_archive_refresh(&s, &original_request, 0u, 512u,
                                          &refresh, err, sizeof(err)));
    ds41_scr_archive *old = g->archive;
    const int old_fd = old->raw->fd;
    const uint64_t old_charge = e.ds41_session_bytes;
    const uint32_t old_position = g->pos;
    ds4_tokens full = {.v = tokens, .len = 315, .cap = 315};
    ds41_scr_coordinates pending_map = {0};
    uint32_t pads[] = {8u, 314u};
    CHECK(ds41_coordinates_build(&pending_map, tokens, 315u, DS4_N_VOCAB, pads, 2u,
                                 s.scr_coordinate_serial + 1u));
    size_t heap;
    CHECK(ds41_coordinates_heap_bytes(pending_map.capacity, &heap));
    ds41_policy_owner owner = {.session = &s, .identity = &owner, .full = &full,
        .bytes = tokens, .length = 315, .capacity = 315, .archive_capacity = 512u,
        .live_host_bytes = sizeof(tokens) + sizeof(owner) + heap, .active = true};
    s.scr_policy_active = true; s.scr_dense_owner = &owner;
    CHECK(ds41_policy_bind_coordinates(&owner, &pending_map, heap));
    controls(tokens, 315u);
    CHECK(ds41_session_rebase_prepare(&s, &full, &owner, UINT64_MAX, err, sizeof(err)));
    CHECK(!opens && !g->archive_pending && e.ds41_session_bytes == old_charge);
    fail_alloc = true;
    CHECK(ds41_session_rebase_prepare(&s, &full, &owner, 0u, err, sizeof(err)));
    CHECK(!opens && !g->archive_pending && g->archive == old && g->pos == old_position);
    reject_open = true;
    CHECK(ds41_session_rebase_prepare(&s, &full, &owner, 0u, err, sizeof(err)));
    CHECK(opens == 1u && !g->archive_pending && e.ds41_session_bytes == old_charge);
    reject_after_open = reject_close = true;
    CHECK(ds41_session_rebase_prepare(&s, &full, &owner, 0u, err, sizeof(err)));
    CHECK(opens == 2u && g->archive_pending && g->archive_pending->retiring &&
          g->archive == old && !old->retiring && g->pos == old_position &&
          !calls && !heads && fcntl(destination_fd, F_GETFD) >= 0 && fcntl(old_fd, F_GETFD) >= 0);
    const uint64_t retained = g->archive_pending->allocation_bytes;
    CHECK(e.ds41_session_bytes == old_charge + retained);
    CHECK(!ds41_session_archive_release_slot(&s, true));
    CHECK(e.ds41_session_bytes == old_charge && !g->archive_pending);
    CHECK(!ds41_session_rebase_prepare(&s, &full, &owner, 0u, err, sizeof(err)));
    ds41_scr_archive *next = g->archive_pending;
    CHECK(next && g->archive == old && g->pos == old_position && s.checkpoint_valid &&
          next->allocation_bytes == ds41_archive_bytes(512u) + heap &&
          e.ds41_session_bytes == old_charge + next->allocation_bytes &&
          next->coordinates.pads != pending_map.pads && next->coordinates.count == 1u &&
          next->coordinates.capacity == 2u && next->coordinates.physical_tokens == 312u &&
          next->coordinates.pads[1] == 314u && next->writer_future_pads == 2u &&
          next->writer_known_end == 315u && !next->raw->file_bytes && !next->sealed);
    CHECK(!ds41_session_rebase_capture_start(&s));
    s.scr_multi_active = true; s.progress = s.display_progress = NULL;
    const ds4_tokens prefix = {.v = tokens, .len = 205, .cap = 315};
    ds41_sync_frontier frontier = {0};
    CHECK(!ds41_session_prefix(&s, &prefix, true, &frontier, err, sizeof(err)));
    CHECK(frontier.completed == 205u && !frontier.heads && !frontier.unknown &&
          !heads && !deferred && !sweeps && ds41_graph_capture_archive(g) == next &&
          g->archive == old);
    CHECK(next->captured_end == 205u && next->raw_origin == 200u &&
          next->raw->committed_tokens == 200u && !next->sealed && !s.checkpoint_valid);
    omit_pos = 205;
    forward(&s, 8u);
    CHECK(!ds41_session_rebase_forward_ready(&s, 205u, 213u));
    CHECK(next->captured_end == 205u && next->raw->committed_tokens == 200u &&
          next->history_hole && next->lost &&
          g->archive == old && old->end == 512u && fcntl(old_fd, F_GETFD) >= 0);
    g->archive_capture_pending = false; s.checkpoint_valid = g->valid = false;
    CHECK(!ds41_session_archive_release_slot(&s, true));
    CHECK(!g->archive_pending && e.ds41_session_bytes == old_charge &&
          ds41_archive_published_valid(old));
    s.scr_multi_active = s.scr_policy_active = false; s.scr_dense_owner = NULL; owner.active = false;
    ds41_coordinates_free(&pending_map);
    owned_splice(&s, old_tokens);
    CHECK(!ds4_session_scr_archive_release(&s));
    CHECK(e.ds41_session_bytes == g->allocation_bytes);
    e.ds41_session_bytes -= g->allocation_bytes;
    ds41_graph_free(g); CHECK(!e.ds41_session_bytes);
    token_vec_free(&s.checkpoint); free(s.logits);
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(index_rows);
    CHECK(ds4_gpu_synchronize()); ds4_gpu_cleanup();
    printf("rebase-owned: preparation/readiness plus actual multi splice, exactM312/N315 heads, clip296to292, independent pad maps, checked commit and new-source staging/append; %llu canonical components; model arithmetic mocked, no scale/RAM/time acceptance\n", (unsigned long long)components);
    return 0;
}
