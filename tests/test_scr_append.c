/* Real cold file/Metal capture and staging. Model rows and headed checkpoint
 * boundaries are simulated; this is not inference, scale or timing evidence. */
#include "../ds4.c"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

static float values[128u * 512u];
static ds4_gpu_tensor *rows, *indices;
static uint64_t compared;
static bool omit_layer;
static uint32_t omit_owner = UINT32_MAX;

static float canonical(uint32_t position, uint32_t domain, uint32_t column) {
    return (float)(position + domain + column % 17u) / 1024.0f;
}

static void capture(ds41_scr_archive *a, uint32_t from, uint32_t count) {
    CHECK(count <= 128u);
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        if (owner == omit_owner) continue;
        uint32_t ratio = owner < 3u ? 2u : 1u;
        uint32_t first = from / ratio * ratio;
        uint32_t complete_end = (from + count) / ratio * ratio;
        uint32_t n = (complete_end - first) / ratio;
        if (!n) continue;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < 512u; ++j)
                values[i * 512u + j] = canonical(first + i * ratio, owner, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)n * 512u * 4u));
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < 128u; ++j)
                values[i * 128u + j] = canonical(first + i * ratio, owner + 4u, j);
        CHECK(ds4_gpu_tensor_write(indices, 0, values, (uint64_t)n * 128u * 4u));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds41_archive_capture_owner(a, owner, first, n, rows, indices));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
    for (uint32_t layer = 0; layer < 40u; ++layer) {
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < 512u; ++j)
                values[i * 512u + j] = canonical(from + i, layer + 8u, j);
        CHECK(ds4_gpu_tensor_write(rows, 0, values, (uint64_t)count * 512u * 4u));
        CHECK(ds4_gpu_begin_commands());
        if (!omit_layer || layer != 17u)
            CHECK(ds41_archive_capture_window(a, layer, from, count, rows));
        CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    }
}

static void headed(ds4_session *s, uint32_t n) {
    s->ds41_graph.pos = n;
    s->checkpoint.len = (int)n;
    s->checkpoint_valid = s->ds41_graph.valid = true;
}

#ifdef DS4_SCR_ENGINE_ARCHIVE_DONE
static bool cancel_after_commit(void *opaque) {
    const ds41_scr_archive *a = opaque;
    return a->raw->committed_tokens == 400u;
}
#endif

static void check_raw(ds41_scr_archive *a, uint32_t n) {
    for (uint32_t block = 0; block < n; block += 8u) {
        CHECK(ds41_raw_read(a->raw, block / 8u, 1u, a->raw_io,
                           DS41_ARCHIVE_READ_BYTES, NULL, NULL));
        const float *raw = (const float *)a->raw_io;
        for (uint32_t layer = 0; layer < 40u; ++layer)
            for (uint32_t t = 0; t < 8u; ++t)
                for (uint32_t column = 0; column < 512u; ++column) {
                    CHECK(raw[(layer * 8u + t) * 512u + column] ==
                          canonical(block + t, layer + 8u, column));
                    ++compared;
                }
    }
}

int main(void) {
    CHECK(ds4_gpu_init());
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_engine e = {.backend = DS4_BACKEND_METAL, .ds41_session_bytes = 4096};
    ds4_session s = {.engine = &e, .ctx_size = 1024, .ds41_graph_ready = true};
    ds41_gpu_graph *g = &s.ds41_graph;
    g->ctx = 1024; g->tp_world = 1; g->valid = true; g->allocation_bytes = 4096;
    g->table[0].fd = g->table[1].fd = -1;
    int tokens[1024];
    for (uint32_t i = 0; i < 1024u; ++i) tokens[i] = (int)(i % 211u);
    s.checkpoint.v = tokens; s.checkpoint.len = 0; s.checkpoint.cap = 1024;
    rows = ds4_gpu_tensor_alloc(128u * 512u * 4u);
    indices = ds4_gpu_tensor_alloc(128u * 128u * 4u);
    CHECK(rows && indices && !ds4_session_scr_archive_prepare(&s, 0, 512));
    ds41_scr_archive *a = g->archive;
    for (uint32_t from = 0; from < 256u; from += 128u) {
        capture(a, from, 128u); headed(&s, from + 128u);
        CHECK(!ds4_session_scr_archive_flush(&s));
    }
    CHECK(!ds4_session_scr_archive_seal(&s, tokens, 256));
    CHECK(a->end == 256u && a->raw->committed_tokens == 256u);
#ifdef DS4_SCR_ENGINE_ARCHIVE_DONE
    ds41_scr_coordinates map = {0};
    const uint32_t pads[] = {8u, 300u};
    CHECK(ds41_coordinates_build(&map, tokens, 385u, DS4_N_VOCAB, pads, 2u, 7u));
    map.count = 1u; map.physical_tokens = 256u;
    CHECK(ds41_coordinates_replace_charged(&a->coordinates, &map, tokens, 256u,
        DS4_N_VOCAB, &g->allocation_bytes, &e.ds41_session_bytes, &a->allocation_bytes));
    CHECK(ds41_archive_append_arm(a, &s, &e.model, tokens, 385u, 2u));
    CHECK(a->coordinates.count == 1u && a->coordinates.capacity == 2u);
    values[0] = 99.0f;
    CHECK(ds4_gpu_tensor_write(rows, 0, values, 512u * 4u));
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds41_archive_capture_owner(a, 1u, 0u, 1u, rows, indices));
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    float old_owner;
    CHECK(ds4_gpu_tensor_read(a->source, ds41_archive_owner(a->cap, 1u), &old_owner, 4u));
    CHECK(old_owner == canonical(0u, 1u, 0u));
#endif
    capture(a, 256u, 128u); headed(&s, 384u);
    /* On the old implementation this fails for actual suppressed capture,
     * before any new helper can stand in for the missing behavior. */
    CHECK(a->raw_tags[0][0] == 256u);
#ifdef DS4_SCR_ENGINE_ARCHIVE_DONE
    CHECK(ds41_archive_append_observe(a, &s, &e.model, tokens, 384u, NULL, NULL));
    CHECK(a->end == 384u && a->raw_origin == 384u);
    CHECK(a->coordinates.count == 2u && a->coordinates.physical_tokens == 384u);
    CHECK(a->coordinates.pads[1] == 300u);
    capture(a, 384u, 1u); headed(&s, 385u);
    CHECK(ds41_archive_append_observe(a, &s, &e.model, tokens, 385u, NULL, NULL));
    CHECK(a->end == 384u && a->captured_end == 385u);
    for (uint32_t from = 385u; from < 392u; ++from) {
        capture(a, from, 1u); headed(&s, from + 1u);
        CHECK(ds41_archive_append_observe(a, &s, &e.model, tokens, from + 1u, NULL, NULL));
        CHECK(a->captured_end == from + 1u);
        CHECK(a->end == (from < 391u ? 384u : 392u));
    }
    CHECK(a->raw->committed_tokens == 392u && a->raw_origin == 392u);
    check_raw(a, 392u);
    CHECK(ds41_archive_materialize_aligned(a, &s, &e.model, 0u, 16u, tokens,
                                          256u, 1024u, 2u));
    CHECK(ds41_archive_materialize_aligned(a, &s, &e.model, 128u, 16u, tokens + 128,
                                          264u, 1024u, 2u));
    uint64_t source_version = a->source_generation;
    CHECK(ds41_archive_append_observe(a, &s, &e.model, tokens, 392u, NULL, NULL));
    CHECK(a->source_generation == source_version);
    capture(a, 392u, 8u); headed(&s, 400u);
    CHECK(!ds41_archive_append_observe(a, &s, &e.model, tokens, 400u,
                                       cancel_after_commit, a));
    CHECK(errno == ECANCELED && a->writer_state == DS41_ARCHIVE_BLOCKED);
    CHECK(a->lost && !a->retiring && !a->range.valid);
    CHECK(a->end == 392u && a->source_generation == source_version);
    CHECK(a->raw->committed_tokens == 400u && a->captured_end == 400u);
    CHECK(!ds41_archive_append_observe(a, &s, &e.model, tokens, 400u, NULL, NULL));
    CHECK(ds41_archive_materialize_aligned(a, &s, &e.model, 128u, 16u, tokens + 128,
                                          264u, 1024u, 2u));
    CHECK(!ds41_archive_materialize_aligned(a, &s, &e.model, 128u, 16u, tokens + 128,
                                           272u, 1024u, 2u));
    ds41_archive_invalidate(a);
    CHECK(a->writer_state == DS41_ARCHIVE_BLOCKED);
    CHECK(a->end == 392u && a->source_generation == source_version);
    CHECK(ds41_archive_published_valid(a));
    CHECK(!ds41_archive_append_arm(a, &s, &e.model, tokens, 400u, 2u));
#endif
    CHECK(!ds4_session_scr_archive_release(&s));
    CHECK(!g->archive && g->allocation_bytes == 4096 && e.ds41_session_bytes == 4096);
#ifdef DS4_SCR_ENGINE_ARCHIVE_DONE
    /* Recovery is an explicit new capture, not promotion of retained extras. */
    headed(&s, 0u);
    CHECK(!ds4_session_scr_archive_prepare(&s, 0u, 512u));
    a = g->archive;
    for (uint32_t from = 0u; from < 256u; from += 128u) {
        capture(a, from, 128u); headed(&s, from + 128u);
        CHECK(!ds4_session_scr_archive_flush(&s));
    }
    CHECK(!ds4_session_scr_archive_seal(&s, tokens, 256u));
    CHECK(ds41_archive_append_arm(a, &s, &e.model, tokens, 512u, 0u));
    for (uint32_t from = 256u; from < 512u; from += 128u) {
        capture(a, from, 128u); headed(&s, from + 128u);
        CHECK(ds41_archive_append_observe(a, &s, &e.model, tokens, from + 128u, NULL, NULL));
    }
    CHECK(a->writer_state == DS41_ARCHIVE_SATURATED && a->end == 512u);
    uint64_t written = a->raw->written_bytes;
    capture(a, 512u, 8u); headed(&s, 520u);
    CHECK(a->raw->written_bytes == written && a->end == 512u && !a->lost);
    CHECK(ds41_archive_published_valid(a));
    ds41_archive_invalidate(a);
    CHECK(a->writer_state == DS41_ARCHIVE_INCOMPATIBLE);
    CHECK(ds41_archive_published_valid(a));
    CHECK(!ds4_session_scr_archive_release(&s));
    CHECK(g->allocation_bytes == 4096 && e.ds41_session_bytes == 4096);
    headed(&s, 0u);
    CHECK(!ds4_session_scr_archive_prepare(&s, 0u, 512u));
    a = g->archive;
    for (uint32_t from = 0u; from < 256u; from += 128u) {
        capture(a, from, 128u); headed(&s, from + 128u);
        CHECK(!ds4_session_scr_archive_flush(&s));
    }
    CHECK(!ds4_session_scr_archive_seal(&s, tokens, 256u));
    CHECK(ds41_archive_append_arm(a, &s, &e.model, tokens, 512u, 0u));
    omit_layer = true;
    capture(a, 256u, 8u); headed(&s, 264u);
    CHECK(!ds41_archive_append_observe(a, &s, &e.model, tokens, 264u, NULL, NULL));
    CHECK(a->history_hole && a->writer_state == DS41_ARCHIVE_BLOCKED && !a->retiring);
    CHECK(a->end == 256u && a->raw->committed_tokens == 256u);
    CHECK(ds41_archive_published_valid(a));
    CHECK(!ftruncate(a->raw->fd, (off_t)ds41_raw_bytes(248u)));
    CHECK(!ds41_archive_append_failure(a, false, EIO));
    CHECK(a->retiring && !a->raw->readable);
    CHECK(ds4_session_scr_archive_stage(&s, 0u, 16u, tokens, 256u) == 1);
    CHECK(!ds4_session_scr_archive_release(&s));
    CHECK(!g->archive && g->allocation_bytes == 4096 && e.ds41_session_bytes == 4096);
    for (uint32_t mode = 0u; mode < 2u; ++mode) {
        omit_layer = false; omit_owner = UINT32_MAX;
        headed(&s, 0u);
        CHECK(!ds4_session_scr_archive_prepare(&s, 0u, 512u));
        a = g->archive;
        for (uint32_t from = 0u; from < 256u; from += 128u) {
            capture(a, from, 128u); headed(&s, from + 128u);
            CHECK(!ds4_session_scr_archive_flush(&s));
        }
        CHECK(!ds4_session_scr_archive_seal(&s, tokens, 256u));
        CHECK(ds41_archive_append_arm(a, &s, &e.model, tokens, 512u, 0u));
        if (mode) omit_owner = 2u;
        uint32_t end = mode ? 264u : 265u;
        capture(a, 256u, end - 256u); headed(&s, end);
        CHECK(!ds41_archive_append_observe(a, &s, &e.model, tokens, end, NULL, NULL));
        CHECK(a->writer_state == DS41_ARCHIVE_BLOCKED && !a->retiring);
        CHECK(a->history_hole == (mode != 0u));
        CHECK(a->captured_end == (mode ? 256u : 265u));
        CHECK(a->end == 256u && a->raw->committed_tokens == 256u);
        CHECK(ds41_archive_published_valid(a));
        CHECK(!ds4_session_scr_archive_release(&s));
        CHECK(!g->archive && g->allocation_bytes == 4096 && e.ds41_session_bytes == 4096);
    }
#endif
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(indices);
    printf("append mechanism: %llu exact raw components; model/head boundaries simulated\n",
           (unsigned long long)compared);
    return 0;
}
