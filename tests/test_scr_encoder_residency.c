/* Encoder residency across SCR capture flushes. A >=16384-row rebase prefix
 * (session prefix) and gap (capture chunks) must each keep residency across
 * their 2048-row arena flushes and drop it only for the last 768 rows, before
 * any head. A >=16384-row first-turn refresh with batch capture feeds its
 * source as one call: headless arena flushes, residency, and a single head;
 * without batch capture it keeps 128-token headed calls. Strict host: tests/scr_host_storage.c replaces GPU buffers and
 * model rows are simulated, so this is ownership evidence, not inference. */
#include "../ds4.h"
static void observe(void *graph, uint32_t n);
#define DS41_FIXTURE_CHUNK(g,n,prefill) ((prefill) ? (void)0 : observe((g), (n)))
#include "scr_engine_harness.h"

#define CTX 40960u
#define SOURCE 2048u
#define PREFIX 16400u
#define GAP 16400u
#define TAIL 152u
#define TARGET (PREFIX + SOURCE / 2u + GAP + SOURCE / 2u + TAIL)
#define REFRESH 16400u

/* Per-span residency trace from inside model_rows. */
static uint32_t prefix_resident, gap_resident, resident_flush_crossings, tail_resident;
static uint32_t refresh_resident, refresh_chunks;
static bool was_resident, refreshing;
static uint32_t last_origin;
static void observe(void *graph, uint32_t n) {
    ds41_gpu_graph *g = graph;
    const ds41_scr_archive *p = g->archive_capture_pending ? g->archive_pending :
        refreshing && g->archive && !g->archive->sealed ? g->archive : NULL;
    if (refreshing && p) { CHECK(n <= p->raw_arena_tokens); ++refresh_chunks; }
    if (!g->encoder_resident) { was_resident = false; return; }
    CHECK(p && n <= p->raw_arena_tokens);
    if (refreshing) ++refresh_resident;
    else if (g->pos < PREFIX) ++prefix_resident;
    else if (g->pos >= PREFIX + SOURCE / 2u && g->pos < PREFIX + SOURCE / 2u + GAP) ++gap_resident;
    else ++tail_resident;
    /* Residency held while the arena origin advanced: a flush ran resident. */
    if (was_resident && p->raw_origin != last_origin) ++resident_flush_crossings;
    was_resident = true; last_origin = p->raw_origin;
}

int main(void) {
    CHECK(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    static ds4_engine e; static ds4_session s;
    harness_session_open(&e, &s, CTX, 2048u);
    static int source[SOURCE], v[TARGET];
    for (uint32_t i = 0; i < SOURCE; ++i) source[i] = 1000 + (int)i;
    harness_archive(&s, source, 0u, SOURCE, SOURCE);
    for (uint32_t i = 0; i < TARGET; ++i) v[i] = 90000 + (int)i;
    memcpy(v + PREFIX, source, SOURCE / 2u * sizeof(int));
    memcpy(v + PREFIX + SOURCE / 2u + GAP, source + SOURCE / 2u, SOURCE / 2u * sizeof(int));
    ds4_tokens full = {.v = v, .len = (int)TARGET, .cap = (int)TARGET};
    controls(v);
    ds4_scr_logical_receipt r; char err[256] = {0};
    int rc = ds4_session_scr_logical_policy(&s, &full, 36864u, 32u, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "residency policy: %s\n", err);
    CHECK(!rc && r.published && r.execution.route == DS4_SCR_DENSE_REUSE);
    CHECK(r.planned_ranges == 2u && r.execution.multi.committed_tokens == SOURCE);
    CHECK(r.execution.multi.rebase_committed);
    /* One acquisition each for the prefix and the gap; every one released. */
    CHECK(encoder_acquires == 2u && encoder_releases == 2u && !s.ds41_graph.encoder_resident);
    /* (16400 - 768) / 2048 rounds to 8 resident chunks per span; the old
     * release-at-flush prefix held one, the old gap none. */
    CHECK(prefix_resident >= 7u && gap_resident >= 7u && !tail_resident);
    CHECK(resident_flush_crossings >= 12u);
    CHECK(s.checkpoint_valid && s.checkpoint.len == (int)TARGET &&
          !memcmp(s.checkpoint.v, v, (size_t)TARGET * sizeof(int)));
    printf("encoder residency: prefix %u + gap %u resident chunks across %u arena flushes, "
           "%u acquire/%u release, heads=%u never resident\n",
           prefix_resident, gap_resident, resident_flush_crossings,
           encoder_acquires, encoder_releases, heads);
    harness_session_close(&s);

    /* First-turn refresh of a 16400-token prompt with batch capture. */
    static ds4_engine fe; static ds4_session fs;
    harness_session_open(&fe, &fs, CTX, 2048u);
    static int fresh[REFRESH];
    for (uint32_t i = 0; i < REFRESH; ++i) fresh[i] = 20000 + (int)i;
    ds4_tokens first = {.v = fresh, .len = (int)REFRESH, .cap = (int)REFRESH};
    controls(fresh);
    refreshing = true; resident_flush_crossings = 0; was_resident = false;
    rc = ds4_session_scr_logical_policy(&fs, &first, 36864u, 32u, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "refresh policy: %s\n", err);
    const ds4_scr_refresh_receipt *f = &r.execution.refresh;
    CHECK(!rc && r.published && r.execution.route == DS4_SCR_DENSE_REFRESH && f->sealed);
    CHECK(f->source_start == 0u && f->source_end == REFRESH && f->source_completed == REFRESH);
    CHECK(f->source_chunks == 1u && f->normal_calls == 1u && f->completed_heads == 1u && heads == 1u);
    CHECK(encoder_acquires == 1u && encoder_releases == 1u && !fs.ds41_graph.encoder_resident);
    CHECK(refresh_resident >= 7u && resident_flush_crossings >= 6u);
    const ds41_scr_archive *sealed = fs.ds41_graph.archive;
    CHECK(sealed && sealed->sealed && ds41_archive_count(sealed) == REFRESH &&
          sealed->raw->committed_tokens == REFRESH);
    printf("refresh: %u-token source in %u call, %u chunks (%u resident) across %u arena flushes, "
           "%u acquire/%u release, heads=%u\n", REFRESH, f->normal_calls, refresh_chunks,
           refresh_resident, resident_flush_crossings, encoder_acquires, encoder_releases, heads);
    harness_session_close(&fs);
    refreshing = false;
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(indices);
    puts("SCR encoder residency spans capture flushes, refresh included (host storage, simulated rows)");
    return 0;
}
