/* Persistent SCR checkpoint pool through the logical policy:
 *   1. cross-restart reuse: a turn on session A seals and saves its archive;
 *      after A and its pool close, session B opens the same directory and its
 *      first request splices multiple ranges from the restored checkpoint;
 *   2. branch fan-out: an unrelated prompt displaces the hot archive, and an
 *      edit of the first branch selects and restores that branch's entry;
 *   3. same-session edit: tokens shared with the live transcript are covered
 *      from the archive (planned and pool-scored), not recomputed, because a
 *      planned request resets the live graph at token 0;
 *   4. two sessions on one pool (batched server slots): each live archive
 *      keeps its own saved-state record, so another session's save neither
 *      makes it a restore candidate of its own nor forces a rewrite.
 * Strict host: tests/scr_host_storage.c replaces GPU buffers and model rows
 * are simulated. Restored owner rows and raw records are checked against the
 * values the fixture captured; this is ownership evidence, not inference. */
#include "scr_engine_harness.h"
#include <dirent.h>
#include <sys/stat.h>

#define CTX 8192u
#define CAP 4096u
#define BASE 1400u
#define EDIT 64u

static char pool_dir[PATH_MAX];

static void check_restored(ds4_session *s, uint32_t count);
static void policy(ds4_session *s, ds4_scr_pool *p, int *v, uint32_t n,
                   ds4_scr_pool_receipt *pr, ds4_scr_logical_receipt *r, uint32_t check) {
    ds4_tokens full = {.v = v, .len = (int)n, .cap = (int)n};
    session = s; /* the harness model rows check the active session */
    controls(v);
    CHECK(!ds4_session_scr_pool_select(s, p, &full, 32u, 201, pr));
    if (check) check_restored(s, check);
    char err[256] = {0};
    int rc = ds4_session_scr_logical_policy(s, &full, CAP, 32u, 201, r, err, sizeof(err));
    if (rc) fprintf(stderr, "pool policy: %s\n", err);
    CHECK(!rc && r->published && s->checkpoint.len == (int)n &&
          !memcmp(s->checkpoint.v, v, (size_t)n * sizeof(int)));
    ds4_scr_pool_receipt saved = {0};
    CHECK(!ds4_session_scr_pool_save(s, p, false, &saved));
    pr->saved |= saved.saved;
    pr->entries = saved.entries;
}
static uint32_t files_named(const char *suffix) {
    DIR *d = opendir(pool_dir); CHECK(d);
    uint32_t n = 0; size_t k = strlen(suffix);
    for (struct dirent *e; (e = readdir(d));) {
        size_t m = strlen(e->d_name);
        n += m >= k && !strcmp(e->d_name + m - k, suffix);
    }
    closedir(d);
    return n;
}
/* The restored owners and raw records hold exactly the captured values. */
static void check_restored(ds4_session *s, uint32_t count) {
    ds41_scr_archive *a = s->ds41_graph.archive;
    CHECK(a && a->sealed && a->start == 0u && ds41_archive_count(a) == count && a->cap == CAP);
    float row[512];
    for (uint32_t o = 0; o < 4; ++o) {
        const uint32_t ratio = o < 3 ? 2u : 1u;
        for (uint32_t r = 0; r < count / ratio; r += 97u) {
            CHECK(ds4_gpu_tensor_read(a->source, ds41_archive_owner(CAP, o) + (uint64_t)r * 2048u, row, 2048u));
            for (uint32_t j = 0; j < 512; j += 61u) CHECK(row[j] == row_value(r * ratio, o, j));
            CHECK(ds4_gpu_tensor_read(a->source, ds41_archive_owner(CAP, o) +
                (uint64_t)(CAP / ratio) * 2048u + (uint64_t)r * 512u, row, 512u));
            for (uint32_t j = 0; j < 128; j += 13u) CHECK(row[j] == row_value(r * ratio, o + 4u, j));
        }
    }
    CHECK(a->raw && a->raw->committed_tokens == count);
    for (uint32_t record = 0; record < count / 8u; record += 37u) {
        CHECK(ds41_raw_read(a->raw, record, 1, a->raw_io, DS41_ARCHIVE_READ_BYTES, NULL, NULL));
        for (uint32_t l = 0; l < 40; l += 7u)
            for (uint32_t k = 0; k < 8; k += 3u) {
                memcpy(row, a->raw_io + (uint64_t)(l * 8u + k) * DS41_RAW_ROW_BYTES, sizeof(row));
                for (uint32_t j = 0; j < 512; j += 61u) CHECK(row[j] == row_value(record * 8u + k, l + 8u, j));
            }
    }
}

/* A mid-prompt edit on the same session: the request shares 600 tokens with
 * the live transcript, then 64 fresh tokens and an 800-token trailing match.
 * Both the shared prefix (from token 8) and the suffix come from the archive. */
static void same_session_edit(const int *a1, int *a2) {
    static ds4_engine e; static ds4_session s;
    harness_session_open(&e, &s, CTX, 512u);
    ds4_tokens first = {.v = (int *)a1, .len = (int)BASE, .cap = (int)BASE};
    ds4_tokens edit = {.v = a2, .len = (int)(BASE + EDIT), .cap = (int)(BASE + EDIT)};
    ds4_scr_logical_receipt r; char err[256] = {0};
    controls(a1);
    CHECK(!ds4_session_scr_logical_policy(&s, &first, CAP, 32u, 201, &r, err, sizeof(err)));
    CHECK(r.execution.route == DS4_SCR_DENSE_REFRESH && s.checkpoint.len == (int)BASE);
    controls(a2);
    int rc = ds4_session_scr_logical_policy(&s, &edit, CAP, 32u, 201, &r, err, sizeof(err));
    if (rc) fprintf(stderr, "same-session edit: %s\n", err);
    CHECK(!rc && r.published && r.execution.route == DS4_SCR_DENSE_REUSE);
    CHECK(r.planned_ranges == 2u && r.execution.multi.committed_tokens >= 1200u);
    const ds4_scr_range *head = &r.physical_plan[0], *tail = &r.physical_plan[1];
    CHECK(head->destination_start <= 8u && head->destination_start + head->count >= 590u);
    CHECK(tail->destination_start >= 600u + EDIT && tail->count >= 600u);
    CHECK(r.execution.multi.prefix_completed <= 8u);
    CHECK(s.checkpoint.len == (int)(BASE + EDIT) &&
          !memcmp(s.checkpoint.v, a2, (size_t)(BASE + EDIT) * sizeof(int)));
    printf("same-session edit: shared prefix [%u,%u) and suffix [%u,%u) reused, %u of %u tokens\n",
           head->destination_start, head->destination_start + head->count,
           tail->destination_start, tail->destination_start + tail->count,
           r.execution.multi.committed_tokens, BASE + EDIT);
    harness_session_close(&s);
}

/* Two live sessions share one pool. Saving B must not make A's own saved
 * state look cold to A: A's edit is scored against its live archive, its own
 * entry is no candidate and nothing is restored or rewritten. */
static void two_sessions(const int *a1, int *a2, int *b1) {
    char dir[PATH_MAX + 16], err[256];
    snprintf(dir, sizeof(dir), "%s-two", pool_dir);
    CHECK(!mkdir(dir, 0700) || errno == EEXIST);
    static ds4_engine ea, eb; static ds4_session sa, sb;
    harness_session_open(&ea, &sa, CTX, 512u);
    harness_session_open(&eb, &sb, CTX, 512u);
    ds4_scr_pool *pool = NULL;
    CHECK(!ds4_scr_pool_open(&pool, dir, CTX, CAP, 0, err, sizeof(err)) && !ds4_scr_pool_entries(pool));
    ds4_scr_pool_receipt pr; ds4_scr_logical_receipt r;
    policy(&sa, pool, (int *)a1, BASE, &pr, &r, 0);
    CHECK(pr.saved && pr.entries == 1u);
    policy(&sb, pool, b1, BASE, &pr, &r, 0);
    CHECK(pr.saved && pr.entries == 2u);
    session = &sa;
    ds4_tokens edited = {.v = a2, .len = (int)(BASE + EDIT), .cap = (int)(BASE + EDIT)};
    controls(a2);
    CHECK(!ds4_session_scr_pool_select(&sa, pool, &edited, 32u, 201, &pr));
    CHECK(pr.selected && !pr.restored && pr.candidates == 1u && pr.hot_covered >= 1200u);
    /* A's unchanged archive is already held: a shutdown save, which skips
     * nothing by owner, finds it saved and writes nothing. */
    CHECK(!ds4_session_scr_pool_save(&sa, pool, true, &pr) && !pr.saved && pr.entries == 2u);
    CHECK(sa.ds41_graph.archive->tokens[0] == a1[0] && sb.ds41_graph.archive->tokens[0] == b1[0]);
    ds4_scr_pool_close(pool);
    harness_session_close(&sb);
    harness_session_close(&sa);
    printf("two sessions: each live archive keeps its saved state on a shared pool\n");
}

int main(void) {
    CHECK(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    const char *tmp = getenv("TMPDIR"); CHECK(tmp && *tmp);
    snprintf(pool_dir, sizeof(pool_dir), "%s/scr-pool", tmp);
    CHECK(!mkdir(pool_dir, 0700) || errno == EEXIST);
    static int a1[BASE], a2[BASE + EDIT], b1[BASE], a3[BASE + 2u * EDIT];
    for (uint32_t i = 0; i < BASE; ++i) { a1[i] = 1000 + (int)i; b1[i] = 50000 + (int)i; }
    /* An edit inserts fresh tokens in the middle of the earlier prompt. */
    memcpy(a2, a1, 600u * sizeof(int));
    for (uint32_t i = 0; i < EDIT; ++i) a2[600u + i] = 90000 + (int)i;
    memcpy(a2 + 600u + EDIT, a1 + 600u, (BASE - 600u) * sizeof(int));
    memcpy(a3, a2, 300u * sizeof(int));
    for (uint32_t i = 0; i < EDIT; ++i) a3[300u + i] = 95000 + (int)i;
    memcpy(a3 + 300u + EDIT, a2 + 300u, (BASE + EDIT - 300u) * sizeof(int));
    ds4_scr_pool_receipt pr; ds4_scr_logical_receipt r; char err[256];

    /* Session A: the first turn refreshes and the pool keeps its archive. */
    static ds4_engine ea; static ds4_session sa;
    harness_session_open(&ea, &sa, CTX, 512u);
    ds4_scr_pool *pool = NULL;
    CHECK(!ds4_scr_pool_open(&pool, pool_dir, CTX, CAP, 0, err, sizeof(err)) && !ds4_scr_pool_entries(pool));
    policy(&sa, pool, a1, BASE, &pr, &r, 0);
    CHECK(r.execution.route == DS4_SCR_DENSE_REFRESH && !pr.selected && !pr.restored);
    CHECK(pr.saved && pr.entries == 1u && files_named(".scr.meta") == 1u && files_named(".tmp") == 0u);
    /* An unchanged archive is not written twice. */
    CHECK(!ds4_session_scr_pool_save(&sa, pool, true, &pr) && !pr.saved && pr.entries == 1u);
    /* Scoring the edit against the live session's archive counts the 600
     * tokens shared with the live transcript as coverage, as the policy plans. */
    ds4_tokens edited = {.v = a2, .len = (int)(BASE + EDIT), .cap = (int)(BASE + EDIT)};
    controls(a2);
    CHECK(!ds4_session_scr_pool_select(&sa, pool, &edited, 32u, 201, &pr));
    CHECK(pr.selected && !pr.restored && !pr.candidates && pr.hot_covered >= 1200u);
    ds4_scr_pool_close(pool);
    harness_session_close(&sa);

    /* Session B after a restart: stale pool temporaries are cleared, other
     * files survive, and the first request splices from the restored entry. */
    char path[PATH_MAX + 64];
    snprintf(path, sizeof(path), "%s/0123456789abcdef.scr.raw.tmp", pool_dir);
    FILE *f = fopen(path, "w"); CHECK(f); fclose(f);
    snprintf(path, sizeof(path), "%s/unrelated.kv", pool_dir);
    f = fopen(path, "w"); CHECK(f); fclose(f);
    static ds4_engine eb; static ds4_session sb;
    harness_session_open(&eb, &sb, CTX, 512u);
    CHECK(!ds4_scr_pool_open(&pool, pool_dir, CTX, CAP, 0, err, sizeof(err)) && ds4_scr_pool_entries(pool) == 1u);
    CHECK(files_named(".tmp") == 0u && files_named("unrelated.kv") == 1u);
    CHECK(!sb.ds41_graph.archive);
    policy(&sb, pool, a2, BASE + EDIT, &pr, &r, BASE);
    CHECK(pr.selected && pr.restored && pr.candidates == 1u && !pr.hot_covered && pr.best_covered >= 1200u);
    CHECK(r.execution.route == DS4_SCR_DENSE_REUSE && r.planned_ranges >= 2u &&
          r.execution.multi.committed_tokens >= 1200u && r.execution.multi.rebase_committed);
    printf("cross-restart: restored %u-token checkpoint, %u ranges reused %u of %u tokens\n",
           BASE, r.planned_ranges, r.execution.multi.committed_tokens, BASE + EDIT);
    /* An edit is not an extension: the earlier entry stays alongside it. */
    CHECK(pr.saved && pr.entries == 2u);

    /* Branch B displaces branch A's hot archive; both stay in the pool. */
    policy(&sb, pool, b1, BASE, &pr, &r, 0);
    CHECK(pr.selected && !pr.restored && !pr.best_covered);
    CHECK(r.execution.route == DS4_SCR_DENSE_REFRESH && pr.saved && pr.entries == 3u);
    CHECK(sb.ds41_graph.archive->tokens[0] == b1[0]);

    /* An edit of branch A restores A's entry and reuses nearly all of it. */
    policy(&sb, pool, a3, BASE + 2u * EDIT, &pr, &r, 0);
    CHECK(pr.selected && pr.restored && pr.candidates == 2u && pr.best_covered > pr.hot_covered);
    CHECK(r.execution.route == DS4_SCR_DENSE_REUSE && r.planned_ranges >= 2u &&
          r.execution.multi.committed_tokens >= 1280u);
    printf("fan-out: branch A restored over branch B, %u ranges reused %u of %u tokens\n",
           r.planned_ranges, r.execution.multi.committed_tokens, BASE + 2u * EDIT);
    CHECK(pr.entries == 4u);
    /* A continuation only appends: it is not rewritten after the request, and
     * the shutdown save replaces the entry it extends. */
    static int a4[BASE + 2u * EDIT + 256u];
    memcpy(a4, a3, sizeof(a3));
    for (uint32_t i = 0; i < 256u; ++i) a4[BASE + 2u * EDIT + i] = 70000 + (int)i;
    const uint32_t end = sb.ds41_graph.archive->end;
    policy(&sb, pool, a4, BASE + 2u * EDIT + 256u, &pr, &r, 0);
    CHECK(r.execution.route == DS4_SCR_DENSE_CONTINUE && !pr.selected && !pr.saved &&
          sb.ds41_graph.archive->end > end);
    CHECK(!ds4_session_scr_pool_save(&sb, pool, true, &pr) && pr.saved && pr.entries == 4u);
    ds4_scr_pool_close(pool);

    /* A mismatched geometry ignores the entries; a smaller byte budget evicts
     * least recently used entries at open. */
    CHECK(!ds4_scr_pool_open(&pool, pool_dir, CTX, 2048u, 0, err, sizeof(err)) && !ds4_scr_pool_entries(pool));
    ds4_scr_pool_close(pool);
    CHECK(files_named(".scr.meta") == 4u);
    CHECK(!ds4_scr_pool_open(&pool, pool_dir, CTX, CAP, 200ull << 20, err, sizeof(err)) &&
          ds4_scr_pool_entries(pool) == 1u);
    ds4_scr_pool_close(pool);
    CHECK(files_named(".scr.meta") == 1u && files_named(".scr.raw") == 1u && files_named("unrelated.kv") == 1u);

    harness_session_close(&sb);
    same_session_edit(a1, a2);
    two_sessions(a1, a2, b1);
    ds4_gpu_tensor_free(rows); ds4_gpu_tensor_free(indices);
    puts("SCR checkpoint pool: cross-restart reuse, branch fan-out, roundtrip, budget, same-session shared prefix, two sessions (host storage, simulated rows)");
    return 0;
}
