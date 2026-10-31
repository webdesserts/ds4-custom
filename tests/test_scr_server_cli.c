/* Pure option/config/startup consumer for the collapsed SCR CLI. Session APIs
 * are non-model stubs; no listener or model is started. */
#include "../ds4.h"
#include <assert.h>
#include <sys/wait.h>
#include <string.h>
static int order[4], calls, failure = -1; static ds4_session *expected;
static int record_selection(ds4_session *s, int kind) {
    assert(s == expected && calls < 4);
    order[calls++] = kind;
    return failure == kind ? kind + 4 : 0;
}
static int fixture_budget(ds4_session *s, uint32_t ms) {
    assert(ms == 5000u);
    return record_selection(s, 0);
}
static int fixture_padding(ds4_session *s, ds4_scr_padding_policy p, char *e, size_t n) {
    (void)e; (void)n;
    assert(p == DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE);
    return record_selection(s, 1);
}
static int fixture_batch(ds4_session *s, char *e, size_t n) {
    (void)e; (void)n;
    return record_selection(s, 2);
}
#define ds4_session_set_scr_planner_budget fixture_budget
#define ds4_session_scr_set_padding_policy fixture_padding
#define ds4_session_scr_enable_capture_batch_chunks fixture_batch
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
static int cli_exit(int argc, char **argv) {
    pid_t p = fork(); assert(p >= 0);
    if (!p) {
        server_config c = parse_options(argc, argv);
        server_scr_resolve_config(&c);
        _exit(server_scr_config_valid(&c, false) ? 0 : 2);
    }
    int status; assert(waitpid(p, &status, 0) == p && WIFEXITED(status));
    return WEXITSTATUS(status);
}
static server_config resolved(int argc, char **argv) {
    server_config c = parse_options(argc, argv);
    server_scr_resolve_config(&c);
    return c;
}
static void startup_selects_full_stack(const server_config *c) {
    int tag; expected = (ds4_session *)&tag; char err[128];
    calls = 0; failure = -1;
    assert(!server_scr_apply_startup(c, expected, err, sizeof(err)) &&
        calls == 3 && order[0] == 0 && order[1] == 1 && order[2] == 2);
}
#define ARGC(a) ((int)(sizeof(a) / sizeof((a)[0])))
int main(void) {
    /* --scr alone selects the production stack with context-sized archive. */
    char *scr[] = {"server", "--metal", "--ctx", "262144", "--scr"};
    server_config c = resolved(ARGC(scr), scr);
    assert(c.scr_enabled && c.scr_tail_tokens == 262144 &&
        c.scr_max_ranges == (int)DS4_SCR_MULTI_MAX_RANGES && !c.scr_max_ranges_present &&
        c.scr_planner_budget_ms == 5000 &&
        server_scr_config_valid(&c, false));
    startup_selects_full_stack(&c);
    /* Each selector failure stops startup at that selector, without retry. */
    int tag; expected = (ds4_session *)&tag; char err[128];
    for (int fail = 0; fail < 3; ++fail) {
        failure = fail; calls = 0;
        assert(server_scr_apply_startup(&c, expected, err, sizeof(err)) == fail + 4 && calls == fail + 1);
    }
    failure = -1;
    /* Default context, explicit tuning flags. */
    char *tuned[] = {"server", "--metal", "--scr", "--scr-tail-tokens", "4096",
        "--scr-max-ranges", "3", "--scr-planner-budget-ms", "5000"};
    server_config t = resolved(ARGC(tuned), tuned);
    assert(t.scr_tail_tokens == 4096 && t.scr_max_ranges == 3 && t.scr_max_ranges_present &&
        server_scr_config_valid(&t, false));
    char *default_ctx[] = {"server", "--metal", "--scr"};
    assert(resolved(ARGC(default_ctx), default_ctx).scr_tail_tokens == 32768);
    /* The deployed launch line, retired selectors included, still starts and
     * resolves to the same production stack. */
    char *production[] = {"server", "--model", "/inert/not-opened.gguf", "--backend", "metal",
        "--host", "127.0.0.1", "--port", "5800", "--ctx", "262144", "--scr-tail-tokens", "262144",
        "--scr-max-ranges", "10", "--scr-logical-coordinates", "--scr-padding-policy",
        "planner-fresh-gap-space-v1", "--scr-capture-batch-trial", "--scr-planner-budget-ms", "5000"};
    server_config pinned = resolved(ARGC(production), production);
    assert(pinned.scr_enabled && pinned.ctx_size == 262144 && pinned.scr_tail_tokens == 262144 &&
        pinned.scr_max_ranges == 10 &&
        server_scr_config_valid(&pinned, false));
    startup_selects_full_stack(&pinned);
    assert(cli_exit(ARGC(production), production) == 0);
    /* Retired selectors are accepted quietly alongside --scr. */
    const char *legacy[] = {"--scr-logical-coordinates", "--scr-capture-batch", "--scr-capture-batch-trial",
        "--scr-batch-suffix", "--scr-noncapturing-fresh-once"};
    for (size_t i = 0; i < sizeof(legacy) / sizeof(legacy[0]); ++i) {
        char *args[] = {"server", "--metal", "--scr", (char *)legacy[i]};
        assert(cli_exit(4, args) == 0);
    }
    /* The retired padding selector consumes its value; only the historical
     * fresh-gap value is accepted, and the removed renderer mode is refused. */
    char *fresh[] = {"server", "--metal", "--scr", "--scr-padding-policy", "planner-fresh-gap-space-v1"};
    assert(cli_exit(ARGC(fresh), fresh) == 0);
    char *joins[] = {"server", "--metal", "--scr", "--scr-padding-policy", "renderer-user-joins-v1"};
    assert(cli_exit(ARGC(joins), joins) == 1);
    /* The long-retired row option remains an unknown option. */
    char *row[] = {"server", "--metal", "--scr", "--scr-capture-row-chunks"};
    assert(cli_exit(ARGC(row), row) == 2);
    /* SCR off: no selection, and tuning flags without SCR are refused. */
    char *off[] = {"server", "--metal"};
    server_config o = resolved(ARGC(off), off);
    assert(!o.scr_enabled && !o.scr_tail_tokens && server_scr_config_valid(&o, false));
    calls = 0; assert(!server_scr_apply_startup(&o, expected, err, sizeof(err)) && !calls);
    char *k_only[] = {"server", "--metal", "--scr-max-ranges", "6"};
    assert(cli_exit(ARGC(k_only), k_only) == 2);
    char *zero[] = {"server", "--metal", "--scr-tail-tokens", "0"};
    assert(cli_exit(ARGC(zero), zero) == 0 && !resolved(ARGC(zero), zero).scr_enabled);
    /* K bounds. */
    for (int k = 1; k <= (int)DS4_SCR_MULTI_MAX_RANGES; ++k) {
        char kb[4]; snprintf(kb, sizeof(kb), "%d", k);
        char *ka[] = {"server", "--metal", "--scr", "--scr-max-ranges", kb};
        assert(cli_exit(ARGC(ka), ka) == 0);
    }
    char over[4]; snprintf(over, sizeof(over), "%u", DS4_SCR_MULTI_MAX_RANGES + 1u);
    char *bad_k[] = {"server", "--metal", "--scr", "--scr-max-ranges", over};
    assert(cli_exit(ARGC(bad_k), bad_k) == 2);
    char *zero_k[] = {"server", "--metal", "--scr", "--scr-max-ranges", "0"};
    assert(cli_exit(ARGC(zero_k), zero_k) == 2);
    /* The disk directory and budget select the checkpoint pool; the disk KV
     * cache tuning options stay incompatible with SCR. */
    char *pool[] = {"server", "--metal", "--scr", "--kv-disk-dir", "/inert/not-created",
        "--kv-disk-space-mb", "65536"};
    server_config pc = resolved(ARGC(pool), pool);
    assert(pc.kv_disk_dir && !strcmp(pc.kv_disk_dir, "/inert/not-created") &&
        pc.kv_disk_space_mb == 65536u && cli_exit(ARGC(pool), pool) == 0);
    const char *kv_tuning[] = {"--kv-cache-min-tokens", "--kv-cache-continued-interval-tokens"};
    for (size_t i = 0; i < sizeof(kv_tuning) / sizeof(kv_tuning[0]); ++i) {
        char *args[] = {"server", "--metal", "--scr", "--kv-disk-dir", "/inert/not-created",
            (char *)kv_tuning[i], "4096"};
        assert(cli_exit(ARGC(args), args) == 2);
    }
    /* DSpark coexists with SCR only as the --dspark/--mtp-model pair. */
    char *dspark[] = {"server", "--metal", "--scr", "--dspark", "--mtp-model", "/inert/not-opened.gguf"};
    server_config dc = resolved(ARGC(dspark), dspark);
    assert(dc.scr_enabled && dc.engine.dspark && dc.engine.mtp_path &&
        server_scr_config_valid(&dc, false) && cli_exit(ARGC(dspark), dspark) == 0);
    char *dspark_alone[] = {"server", "--metal", "--scr", "--dspark"};
    char *mtp_alone[] = {"server", "--metal", "--scr", "--mtp-model", "/inert/not-opened.gguf"};
    /* Batched sessions coexist with SCR and DSpark: slots decode in executor
     * turns. The mixed prefill quantum stays a refused batch tuning flag. */
    char *batched[] = {"server", "--metal", "--scr", "--batched-session", "4", "--dspark",
        "--mtp-model", "/inert/not-opened.gguf"};
    server_config bc = resolved(ARGC(batched), batched);
    assert(bc.batched_sessions == 4 && server_scr_config_valid(&bc, false) &&
        cli_exit(ARGC(batched), batched) == 0);
    char *mixed[] = {"server", "--metal", "--scr", "--batched-session", "2",
        "--mixed-prefill-quantum", "256"};
    assert(cli_exit(ARGC(mixed), mixed) == 2);
    char *glm_mtp[] = {"server", "--metal", "--scr", "--mtp-timing"};
    assert(cli_exit(ARGC(dspark_alone), dspark_alone) == 2 &&
        cli_exit(ARGC(mtp_alone), mtp_alone) == 2 &&
        cli_exit(ARGC(glm_mtp), glm_mtp) == 2);
    /* Incompatible engine modes. */
    for (unsigned mode = 0; mode < 3; mode++) {
        server_config d = c;
        if (mode == 0) d.engine.dspark = true;
        if (mode == 1) d.engine.mtp_path = "/inert/not-opened.gguf";
        if (mode == 2) d.engine.glm_mtp = true;
        assert(!server_scr_config_valid(&d, false));
    }
    assert(DS4_DEFAULT_TEMPERATURE == 1.0f && DS4_DEFAULT_TOP_P == 1.0f &&
        DS4_DEFAULT_MIN_P == 0.05f);
    puts("scr cli: --scr production stack, tuning flags, retired selectors accepted, deployed argv, K bounds, checkpoint pool options, DSpark pair, batched sessions; stub sessions, no model");
    return 0;
}
