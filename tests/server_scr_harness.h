/* Model-free SCR server harness: the production HTTP/parser/native renderer/
 * worker/prefill/generated-message parser/usage/SSE/telemetry code runs from
 * ../ds4_server.c. Only tokenizer, identity, session policy, retirement and
 * decode/session storage are substituted; no listener or engine is opened.
 * Include once per test binary, before the test bodies. */
#ifndef DS4_TEST_SERVER_SCR_HARNESS_H
#define DS4_TEST_SERVER_SCR_HARNESS_H
#include "ds4.h"
#include "ds4_kvstore.h"
#include "../adapter/include/cache_reuse_utils.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

static struct {
    ds4_tokens live;
    int pos, policy_calls, sync_calls, tokenize_calls, releases, samples;
    int policy_rc, sync_rc, decode_mode, eval_rc;
    int physical_sync_calls, compare_calls, compare_rc, rewrite_calls, logical_calls;
    bool mapped, map_after_policy, same_logical_text;
    bool source, source_retiring, cancel_in_policy, poison;
    uint32_t source_capacity, source_count, configured_capacity;
    bool logical; /* Synthesize a padded logical plan instead of passing the route through. */
    int configured_k, route, cancel_sets, frees, engine_closes, retire_calls;
    ds4_scr_retire_outcome retire_outcome;
    bool force_retire, observation, dispose_refusal;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
    void *request_job;
    bool probe_liveness, headers_during_policy;
    int probe_fd;
    char *rendered;
    const char *output;
    ds4_scr_dense_policy_receipt last;
} fixture;

static void fixture_tokenize(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;
    fixture.tokenize_calls++;
    free(fixture.rendered);
    fixture.rendered = strdup(text);
    size_t n = strlen(text);
    out->v = malloc(n * sizeof(int));
    assert(out->v);
    out->len = out->cap = (int)n;
    for (size_t i = 0; i < n; i++) out->v[i] = (unsigned char)text[i];
}
static void fixture_install(const ds4_tokens *p) {
    int n = p->len;
    int *ids = malloc((size_t)n * sizeof(int));
    assert(ids);
    memcpy(ids, p->v, (size_t)n * sizeof(int));
    ds4_tokens_free(&fixture.live);
    fixture.live.v = ids;
    fixture.live.len = fixture.live.cap = fixture.pos = n;
}
static void fixture_cancel_job(void *ud);
static void fixture_probe_liveness(void);
static int fixture_policy(ds4_session *s, const ds4_tokens *p, uint32_t tail,
                          uint32_t k, ds4_scr_dense_policy_receipt *r, char *err, size_t errlen) {
    (void)s;
    assert(tail == (fixture.configured_capacity ? fixture.configured_capacity : 128u) &&
           k == (uint32_t)fixture.configured_k);
    assert(p->len == (int)strlen(fixture.rendered));
    for (int i = 0; i < p->len; i++) assert(p->v[i] == (unsigned char)fixture.rendered[i]);
    if (fixture.poison) assert(fixture.pos == 77 && fixture.live.v[0] == -999);
    fixture.policy_calls++;
    *r = (ds4_scr_dense_policy_receipt){
        .requested_tokens = (uint32_t)p->len, .configured_cap = tail, .configured_k = k,
        .route = fixture.policy_rc ? DS4_SCR_DENSE_ERROR :
            fixture.route ? (ds4_scr_dense_route)fixture.route :
            fixture.source ? DS4_SCR_DENSE_REUSE : DS4_SCR_DENSE_REFRESH,
        .reason = fixture.source ? DS4_SCR_DENSE_HIT : DS4_SCR_DENSE_MISSING,
        .complete_n = !fixture.policy_rc, .source_start = 8, .source_end = 136,
        .continuation_work_unknown = fixture.policy_rc != 0,
        .multi = { .requested_tokens = (uint32_t)p->len,
            .committed_tokens = fixture.source && !fixture.policy_rc ? 128 : 0,
            .discarded_tokens = fixture.policy_rc ? 8 : 0,
            .partial_work_unknown = fixture.policy_rc != 0 },
    };
    fixture.last = *r;
    if (fixture.probe_liveness) fixture_probe_liveness();
    if (fixture.cancel_in_policy) {
        /* The installed production callback owns this job, not a fake router. */
        assert(fixture.cancel_ud);
        fixture_cancel_job(fixture.request_job);
        assert(fixture.cancel(fixture.cancel_ud));
        r->route = DS4_SCR_DENSE_ERROR; r->reason = DS4_SCR_DENSE_CANCELLED;
        r->complete_n = false; r->error_code = DS4_SESSION_SYNC_INTERRUPTED;
        fixture.last = *r;
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (fixture.policy_rc) {
        snprintf(err, errlen, "substituted policy failure");
        return fixture.policy_rc;
    }
    fixture_install(p);
    fixture.mapped = fixture.map_after_policy;
    fixture.source = true;
    if (r->route == DS4_SCR_DENSE_REFRESH) {
        uint32_t covered = fixture.source_count ? fixture.source_count : 128u;
        uint32_t capacity = fixture.source_capacity ? fixture.source_capacity : covered;
        r->refresh = (ds4_scr_refresh_receipt){ .requested_tokens = (uint32_t)p->len,
            .source_start = 8, .source_end = 8u + covered, .source_count = covered,
            .reserved_capacity = capacity, .published_end = 8u + covered, .published_count = covered,
            .sealed = true, .source_usable = true, .complete_n = true };
    }
    fixture.last = *r;
    return 0;
}
static int fixture_logical_policy(ds4_session *s, const ds4_tokens *p, uint32_t tail,
        uint32_t k, int pad, ds4_scr_logical_receipt *r, char *err, size_t errlen) {
    assert(pad == 201); ++fixture.logical_calls;
    memset(r, 0, sizeof(*r)); r->logical_known = true; r->logical_tokens = (uint32_t)p->len;
    int rc = fixture_policy(s, p, tail, k, &r->execution, err, errlen);
    r->execution.error_code = rc;
    if (rc) { r->execution.requested_tokens = 0; return rc; }
    if (!fixture.logical) {
        /* Unpadded identity map: the substituted route passes straight through. */
        r->physical_known = r->published = true;
        r->physical_tokens = (uint32_t)p->len; r->map_identity = 1;
        r->target_mutated = r->execution.route != DS4_SCR_DENSE_CONTINUE;
        uint32_t committed = r->execution.multi.committed_tokens;
        if (committed) {
            assert(p->len > 16 + (int)committed);
            r->planned = true; r->planned_ranges = 1; r->useful_committed = committed;
            r->physical_plan[0] = (ds4_scr_range){8, 8, committed};
        }
        fixture.last = r->execution;
        return 0;
    }
    assert(p->len > 146);
    ds4_tokens physical = {.len = p->len + 1, .cap = p->len + 1};
    physical.v = malloc((size_t)physical.len * sizeof(int)); assert(physical.v);
    memcpy(physical.v, p->v, 17u * sizeof(int)); physical.v[17] = 201;
    memcpy(physical.v + 18, p->v + 17, (size_t)(p->len - 17) * sizeof(int));
    fixture_install(&physical); ds4_tokens_free(&physical); fixture.mapped = true;
    r->physical_known = r->planned = r->published = r->target_mutated = true;
    r->physical_tokens = (uint32_t)p->len + 1u; r->new_pads = 1; r->map_identity = 41;
    r->useful_committed = 128; r->planned_ranges = 1;
    /* Synthetic transport receipt; actual permission discovery is tested elsewhere. */
    r->boundary_policy = DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1;
    r->planner_status_known = r->planner_timing_known = true;
    r->planner_status = CACHE_REUSE_COMPLETE; r->planner_best_useful = 128;
    r->planner_discovery_known = true; r->planner_discovered_matches = 1; r->planner_longest_match = 128;
    r->planner_elapsed_ns = 2000000;
    r->logical_match.selected_ranges = 1;
    r->logical_match.plan[0] = (ds4_scr_range){8, 17, 128};
    r->physical_plan[0] = (ds4_scr_range){8, 18, 128};
    r->execution.route = DS4_SCR_DENSE_REUSE; r->execution.reason = r->execution.trigger = DS4_SCR_DENSE_HIT;
    r->execution.requested_tokens = r->physical_tokens; memset(&r->execution.refresh, 0, sizeof(r->execution.refresh));
    r->execution.multi = (ds4_scr_multi_receipt){.outcome = DS4_SCR_REUSED,
        .phase = DS4_SCR_MULTI_PUBLICATION, .requested_tokens = r->physical_tokens,
        .input_spans = 1, .normalized_spans = 1, .attempted_spans = 1, .materialized_spans = 1,
        .installed_spans = 1, .committed_spans = 1, .attempted_tokens = 128, .materialized_tokens = 128,
        .installed_tokens = 128, .committed_tokens = 128, .prefix_completed = 18,
        .suffix_completed = r->physical_tokens - 146u, .head_completed = true, .published = true,
        .target_mutated = true, .prefix_seconds = 0.25, .stage_seconds = 0.125,
        .install_seconds = 0.0625, .suffix_seconds = 0.5, .head_seconds = 0.03125};
    r->execution.source_generation = 1; r->execution.source_start = 8; r->execution.source_end = 136;
    r->execution.multi.ranges[0] = r->physical_plan[0]; fixture.last = r->execution;
    return 0;
}

static int fixture_sync(ds4_session *s, const ds4_tokens *p, char *err, size_t n) {
    (void)s;
    assert(!fixture.mapped);
    fixture.sync_calls++;
    if (fixture.sync_rc) { snprintf(err, n, "substituted repair sync failure"); return fixture.sync_rc; }
    fixture_install(p);
    return 0;
}
static bool fixture_mapped(const ds4_session *s) { (void)s; return fixture.mapped; }
static int fixture_physical_sync(ds4_session *s, const ds4_tokens *p, char *err, size_t n) {
    (void)s; (void)err; (void)n;
    assert(fixture.mapped); ++fixture.physical_sync_calls; fixture_install(p); return 0;
}
static int fixture_compare(ds4_session *s, const ds4_tokens *p, const char *text,
                           size_t length, ds4_logical_comparison *out, char *err, size_t n) {
    (void)s; (void)p;
    assert(fixture.mapped && text && length == strlen(text));
    ++fixture.compare_calls;
    if (fixture.compare_rc) { snprintf(err, n, "substituted logical comparison failure"); return 1; }
    *out = (ds4_logical_comparison){.live_tokens = fixture.pos,
        .common_tokens = (int)fixture.last.requested_tokens, .text_equal = fixture.same_logical_text};
    return 0;
}
static int fixture_pos(ds4_session *s) { (void)s; return fixture.pos; }
static int fixture_ctx(ds4_session *s) { (void)s; return 32768; }
static const ds4_tokens *fixture_tokens(ds4_session *s) { (void)s; return &fixture.live; }
static int fixture_common(ds4_session *s, const ds4_tokens *p) {
    (void)s;
    int n = 0;
    while (n < fixture.live.len && n < p->len && fixture.live.v[n] == p->v[n]) n++;
    return n;
}
static bool fixture_identity(ds4_engine *e) { (void)e; return true; }
static bool fixture_checkpoint_valid(const ds4_session *s) { (void)s; return fixture.live.len > 0; }
static bool fixture_fingerprint_prefix(const ds4_session *s, const ds4_vision_span *v, size_t n) {
    (void)s; (void)v; (void)n; return true; /* Synthetic identity, never image/model data. */
}
static bool fixture_vision(ds4_session *s, const ds4_vision_span *v, size_t n) {
    (void)s; (void)v; return n == 0;
}
static void fixture_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    (void)s; (void)fn; (void)ud;
}
static void fixture_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    (void)s;
    if (fn && !fixture.cancel) fixture.request_job = ud;
    fixture.cancel_sets++; fixture.cancel = fn; fixture.cancel_ud = ud;
}
static ds4_scr_retire_outcome fixture_release(ds4_session *s, ds4_scr_retire_receipt *r) {
    (void)s; fixture.retire_calls++;
    *r = (ds4_scr_retire_receipt){ .outcome = fixture.force_retire ?
        fixture.retire_outcome : fixture.source ? DS4_SCR_RETIRE_RELEASED : DS4_SCR_RETIRE_ABSENT };
    if (r->outcome == DS4_SCR_RETIRE_INVALID) return r->outcome;
    r->charge_known = true;
    r->charged_bytes_before = fixture.source ? 4096 : 0;
    r->charged_bytes_after = r->outcome == DS4_SCR_RETIRE_RELEASED ? 0 : r->charged_bytes_before;
    r->attempted = r->outcome == DS4_SCR_RETIRE_RELEASED || r->outcome == DS4_SCR_RETIRE_FAILED_RETAINED;
    if (r->outcome == DS4_SCR_RETIRE_FAILED_RETAINED || r->outcome == DS4_SCR_RETIRE_ALREADY_FENCED)
        fixture.source_retiring = true;
    if (r->outcome == DS4_SCR_RETIRE_RELEASED) { fixture.releases++; fixture.source = false; fixture.source_retiring = false; }
    return r->outcome;
}
static void fixture_free(ds4_session *s) { (void)s; fixture.frees++; }
static uint64_t fixture_archive_bytes(const ds4_session *s) {
    (void)s; return fixture.source ? 4096u : 0u;
}
static int fixture_archive_release(ds4_session *s) {
    ds4_scr_retire_receipt receipt;
    ds4_scr_retire_outcome outcome = fixture_release(s, &receipt);
    return outcome == DS4_SCR_RETIRE_ABSENT || outcome == DS4_SCR_RETIRE_RELEASED ? 0 : 1;
}
static int fixture_dispose(ds4_session **s) {
    if (!s) return 1;
    if (!*s) return 0;
    if (fixture.dispose_refusal || (fixture.source && fixture.source_retiring)) return 1;
    fixture_free(*s); *s = NULL; return 0;
}
static void fixture_close(ds4_engine *e) { (void)e; fixture.engine_closes++; }
static int fixture_sample(ds4_session *s, float t, int k, float p, float m, uint64_t *r) {
    (void)s; (void)t; (void)k; (void)p; (void)m; (void)r;
    fixture.samples++;
    if (fixture.decode_mode == -1) return -1;
    return fixture.output && fixture.samples == 1 ? 42 : 0;
}
static int fixture_eos(ds4_engine *e) { (void)e; return 0; }
static bool fixture_stop(ds4_engine *e, int token, ds4_think_mode m) {
    (void)e; (void)m; return token == 0;
}
static int fixture_eval(ds4_session *s, int token, char *err, size_t n) {
    (void)s; (void)token;
    if (fixture.eval_rc) { snprintf(err, n, "substituted decode failure"); return fixture.eval_rc; }
    fixture.pos++;
    return 0;
}
static char *fixture_text(ds4_engine *e, int t, size_t *n) {
    (void)e; (void)t;
    const char *p = fixture.output ? fixture.output : "";
    *n = strlen(p);
    return strdup(p);
}
static char *fixture_render_tokens(ds4_engine *e, const ds4_tokens *p, size_t *n) {
    (void)e;
    char *text = malloc((size_t)p->len + 1); assert(text);
    for (int i = 0; i < p->len; i++) text[i] = (char)p->v[i];
    text[p->len] = 0; *n = (size_t)p->len;
    return text;
}
static void fixture_invalidate(ds4_session *s) { (void)s; fixture.pos = 0; fixture.mapped = false; }
static ds4_session_rewrite_result fixture_rewrite(ds4_session *s, const ds4_tokens *p,
                                                  int common, char *err, size_t n) {
    (void)s; (void)p; (void)common; (void)err; (void)n;
    ++fixture.rewrite_calls;
    return DS4_SESSION_REWRITE_REBUILD_NEEDED;
}
/* The separately linked kvstore helper would call the real tokenizer with the
 * inert engine. Keep its exact-prefix contract while substituting tokenization. */
static void fixture_build_suffix(ds4_engine *e, const ds4_tokens *prefix, const char *suffix, ds4_tokens *out) {
    (void)e; ds4_tokens_copy(out, prefix);
    for (const unsigned char *p = (const unsigned char *)(suffix ? suffix : ""); *p; ++p)
        ds4_tokens_push(out, (int)*p);
}
#define ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix fixture_build_suffix
#define ds4_kvstore_render_tokens_text fixture_render_tokens
#define ds4_tokenize_rendered_chat fixture_tokenize
#define ds4_session_scr_logical_policy fixture_logical_policy
#define ds4_session_sync fixture_sync
#define ds4_session_sync_physical fixture_physical_sync
#define ds4_session_has_physical_coordinates fixture_mapped
#define ds4_session_compare_logical fixture_compare
#define ds4_session_pos fixture_pos
#define ds4_session_ctx fixture_ctx
#define ds4_session_tokens fixture_tokens
#define ds4_session_common_prefix fixture_common
#define ds4_session_checkpoint_valid fixture_checkpoint_valid
#define ds4_session_vision_fingerprint_prefix_matches fixture_fingerprint_prefix
#define ds4_engine_is_deepseek41 fixture_identity
#define ds4_session_vision_prefix_matches fixture_vision
#define ds4_session_set_progress fixture_progress
#define ds4_session_set_display_progress fixture_progress
#define ds4_session_set_cancel fixture_cancel
/* Synthetic cumulative counters that advance with policy calls and samples, so
 * the telemetry writer's admission-to-emission deltas are exercised. */
static ds4_scr_archive_snapshot fixture_observation_values(ds4_scr_archive_snapshot v) {
    const uint64_t rows = (uint64_t)fixture.policy_calls * 32 + (uint64_t)fixture.samples * 16;
    v.observation_known = true;
    v.engram_io = (ds4_engram_observation){.enabled = true, .rows_completed = rows,
        .read_bytes = rows * 264,
        .call_wall_nanoseconds = (uint64_t)fixture.policy_calls * 250000000u + (uint64_t)fixture.samples * 20000000u};
    if (v.present) v.active_io = (ds4_scr_owner_observation){.known = true, .raw_present = true,
        .owner_id = 42, .raw_readback_seconds = 0.0625 * fixture.policy_calls,
        .raw_write_seconds = 0.125 * fixture.policy_calls};
    return v;
}
static ds4_scr_archive_snapshot fixture_archive_snapshot_base(const ds4_session *s) {
    (void)s;
    uint32_t capacity = fixture.source_capacity ? fixture.source_capacity : 128;
    uint32_t covered = fixture.source_count ? fixture.source_count : 128;
    return (ds4_scr_archive_snapshot){ .presence_known = true, .present = fixture.source,
        .sealed = fixture.source, .retiring = fixture.source_retiring, .capacity = fixture.source ? capacity : 0,
        .start = fixture.source ? 8 : 0, .end = fixture.source ? 8 + covered : 0,
        .sealed_tokens = fixture.source ? covered : 0,
        .generation = fixture.configured_capacity && fixture.source ? 3 : 0,
        .source_generation = fixture.source ? (fixture.configured_capacity ? 3 : 1) : 0 };
}
static ds4_scr_archive_snapshot fixture_archive_snapshot(const ds4_session *s) {
    ds4_scr_archive_snapshot v = fixture_archive_snapshot_base(s);
    return fixture.observation ? fixture_observation_values(v) : v;
}
#define ds4_session_scr_archive_snapshot fixture_archive_snapshot
#define ds4_session_scr_archive_retire_once fixture_release
#define ds4_session_scr_archive_bytes fixture_archive_bytes
#define ds4_session_scr_archive_release fixture_archive_release
#define ds4_session_dispose_checked fixture_dispose
#define ds4_session_free fixture_free
#define ds4_engine_close fixture_close
#define ds4_session_sample fixture_sample
#define ds4_token_eos fixture_eos
#define ds4_token_is_stop_for_think_mode fixture_stop
#define ds4_session_eval fixture_eval
#define ds4_token_text fixture_text
#define ds4_session_invalidate fixture_invalidate
#define ds4_session_rewrite_from_common fixture_rewrite
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"

static void fixture_cancel_job(void *ud) { job_mark_cancelled(ud); }

static void fixture_probe_liveness(void) {
    assert(fixture.cancel && !fixture.cancel(fixture.cancel_ud));
    char bytes[4096];
    ssize_t n = recv(fixture.probe_fd, bytes, sizeof(bytes) - 1,
                     MSG_PEEK | MSG_DONTWAIT);
    if (n > 0) {
        bytes[n] = 0;
        fixture.headers_during_policy = strstr(bytes, "HTTP/1.1 200") &&
            strstr(bytes, "text/event-stream");
    }
}

static const char *chat =
    "{\"model\":\"deepseek-v4.1-flash\",\"messages\":["
    "{\"role\":\"system\",\"content\":\"dedicated prototype\"},"
    "{\"role\":\"user\",\"content\":\"image session_id are ordinary text; explain this task\"}],"
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"image\":{\"type\":\"string\"}}}}}],"
    "\"thinking\":false,\"stream\":true,\"stream_options\":{\"include_usage\":true}}";

static void fixture_reset(void) {
    ds4_tokens_free(&fixture.live);
    free(fixture.rendered);
    memset(&fixture, 0, sizeof(fixture));
    fixture.configured_k = 1;
}
static void setup(server *s, server_slot *slot, bool scr) {
    memset(s, 0, sizeof(*s));
    memset(slot, 0, sizeof(*slot));
    s->slots = slot; s->slot_count = 1; s->ctx_size = 32768;
    s->default_tokens = 8;
    s->scr_tail_tokens = scr ? (fixture.configured_capacity ? fixture.configured_capacity : 128u) : 0;
    s->scr_max_ranges = fixture.configured_k;
    slot->srv = s; slot->session = (ds4_session *)&fixture;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    pthread_cond_init(&s->clients_cv, NULL);
    pthread_mutex_init(&s->tool_mu, NULL);
    pthread_mutex_init(&s->kv_mu, NULL);
    pthread_mutex_init(&s->inference_mu, NULL);
    pthread_mutex_init(&s->model_mu, NULL);
    pthread_cond_init(&s->model_cv, NULL);
    pthread_mutex_init(&s->trace_mu, NULL);
}
static void teardown(server *s, server_slot *slot) {
    server_reuse_close(s);
    server_scr_retire_owned(s, slot);
    live_tool_state_free(&slot->responses_live);
    live_tool_state_free(&slot->anthropic_live);
    visible_live_free(&slot->thinking_live);
    tool_memory_free(&s->tool_mem);
    pthread_mutex_destroy(&s->mu); pthread_cond_destroy(&s->cv);
    pthread_cond_destroy(&s->clients_cv); pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->kv_mu); pthread_mutex_destroy(&s->inference_mu);
    pthread_mutex_destroy(&s->model_mu); pthread_cond_destroy(&s->model_cv);
    pthread_mutex_destroy(&s->trace_mu);
}
static char *read_response(int fd) {
    buf b = {0}; char chunk[4096]; ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0) buf_append(&b, chunk, (size_t)n);
    assert(n == 0);
    close(fd);
    return buf_take(&b);
}
static char *http_route(server *s, const char *path, const char *body) {
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    buf wire = {0};
    buf_printf(&wire, "POST %s HTTP/1.1\r\nHost: localhost\r\nContent-Length: %zu\r\n\r\n%s", path, strlen(body), body);
    assert(write(pair[0], wire.ptr, wire.len) == (ssize_t)wire.len);
    buf_free(&wire);
    client_arg *arg = malloc(sizeof(*arg)); assert(arg);
    *arg = (client_arg){ .srv = s, .fd = pair[1] };
    s->clients++;
    pthread_t worker, client;
    s->stopping = false;
    assert(pthread_create(&worker, NULL, worker_main, s) == 0);
    assert(pthread_create(&client, NULL, client_main, arg) == 0);
    char *response = read_response(pair[0]);
    assert(pthread_join(client, NULL) == 0);
    pthread_mutex_lock(&s->mu); s->stopping = true;
    pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu);
    assert(pthread_join(worker, NULL) == 0);
    return response;
}
static void poison(server_slot *slot) {
    fixture.live.v = malloc(sizeof(int)); assert(fixture.live.v);
    fixture.live.len = fixture.live.cap = 1; fixture.live.v[0] = -999;
    fixture.pos = 77; fixture.poison = true;
    slot->continued_last_store_tokens = 123;
    slot->thinking_live.valid = true;
    slot->thinking_live.live_tokens = 77;
    slot->thinking_live.visible_text = strdup("poisoned effective prompt");
}

/* The shared chat body with stream_options.include_ds4_scr enabled. */
static void chat_with_reuse_option(buf *body) {
    const char *option = strstr(chat, "\"include_usage\":true"); assert(option);
    buf_printf(body, "%.*s\"include_usage\":true,\"include_ds4_scr\":true%s",
        (int)(option - chat), chat, option + strlen("\"include_usage\":true"));
}
#endif
