/* ds4.reuse.v1 telemetry through the production writer and HTTP worker: record
 * schema, the opt-in SSE chunk, the append-only JSONL store and error phases.
 * Engine execution is substituted by the model-free SCR harness. */
#include "server_scr_harness.h"
#include <sys/stat.h>

static size_t count_of(const char *text, const char *needle) {
    size_t n = 0;
    for (const char *p = strstr(text, needle); p; p = strstr(p + 1, needle)) ++n;
    return n;
}

/* Exactly one JSON value; `null` appears only as the success error marker. */
static void assert_record(const char *record, size_t length) {
    const char *cursor = record;
    assert(json_skip_value(&cursor) && (size_t)(cursor - record) == length);
    assert(!strncmp(record, "{\"v\":1,\"schema\":\"ds4.reuse.v1\",", 31));
    assert(count_of(record, "null") == count_of(record, "\"error\":null"));
    assert(!strstr(record, "dedicated prototype") && !strstr(record, "explain this task"));
}

/* The record's request field quotes the response's first `id`. */
static void assert_shares_response_id(const char *record, const char *wire) {
    char id[128]; const char *at = strstr(wire, "\"id\":\""); assert(at);
    snprintf(id, sizeof(id), "\"request\":\"%.*s\"", (int)strcspn(at + 6, "\""), at + 6);
    assert(strstr(record, id));
}

static char *reuse_chunk(const char *wire) {
    const char *key = strstr(wire, "\"ds4_reuse\":");
    if (!key) return NULL;
    const char *start = key + strlen("\"ds4_reuse\":"), *end = start;
    assert(json_skip_value(&end));
    return xstrndup(start, (size_t)(end - start));
}

static void test_record_schema(void) {
    server s; server_slot slot;
    fixture_reset(); fixture.configured_k = (int)DS4_SCR_MULTI_MAX_RANGES;
    setup(&s, &slot, true); server_reuse_init(&s, NULL);
    job j = {0}; request_init(&j.req, REQ_CHAT, 8);
    j.req.prompt.len = 4096;
    j.reuse = (job_reuse_telemetry){.seq = 7, .enqueued = 10, .started = 10.5,
        .first_token = 12, .decode_started = 11.5, .decode_finished = 13.5,
        .generated = 4, .prefill_known = true};
    snprintf(j.reuse.request, sizeof(j.reuse.request), "chatcmpl-fixture");
    ds4_scr_logical_receipt *l = &j.reuse.logical;
    l->physical_known = true; l->physical_tokens = 4096 + 32;
    l->new_pads = 32; l->planned_ranges = DS4_SCR_MULTI_MAX_RANGES;
    l->planner_status_known = l->planner_timing_known = true;
    l->planner_status = CACHE_REUSE_DEADLINE_PARTIAL; l->planner_best_useful = 1900;
    l->planner_discovery_known = true; l->planner_discovered_matches = 3; l->planner_longest_match = 2048;
    l->planner_elapsed_ns = 1500000;
    for (uint32_t i = 0; i < DS4_SCR_MULTI_MAX_RANGES; ++i)
        l->physical_plan[i] = (ds4_scr_range){8 + 64 * i, 40 + 65 * i, 64};
    l->execution = (ds4_scr_dense_policy_receipt){.route = DS4_SCR_DENSE_REUSE,
        .complete_n = true, .source_generation = UINT64_MAX, .source_start = 8, .source_end = 2056,
        .multi = {.committed_tokens = 2048, .prefix_completed = 40, .suffix_completed = 2040,
            .stage_seconds = 0.25, .install_seconds = 0.125, .prefix_seconds = 0.5}};
    ds4_scr_archive_snapshot view = {.presence_known = true, .present = true, .sealed = true,
        .capacity = 4096, .sealed_tokens = 2048, .accounted_bytes = 3u << 20};
    buf b = {0};
    server_reuse_json(&b, &s, &slot, &j, &view);
    assert_record(b.ptr, b.len);
    assert(strstr(b.ptr, "\"request\":\"chatcmpl-fixture\",\"slot\":0,\"seq\":7"));
    assert(strstr(b.ptr, "\"tokens\":{\"prompt\":4096,\"prepared\":4128,\"reused\":2048,"
                         "\"live_prefix\":0,\"fresh\":2080,\"pad\":32,\"discovered_overlap\":2048,\"generated\":4}"));
    assert(strstr(b.ptr, "\"plan\":{\"status\":\"deadline\",\"k\":32,\"range_count\":32,\"ranges\":["));
    assert(count_of(b.ptr, "{\"src\":") == DS4_SCR_MULTI_MAX_RANGES);
    assert(strstr(b.ptr, "{\"src\":1992,\"dst\":2055,\"n\":64}]"));
    assert(strstr(b.ptr, "\"source\":{\"kind\":\"archive\",\"generation\":18446744073709551615,\"tokens\":2048}"));
    const char *ms[] = {"\"ms\":{\"queue\":500.000", "\"plan\":1.500", "\"stage\":250.000",
        "\"install\":125.000", "\"prefill\":500.000", "\"head\":0.000", "\"first_token\":1500.000",
        "\"decode\":2000.000", "\"total\":"};
    for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i) assert(strstr(b.ptr, ms[i]));
    assert(!strstr(b.ptr, "\"capture\":")); /* No observation: omitted, not zero. */
    assert(strstr(b.ptr, "\"archive\":{\"capacity\":4096,\"occupancy\":2048,\"action\":\"retained\",\"device_mib\":3.0}"));
    assert(strstr(b.ptr, "\"decode\":{\"steps\":4,\"tps\":2.00}") && strstr(b.ptr, "\"error\":null}"));
    assert(!strstr(b.ptr, "\"io\":"));
    assert(!strstr(b.ptr, "\"pool\":") && !strstr(b.ptr, "\"rebase\":"));
    buf_free(&b);
    /* Planner without a discovery stage: the overlap is unknown, so omitted. */
    l->planner_discovery_known = false;
    server_reuse_json(&b, &s, &slot, &j, &view);
    assert_record(b.ptr, b.len);
    assert(!strstr(b.ptr, "discovered_overlap") && strstr(b.ptr, "\"pad\":32,\"generated\":4}"));
    buf_free(&b);
    /* With the checkpoint pool, the archive reports a restore and the catalog size. */
    j.reuse.pool_known = true;
    j.reuse.pool = (ds4_scr_pool_receipt){.selected = true, .restored = true, .entries = 3,
        .seconds = 2.0};
    /* A rebased archive also reports its prepare and per-range import time. */
    l->execution.multi.rebase_prepared = true;
    l->execution.multi.rebase_prepare_seconds = 1.0;
    l->execution.multi.import_seconds = 0.5;
    server_reuse_json(&b, &s, &slot, &j, &view);
    assert_record(b.ptr, b.len);
    assert(strstr(b.ptr, "\"device_mib\":3.0,\"pool_restored\":true,\"pool_entries\":3}"));
    assert(strstr(b.ptr, "\"ms\":{\"queue\":500.000,\"pool\":2000.000,\"plan\":1.500,"
                         "\"rebase\":1500.000,\"stage\":250.000,"));
    buf_free(&b);

    /* Before prefill: unknown values are omitted and ranges stay variable-length. */
    j.reuse = (job_reuse_telemetry){.seq = 8, .started = 20, .error_phase = "quarantine", .error_code = 503};
    view = (ds4_scr_archive_snapshot){0};
    server_reuse_json(&b, &s, &slot, &j, &view);
    assert_record(b.ptr, b.len);
    assert(strstr(b.ptr, "\"tokens\":{\"prompt\":4096,\"generated\":0}"));
    assert(strstr(b.ptr, "\"plan\":{\"status\":\"none\",\"k\":32,\"range_count\":0,\"ranges\":[]}"));
    assert(!strstr(b.ptr, "\"archive\":") && !strstr(b.ptr, "\"decode\":{"));
    assert(strstr(b.ptr, "\"error\":{\"phase\":\"quarantine\",\"code\":503}}"));
    buf_free(&b); request_free(&j.req); teardown(&s, &slot);
}

static void test_worker_stream_chunk(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); server_reuse_init(&s, NULL);
    fixture.logical = fixture.source = fixture.same_logical_text = fixture.observation = true;
    fixture.output = "ok";
    buf body = {0}; chat_with_reuse_option(&body);
    char *wire = http_route(&s, "/v1/chat/completions", body.ptr);
    assert(count_of(wire, "\"ds4_reuse\":") == 1);
    const char *chunk_at = strstr(wire, "\"ds4_reuse\":");
    const char *finish = strstr(wire, "\"finish_reason\":\"stop\""), *done = strstr(wire, "[DONE]");
    assert(finish && done && chunk_at < finish && finish < done);
    char *record = reuse_chunk(wire);
    assert_record(record, strlen(record));
    int logical_n = (int)strlen(fixture.rendered);
    char expected[256];
    snprintf(expected, sizeof(expected),
             "\"tokens\":{\"prompt\":%d,\"prepared\":%d,\"reused\":128,\"live_prefix\":0,"
             "\"fresh\":%d,\"pad\":1,\"discovered_overlap\":128,\"generated\":1}",
             logical_n, logical_n + 1, logical_n + 1 - 128);
    assert(strstr(record, expected));
    assert(strstr(record, "\"plan\":{\"status\":\"complete\",\"k\":1,\"range_count\":1,"
                          "\"ranges\":[{\"src\":8,\"dst\":18,\"n\":128}],"
                          "\"source\":{\"kind\":\"archive\",\"generation\":1,\"tokens\":128}}"));
    const char *ms[] = {"\"queue\":", "\"plan\":2.000", "\"stage\":125.000", "\"install\":62.500",
        "\"prefill\":562.500", "\"capture\":187.500", "\"head\":31.250", "\"first_token\":",
        "\"decode\":", "\"total\":"};
    for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); ++i) assert(strstr(record, ms[i]));
    /* Admission (no policy call yet) to emission (one policy call, one sample). */
    assert(strstr(record, "\"io\":{\"engram\":{\"rows\":64,\"bytes\":16896,\"ms\":290.000}}"));
    assert(strstr(record, "\"archive\":{\"capacity\":128,\"occupancy\":128,\"action\":\"retained\""));
    assert(strstr(record, "\"decode\":{\"steps\":1") && strstr(record, "\"error\":null}"));
    assert_shares_response_id(record, wire);
    free(record); free(wire);

    /* Without the opt-in, or without streaming, no chunk is invented. */
    fixture.samples = 0;
    wire = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(wire, "[DONE]") && !strstr(wire, "ds4_reuse")); free(wire);
    const char *stream = strstr(body.ptr, "\"stream\":true"); assert(stream);
    buf nonstream = {0};
    buf_printf(&nonstream, "%.*s\"stream\":false%s", (int)(stream - body.ptr), body.ptr,
               stream + strlen("\"stream\":true"));
    fixture.samples = 0;
    wire = http_route(&s, "/v1/chat/completions", nonstream.ptr);
    assert(strstr(wire, "\"finish_reason\":\"stop\"") && !strstr(wire, "ds4_reuse")); free(wire);

    /* A prefill failure reports the error and carries no reuse chunk. */
    fixture.samples = 0; fixture.policy_rc = 1;
    wire = http_route(&s, "/v1/chat/completions", body.ptr);
    assert(strstr(wire, "substituted policy failure") && !strstr(wire, "[DONE]"));
    assert(!strstr(wire, "ds4_reuse")); free(wire);
    buf_free(&nonstream); buf_free(&body); teardown(&s, &slot);
}

static char *read_file(int dir, const char *name) {
    int fd = openat(dir, name, O_RDONLY | O_NOFOLLOW); assert(fd >= 0);
    return read_response(fd);
}

static void test_jsonl_store(void) {
    char directory[] = "/tmp/scr-reuse-telemetry-XXXXXX";
    assert(mkdtemp(directory));
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); server_reuse_init(&s, directory);
    assert(s.reuse.initialized && s.reuse.requested && s.reuse.fd >= 0);
    char *first = http_route(&s, "/v1/chat/completions", chat);
    fixture.samples = 0; fixture.policy_rc = 1;
    char *second = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(second, "substituted policy failure") && !strstr(second, "[DONE]"));

    /* A worker admitted into quarantine still leaves one record. */
    fixture.policy_rc = 0; s.scr_ownership_fatal = true;
    int sockets[2]; assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, sockets));
    job j = {.fd = sockets[1]}; pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    generate_job(&s, &slot, &j);
    assert(j.reuse.seq == 3 && j.reuse.emitted);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
    close(sockets[0]); close(sockets[1]); s.scr_ownership_fatal = false;
    teardown(&s, &slot);

    int dir = open(directory, O_RDONLY | O_DIRECTORY); assert(dir >= 0);
    struct stat st; assert(!fstatat(dir, "reuse.v1.jsonl", &st, AT_SYMLINK_NOFOLLOW));
    assert((st.st_mode & 0777) == 0600);
    char *records = read_file(dir, "reuse.v1.jsonl");
    char *lines[4] = {0}; size_t count = 0;
    for (char *line = records, *end; (end = strchr(line, '\n')); line = end + 1) {
        assert(count < 4); *end = 0; lines[count++] = line;
        assert_record(line, strlen(line));
    }
    assert(count == 3);
    assert(strstr(lines[0], "\"seq\":1,") && strstr(lines[0], "\"error\":null}"));
    assert(strstr(lines[0], "\"action\":\"sealed\""));
    assert_shares_response_id(lines[0], first);
    assert(strstr(lines[1], "\"seq\":2,") && strstr(lines[1], "\"error\":{\"phase\":\"prefill\",\"code\":1}"));
    assert(strstr(lines[2], "\"seq\":3,") && strstr(lines[2], "\"error\":{\"phase\":\"quarantine\",\"code\":503}"));
    assert(!strstr(records, "lookup"));
    free(records); free(first); free(second);
    assert(!unlinkat(dir, "reuse.v1.jsonl", 0)); close(dir); assert(!rmdir(directory));

    /* A missing store directory is telemetry loss only. */
    char parent[] = "/tmp/scr-reuse-missing-XXXXXX"; assert(mkdtemp(parent));
    char missing[256]; snprintf(missing, sizeof(missing), "%s/missing", parent);
    fixture_reset(); setup(&s, &slot, true); server_reuse_init(&s, missing);
    assert(s.reuse.requested && s.reuse.fd < 0);
    char *response = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(response, "[DONE]") && s.reuse.dropped == 1);
    free(response); teardown(&s, &slot); assert(!rmdir(parent));
}

/* The shared streaming chat body with the record opt-in and a "reuse" mode. */
static void chat_with_reuse_mode(buf *body, const char *mode) {
    buf opted = {0}; chat_with_reuse_option(&opted);
    buf_printf(body, "{\"reuse\":\"%s\",%s", mode, opted.ptr + 1);
    buf_free(&opted);
}

/* "reuse":"exact" runs ordinary prefill, never the planner, and still emits a
 * record; "scr" is the default; unknown modes are refused before any work. */
static void test_reuse_mode(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); server_reuse_init(&s, NULL);
    fixture.observation = true; fixture.output = "ok";
    buf body = {0}; chat_with_reuse_mode(&body, "exact");
    char *wire = http_route(&s, "/v1/chat/completions", body.ptr);
    assert(strstr(wire, "[DONE]") && !fixture.logical_calls && fixture.sync_calls == 1);
    char *record = reuse_chunk(wire); assert(record);
    assert_record(record, strlen(record));
    int n = (int)strlen(fixture.rendered);
    char expected[256];
    snprintf(expected, sizeof(expected), "\"tokens\":{\"prompt\":%d,\"prepared\":%d,\"reused\":0,"
             "\"live_prefix\":0,\"fresh\":%d,\"pad\":0,\"generated\":1}", n, n, n);
    assert(strstr(record, expected));
    assert(strstr(record, "\"plan\":{\"status\":\"exact\",\"k\":1,\"range_count\":0,\"ranges\":[]}"));
    assert(strstr(record, "\"prefill\":") && !strstr(record, "\"stage\":") && !strstr(record, "\"plan\":0"));
    assert(strstr(record, "\"error\":null}"));
    free(record); free(wire);

    /* The same prompt again: the live transcript is the exact prefix. The
     * sampled token extended live state, so restore the prompt-only frontier. */
    char *rendered = strdup(fixture.rendered); assert(rendered);
    ds4_tokens prompt = {0}; fixture_tokenize(NULL, rendered, &prompt); free(rendered);
    fixture_install(&prompt); ds4_tokens_free(&prompt);
    wire = http_route(&s, "/v1/chat/completions", body.ptr);
    record = reuse_chunk(wire); assert(record);
    snprintf(expected, sizeof(expected), "\"live_prefix\":%d,\"fresh\":0,\"pad\":0", n);
    assert(strstr(record, expected) && fixture.sync_calls == 2 && !fixture.logical_calls);
    snprintf(expected, sizeof(expected), "\"status\":\"exact\",\"k\":1,\"range_count\":0,\"ranges\":[],"
             "\"source\":{\"kind\":\"live\",\"tokens\":%d}}", n);
    assert(strstr(record, expected));
    free(record); free(wire); buf_free(&body);

    /* "scr" is explicit default: the planner runs and maps the transcript. */
    fixture.logical = fixture.source = fixture.same_logical_text = true;
    chat_with_reuse_mode(&body, "scr");
    wire = http_route(&s, "/v1/chat/completions", body.ptr);
    record = reuse_chunk(wire); assert(record);
    assert(fixture.logical_calls == 1 && fixture.mapped && strstr(record, "\"status\":\"complete\""));
    free(record); free(wire); buf_free(&body);

    /* Exact after a padded transcript: no pad provenance survives ordinary
     * sync, so the mapped live state is invalidated and fully prefilled. */
    chat_with_reuse_mode(&body, "exact");
    wire = http_route(&s, "/v1/chat/completions", body.ptr);
    record = reuse_chunk(wire); assert(record);
    assert(fixture.logical_calls == 1 && fixture.sync_calls == 3 && !fixture.mapped &&
           !fixture.physical_sync_calls);
    snprintf(expected, sizeof(expected), "\"reused\":0,\"live_prefix\":0,\"fresh\":%d,\"pad\":0", n);
    assert(strstr(record, expected) && strstr(record, "\"status\":\"exact\""));
    free(record); free(wire); buf_free(&body);

    /* Unknown, misspelled or non-string modes: HTTP 400 before tokenization. */
    const char *bad[] = {"{\"reuse\":\"fast\",", "{\"reuse\":\"SCR\",", "{\"reuse\":1,", "{\"reuse\":null,"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        const int tokenized = fixture.tokenize_calls;
        buf_printf(&body, "%s%s", bad[i], chat + 1);
        wire = http_route(&s, "/v1/chat/completions", body.ptr);
        assert(strstr(wire, "HTTP/1.1 400") && strstr(wire, "reuse must be") &&
               fixture.tokenize_calls == tokenized && fixture.logical_calls == 1);
        free(wire); buf_free(&body);
    }
    teardown(&s, &slot);

    /* /v1/completions parses the same field; SCR itself remains chat-only. */
    request req; char err[160];
    assert(parse_completion_request(NULL, "{\"prompt\":\"hi\",\"reuse\":\"exact\"}", 8, 32768,
                                    &req, err, sizeof(err)) && req.reuse_exact);
    request_free(&req);
    assert(parse_completion_request(NULL, "{\"prompt\":\"hi\",\"reuse\":\"scr\"}", 8, 32768,
                                    &req, err, sizeof(err)) && !req.reuse_exact);
    request_free(&req);
    fixture_reset(); setup(&s, &slot, false);
    wire = http_route(&s, "/v1/completions", "{\"prompt\":\"hi\",\"reuse\":\"bogus\"}");
    assert(strstr(wire, "HTTP/1.1 400") && strstr(wire, "reuse must be"));
    free(wire); teardown(&s, &slot);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    test_record_schema();
    test_worker_stream_chunk();
    test_jsonl_store();
    test_reuse_mode();
    fixture_reset();
    puts("ds4.reuse.v1: record schema (K=32 ranges), opt-in SSE chunk, JSONL store, error phases and per-request reuse mode passed");
    return 0;
}
