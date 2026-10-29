/* Model-free SCR server behaviour: routing, usage, request guards, owned
 * retirement, cancellation, transport liveness, configuration, quarantine and
 * mapped-coordinate recovery through the production HTTP worker. */
#include "server_scr_harness.h"
#include <sys/wait.h>

static void test_route_usage_and_baseline(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); poison(&slot);
    char *response = http_route(&s, "/v1/chat/completions", chat);
    assert(fixture.policy_calls == 1 && fixture.sync_calls == 0 && fixture.logical_calls == 1);
    assert(strstr(fixture.rendered, "dedicated prototype") && strstr(fixture.rendered, "lookup"));
    assert(strstr(response, "\"finish_reason\":\"stop\"") && strstr(response, "[DONE]"));
    assert(strstr(response, "\"cached_tokens\":0") && strstr(response, "\"cache_write_tokens\":0"));
    char usage[80]; snprintf(usage, sizeof(usage), "\"prompt_tokens\":%u", fixture.last.requested_tokens);
    assert(strstr(response, usage));
    assert(fixture.source && fixture.releases == 0 && slot.continued_last_store_tokens == 123);
    free(response);
    fixture.poison = false; fixture.samples = 0;
    response = http_route(&s, "/v1/chat/completions", chat);
    assert(fixture.policy_calls == 2 && fixture.logical_calls == 2 && fixture.last.route == DS4_SCR_DENSE_REUSE && fixture.last.multi.committed_tokens == 128);
    assert(strstr(response, "\"cached_tokens\":0")); free(response);
    teardown(&s, &slot); assert(!fixture.source && fixture.releases == 1);

    fixture_reset(); setup(&s, &slot, false);
    response = http_route(&s, "/v1/chat/completions", chat);
    snprintf(usage, sizeof(usage), "\"cache_write_tokens\":%d", fixture.live.len);
    assert(strstr(response, usage)); free(response);
    assert(fixture.policy_calls == 0 && fixture.sync_calls == 1 && fixture.releases == 0);
    fixture.samples = 0;
    response = http_route(&s, "/v1/chat/completions", chat);
    snprintf(usage, sizeof(usage), "\"cached_tokens\":%d", fixture.live.len);
    assert(strstr(response, usage)); assert(strstr(response, "\"cache_write_tokens\":0"));
    free(response); teardown(&s, &slot); assert(fixture.releases == 0);
}
static void test_guards_no_owned_retirement(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); fixture.source = true;
    const char *paths[] = {"/v1/messages", "/v1/responses", "/v1/completions"};
    for (size_t i = 0; i < 3; i++) {
        char *r = http_route(&s, paths[i], chat);
        assert(strstr(r, "400") && fixture.tokenize_calls == 0); free(r);
    }
    const char *bad[] = {
        "{\"session_id\":\"owner\",\"session_id\":null,\"messages\":[]}",
        "{\"sess\\u0069on_id\":\"owner\",\"messages\":[]}",
        "{\"conversation\":{},\"messages\":[]}", "{\"conversation_id\":0,\"messages\":[]}",
        "{\"previous_response_id\":false,\"messages\":[]}",
        "{\"store\":true,\"store\":false,\"messages\":[]}",
        "{\"store\":0,\"messages\":[]}", "{\"store\":\"false\",\"messages\":[]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}}]}],\"messages\":[]}",
    };
    for (size_t i = 0; i < sizeof(bad)/sizeof(bad[0]); i++) {
        char *r = http_route(&s, "/v1/chat/completions", bad[i]);
        assert(strstr(r, "400") && fixture.tokenize_calls == 0);
        if (i == 8) assert(strstr(r, "SCR requires text-only chat"));
        free(r);
        assert(fixture.source && fixture.releases == 0);
    }
    request req; char err[160];
    assert(parse_chat_request(NULL, &s,
        "{\"session_id\":null,\"conversation\":null,\"conversation_id\":null,\"previous_response_id\":null,\"store\":false,"
        "\"metadata\":{\"session_id\":\"nested\",\"store\":true},\"messages\":[{\"role\":\"user\",\"content\":\"image\"}]}",
        8, 32768, &req, err, sizeof(err))); request_free(&req);
    assert(server_scr_chat_guard(&s, "{\"store\":null}"));
    assert(server_scr_chat_guard(&s, "{\"store\": false ,\"session_id\": null }"));
    assert(parse_chat_request(NULL, &s,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"lookup please\"},{\"role\":\"assistant\",\"reasoning_content\":\"native reasoning\",\"tool_calls\":["
        "{\"id\":\"call_preserved\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"image\\\":\\\"session_id\\\"}\"}}]},"
        "{\"role\":\"tool\",\"tool_call_id\":\"call_preserved\",\"content\":\"result\"},{\"role\":\"user\",\"content\":\"continue\"}]}",
        8, 32768, &req, err, sizeof(err)));
    assert(req.model_syntax == SERVER_MODEL_SYNTAX_DEEPSEEK41 && ds4_think_mode_enabled(req.think_mode));
    assert(strstr(req.prompt_text, "native reasoning") && strstr(req.prompt_text, "lookup"));
    request_free(&req);
    job queued = {0}; pthread_mutex_init(&queued.mu, NULL); pthread_cond_init(&queued.cv, NULL);
    s.stopping = false; assert(enqueue(&s, &queued)); server_cancel_job(&s, &queued);
    assert(queued.done && !s.head && fixture.source && fixture.releases == 0);
    pthread_mutex_destroy(&queued.mu); pthread_cond_destroy(&queued.cv);
    s.scr_tail_tokens = 0;
    assert(parse_chat_request(NULL, &s, "{\"session_id\":\"legacy\",\"store\":true,\"messages\":[]}",
        8, 32768, &req, err, sizeof(err))); request_free(&req);
    s.scr_tail_tokens = 128; teardown(&s, &slot);
}
static void test_errors_and_owned_cleanup(void) {
    const int failures[] = {1, DS4_SESSION_SYNC_INTERRUPTED};
    for (size_t i = 0; i < 2; i++) {
        server s; server_slot slot;
        fixture_reset(); setup(&s, &slot, true);
        fixture.source = true; fixture.policy_rc = failures[i];
        char *r = http_route(&s, "/v1/chat/completions", chat);
        assert(fixture.policy_calls == 1 && fixture.sync_calls == 0 && fixture.samples == 0);
        assert(fixture.releases == 1 && !fixture.source && fixture.last.multi.partial_work_unknown && fixture.last.multi.discarded_tokens == 8);
        assert(!strstr(r, "[DONE]") && !strstr(r, "\"finish_reason\":\"stop\"")); free(r);
        teardown(&s, &slot);
    }
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); fixture.cancel_in_policy = true;
    char *cancelled_response = http_route(&s, "/v1/chat/completions", chat);
    assert(fixture.policy_calls == 1 && fixture.releases == 0 && fixture.retire_calls == 1 && fixture.samples == 0 && !fixture.source);
    free(cancelled_response); teardown(&s, &slot);
    fixture_reset(); setup(&s, &slot, true); fixture.decode_mode = -1;
    char *r = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(r, "\"finish_reason\":\"error\""));
    assert(fixture.releases == 1 && !fixture.source); free(r);
    fixture.decode_mode = 0; fixture.samples = 0;
    r = http_route(&s, "/v1/chat/completions", chat);
    assert(fixture.policy_calls == 2 && fixture.last.route == DS4_SCR_DENSE_REFRESH && fixture.source); free(r);
    /* Assigned cancellation and failed delivery use the actual owning wrapper. */
    job j = {0}; pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    j.cancelled = true; generate_job(&s, &slot, &j);
    assert(!fixture.source && fixture.releases == 2);
    j.cancelled = false; fixture.samples = 0;
    char err[160]; assert(parse_chat_request(NULL, &s, chat, 8, 32768, &j.req, err, sizeof(err)));
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    close(pair[0]); j.fd = pair[1]; generate_job(&s, &slot, &j); close(pair[1]);
    assert(!fixture.source && fixture.releases == 3 && job_cancelled(&j)); request_free(&j.req);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
    teardown(&s, &slot);
}
static void test_tool_and_generic_sync(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true);
    fixture.output = "\n\n" DS41_TOOL_CALLS_START "\n" DS41_INVOKE_START " name=\"lookup\">\n"
        DS41_PARAM_START " name=\"image\" string=\"true\">session_id" DS41_PARAM_END "\n"
        DS41_INVOKE_END "\n" DS41_TOOL_CALLS_END;
    char *r = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(r, "\"finish_reason\":\"tool_calls\"") && strstr(r, "lookup"));
    assert(fixture.policy_calls == 1 && fixture.source && fixture.releases == 0); free(r);
    char err[160]; ds4_tokens full = fixture.live;
    assert(server_session_sync(&s, &slot, &full, err, sizeof(err)) == 0);
    assert(fixture.policy_calls == 1 && fixture.sync_calls == 1 && fixture.source);
    /* Force the existing canonical rebuild path, not a policy fixture route. */
    s.disable_exact_dsml_tool_replay = true; fixture.samples = 0;
    r = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(r, "\"finish_reason\":\"tool_calls\"")); free(r);
    assert(fixture.policy_calls == 2 && fixture.sync_calls == 2 && fixture.source);
    fixture.samples = 0; fixture.sync_rc = 1;
    r = http_route(&s, "/v1/chat/completions", chat);
    assert(strstr(r, "\"finish_reason\":\"error\"")); free(r);
    assert(fixture.policy_calls == 3 && fixture.sync_calls == 3 && !fixture.source && fixture.releases == 1);
    fixture.samples = 0; fixture.sync_rc = 0; fixture.output = "hello";
    r = http_route(&s, "/v1/chat/completions",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"length\"}],\"thinking\":false,\"max_tokens\":1}");
    assert(strstr(r, "\"finish_reason\":\"length\"") && strstr(r, "\"completion_tokens\":1")); free(r);
    assert(fixture.source && fixture.releases == 1);
    assert(server_scr_terminal_success(true, "length"));
    assert(!server_scr_terminal_success(true, "error"));
    assert(!server_scr_terminal_success(false, "stop"));
    teardown(&s, &slot);
}
static void test_scr_liveness_during_policy(void) {
    {
        server s; server_slot slot;
        fixture_reset(); setup(&s, &slot, true);
        fixture.probe_liveness = true;
        job j = {0};
        pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
        char err[160];
        assert(parse_chat_request(NULL, &s, chat, 8, 32768, &j.req, err, sizeof(err)));
        int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
        j.fd = pair[1]; fixture.probe_fd = pair[0];
        generate_job(&s, &slot, &j);
        close(pair[1]);
        char *response = read_response(pair[0]);
        assert(fixture.headers_during_policy);
        assert(strstr(response, "[DONE]") && !fixture.cancel && !fixture.cancel_ud);
        free(response); request_free(&j.req);
        pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
        teardown(&s, &slot);
    }
}

static void *probe_nonowner_cancel(void *ud) {
    assert(!server_scr_cancel_with_keepalive(ud));
    return NULL;
}

static void test_transport_clock_and_owner(void) {
    job j = {0};
    pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    j.fd = pair[1];
    server_prefill_progress progress = {
        .request_job = &j, .fd = pair[1], .stream = true, .cached_tokens = 88,
    };
    server_scr_transport_context context = { &progress, pthread_self() };
    pthread_t other;
    assert(pthread_create(&other, NULL, probe_nonowner_cancel, &context) == 0);
    assert(pthread_join(other, NULL) == 0);
    assert(!progress.headers_sent);
    char bytes[4096];
    assert(recv(pair[0], bytes, sizeof(bytes), MSG_PEEK | MSG_DONTWAIT) == -1);
    assert(errno == EAGAIN || errno == EWOULDBLOCK);
    assert(!server_scr_cancel_with_keepalive(&context));
    assert(progress.headers_sent && !progress.seen && progress.cached_tokens == 88);
    double first = progress.last_keepalive;
    assert(server_prefill_keepalive(&progress, first + 4.0));
    assert(progress.last_keepalive == first);
    assert(server_prefill_keepalive(&progress, first + 6.0));
    ssize_t n = recv(pair[0], bytes, sizeof(bytes) - 1, MSG_PEEK | MSG_DONTWAIT);
    assert(n > 0); bytes[n] = 0;
    assert(strstr(bytes, ": prefill\n\n") && !progress.seen);
    close(pair[0]);
    assert(!server_prefill_keepalive(&progress, first + 12.0));
    assert(progress.stream_failed && job_cancelled(&j));
    assert(server_scr_cancel_with_keepalive(&context));
    close(pair[1]);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
}

static void test_prefill_error_after_headers(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true);
    fixture.probe_liveness = true; fixture.policy_rc = 1;
    job j = {0};
    pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    char err[160];
    assert(parse_chat_request(NULL, &s, chat, 8, 32768, &j.req, err, sizeof(err)));
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    j.fd = pair[1]; fixture.probe_fd = pair[0];
    generate_job(&s, &slot, &j);
    close(pair[1]); char *response = read_response(pair[0]);
    assert(fixture.headers_during_policy && fixture.samples == 0);
    assert(strstr(response, "event: error\n") && strstr(response, "substituted policy failure"));
    assert(!strstr(response, "[DONE]") && !strstr(response, "\"finish_reason\":\"stop\""));
    assert(!fixture.cancel && !fixture.cancel_ud);
    free(response); request_free(&j.req);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
    teardown(&s, &slot);
}

static void test_context_bounded_output_default(void) {
    char *args[] = {"server", "--metal", "--ctx", "262144",
        "--scr-tail-tokens", "65536", "--scr-batch-suffix"};
    server_config config = parse_options(7, args);
    assert(config.default_tokens == 393216 && config.ctx_size == 262144);
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true);
    request req; char err[160];
    assert(parse_chat_request(NULL, &s, chat, config.default_tokens,
                              config.ctx_size, &req, err, sizeof(err)));
    assert(server_decode_budget(req.max_tokens, 0, 262144 - 80000) == 182144);
    assert(server_decode_budget(req.max_tokens, 0, 262144 - 200000) == 62144);
    assert(server_decode_budget(1024, 0, 62144) == 1024);
    assert(server_decode_budget(req.max_tokens, 0, 0) == 0);
    request_free(&req);
    const char *think_body = "{\"model\":\"deepseek-v4.1-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"inert fixture\"}],\"think\":true,\"stream\":true}";
    assert(parse_chat_request(NULL, &s, think_body, config.default_tokens, config.ctx_size, &req, err, sizeof(err)));
    assert(req.think_mode == DS4_THINK_HIGH && req.temperature == 1.0f && req.top_p == 1.0f && req.min_p == 0.05f && req.top_k == 0);
    request_free(&req); teardown(&s, &slot);
}

static int cli_exit(int argc, char **argv) {
    pid_t pid = fork(); assert(pid >= 0);
    if (!pid) { (void)parse_options(argc, argv); _exit(0); }
    int status; assert(waitpid(pid, &status, 0) == pid); assert(WIFEXITED(status));
    return WEXITSTATUS(status);
}
static void test_configuration_and_help(void) {
    char *valid[] = {"server", "--metal", "--scr-tail-tokens", "128"};
    assert(cli_exit(4, valid) == 0);
    const char *larger[] = {"65544", "200000"};
    for (size_t i = 0; i < 2; ++i) {
        char *args[] = {"server", "--metal", "--ctx", "262144", "--scr-tail-tokens", (char *)larger[i]};
        assert(cli_exit(6, args) == 0);
        server_config config = parse_options(6, args);
        assert(config.scr_tail_tokens == atoi(larger[i]) && config.ctx_size == 262144);
    }
    char *too_small[] = {"server", "--metal", "--ctx", "131072", "--scr-tail-tokens", "200000"};
    assert(cli_exit(6, too_small) == 2);
    const char *pairs[][2] = {
        {"--scr-tail-tokens", "127"}, {"--scr-tail-tokens", "129"}, {"--scr-tail-tokens", "1048576"},
        {"--scr-tail-tokens", "oops"}, {"--ctx", "64"}, {"--host", "0.0.0.0"},
        {"--host", "::1"}, {"--host", "example.test"}, {"--batched-session", "1"},
        {"--kv-cache-min-tokens", "512"}, {"--kv-cache-cold-max-tokens", "0"},
        {"--vision", "not-opened"}, {"--gpu-devices", "0"}, {"--gpu-vram", "auto"},
        {"--backend", "cpu"}, {"--role", "worker"}, {"--dist-prefill-chunk", "128"},
    };
    for (size_t i = 0; i < sizeof(pairs)/sizeof(pairs[0]); i++) {
        char *args[] = {"server", "--scr-tail-tokens", "128", (char *)pairs[i][0], (char *)pairs[i][1], "--metal"};
        assert(cli_exit(6, args) == 2);
    }
    char *cpu[] = {"server", "--scr-tail-tokens", "128", "--cpu", "--metal"};
    assert(cli_exit(5, cpu) == 2);
    const char *ks[] = {"1", "2", "3", "6", "7", "10"};
    for (size_t i = 0; i < sizeof(ks)/sizeof(ks[0]); ++i) {
        char *args[] = {"server", "--metal", "--scr-tail-tokens", "128", "--scr-max-ranges", (char *)ks[i]};
        assert(cli_exit(6, args) == 0);
        server_config k = parse_options(6, args);
        assert(k.scr_max_ranges == atoi(ks[i]) && k.scr_max_ranges_present);
    }
    char over[8]; snprintf(over, sizeof(over), "%u", DS4_SCR_MULTI_MAX_RANGES + 1u);
    const char *badks[] = {"0", over, "oops", "-1"};
    for (size_t i = 0; i < sizeof(badks)/sizeof(badks[0]); ++i) {
        char *args[] = {"server", "--metal", "--scr-tail-tokens", "128", "--scr-max-ranges", (char *)badks[i]};
        assert(cli_exit(6, args) == 2);
    }
    char *absent[] = {"server", "--scr-max-ranges", "1"}; assert(cli_exit(3, absent) == 2);
    char *zero[] = {"server", "--scr-tail-tokens", "0", "--scr-max-ranges", "1"}; assert(cli_exit(5, zero) == 2);
    /* Retired selectors are accepted quietly and select nothing extra. */
    char *legacy[] = {"server", "--metal", "--scr-tail-tokens", "128", "--scr-logical-coordinates", "--scr-batch-suffix"};
    assert(cli_exit(6, legacy) == 0);
    server_config logical_config = parse_options(4, valid); logical_config.engine.glm_mtp = true;
    assert(!server_scr_config_valid(&logical_config, false));
    logical_config.engine.glm_mtp = false; logical_config.engine.dspark = true;
    assert(!server_scr_config_valid(&logical_config, false));
    char *help[] = {"server", "--help"}; assert(cli_exit(2, help) == 0);
    assert(server_scr_loopback("localhost") && server_scr_loopback("127.99.2.3"));
    assert(!server_scr_loopback("localhost.localdomain") && !server_scr_loopback("128.0.0.1"));
    server_config c = parse_options(4, valid);
    assert(c.scr_enabled);
    assert(c.kv_cache.min_tokens > 0 && c.scr_tail_tokens == 128 &&
        c.scr_max_ranges == (int)DS4_SCR_MULTI_MAX_RANGES && !c.scr_max_ranges_present);
}
static void test_dense_routes_and_retire_results(void) {
    const int ks[] = {1, 2, 3, 6};
    const ds4_scr_dense_route routes[] = {DS4_SCR_DENSE_CONTINUE, DS4_SCR_DENSE_FRESH,
        DS4_SCR_DENSE_REUSE, DS4_SCR_DENSE_REFRESH};
    for (size_t i = 0; i < 4; ++i) {
        server s; server_slot slot;
        fixture_reset(); fixture.configured_k = ks[i]; fixture.route = routes[i];
        setup(&s, &slot, true); poison(&slot);
        char *r = http_route(&s, "/v1/chat/completions", chat);
        assert(fixture.policy_calls == 1 && fixture.last.configured_k == (uint32_t)ks[i]);
        assert(fixture.last.route == routes[i] && fixture.last.complete_n);
        assert(strstr(r, "\"cached_tokens\":0") && strstr(r, "\"cache_write_tokens\":0"));
        free(r); teardown(&s, &slot);
    }
    const ds4_scr_retire_outcome outcomes[] = {DS4_SCR_RETIRE_ABSENT,
        DS4_SCR_RETIRE_RELEASED, DS4_SCR_RETIRE_ALREADY_FENCED, DS4_SCR_RETIRE_FAILED_RETAINED};
    for (size_t i = 0; i < 4; ++i) {
        server s; server_slot slot;
        fixture_reset(); setup(&s, &slot, true); fixture.source = i != 0;
        fixture.force_retire = true; fixture.retire_outcome = outcomes[i];
        ds4_scr_retire_receipt r = server_scr_retire_owned(&s, &slot);
        assert(r.outcome == outcomes[i] && r.charge_known && r.charged_bytes_before == (i ? 4096u : 0u));
        assert(r.attempted == (i == 1 || i == 3));
        assert(r.charged_bytes_after == (i <= 1 ? 0u : 4096u));
        assert(!s.scr_ownership_fatal); teardown(&s, &slot);
    }
}
static void test_owned_final_cleanup(void) {
    const ds4_scr_retire_outcome outcomes[] = {DS4_SCR_RETIRE_ABSENT,
        DS4_SCR_RETIRE_RELEASED, DS4_SCR_RETIRE_ALREADY_FENCED, DS4_SCR_RETIRE_FAILED_RETAINED};
    for (size_t i = 0; i < 4; ++i) {
        fixture_reset(); server s; server_slot *slot = malloc(sizeof(*slot)); assert(slot);
        setup(&s, slot, true); fixture.source = i != 0;
        fixture.force_retire = true; fixture.retire_outcome = outcomes[i];
        if (i >= 2) {
            ds4_session *owned = slot->session;
            assert(server_close_resources(&s) == 1 && s.slots == slot && slot->session == owned);
            assert(fixture.retire_calls == 1 && !fixture.frees && !fixture.engine_closes);
            fixture.retire_outcome = DS4_SCR_RETIRE_RELEASED;
        }
        assert(server_close_resources(&s) == 0);
        assert(fixture.retire_calls == (i >= 2 ? 2 : 1) && fixture.frees == 1 && fixture.engine_closes == 1);
    }
    fixture_reset(); server s; server_slot *slot = malloc(sizeof(*slot)); assert(slot);
    setup(&s, slot, false);
    assert(server_close_resources(&s) == 0 && fixture.retire_calls == 0 && fixture.frees == 1 && fixture.engine_closes == 1);
    fixture_reset(); slot = malloc(sizeof(*slot)); assert(slot); setup(&s, slot, false);
    slot->live_text = strdup("owned"); slot->live_text_len = 5; slot->live_text_pos = 7;
    fixture.dispose_refusal = true;
    assert(server_close_resources(&s) == 1 && slot->session && !fixture.engine_closes);
    assert(!slot->live_text && !slot->live_text_len && !slot->live_text_pos);
    fixture.dispose_refusal = false;
    assert(server_close_resources(&s) == 0 && fixture.frees == 1 && fixture.engine_closes == 1);
    fixture_reset(); slot = malloc(sizeof(*slot)); assert(slot); setup(&s, slot, true);
    slot->session = NULL; /* Startup failure before a session exists. */
    assert(server_close_resources(&s) == 0 && fixture.retire_calls == 0 && fixture.frees == 0 && fixture.engine_closes == 1);
}
static void test_invalid_terminal(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); fixture.source = true;
    fixture.force_retire = true; fixture.retire_outcome = DS4_SCR_RETIRE_INVALID;
    job j = {.cancelled = true}; pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0); j.fd = pair[1];
    generate_job(&s, &slot, &j);
    assert(s.scr_ownership_fatal && !slot.running && fixture.cancel_sets == 1 && fixture.retire_calls == 1);
    /* No post-INVALID callback clear or next-assignment callback installation. */
    generate_job(&s, &slot, &j);
    assert(fixture.cancel_sets == 1 && fixture.retire_calls == 1 && fixture.policy_calls == 0);
    assert(server_close_resources(&s) == 1 && fixture.frees == 0 && fixture.engine_closes == 0);
    close(pair[1]); char *r = read_response(pair[0]); assert(strstr(r, "503")); free(r);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv); teardown(&s, &slot);
}
static void test_dense_receipt_log(void) {
    server_slot slot = {.id = 7}; job j = {0};
    ds4_scr_dense_policy_receipt r = {
        .route = DS4_SCR_DENSE_ERROR, .requested_tokens = 1001,
        .configured_cap = 128, .configured_k = 6, .native_lcp = 55,
        .retained_tokens = 44, .continuation_completed = 12, .continuation_heads = 2,
        .continuation_work_unknown = true, .continuation_head_unknown = true,
        .source_start = 8, .source_end = 136, .source_generation = 9,
        .match = {.selected_ranges = 1, .plan = {{8, 16, 128}}},
        .multi = {.requested_tokens = 1001, .installed_spans = 2, .committed_spans = 0,
            .installed_tokens = 256, .committed_tokens = 0, .discarded_tokens = 31,
            .prefix_completed = 17, .suffix_completed = 19, .fresh_completed = 21,
            .partial_work_unknown = true, .gap_completed = {23}},
        .refresh = {.requested_tokens = 1001, .source_count = 128,
            .old_retirement_attempted = true, .old_retired = true, .prepared = true,
            .sealed = true, .source_completed = 128, .remainder_completed = 7,
            .work_unknown = true, .source_charged_bytes = 4096},
    };
    FILE *capture = tmpfile(); assert(capture);
    fflush(stderr); int saved = dup(STDERR_FILENO); assert(saved >= 0);
    assert(dup2(fileno(capture), STDERR_FILENO) >= 0);
    server_scr_receipt_log(&slot, &j, 1, &r);
    fflush(stderr); assert(dup2(saved, STDERR_FILENO) >= 0); close(saved);
    assert(fseek(capture, 0, SEEK_END) == 0); long n = ftell(capture); assert(n > 0);
    rewind(capture); char *text = malloc((size_t)n + 1); assert(text);
    assert(fread(text, 1, (size_t)n, capture) == (size_t)n); text[n] = 0; fclose(capture);
    const char *fields[] = {"N=1001 C=128 K=6 lcp=55 live_retained=44",
        "continuation_work=12 continuation_heads=2 work_unknown=1 head_unknown=1",
        "complete_N=0 source=[8,136) generation=9", "scr match request=",
        "selected=1", "installed_spans=2 committed_spans=0",
        "installed_tokens=256 committed_tokens=0", "prefix_work=17 suffix_work=19 fresh_work=21 discarded=31 unknown=1",
        "plan_source=8 plan_destination=16 plan_count=128", "gap_work=23",
        "count=128", "source_work=128 remainder_work=7", "retire_attempt=1 retired=1 prepared=1 sealed=1",
        "charged=4096", "stage_s=", "cleanup_s=", "total_s="};
    for (size_t i = 0; i < sizeof(fields)/sizeof(fields[0]); ++i) assert(strstr(text, fields[i]));
    assert(!strstr(text, "forwarded=") && !strstr(text, "cache_read_tokens="));
    /* The retired overlap search reported pair and discovery counters. */
    assert(!strstr(text, "pairs=") && !strstr(text, "discovered=") && !strstr(text, "eligible="));
    free(text);
}
static void test_invalid_quarantine(void) {
    server s; server_slot slot;
    fixture_reset(); setup(&s, &slot, true); fixture.source = true;
    fixture.force_retire = true; fixture.retire_outcome = DS4_SCR_RETIRE_INVALID;
    /* An unrelated global descriptor must survive fixture quarantine. */
    int pair[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
    g_listen_fd = pair[0];
    ds4_scr_retire_receipt r = server_scr_retire_owned(&s, &slot);
    assert(r.outcome == DS4_SCR_RETIRE_INVALID && !r.attempted && !r.charge_known);
    assert(s.scr_ownership_fatal && s.stopping && fixture.source);
    assert(write(pair[1], "x", 1) == 1); char c; assert(read(pair[0], &c, 1) == 1);
    close(pair[0]); close(pair[1]); g_listen_fd = -1;
    job j = {0}; pthread_mutex_init(&j.mu, NULL); pthread_cond_init(&j.cv, NULL);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0); j.fd = pair[1];
    generate_job(&s, &slot, &j);
    assert(!slot.running && fixture.cancel_sets == 0 && fixture.policy_calls == 0);
    /* Stopping ordinarily drains queued work: exercise the real worker handshake. */
    s.head = s.tail = &j; worker_main(&s);
    assert(j.done && !s.head && fixture.cancel_sets == 0 && fixture.retire_calls == 1);
    close(pair[1]); char *response = read_response(pair[0]); assert(strstr(response, "503")); free(response);
    assert(server_close_resources(&s) == 1 && fixture.frees == 0 && fixture.engine_closes == 0);
    pthread_mutex_destroy(&j.mu); pthread_cond_destroy(&j.cv);
    teardown(&s, &slot);
    /* Invalid discovered during owned teardown also returns before destructors. */
    fixture_reset(); setup(&s, &slot, true);
    fixture.force_retire = true; fixture.retire_outcome = DS4_SCR_RETIRE_INVALID;
    assert(server_close_resources(&s) == 1 && fixture.frees == 0 && fixture.engine_closes == 0);
    teardown(&s, &slot);
}
static void test_mapped_recovery_and_logical_canonicalization(void) {
    server s; server_slot slot; char err[160];
    fixture_reset(); setup(&s, &slot, true);
    int ids[] = {1, 2, 3}; ds4_tokens input = {.v = ids, .len = 3, .cap = 3};
    fixture.mapped = true;
    assert(!server_session_sync_physical(&s, &slot, &input, err, sizeof(err)));
    assert(fixture.physical_sync_calls == 1 && !fixture.sync_calls);
    s.batched_mode = true;
    assert(server_session_sync_physical(&s, &slot, &input, err, sizeof(err)) == 1);
    assert(fixture.physical_sync_calls == 1);
    s.batched_mode = false; fixture.mapped = false;
    assert(!server_session_sync_physical(&s, &slot, &input, err, sizeof(err)));
    assert(fixture.sync_calls == 1);
    teardown(&s, &slot);
    for (int mode = 0; mode < 3; ++mode) {
        fixture_reset(); setup(&s, &slot, true);
        fixture.map_after_policy = true;
        fixture.same_logical_text = mode == 0;
        fixture.compare_rc = mode == 2;
        s.disable_exact_dsml_tool_replay = true;
        fixture.output = "\n\n" DS41_TOOL_CALLS_START "\n" DS41_INVOKE_START " name=\"lookup\">\n"
            DS41_PARAM_START " name=\"image\" string=\"true\">session_id" DS41_PARAM_END "\n"
            DS41_INVOKE_END "\n" DS41_TOOL_CALLS_END;
        char *response = http_route(&s, "/v1/chat/completions", chat);
        assert(fixture.compare_calls == 1 && !fixture.rewrite_calls && !fixture.physical_sync_calls);
        assert(fixture.sync_calls == (mode == 1 ? 1 : 0));
        assert(strstr(response, mode == 2 ? "\"finish_reason\":\"error\"" :
                                               "\"finish_reason\":\"tool_calls\""));
        assert(mode != 0 || fixture.mapped);
        assert(mode != 1 || !fixture.mapped);
        free(response); teardown(&s, &slot);
    }
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    test_mapped_recovery_and_logical_canonicalization();
    test_configuration_and_help();
    test_route_usage_and_baseline();
    test_guards_no_owned_retirement();
    test_errors_and_owned_cleanup();
    test_tool_and_generic_sync();
    test_dense_routes_and_retire_results();
    test_invalid_quarantine();
    test_dense_receipt_log();
    test_invalid_terminal();
    test_owned_final_cleanup();
    test_scr_liveness_during_policy();
    test_transport_clock_and_owner();
    test_prefill_error_after_headers();
    test_context_bounded_output_default();
    fixture_reset();
    puts("test_server_contract: 15 scenario groups passed (model/tokenizer/session/coordinate comparison/dense-policy/retirement/decode substituted)");
    return 0;
}
