/* Selected upstream routing/valid-frontier regressions use heap-only checkpoint
 * shells, not a loaded engine. No listener, native execution or GPU is opened. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
/* Migrate material facts from the retired duplicate helper to the real probe.
 * Stock upstream tests below already cover wrong frontier, pixel identity,
 * shifted image offsets, tier precedence and staleness; do not duplicate those. */
static void test_migrated_visible_facts(void) {
    server s = {0}; server_slot slot = {0}; request req = {.kind = REQ_CHAT};
    int ids[] = {10, 11};
    slot.session = ds4_session_new_test_checkpoint(ids, 2);
    slot.responses_live = (live_tool_state){.valid = true, .live_tokens = 2,
        .visible_text = xstrdup("ab"), .visible_len = 2};
    slot.thinking_live = (visible_live_state){.valid = true, .live_tokens = 2,
        .visible_text = xstrdup("ab"), .visible_len = 2};
    for (int i = 0; i < 16; ++i) ds4_tokens_push(&req.prompt, 999 + i);
    for (int api = 0; api < 2; ++api) {
        req.api = api ? API_OPENAI : API_RESPONSES; req.prompt_text = "abcd";
        slot_reuse p = slot_probe_reuse_locked(&s, &slot, &req);
        TEST_ASSERT(p.kind == (api ? REUSE_THINKING_VISIBLE : REUSE_RESPONSES_VISIBLE));
        TEST_ASSERT(p.reuse_tokens == 2 && p.suffix_off == 2);
        TEST_ASSERT(!slot.responses_live.call_ids.len); /* Visible binding needs no call ID. */
        req.prompt_text = "ab"; TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        req.prompt_text = "zzcd"; TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        req.prompt_text = NULL; TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        req.prompt_text = "abcd";
        slot.responses_live.valid = slot.thinking_live.valid = false;
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        slot.responses_live.valid = slot.thinking_live.valid = true;
        char *saved = api ? slot.thinking_live.visible_text : slot.responses_live.visible_text;
        if (api) slot.thinking_live.visible_text = NULL; else slot.responses_live.visible_text = NULL;
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        if (api) slot.thinking_live.visible_text = saved; else slot.responses_live.visible_text = saved;
    }
    req.api = API_RESPONSES; slot.responses_live.valid = false;
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE); /* No thinking fallback. */
    slot.responses_live.valid = true; req.api = API_OPENAI; req.kind = REQ_COMPLETION;
    TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
    req.kind = REQ_CHAT;
    ds4_session_free_test_checkpoint(slot.session);
    slot.session = ds4_session_new_test_checkpoint(NULL, 0);
    TEST_ASSERT(ds4_session_checkpoint_valid(slot.session) && ds4_session_tokens(slot.session)->len == 0);
    slot.responses_live.live_tokens = slot.thinking_live.live_tokens = 0;
    for (int api = 0; api < 2; ++api) {
        req.api = api ? API_OPENAI : API_RESPONSES;
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
    }
    ds4_session_free_test_checkpoint(slot.session);
    slot.session = ds4_session_new_test_checkpoint(ids, 2);
    slot.responses_live.live_tokens = slot.thinking_live.live_tokens = 2;
    free(slot.responses_live.visible_text); free(slot.thinking_live.visible_text);
    slot.responses_live.visible_text = xstrdup("a\036b"); slot.responses_live.visible_len = 3;
    slot.thinking_live.visible_text = xstrdup("a\036b"); slot.thinking_live.visible_len = 3;
    slot.responses_live.images = slot.thinking_live.images = (visible_image_key){.count=1,.offsets={1}};
    ds4_vision_span historical = {.token_start=1,.embedding={.token_count=1,.fingerprint={7}}};
    ds4_session_set_test_images(slot.session, &historical, 1);
    req.images = calloc(2, sizeof(*req.images)); TEST_ASSERT(req.images != NULL);
    req.images[0] = historical; req.images[1] = (ds4_vision_span){.token_start=5,.embedding={.token_count=1,.fingerprint={9}}};
    req.image_markers = calloc(2, sizeof(*req.image_markers)); TEST_ASSERT(req.image_markers != NULL);
    snprintf(req.image_markers[0], sizeof(req.image_markers[0]), "X");
    snprintf(req.image_markers[1], sizeof(req.image_markers[1]), "Y");
    for (int api = 0; api < 2; ++api) {
        req.api = api ? API_OPENAI : API_RESPONSES;
        req.image_count = 1; req.prompt_text = "aXbmore";
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == (api ? REUSE_THINKING_VISIBLE : REUSE_RESPONSES_VISIBLE));
        req.prompt_text = "Xa bmore"; /* Historical image offset shifted. */
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        req.prompt_text = "aXbmore";
        req.image_count = 0; TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
        req.image_count = 2; req.prompt_text = "aXbYmore"; /* Exactly at boundary. */
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == (api ? REUSE_THINKING_VISIBLE : REUSE_RESPONSES_VISIBLE));
        req.prompt_text = "aXbcdYmore";
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == (api ? REUSE_THINKING_VISIBLE : REUSE_RESPONSES_VISIBLE));
        req.prompt_text = "aXYbmore"; /* New image inside the remembered boundary. */
        TEST_ASSERT(slot_probe_reuse_locked(&s, &slot, &req).kind == REUSE_NONE);
    }
    req.prompt_text = NULL; request_free(&req);
    live_tool_state_free(&slot.responses_live); visible_live_free(&slot.thinking_live);
    ds4_session_free_test_checkpoint(slot.session);
}
int main(void) {
    test_migrated_visible_facts();
    test_live_continuation_contract();
    test_slot_probe_and_routing_scores();
    test_slot_probe_live_state_tiers();
    test_slot_probe_vision_tiers();
    test_slot_routing_staleness_tiers();
    test_dispatch_routes_alien_request_to_empty_slot();
    test_request_defaults_use_min_p_filtering();
    test_reasoning_effort_mapping();
    if (test_failures) return 1;
    puts("upstream model-free reuse/vision/frontier/staleness/dispatch/default-controls: PASS");
    return 0;
}
