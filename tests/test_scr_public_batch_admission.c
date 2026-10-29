/* Public admission regression: otherwise eligible resident Flash41 sessions.
 * Strict host storage only; no Metal/model backend is linked or executed. */
#include "../ds4.h"
static bool scalar_step(void *, const void *, const void *, int, float *);
#define DS41_EVAL_STEP scalar_step
#include "../ds4.c"
#include <assert.h>
extern uint64_t scr_host_command_begins(void);
extern void scr_host_fail_next_begin(void);
static unsigned scalar_calls;
static bool scalar_step(void *opaque, const void *model, const void *weights, int token, float *logits) {
    (void)model; (void)weights;
    ds41_gpu_graph *g = opaque;
    ++scalar_calls; ++g->pos; g->valid = true;
    logits[0] = (float)token;
    return true;
}

typedef struct {
    ds4_engine engine;
    ds4_session sessions[2];
    ds41_scr_archive archive;
    uint32_t archive_tags[640];
    int tokens[2][8];
    float logits[2];
} fixture;

static fixture *prepare(void) {
    fixture *f = calloc(1, sizeof(*f)); assert(f);
    f->engine.backend = DS4_BACKEND_METAL;
    f->engine.ds41_session_bytes = 4096;
    for (int i = 0; i < 2; ++i) {
        ds4_session *s = &f->sessions[i];
        s->engine = &f->engine; s->ctx_size = 64;
        s->checkpoint_valid = s->ds41_graph_ready = true;
        s->checkpoint = (ds4_tokens){f->tokens[i], 4, 8};
        for (int t = 0; t < 8; ++t) f->tokens[i][t] = t + i + 1;
        s->logits = &f->logits[i];
        ds41_gpu_graph *g = &s->ds41_graph;
        g->valid = true; g->pos = 4; g->ctx = 64; g->prefill_cap = 8;
        g->tp_world = 1; g->allocation_bytes = 2048;
        assert(!ds41_coordinates_restricted(s));
    }
    f->archive.tags = f->archive_tags;
    f->archive.cap = f->archive.end = 128;
    f->archive.sealed = true; f->archive.generation = 7;
    f->archive.writer_state = DS41_ARCHIVE_ARMED;
    f->archive.allocation_bytes = 1024;
    return f;
}
static void attach(fixture *f, unsigned state, unsigned owner) {
    ds41_gpu_graph *g = &f->sessions[owner].ds41_graph;
    f->archive.session = &f->sessions[owner];
    f->archive.model = &f->engine.model;
    if (state == 0) g->archive = &f->archive;
    if (state == 1) g->archive_pending = &f->archive;
    if (state == 2) g->archive_capture_pending = true;
    assert(!ds41_coordinates_restricted(&f->sessions[owner])); /* Deliberately unmapped. */
}
static int owning_case(bool mixed, unsigned state, unsigned owner) {
    fixture *f = prepare(); attach(f, state, owner);
    fixture *before = malloc(sizeof(*before)); assert(before); memcpy(before, f, sizeof(*f));
    ds4_decode_item items[2] = {{&f->sessions[0], 11}, {&f->sessions[1], 12}};
    ds4_tokens prompt = {f->tokens[0], 5, 8};
    /* Prove the fixture meets the native upstream support selector, rather than
     * accidentally passing because another prerequisite rejects it. */
    assert(ds4_sessions_eval_batch_metal_supported(items, 2, &f->engine));
    if (mixed) assert(ds41_mixed_batch_supported(&items[1], 1, &f->sessions[0], &prompt));
    char error[256] = {0};
    const uint64_t commands = scr_host_command_begins();
    const unsigned scalars = scalar_calls;
    int rc = mixed ? ds4_sessions_eval_batch_with_prefill(&items[1], 1,
        &f->sessions[0], &prompt, error, sizeof(error)) : ds4_sessions_eval_batch(items, 2, error, sizeof(error));
    const bool unchanged = !memcmp(f, before, sizeof(*f));
    const uint64_t began = scr_host_command_begins() - commands;
    const bool scalar_fallback = scalar_calls != scalars;
    const bool pass = rc != 0 && unchanged && !began && !scalar_fallback;
    printf("{\"mixed\":%s,\"owner\":%u,\"state\":%u,\"rc\":%d,\"commands_started\":%llu,\"unchanged\":%s,\"scalar_fallback\":%s,\"passed\":%s}\n",
        mixed ? "true" : "false", owner, state, rc, (unsigned long long)began,
        unchanged ? "true" : "false", scalar_fallback ? "true" : "false", pass ? "true" : "false");
    free(before); free(f); return !pass;
}
static void ordinary_controls(void) {
    fixture *f = prepare();
    ds4_decode_item items[2] = {{&f->sessions[0], 11}, {&f->sessions[1], 12}};
    assert(ds4_sessions_eval_batch_metal_supported(items, 2, &f->engine));
    char error[256];
    uint64_t commands = scr_host_command_begins();
    /* An injected backend refusal proves an ordinary public batch still reaches
     * the existing dispatch. It does not stand in for successful GPU arithmetic. */
    scr_host_fail_next_begin();
    assert(ds4_sessions_eval_batch(items, 2, error, sizeof(error)) != 0);
    assert(scr_host_command_begins() == commands + 1);
    assert(!f->sessions[0].checkpoint_valid && !f->sessions[1].checkpoint_valid);
    free(f); f = prepare();
    attach(f, 0, 0); /* Owning single-session scalar route remains supported. */
    items[0] = (ds4_decode_item){&f->sessions[0], 13};
    commands = scr_host_command_begins();
    unsigned scalars = scalar_calls;
    assert(!ds4_sessions_eval_batch(items, 1, error, sizeof(error)));
    assert(scalar_calls == scalars + 1 && scr_host_command_begins() == commands);
    assert(f->sessions[0].checkpoint_valid && f->sessions[0].checkpoint.len == 5);
    assert(f->sessions[0].ds41_graph.archive == &f->archive);
    free(f);
    puts("ordinary native selection/public dispatch and owning single-session scalar controls passed");
}
int main(void) {
    assert(ds4_gpu_init()); g_ds4_shape = DS4_SHAPE_FLASH41;
    int failures = 0;
    for (unsigned mixed = 0; mixed < 2; ++mixed)
        for (unsigned state = 0; state < 3; ++state)
            for (unsigned owner = 0; owner < 2; ++owner)
                failures += owning_case(mixed, state, owner);
    ordinary_controls(); ds4_gpu_cleanup();
    printf("public ownership admission failures=%d of12\n", failures);
    return failures ? 1 : 0;
}
