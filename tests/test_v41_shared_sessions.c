/* Model-backed multi-session equivalence for the V4.1 shared workspace.
 *
 *   DS4_V41_SHARED_WORKSPACE=0|1 ./tests/test_v41_shared_sessions MODEL DRAFTER OUT
 *
 * Runs N sessions (default 4) twice on one engine: "alone" (each session is
 * created, prefilled in its own quanta, decoded with DSpark and plain steps,
 * then freed before the next starts) and "inter" (all sessions live, every
 * operation round-robin, then batched decode steps and a mixed step of four
 * suffix rows plus three decode rows). Writes one line per checkpoint: tokens
 * and an FNV-1a hash of the logits. Run once unshared and once shared and
 * compare the files: every
 * line must match, and the per-session lines of "inter" must match "alone".
 * Session 0 has a >40k-token prompt so its single sync defers the decoder
 * across an encoder-only sweep (the carry outlives one sweep).
 * Env: V41_SESSIONS (1..4), V41_CTX (default 262144), V41_ROUNDS (12). */
#include "ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SESSIONS 4
#define MAX_GEN 4096

static const char *prompt_files[MAX_SESSIONS] = {
    "/tmp/ds4_policy/prompts/summarize.txt",
    "/tmp/ds4_policy/prompts/ren_agent.txt",
    "/tmp/ds4_policy/prompts/json_tool.txt",
    "/tmp/ds4_policy/prompts/ren_toolcall.txt",
};
/* Server-like quanta per session; 0 syncs the whole prompt in one call. */
static const int quanta[MAX_SESSIONS] = {0, 2048, 128, 4096};
static const int long_tokens = 45000;

typedef struct {
    ds4_session *s;
    ds4_tokens prompt;
    int done;
    int gen[MAX_GEN];
    int ngen;
} spec;

static ds4_engine *engine;
static int eos, nsessions = MAX_SESSIONS, rounds = 12, ctx = 262144;
static FILE *out;
static float *logits;

static void die(const char *what, int i, const char *err) {
    fprintf(stderr, "FAIL: %s session=%d %s\n", what, i, err ? err : "");
    exit(1);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open prompt", -1, path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) die("read prompt", -1, path);
    buf[n] = 0;
    fclose(f);
    return buf;
}

static uint64_t logits_hash(spec *p) {
    const int n = ds4_session_copy_logits(p->s, logits, 1 << 20);
    if (n <= 0) die("copy logits", -1, NULL);
    uint64_t h = 1469598103934665603ull;
    const unsigned char *b = (const unsigned char *)logits;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static void checkpoint(const char *mode, int i, const char *phase, spec *p, int from) {
    fprintf(out, "%s s%d %s pos=%d logits=%016llx tokens", mode, i, phase,
            ds4_session_pos(p->s), (unsigned long long)logits_hash(p));
    for (int k = from; k < p->ngen; k++) fprintf(out, " %d", p->gen[k]);
    fputc('\n', out);
    fflush(out);
}

static void push(spec *p, int token) {
    if (p->ngen >= MAX_GEN) die("generation overflow", -1, NULL);
    p->gen[p->ngen++] = token;
    ds4_tokens_push(&p->prompt, token);
}

/* One prefill quantum; returns true when the prompt is complete. */
static bool prefill_step(spec *p, int i, int len) {
    char err[256];
    int target = quanta[i] ? p->done + quanta[i] : len;
    if (target > len) target = len;
    ds4_tokens prefix = p->prompt;
    prefix.len = target;
    if (ds4_session_sync(p->s, &prefix, err, sizeof(err)) != 0) die("sync", i, err);
    p->done = ds4_session_pos(p->s);
    if (p->done != target) die("sync frontier", i, NULL);
    return p->done >= len;
}

/* One DSpark cycle and one plain decode step. */
static void decode_round(spec *p, int i) {
    char err[256];
    int acc[17];
    int token = ds4_session_argmax_excluding(p->s, eos);
    const int n = ds4_session_eval_speculative_argmax(p->s, token, 16, eos, acc, 17, err, sizeof(err));
    if (n <= 0) die("speculative", i, err);
    for (int k = 0; k < n; k++) push(p, acc[k]);
    token = ds4_session_argmax_excluding(p->s, eos);
    if (ds4_session_eval(p->s, token, err, sizeof(err)) != 0) die("eval", i, err);
    push(p, token);
}

static void make_prompt(spec *p, int i) {
    char *text = read_file(prompt_files[i]);
    ds4_tokenize_rendered_chat(engine, text, &p->prompt);
    free(text);
    if (i == 0) {
        const int base = p->prompt.len;
        for (int k = 0; p->prompt.len < long_tokens; k++) ds4_tokens_push(&p->prompt, p->prompt.v[k % base]);
    }
}

static void create(spec *p, int i) {
    if (ds4_session_create(&p->s, engine, ctx) != 0) die("session create", i, NULL);
}

static void run_alone(spec *sp) {
    for (int i = 0; i < nsessions; i++) {
        spec *p = &sp[i];
        create(p, i);
        const int len = p->prompt.len;
        while (!prefill_step(p, i, len)) {}
        checkpoint("alone", i, "prefill", p, 0);
        for (int r = 0; r < rounds; r++) decode_round(p, i);
        checkpoint("alone", i, "decode", p, 0);
        ds4_session_free(p->s);
        p->s = NULL;
    }
}

static void run_inter(spec *sp, int *len) {
    for (int i = 0; i < nsessions; i++) create(&sp[i], i);
    bool busy = true;
    while (busy) {
        busy = false;
        for (int i = 0; i < nsessions; i++)
            if (sp[i].done < len[i]) busy |= !prefill_step(&sp[i], i, len[i]);
    }
    for (int i = 0; i < nsessions; i++) checkpoint("inter", i, "prefill", &sp[i], 0);
    for (int r = 0; r < rounds; r++)
        for (int i = 0; i < nsessions; i++) decode_round(&sp[i], i);
    for (int i = 0; i < nsessions; i++) checkpoint("inter", i, "decode", &sp[i], 0);
    if (nsessions < 2) return;

    char err[256];
    int mark[MAX_SESSIONS];
    for (int i = 0; i < nsessions; i++) mark[i] = sp[i].ngen;
    ds4_decode_item items[MAX_SESSIONS];
    for (int step = 0; step < 6; step++) {
        for (int i = 0; i < nsessions; i++)
            items[i] = (ds4_decode_item){sp[i].s, ds4_session_argmax_excluding(sp[i].s, eos)};
        if (ds4_sessions_eval_batch(items, nsessions, err, sizeof(err)) != 0) die("batch", -1, err);
        for (int i = 0; i < nsessions; i++) push(&sp[i], items[i].token);
    }
    /* Session 0 appends a short user suffix while the others decode. */
    ds4_tokens suffix = {0};
    ds4_tokenize_text(engine, " Answer in one short sentence.", &suffix);
    for (int k = 0; k < suffix.len && k < 4; k++) ds4_tokens_push(&sp[0].prompt, suffix.v[k]);
    ds4_tokens_free(&suffix);
    for (int i = 1; i < nsessions; i++)
        items[i - 1] = (ds4_decode_item){sp[i].s, ds4_session_argmax_excluding(sp[i].s, eos)};
    if (ds4_sessions_eval_batch_with_prefill(items, nsessions - 1, sp[0].s, &sp[0].prompt,
                                             err, sizeof(err)) != 0) die("mixed batch", 0, err);
    if (ds4_session_pos(sp[0].s) != sp[0].prompt.len) die("mixed prefill frontier", 0, NULL);
    for (int i = 1; i < nsessions; i++) push(&sp[i], items[i - 1].token);
    for (int step = 0; step < 4; step++) {
        for (int i = 0; i < nsessions; i++)
            items[i] = (ds4_decode_item){sp[i].s, ds4_session_argmax_excluding(sp[i].s, eos)};
        if (ds4_sessions_eval_batch(items, nsessions, err, sizeof(err)) != 0) die("batch", -1, err);
        for (int i = 0; i < nsessions; i++) push(&sp[i], items[i].token);
    }
    for (int i = 0; i < nsessions; i++) checkpoint("inter", i, "batch", &sp[i], mark[i]);
    for (int i = 0; i < nsessions; i++) { ds4_session_free(sp[i].s); sp[i].s = NULL; }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL DRAFTER OUT\n", argv[0]);
        return 2;
    }
    if (getenv("V41_SESSIONS")) nsessions = atoi(getenv("V41_SESSIONS"));
    if (getenv("V41_ROUNDS")) rounds = atoi(getenv("V41_ROUNDS"));
    if (getenv("V41_CTX")) ctx = atoi(getenv("V41_CTX"));
    if (nsessions < 1 || nsessions > MAX_SESSIONS || rounds < 0) return 2;
    ds4_engine_options opt = {
        .model_path = argv[1], .mtp_path = argv[2], .backend = DS4_BACKEND_METAL,
        .context_size = ctx, .dspark = true,
        .placement_session_count_hint = nsessions,
        .share_session_prefill_workspace = true,
    };
    if (ds4_engine_open(&engine, &opt) != 0) die("engine open", -1, NULL);
    out = fopen(argv[3], "w");
    logits = malloc((size_t)4 << 20);
    if (!out || !logits) die("output", -1, argv[3]);
    eos = ds4_token_eos(engine);
    spec sp[MAX_SESSIONS] = {0}, again[MAX_SESSIONS] = {0};
    int len[MAX_SESSIONS];
    for (int i = 0; i < nsessions; i++) {
        make_prompt(&sp[i], i);
        ds4_tokens_copy(&again[i].prompt, &sp[i].prompt);
        len[i] = sp[i].prompt.len;
        fprintf(stderr, "session %d: %d prompt tokens, quantum %d\n", i, len[i], quanta[i]);
    }
    run_alone(sp);
    run_inter(again, len);
    for (int i = 0; i < nsessions; i++) { ds4_tokens_free(&sp[i].prompt); ds4_tokens_free(&again[i].prompt); }
    fclose(out);
    free(logits);
    ds4_engine_close(engine);
    fprintf(stderr, "test_v41_shared_sessions: done\n");
    return 0;
}
