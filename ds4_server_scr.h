/* SCR server integration, selected by inclusion section so each part lands at
 * the point in ds4_server.c where its dependencies are complete:
 *
 *   DS4_SERVER_SCR_TYPES      before the server types: trace ids and the
 *                             ds4.reuse.v1 telemetry state.
 *   DS4_SERVER_SCR_GUARDS     request guards used by the API parsers.
 *   DS4_SERVER_SCR_RUNTIME    prefill admission, owned retirement and the
 *                             ds4.reuse.v1 writer, after server/slot/job types.
 *   DS4_SERVER_SCR_TRANSPORT  keepalive-serviced SCR prefill, after the
 *                             prefill-progress type.
 *   DS4_SERVER_SCR_CONFIG     CLI configuration, after server_config.
 *
 * ds4.reuse.v1 is one compact SCR telemetry record per request. The record is
 * written once, finalized, to an optional append-only JSONL file
 * (<DS4_SCR_METADATA_DIR>/reuse.v1.jsonl) and, when a streaming chat request
 * sets stream_options.include_ds4_scr, carried once as a `ds4_reuse` chunk
 * with an empty `choices` array before the final SSE events. Numbers are plain
 * JSON numbers, arrays are variable-length and unknown values are omitted
 * rather than nulled. The record never contains prompt text, generated text or
 * token ids. */

/* ---- Types: trace ids and telemetry state ------------------------------- */
#if defined(DS4_SERVER_SCR_TYPES) && !defined(DS4_SERVER_SCR_TYPES_DONE)
#define DS4_SERVER_SCR_TYPES_DONE
#include "ds4.h"
#include "ds4_kvstore.h"
#include <stdlib.h>

#define DS4_TRACE_IDS_FORMAT "sha1-le64count-le32ids-v1"
#define DS4_TRACE_IDS_LIMIT 1048576

/* Non-adversarial diagnostic checksum, not authentication. The ceiling bounds
 * diagnostic scratch only; it does not change request admission or context. */
static const char *trace_ids_sha1(const ds4_tokens *tokens, char hex[41]) {
    if (!tokens || tokens->len < 0) return "invalid-count";
    if (tokens->len > DS4_TRACE_IDS_LIMIT) return "oversize";
    if (tokens->len && !tokens->v) return "invalid-pointer";
    for (int i = 0; i < tokens->len; ++i) {
        if (tokens->v[i] < 0) return "negative-id";
    }
    uint64_t count = (uint64_t)tokens->len;
    if (count > (SIZE_MAX - 8) / 4) return "oversize";
    size_t bytes = 8 + (size_t)count * 4;
    unsigned char *encoded = malloc(bytes);
    if (!encoded) return "allocation-failed";
    for (unsigned i = 0; i < 8; ++i)
        encoded[i] = (unsigned char)(count >> (8 * i));
    for (int i = 0; i < tokens->len; ++i) {
        uint32_t id = (uint32_t)tokens->v[i];
        for (unsigned b = 0; b < 4; ++b)
            encoded[8 + (size_t)i * 4 + b] = (unsigned char)(id >> (8 * b));
    }
    ds4_kvstore_sha1_bytes_hex(encoded, bytes, hex);
    free(encoded);
    return NULL;
}

static void trace_ids_record(FILE *fp, const char *role, const ds4_tokens *tokens) {
    char hex[41];
    const char *reason = trace_ids_sha1(tokens, hex);
    fprintf(fp, "%s_ids: count=%d format=%s ", role,
            tokens ? tokens->len : -1, DS4_TRACE_IDS_FORMAT);
    if (reason) fprintf(fp, "unavailable=%s\n", reason);
    else fprintf(fp, "sha1=%s\n", hex);
}

static void trace_prompt_ids(FILE *fp, const ds4_tokens *canonical,
                             const ds4_tokens *selected) {
    trace_ids_record(fp, "canonical_prompt", canonical);
    /* Object identity binds the actual selected vector, never equal lengths. */
    if (canonical == selected)
        fputs("effective_prompt_ids: same-vector=canonical_prompt_ids\n", fp);
    else
        trace_ids_record(fp, "effective_prompt", selected);
}


#include <pthread.h>

typedef struct {
    bool initialized, requested, mutex_ready;
    int fd;              /* Append-only record file; -1 when absent or failed. */
    uint64_t sequence;   /* Last admitted request sequence. */
    uint64_t dropped;    /* Records that could not be written. */
    pthread_mutex_t mutex;
} server_reuse_telemetry;

typedef struct {
    uint64_t seq;        /* 0: not admitted (telemetry off or exhausted). */
    char request[96];    /* Response id, shared with the client response. */
    double enqueued, started, first_token, decode_started, decode_finished;
    int prefill_rc, generated;
    bool prefill_known, emitted;
    const char *error_phase; /* NULL on success. */
    int error_code;
    /* "reuse":"exact": ordinary prefill reused this many live tokens. */
    bool exact;
    int exact_live_prefix;
    double exact_prefill_seconds;
    ds4_scr_logical_receipt logical;
    ds4_scr_archive_snapshot admission, after_prefill;
    /* Checkpoint pool: selection before prefill, save after success. */
    bool pool_known;
    ds4_scr_pool_receipt pool;
} job_reuse_telemetry;
#endif

/* ---- Guards ------------------------------------------------------------- */
#ifdef DS4_SERVER_SCR_GUARDS
static bool server_scr_enabled(const server *s);

/* Decode structural keys with the production JSON reader. Every occurrence is
 * checked; nested metadata and text are deliberately not conversation state. */
static bool server_scr_chat_guard(server *s, const char *body) {
    if (!server_scr_enabled(s)) return true;
    const char *p = body;
    json_ws(&p);
    if (*p++ != '{') return false;
    for (;;) {
        json_ws(&p);
        if (*p == '}') return true;
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p++ != ':') { free(key); return false; }
        json_ws(&p);
        bool state = !strcmp(key, "previous_response_id") ||
                     !strcmp(key, "conversation") ||
                     !strcmp(key, "conversation_id") || !strcmp(key, "session_id");
        bool store = !strcmp(key, "store");
        free(key);
        const char *value = p;
        if (!json_skip_value(&p)) return false;
        size_t n = (size_t)(p - value);
        if (state && !(n == 4 && !memcmp(value, "null", 4))) return false;
        if (store && !((n == 4 && !memcmp(value, "null", 4)) ||
                       (n == 5 && !memcmp(value, "false", 5)))) return false;
        json_ws(&p);
        if (*p == '}') return true;
        if (*p++ != ',') return false;
    }
}

static bool server_scr_images_guard(server *s, const chat_msgs *msgs) {
    if (!server_scr_enabled(s)) return true;
    for (int i = 0; i < msgs->len; i++)
        if (msgs->v[i].images.len) return false;
    return true;
}
#endif

/* ---- Runtime: admission, retirement and the ds4.reuse.v1 writer --------- */
#ifdef DS4_SERVER_SCR_RUNTIME
static bool server_scr_enabled(const server *s) {
    return s && s->scr_tail_tokens != 0;
}

/* Timers overlap; live retention may be SCR-conditioned, not exact cache reuse.
 * Installed spans are not committed work. Seal is not terminal source survival. */
static void server_scr_receipt_log(server_slot *slot, job *j, int rc,
                                    const ds4_scr_dense_policy_receipt *r) {
#define SCR_LOG(...) server_log(DS4_LOG_PREFILL, __VA_ARGS__)
#define SCR_U(x) ((unsigned long long)(x))
    SCR_LOG("ds4-server: scr dense slot=%d request=%p rc=%d route=%d reason=%d trigger=%d error=%d N=%u C=%u K=%u lcp=%u live_retained=%u continuation_work=%u continuation_heads=%u work_unknown=%d head_unknown=%d partial_valid=%d complete_N=%d source=[%u,%u) generation=%llu outer_host=%llu peak_outer_host=%llu total_s=%.6f",
        slot->id, (void *)j, rc, r->route, r->reason, r->trigger, r->error_code,
        r->requested_tokens, r->configured_cap, r->configured_k, r->native_lcp,
        r->retained_tokens, r->continuation_completed, r->continuation_heads,
        r->continuation_work_unknown, r->continuation_head_unknown, r->partial_valid,
        r->complete_n, r->source_start, r->source_end, SCR_U(r->source_generation),
        SCR_U(r->outer_host_bytes), SCR_U(r->peak_outer_host_bytes), r->total_seconds);
    const ds4_scr_dense_match_receipt *m = &r->match;
    SCR_LOG("ds4-server: scr match request=%p selected=%u", (void *)j, m->selected_ranges);
    const ds4_scr_multi_receipt *u = &r->multi;
    SCR_LOG("ds4-server: scr multi request=%p outcome=%d reason=%d phase=%d N=%u input=%u normalized=%u attempted_spans=%u materialized_spans=%u installed_spans=%u committed_spans=%u attempted_tokens=%u materialized_tokens=%u installed_tokens=%u committed_tokens=%u prefix_work=%u suffix_work=%u fresh_work=%u discarded=%u unknown=%d head=%d published=%d host=%llu growth=%llu archive=%llu prefix_s=%.6f stage_s=%.6f install_s=%.6f gap_s=%.6f suffix_s=%.6f fresh_s=%.6f head_s=%.6f publication_s=%.6f total_s=%.6f",
        (void *)j, u->outcome, u->reason, u->phase, u->requested_tokens,
        u->input_spans, u->normalized_spans, u->attempted_spans, u->materialized_spans,
        u->installed_spans, u->committed_spans, u->attempted_tokens, u->materialized_tokens,
        u->installed_tokens, u->committed_tokens, u->prefix_completed, u->suffix_completed,
        u->fresh_completed, u->discarded_tokens, u->partial_work_unknown, u->head_completed,
        u->published, SCR_U(u->temporary_host_bytes), SCR_U(u->publication_growth_bytes),
        SCR_U(u->archive_bytes), u->prefix_seconds, u->stage_seconds, u->install_seconds,
        u->gap_seconds, u->suffix_seconds, u->fresh_seconds, u->head_seconds,
        u->publication_seconds, u->total_seconds);
    for (uint32_t i = 0; i < r->configured_k && i < DS4_SCR_MULTI_MAX_RANGES; ++i)
        SCR_LOG("ds4-server: scr range request=%p i=%u plan_source=%u plan_destination=%u plan_count=%u multi_source=%u multi_destination=%u multi_count=%u gap_work=%u",
            (void *)j, i, m->plan[i].source_start, m->plan[i].destination_start,
            m->plan[i].count, u->ranges[i].source_start, u->ranges[i].destination_start,
            u->ranges[i].count, u->gap_completed[i]);
    const ds4_scr_refresh_receipt *f = &r->refresh;
    SCR_LOG("ds4-server: scr refresh request=%p phase=%d reason=%d error=%d N=%u source=[%u,%u) count=%u calls=%u chunks=%u prefix_work=%u source_work=%u remainder_work=%u heads=%u unknown=%d head_unknown=%d complete_N=%d retire_attempt=%d retired=%d prepared=%d sealed=%d cleanup_attempt=%d cleanup_retired=%d usable=%d fenced=%d host=%llu growth=%llu prospective=%llu charged=%llu engine_before=%llu engine_retired=%llu engine_final=%llu retire_s=%.6f prepare_s=%.6f prefix_s=%.6f source_s=%.6f seal_s=%.6f remainder_s=%.6f complete_s=%.6f cleanup_s=%.6f total_s=%.6f",
        (void *)j, f->phase, f->reason, f->error_code, f->requested_tokens,
        f->source_start, f->source_end, f->source_count, f->normal_calls, f->source_chunks,
        f->prefix_completed, f->source_completed, f->remainder_completed, f->completed_heads,
        f->work_unknown, f->head_unknown, f->complete_n, f->old_retirement_attempted,
        f->old_retired, f->prepared, f->sealed, f->failed_source_retirement_attempted,
        f->failed_source_retired, f->source_usable, f->source_retiring,
        SCR_U(f->temporary_host_bytes), SCR_U(f->publication_growth_bytes),
        SCR_U(f->prospective_archive_bytes), SCR_U(f->source_charged_bytes),
        SCR_U(f->engine_charged_bytes_before), SCR_U(f->engine_charged_bytes_after_retirement),
        SCR_U(f->engine_charged_bytes_final), f->retire_seconds, f->prepare_seconds,
        f->prefix_seconds, f->source_seconds, f->seal_seconds, f->remainder_seconds,
        f->complete_seconds, f->cleanup_seconds, f->total_seconds);
#undef SCR_U
#undef SCR_LOG
}

/* --kv-disk-dir under --scr holds the persistent checkpoint pool instead of
 * the disk KV cache, which SCR bypasses. An unusable directory disables the
 * pool only, as an unusable disk KV directory disables that cache. */
static void server_scr_pool_open(server *s, const char *directory, uint64_t budget_mb) {
    if (!server_scr_enabled(s) || !directory || !*directory) return;
    char path[PATH_MAX];
    int n = snprintf(path, sizeof(path), "%s", directory);
    for (char *at = path + 1; n > 0 && (size_t)n < sizeof(path) && *at; ++at)
        if (*at == '/') { *at = '\0'; (void)mkdir(path, 0700); *at = '/'; }
    (void)mkdir(directory, 0700);
    char err[PATH_MAX + 64] = {0};
    const uint64_t budget = budget_mb > UINT64_MAX >> 20 ? UINT64_MAX : budget_mb << 20;
    if (ds4_scr_pool_open(&s->scr_pool, directory, (uint32_t)s->ctx_size,
                          (uint32_t)s->scr_tail_tokens, budget, err, sizeof(err))) {
        server_log(DS4_LOG_WARNING, "ds4-server: SCR checkpoint pool disabled: %s", err);
        return;
    }
    server_log(DS4_LOG_DEFAULT, "ds4-server: SCR checkpoint pool %s entries=%u budget_mib=%llu",
               directory, ds4_scr_pool_entries(s->scr_pool),
               (unsigned long long)((budget ? budget : DS4_SCR_POOL_DEFAULT_BYTES) >> 20));
}

static void server_scr_pool_log(server_slot *slot, const char *phase, int rc,
                                const ds4_scr_pool_receipt *r) {
    server_log(DS4_LOG_PREFILL, "ds4-server: scr pool slot=%d phase=%s rc=%d selected=%d candidates=%u hot_covered=%u best_covered=%u restored=%d saved=%d written_bytes=%llu entries=%u seconds=%.6f",
        slot->id, phase, rc, r->selected, r->candidates, r->hot_covered, r->best_covered,
        r->restored, r->saved, (unsigned long long)r->written_bytes, r->entries, r->seconds);
}

/* Only normal canonical prefill calls policy. Repair/rewind keep ordinary sync. */
static int server_scr_normal_prefill(server *s, server_slot *slot, job *j,
                                      char *err, size_t errlen) {
    j->req.cache_read_tokens = 0;
    j->req.cache_write_tokens = 0;
    ds4_scr_logical_receipt logical = {.logical_known = true,
        .logical_tokens = (uint32_t)j->req.prompt.len,
        .execution = {
            .configured_cap = (uint32_t)s->scr_tail_tokens,
            .configured_k = (uint32_t)s->scr_max_ranges,
            .route = DS4_SCR_DENSE_ERROR, .reason = DS4_SCR_DENSE_CANCELLED,
            .error_code = DS4_SESSION_SYNC_INTERRUPTED,
        }};
    int rc = DS4_SESSION_SYNC_INTERRUPTED;
    if (server_prefill_enter(s, slot)) {
        if (s->scr_pool) {
            /* A failed swap leaves no archive; the policy then refreshes. */
            int pool_rc = ds4_session_scr_pool_select(slot->session, s->scr_pool, &j->req.prompt,
                (uint32_t)s->scr_max_ranges, 201, &j->reuse.pool);
            j->reuse.pool_known = true;
            if (j->reuse.pool.selected || pool_rc)
                server_scr_pool_log(slot, "select", pool_rc, &j->reuse.pool);
        }
        rc = ds4_session_scr_logical_policy(slot->session, &j->req.prompt,
            (uint32_t)s->scr_tail_tokens, (uint32_t)s->scr_max_ranges, 201,
            &logical, err, errlen);
        if (s->reuse.initialized)
            j->reuse.after_prefill = ds4_session_scr_archive_snapshot(slot->session);
        server_prefill_leave(s);
    }
    if (logical.physical_known)
        server_scr_receipt_log(slot, j, rc, &logical.execution);
    else server_log(DS4_LOG_PREFILL, "ds4-server: logical SCR physical request unknown rc=%d logical_N=%d",
        rc, j->req.prompt.len);
    if (s->reuse.initialized) {
        j->reuse.prefill_known = true;
        j->reuse.prefill_rc = rc;
        j->reuse.logical = logical;
    }
    return rc;
}

/* Per-request "reuse":"exact" under --scr: no planner and no grafts. The live
 * transcript is reused only when the prompt extends it, exactly as ordinary
 * sync does; a mapped transcript that does not is invalidated first, since
 * ordinary sync never owns pad provenance. */
static int server_scr_exact_prefill(server *s, server_slot *slot, job *j,
                                    char *err, size_t errlen) {
    j->req.cache_read_tokens = 0;
    j->req.cache_write_tokens = 0;
    int rc = DS4_SESSION_SYNC_INTERRUPTED, live = 0;
    double seconds = -1.0;
    if (server_prefill_enter(s, slot)) {
        ds4_session *session = slot->session;
        const double began = now_sec();
        const int common = ds4_session_common_prefix(session, &j->req.prompt);
        const bool extends = common > 0 && common == ds4_session_pos(session);
        if (ds4_session_has_physical_coordinates(session)) {
            if (!extends) ds4_session_invalidate(session);
            rc = extends ? ds4_session_sync_physical(session, &j->req.prompt, err, errlen) :
                ds4_session_sync(session, &j->req.prompt, err, errlen);
        } else rc = ds4_session_sync(session, &j->req.prompt, err, errlen);
        live = extends ? common : 0;
        seconds = now_sec() - began;
        if (s->reuse.initialized)
            j->reuse.after_prefill = ds4_session_scr_archive_snapshot(session);
        server_prefill_leave(s);
    }
    server_log(DS4_LOG_PREFILL, "ds4-server: scr exact slot=%d request=%p rc=%d N=%d live_prefix=%d prefill_s=%.6f",
        slot->id, (void *)j, rc, j->req.prompt.len, live, seconds);
    if (s->reuse.initialized) {
        j->reuse.prefill_known = true;
        j->reuse.prefill_rc = rc;
        j->reuse.exact = true;
        j->reuse.exact_live_prefix = live;
        j->reuse.exact_prefill_seconds = seconds;
    }
    return rc;
}

static bool server_scr_terminal_success(bool delivered, const char *finish) {
    return delivered && finish && (!strcmp(finish, "stop") ||
        !strcmp(finish, "tool_calls") || !strcmp(finish, "length"));
}

static bool server_scr_quarantined(server *s) {
    pthread_mutex_lock(&s->mu);
    bool fatal = s->scr_ownership_fatal;
    pthread_mutex_unlock(&s->mu);
    return fatal;
}

/* Called only by an assigned worker or owned teardown, never client rejection
 * or queued cancellation. This retires ephemeral source, not client history. */
static ds4_scr_retire_receipt server_scr_retire_owned(server *s, server_slot *slot) {
    ds4_scr_retire_receipt r = {0};
    if (!server_scr_enabled(s)) return r;
    if (!slot->session) {
        /* An owned uncreated slot cannot contain a session archive. */
        r.outcome = DS4_SCR_RETIRE_ABSENT;
        r.charge_known = true;
        return r;
    }
    pthread_mutex_lock(&s->inference_mu);
    if (!server_scr_quarantined(s))
        ds4_session_scr_archive_retire_once(slot->session, &r);
    pthread_mutex_unlock(&s->inference_mu);
    if (r.outcome == DS4_SCR_RETIRE_INVALID) {
        pthread_mutex_lock(&s->mu);
        s->scr_ownership_fatal = true;
        s->stopping = true;
        /* Wake only this server's owned listener, never a global test fd. */
        if (s->scr_listener_owned && g_listen_fd == s->scr_listener_fd)
            shutdown(s->scr_listener_fd, SHUT_RDWR);
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
    }
    return r;
}

/* After a successful request: keep a newly sealed archive. Pool failure is
 * checkpoint loss only. */
static void server_scr_pool_save(server *s, server_slot *slot, job *j) {
    if (!s->scr_pool || server_scr_quarantined(s)) return;
    ds4_scr_pool_receipt r = {0};
    server_inference_lock(s);
    int rc = ds4_session_scr_pool_save(slot->session, s->scr_pool, false, &r);
    server_inference_unlock(s);
    if (r.saved || rc) server_scr_pool_log(slot, "save", rc, &r);
    j->reuse.pool_known = true;
    j->reuse.pool.saved = r.saved;
    j->reuse.pool.entries = r.entries;
}

/* Owned teardown, after workers stopped: keep each slot's archive, appended
 * tail included, then drop the catalog. */
static void server_scr_pool_close(server *s) {
    if (!s->scr_pool) return;
    for (int i = 0; s->slots && !server_scr_quarantined(s) && i < s->slot_count; ++i) {
        if (!s->slots[i].session) continue;
        ds4_scr_pool_receipt r = {0};
        int rc = ds4_session_scr_pool_save(s->slots[i].session, s->scr_pool, true, &r);
        server_scr_pool_log(&s->slots[i], "shutdown", rc, &r);
    }
    ds4_scr_pool_close(s->scr_pool);
    s->scr_pool = NULL;
}

#include <fcntl.h>
#include "adapter/include/cache_reuse_utils.h"

static ds4_scr_archive_snapshot server_scr_snapshot(server *s, server_slot *slot) {
    ds4_scr_archive_snapshot view = {0};
    if (server_scr_quarantined(s)) return view;
    server_inference_lock(s);
    view = ds4_session_scr_archive_snapshot(slot->session);
    server_inference_unlock(s);
    return view;
}

/* The directory must already exist; a missing or unwritable store is telemetry
 * loss only and never refuses startup or a request. */
static void server_reuse_init(server *s, const char *directory) {
    if (!server_scr_enabled(s)) return;
    server_reuse_telemetry *t = &s->reuse;
    *t = (server_reuse_telemetry){.initialized = true, .fd = -1};
    t->mutex_ready = pthread_mutex_init(&t->mutex, NULL) == 0;
    t->requested = directory && directory[0];
    if (!t->requested || !t->mutex_ready) return;
    int dir = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir >= 0) {
        t->fd = openat(dir, "reuse.v1.jsonl",
                       O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC, 0600);
        close(dir);
    }
    if (t->fd < 0)
        server_log(DS4_LOG_WARNING, "ds4-server: SCR reuse telemetry store unavailable errno=%d", errno);
}

/* The owner closes only after its workers have stopped. */
static void server_reuse_close(server *s) {
    server_reuse_telemetry *t = &s->reuse;
    if (!t->initialized) return;
    if (t->fd >= 0) close(t->fd);
    if (t->mutex_ready) pthread_mutex_destroy(&t->mutex);
    *t = (server_reuse_telemetry){.fd = -1};
}

static void server_reuse_admit(server *s, server_slot *slot, job *j) {
    if (!s->reuse.initialized) return;
    pthread_mutex_lock(&s->mu);
    if (s->reuse.sequence != UINT64_MAX) j->reuse.seq = ++s->reuse.sequence;
    pthread_mutex_unlock(&s->mu);
    if (!j->reuse.seq) return;
    responses_random_id(j->reuse.request, sizeof(j->reuse.request), "chatcmpl-");
    j->reuse.started = now_sec();
    j->reuse.admission = server_scr_snapshot(s, slot);
}

static const char *server_reuse_plan_status(const ds4_scr_logical_receipt *l) {
    if (!l->planner_status_known) return "none";
    switch (l->planner_status) {
    case CACHE_REUSE_COMPLETE: return "complete";
    case CACHE_REUSE_NO_FEASIBLE: return "none";
    case CACHE_REUSE_DEADLINE_PARTIAL: return "deadline";
    case CACHE_REUSE_VARIANT_LIMIT: return "variant_limit";
    case CACHE_REUSE_STATE_LIMIT: return "state_limit";
    case CACHE_REUSE_ERR_CANCELLED: return "cancelled";
    default: return "error";
    }
}

/* Rows the prefill actually forwarded through the model, by executed route. */
static uint64_t server_reuse_fresh_rows(const ds4_scr_dense_policy_receipt *e) {
    const ds4_scr_multi_receipt *m = &e->multi;
    const ds4_scr_refresh_receipt *f = &e->refresh;
    switch (e->route) {
    case DS4_SCR_DENSE_CONTINUE: return e->continuation_completed;
    case DS4_SCR_DENSE_REFRESH:
        return (uint64_t)f->prefix_completed + f->source_completed + f->remainder_completed;
    case DS4_SCR_DENSE_REUSE:
    case DS4_SCR_DENSE_FRESH: {
        uint64_t rows = (uint64_t)m->prefix_completed + m->suffix_completed + m->fresh_completed;
        for (uint32_t i = 0; i < DS4_SCR_MULTI_MAX_RANGES; ++i) rows += m->gap_completed[i];
        return rows;
    }
    default: return 0;
    }
}

/* Prefill compute seconds by route; staging, installation and the head are
 * reported as their own stages. */
static double server_reuse_prefill_seconds(const ds4_scr_dense_policy_receipt *e) {
    const ds4_scr_multi_receipt *m = &e->multi;
    const ds4_scr_refresh_receipt *f = &e->refresh;
    switch (e->route) {
    case DS4_SCR_DENSE_CONTINUE: return e->total_seconds;
    case DS4_SCR_DENSE_REFRESH: return f->prefix_seconds + f->source_seconds + f->remainder_seconds;
    case DS4_SCR_DENSE_REUSE:
    case DS4_SCR_DENSE_FRESH:
        return m->prefix_seconds + m->gap_seconds + m->suffix_seconds + m->fresh_seconds;
    default: return -1.0;
    }
}

/* Raw archive capture (readback + write) performed during this prefill, from
 * the active owner's cumulative counters; negative when unknown. */
static double server_reuse_capture_seconds(const ds4_scr_archive_snapshot *before,
                                           const ds4_scr_archive_snapshot *after) {
    const ds4_scr_owner_observation *a = &after->active_io, *b = &before->active_io;
    if (!after->observation_known || !a->known || !a->raw_present || a->raw_timing_unknown) return -1.0;
    double total = a->raw_readback_seconds + a->raw_write_seconds;
    if (before->observation_known && b->known && b->raw_present && !b->raw_timing_unknown &&
        b->owner_id == a->owner_id) {
        double prior = b->raw_readback_seconds + b->raw_write_seconds;
        if (prior > total) return -1.0;
        total -= prior;
    }
    return total;
}

static const char *server_reuse_archive_action(const ds4_scr_logical_receipt *l,
                                               const ds4_scr_archive_snapshot *before,
                                               const ds4_scr_archive_snapshot *after) {
    const ds4_scr_dense_policy_receipt *e = &l->execution;
    bool was = before->presence_known && before->present && !before->lost;
    bool is = after->presence_known && after->present && !after->lost;
    if (!after->presence_known) return "unknown";
    if (was && !is) return "lost";
    if (!is) return "none";
    if (e->refresh.sealed) return e->refresh.old_retired ? "refreshed" : "sealed";
    if (e->multi.rebase_committed) return "rebased";
    if (was && after->generation == before->generation && after->end > before->end) return "appended";
    return "retained";
}

static void server_reuse_ms(buf *b, bool *first, const char *key, double seconds) {
    if (!(seconds >= 0.0) || !isfinite(seconds)) return;
    buf_printf(b, "%s\"%s\":%.3f", *first ? "" : ",", key, seconds * 1000.0);
    *first = false;
}

/* Builds the record from the frozen prefill receipt and the archive snapshot at
 * the emission boundary. Called outside inference exclusion. */
static void server_reuse_json(buf *b, const server *s, const server_slot *slot,
                              const job *j, const ds4_scr_archive_snapshot *now_view) {
    const job_reuse_telemetry *t = &j->reuse;
    const ds4_scr_logical_receipt *l = &t->logical;
    const ds4_scr_dense_policy_receipt *e = &l->execution;
    const bool planned = t->prefill_known && l->physical_known;
    const bool exact = t->prefill_known && t->exact;
    const double now = now_sec();
    buf_printf(b, "{\"v\":1,\"schema\":\"ds4.reuse.v1\",\"request\":\"%s\",\"slot\":%d,\"seq\":%llu",
               t->request, slot->id, (unsigned long long)t->seq);

    buf_printf(b, ",\"tokens\":{\"prompt\":%d", j->req.prompt.len);
    if (planned) {
        uint64_t prepared = l->physical_tokens, reused = e->multi.committed_tokens;
        uint64_t fresh = server_reuse_fresh_rows(e);
        uint64_t live = e->complete_n && prepared >= reused + fresh ? prepared - reused - fresh : 0;
        buf_printf(b, ",\"prepared\":%llu,\"reused\":%llu,\"live_prefix\":%llu,\"fresh\":%llu,\"pad\":%u",
                   (unsigned long long)prepared, (unsigned long long)reused,
                   (unsigned long long)live, (unsigned long long)fresh, l->new_pads);
    } else if (exact) {
        const int live = t->exact_live_prefix, prompt = j->req.prompt.len;
        buf_printf(b, ",\"prepared\":%d,\"reused\":0,\"live_prefix\":%d,\"fresh\":%d,\"pad\":0",
                   prompt, live, t->prefill_rc == 0 && prompt >= live ? prompt - live : 0);
    }
    if (t->prefill_known && l->planner_discovery_known)
        buf_printf(b, ",\"discovered_overlap\":%u", l->planner_longest_match);
    buf_printf(b, ",\"generated\":%d}", t->generated);

    bool first = true;
    buf_puts(b, ",\"ms\":{");
    if (t->enqueued > 0.0) server_reuse_ms(b, &first, "queue", t->started - t->enqueued);
    if (t->prefill_known && l->planner_timing_known)
        server_reuse_ms(b, &first, "plan", (double)l->planner_elapsed_ns / 1e9);
    if (planned && (e->route == DS4_SCR_DENSE_REUSE)) {
        server_reuse_ms(b, &first, "stage", e->multi.stage_seconds);
        server_reuse_ms(b, &first, "install", e->multi.install_seconds);
    }
    double capture = planned || exact ?
        server_reuse_capture_seconds(&t->admission, &t->after_prefill) : -1.0;
    double prefill = planned ? server_reuse_prefill_seconds(e) :
        exact ? t->exact_prefill_seconds : -1.0;
    if (prefill >= 0.0 && capture >= 0.0) prefill = prefill > capture ? prefill - capture : 0.0;
    server_reuse_ms(b, &first, "prefill", prefill);
    server_reuse_ms(b, &first, "capture", capture);
    if (planned && (e->route == DS4_SCR_DENSE_REUSE || e->route == DS4_SCR_DENSE_FRESH))
        server_reuse_ms(b, &first, "head", e->multi.head_seconds + e->multi.covered_head_seconds);
    if (t->first_token > 0.0) server_reuse_ms(b, &first, "first_token", t->first_token - t->started);
    if (t->decode_finished > 0.0)
        server_reuse_ms(b, &first, "decode", t->decode_finished - t->decode_started);
    server_reuse_ms(b, &first, "total", now - (t->enqueued > 0.0 ? t->enqueued : t->started));
    buf_putc(b, '}');

    buf_printf(b, ",\"plan\":{\"status\":\"%s\",\"k\":%d",
               exact ? "exact" : t->prefill_known ? server_reuse_plan_status(l) : "none",
               s->scr_max_ranges);
    uint32_t ranges = planned && l->planned_ranges <= DS4_SCR_MULTI_MAX_RANGES ? l->planned_ranges : 0;
    buf_printf(b, ",\"range_count\":%u,\"ranges\":[", ranges);
    for (uint32_t i = 0; i < ranges; ++i) {
        const ds4_scr_range *r = &l->physical_plan[i];
        buf_printf(b, "%s{\"src\":%u,\"dst\":%u,\"n\":%u}", i ? "," : "",
                   r->source_start, r->destination_start, r->count);
    }
    buf_putc(b, ']');
    if (t->prefill_known && e->source_generation)
        buf_printf(b, ",\"source\":{\"kind\":\"archive\",\"generation\":%llu,\"tokens\":%u}",
                   (unsigned long long)e->source_generation, e->source_end - e->source_start);
    else if (planned && e->route == DS4_SCR_DENSE_CONTINUE)
        buf_printf(b, ",\"source\":{\"kind\":\"live\",\"tokens\":%u}", e->retained_tokens);
    else if (exact && t->exact_live_prefix > 0)
        buf_printf(b, ",\"source\":{\"kind\":\"live\",\"tokens\":%d}", t->exact_live_prefix);
    buf_putc(b, '}');

    if (now_view->presence_known) {
        buf_printf(b, ",\"archive\":{\"capacity\":%u,\"occupancy\":%u,\"action\":\"%s\",\"device_mib\":%.1f",
                   now_view->present ? now_view->capacity : 0u,
                   now_view->present ? now_view->sealed_tokens : 0u,
                   server_reuse_archive_action(l, &t->admission, now_view),
                   (double)now_view->accounted_bytes / (1024.0 * 1024.0));
        if (t->pool_known)
            buf_printf(b, ",\"pool_restored\":%s,\"pool_entries\":%u",
                       t->pool.restored ? "true" : "false", t->pool.entries);
        buf_putc(b, '}');
    }

    const ds4_engram_observation *ea = &t->admission.engram_io, *ez = &now_view->engram_io;
    if (ez->enabled && !ez->counters_overflow && ea->enabled == ez->enabled &&
        !ea->counters_overflow && ez->rows_completed >= ea->rows_completed &&
        ez->read_bytes >= ea->read_bytes) {
        buf_printf(b, ",\"io\":{\"engram\":{\"rows\":%llu,\"bytes\":%llu",
                   (unsigned long long)(ez->rows_completed - ea->rows_completed),
                   (unsigned long long)(ez->read_bytes - ea->read_bytes));
        if (!ez->clock_failed && !ez->time_overflow && !ea->clock_failed && !ea->time_overflow &&
            ez->call_wall_nanoseconds >= ea->call_wall_nanoseconds)
            buf_printf(b, ",\"ms\":%.3f",
                       (double)(ez->call_wall_nanoseconds - ea->call_wall_nanoseconds) / 1e6);
        buf_puts(b, "}}");
    }

    if (t->decode_finished > 0.0) {
        double seconds = t->decode_finished - t->decode_started;
        buf_printf(b, ",\"decode\":{\"steps\":%d", t->generated);
        if (seconds > 0.0) buf_printf(b, ",\"tps\":%.2f", (double)t->generated / seconds);
        buf_putc(b, '}');
    }
    if (t->error_phase) buf_printf(b, ",\"error\":{\"phase\":\"%s\",\"code\":%d}", t->error_phase, t->error_code);
    else buf_puts(b, ",\"error\":null");
    buf_putc(b, '}');
}

/* One finalized out-of-band record per admitted request. Failure is telemetry
 * loss only. */
static void server_reuse_store(server *s, server_slot *slot, job *j) {
    server_reuse_telemetry *t = &s->reuse;
    if (!t->initialized || !j->reuse.seq || j->reuse.emitted) return;
    j->reuse.emitted = true;
    if (!t->requested) return;
    ds4_scr_archive_snapshot view = server_scr_snapshot(s, slot);
    buf record = {0};
    server_reuse_json(&record, s, slot, j, &view);
    buf_putc(&record, '\n');
    bool ok = false;
    if (t->mutex_ready) {
        pthread_mutex_lock(&t->mutex);
        ok = t->fd >= 0 && record.ptr && write(t->fd, record.ptr, record.len) == (ssize_t)record.len;
        if (!ok && t->dropped != UINT64_MAX) ++t->dropped;
        pthread_mutex_unlock(&t->mutex);
    }
    if (!ok) server_log(DS4_LOG_WARNING, "ds4-server: SCR reuse telemetry record lost");
    buf_free(&record);
}

/* Streams the current record as one SSE chunk when the chat client opted in.
 * Returns false only when the client write fails. */
static bool server_reuse_stream(server *s, server_slot *slot, job *j) {
    if (!s->reuse.initialized || !j->reuse.seq || !j->req.stream ||
        !j->req.stream_include_ds4_scr || j->req.kind != REQ_CHAT) return true;
    ds4_scr_archive_snapshot view = server_scr_snapshot(s, slot);
    buf frame = {0};
    buf_printf(&frame, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"ds4_reuse\":",
               j->reuse.request);
    server_reuse_json(&frame, s, slot, j, &view);
    buf_puts(&frame, "}\n\n");
    bool ok = frame.ptr && send_all(j->fd, frame.ptr, frame.len);
    buf_free(&frame);
    return ok;
}
#endif

/* ---- Transport --------------------------------------------------------- */
#ifdef DS4_SERVER_SCR_TRANSPORT
/* Included after the server's job and prefill-progress types. Transport liveness
 * must not publish transactional SCR token progress or cache checkpoints. */
static bool server_prefill_keepalive(server_prefill_progress *p, double now) {
    if (!p->stream || p->fd < 0 || p->stream_failed) return true;
    if (!p->headers_sent) {
        p->headers_sent = true;
        if (!sse_headers(p->fd, p->enable_cors)) goto failed;
        p->last_keepalive = now;
    } else if (now - p->last_keepalive >= 5.0) {
        static const char comment[] = ": prefill\n\n";
        if (!send_all(p->fd, comment, sizeof(comment) - 1)) goto failed;
        p->last_keepalive = now;
    }
    return true;
failed:
    p->stream_failed = true;
    job_mark_cancelled(p->request_job);
    return false;
}

typedef struct {
    server_prefill_progress *progress;
    pthread_t owner;
} server_scr_transport_context;

static bool server_scr_cancel_with_keepalive(void *ud) {
    server_scr_transport_context *context = ud;
    server_prefill_progress *progress = context->progress;
    if (job_cancelled(progress->request_job)) return true;
    if (pthread_equal(pthread_self(), context->owner))
        server_prefill_keepalive(progress, now_sec());
    return job_cancelled(progress->request_job);
}

static int server_scr_prefill_with_keepalive(server *s, server_slot *slot, job *j,
                                             server_prefill_progress *progress,
                                             char *err, size_t errlen) {
    int (*prefill)(server *, server_slot *, job *, char *, size_t) =
        j->req.reuse_exact ? server_scr_exact_prefill : server_scr_normal_prefill;
    if (!j->req.stream) return prefill(s, slot, j, err, errlen);
    /* Cooperative cancellation remains active when SCR suppresses ordinary
     * progress callbacks. The owning worker services transport at those points. */
    server_scr_transport_context context = { .progress = progress, .owner = pthread_self() };
    ds4_session_set_cancel(slot->session, server_scr_cancel_with_keepalive, &context);
    int rc = prefill(s, slot, j, err, errlen);
    if (!server_scr_quarantined(s))
        ds4_session_set_cancel(slot->session, job_cancelled, j);
    return rc;
}
#endif

/* ---- Configuration ----------------------------------------------------- */
#ifdef DS4_SERVER_SCR_CONFIG
static bool server_scr_loopback(const char *host) {
    struct in_addr addr;
    return !strcmp(host, "localhost") ||
        (inet_pton(AF_INET, host, &addr) == 1 && (ntohl(addr.s_addr) >> 24) == 127);
}

/* --scr selects the production stack; a nonzero --scr-tail-tokens alone keeps
 * enabling it for existing launch scripts. The archive defaults to the whole
 * (eight-aligned) context and K to the native maximum. */
static void server_scr_resolve_config(server_config *c) {
    if (c->scr_tail_tokens) c->scr_enabled = true;
    if (!c->scr_enabled || c->scr_tail_tokens) return;
    uint32_t tail = c->ctx_size > 0 ? (uint32_t)c->ctx_size : 0u;
    if (tail > DS4_SCR_ARCHIVE_MAX_CAPACITY) tail = DS4_SCR_ARCHIVE_MAX_CAPACITY;
    c->scr_tail_tokens = (int)(tail & ~7u);
}
/* Applied once immediately after the owned slot session is created. A failed
 * selector follows the existing startup cleanup path; never retry setup. */
static int server_scr_apply_startup(const server_config *c, ds4_session *session,
                                   char *err, size_t errlen) {
    if (!c->scr_enabled) return 0;
    int rc;
    if ((rc = ds4_session_set_scr_planner_budget(session, (uint32_t)c->scr_planner_budget_ms))) return rc;
    if ((rc = ds4_session_scr_set_padding_policy(session, DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE, err, errlen))) return rc;
    return ds4_session_scr_enable_capture_batch_chunks(session, err, errlen);
}
static bool server_scr_config_valid(const server_config *c, bool incompatible) {
    /* Tuning flags without SCR are a launch mistake, not a silent no-op. */
    if (!c->scr_enabled) return !c->scr_max_ranges_present;
    /* DSpark speculation is the one support model SCR coexists with, and
     * only as the --dspark/--mtp-model pair; legacy MTP stays refused. */
    if (c->engine.glm_mtp || c->engine.dspark != (c->engine.mtp_path != NULL)) return false;
    return c->scr_max_ranges >= 1 && c->scr_max_ranges <= (int)DS4_SCR_MULTI_MAX_RANGES && !incompatible && c->engine.backend == DS4_BACKEND_METAL &&
        !c->engine.cuda_tensor_parallel && c->engine.tp.role == DS4_TP_NONE &&
        c->engine.distributed.role == DS4_DISTRIBUTED_NONE &&
        c->scr_tail_tokens >= 128 && (uint32_t)c->scr_tail_tokens <= DS4_SCR_ARCHIVE_MAX_CAPACITY &&
        c->scr_tail_tokens % 8 == 0 && c->scr_tail_tokens <= c->ctx_size &&
        server_scr_loopback(c->host);
}
#endif
