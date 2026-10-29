/* Exercise the actual ported entry/copy fences with no model and a strict host
 * storage adapter. No Metal object is linked into this executable. */
#include "../ds4.c"
#include <assert.h>
extern uint64_t scr_host_command_begins(void);

static void startup_modes(void) {
    ds4_engine e = {.backend = DS4_BACKEND_METAL};
    ds4_session s = {.engine = &e};
    char error[256];
    assert(ds41_scr_empty_startup(&s));
    e.dspark = true;
    assert(!ds41_scr_empty_startup(&s));
    assert(ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(!s.scr_padding_selected);
    e.dspark = false; e.mtp_ready = true;
    assert(!ds41_scr_empty_startup(&s));
    assert(ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(!s.scr_padding_selected);
    e.mtp_ready = false; e.glm_mtp = true;
    assert(!ds41_scr_empty_startup(&s));
    e.glm_mtp = false;
    /* Compatibility check only: repeatable, selects nothing. */
    assert(!ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(!ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
}

static void capture_and_verifier_owners(void) {
    ds4_engine e = {.backend = DS4_BACKEND_METAL, .dspark = true};
    ds4_session s = {.engine = &e, .ctx_size = 1024, .ds41_graph_ready = true};
    ds41_gpu_graph *g = &s.ds41_graph;
    g->valid = true; g->ctx = 1024; g->tp_world = 1;
    assert(ds4_session_scr_archive_prepare(&s, 0, 128));
    assert(!g->archive && !g->allocation_bytes && !e.ds41_session_bytes);
    e.dspark = false;
    const uint64_t before = scr_host_command_begins();
    ds41_dspark draft = {0};
    ds41_scr_archive archive = {0};
    int tokens[2] = {1, 2};
    ds41_gpu_graph *graphs[2] = {g, g};
    for (unsigned owner = 0; owner < 3; ++owner) {
        g->archive = owner == 0 ? &archive : NULL;
        g->archive_pending = owner == 1 ? &archive : NULL;
        g->archive_capture_pending = owner == 2;
        ds41_gpu_graph snapshot = *g;
        assert(!ds41_dspark_verify(&draft, g, NULL, NULL, tokens, 2));
        assert(!ds41_graph_step_batch(graphs, tokens, 2, 0, NULL, NULL));
        assert(!memcmp(g, &snapshot, sizeof(*g)));
        assert(scr_host_command_begins() == before);
        for (unsigned row = 0; row <= DS41_VERIFY_ROWS; ++row)
            assert(!draft.verified[row].archive && !draft.verified[row].archive_pending);
    }
    g->archive = g->archive_pending = NULL; g->archive_capture_pending = false;
    s.scr_coordinates.identity = 1;
    ds4_session prior = s;
    assert(ds4_session_restore_speculative_prefix(&s, 0) == 1);
    assert(!ds4_session_rewind_speculative(&s, 0));
    assert(!memcmp(&s, &prior, sizeof(s)));
    assert(scr_host_command_begins() == before);
}

static ds4_gpu_tensor *owned[256];
static unsigned owned_count;
static ds4_gpu_tensor *tensor(uint64_t bytes) {
    assert(owned_count < sizeof(owned) / sizeof(owned[0]));
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    assert(t);
    return owned[owned_count++] = t;
}

/* Give the verifier block rows and the target frontier just the state a
 * commit or speculative restore copies; no forward pass runs here. */
static void verifier_state(ds41_gpu_graph *g, ds41_dspark *d) {
    const uint64_t kv = DS4_N_HEAD_DIM * 4u, feature = DS4_N_EMBD * 4u;
    ds41_gpu_graph *frontiers[DS41_VERIFY_ROWS + 2u] = {g};
    for (unsigned row = 0; row <= DS41_VERIFY_ROWS; ++row) frontiers[row + 1u] = &d->verified[row];
    for (unsigned i = 0; i < DS41_VERIFY_ROWS + 2u; ++i) {
        ds41_gpu_graph *f = frontiers[i];
        for (unsigned o = 0; o < 4; ++o) {
            ds4_gpu_tensor **k = i ? &f->verify_previous_kv[o] : &f->previous_kv[o];
            ds4_gpu_tensor **v = i ? &f->verify_previous_score[o] : &f->previous_score[o];
            *k = tensor(kv); *v = tensor(kv);
        }
        f->residual = tensor(DS4_N_HC * feature);
        f->pre = tensor(DS4_N_HC * 4u);
        f->logits = tensor((uint64_t)DS4_N_VOCAB * 4u);
    }
    for (unsigned l = 0; l < 40; ++l) g->window[l] = tensor(DS41_DRAFT_WINDOW * kv);
    for (unsigned stage = 0; stage < 3; ++stage) d->target[stage] = tensor(DS41_DRAFT_WINDOW * feature);
    d->old_raw = tensor(40u * DS41_VERIFY_ROWS * kv);
    d->old_features = tensor(3u * DS41_VERIFY_ROWS * feature);
    d->host_verified = calloc((DS41_VERIFY_ROWS + 1u) * (size_t)DS4_N_VOCAB, sizeof(float));
    assert(d->host_verified);
}

/* V4.1 DSpark coexists with SCR: startup and archive admission succeed, a
 * verified block disarms the sealed archive's writer without aliasing it into
 * verifier rows, commit appends the accepted IDs to the physical map and a
 * speculative rewind truncates it exactly once. */
static void dspark_flash41_coexistence(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_engine e = {.backend = DS4_BACKEND_METAL, .dspark = true,
        .support_kind = DS4_SUPPORT_DSPARK};
    e.dspark_weights.v41 = true;
    ds4_session s = {.engine = &e, .ctx_size = 1024, .ds41_graph_ready = true};
    ds41_gpu_graph *g = &s.ds41_graph;
    g->valid = true; g->ctx = 1024; g->tp_world = 1;
    char error[256];
    assert(ds41_scr_empty_startup(&s));
    assert(!ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(!ds4_session_scr_archive_prepare(&s, 0, 128) && g->archive);
    e.support_kind = DS4_SUPPORT_MTP_LEGACY;
    assert(!ds41_scr_engine_compatible(&s));
    e.support_kind = DS4_SUPPORT_DSPARK; e.dspark_weights.v41 = false;
    assert(!ds41_scr_engine_compatible(&s));
    e.dspark_weights.v41 = true;
    assert(ds41_scr_engine_compatible(&s));

    /* A sealed archive left by a refresh, its writer armed for appends. */
    ds41_scr_archive *a = g->archive;
    a->sealed = true; a->writer_state = DS41_ARCHIVE_ARMED; a->range.valid = true;
    ds41_dspark *d = calloc(1, sizeof(*d));
    assert(d);
    g->dspark = d;
    s.logits = calloc(DS4_N_VOCAB, sizeof(float));
    s.checkpoint.v = calloc(64, sizeof(int)); s.checkpoint.cap = 64;
    assert(s.logits && s.checkpoint.v);
    for (int i = 0; i < 10; ++i) s.checkpoint.v[i] = 100 + i;
    s.checkpoint.len = 10; s.checkpoint_valid = true; g->pos = 10;
    const uint32_t pad = 3;
    assert(ds41_coordinates_build(&s.scr_coordinates, s.checkpoint.v, 10, DS4_N_VOCAB, &pad, 1, 1));
    s.scr_coordinate_serial = 1; s.scr_coordinate_charge = sizeof(uint32_t);
    g->allocation_bytes += sizeof(uint32_t); e.ds41_session_bytes += sizeof(uint32_t);
    assert(ds41_coordinates_restricted(&s) && ds41_coordinates_eval_supported(&s) &&
        ds41_speculation_coordinates_allowed(&s));

    /* Verification runs no model here: with no snapshot storage it fails
     * after admission, where the writer is disarmed and row 0 is copied. */
    int tokens[3] = {200, 201, 202};
    d->verified[0].archive = a;
    assert(!ds41_dspark_verify(d, g, NULL, NULL, tokens, 3));
    assert(a->sealed && a->writer_state == DS41_ARCHIVE_INCOMPATIBLE && !a->range.valid);
    assert(!d->verified[0].archive && !d->verified[0].archive_pending &&
        !d->verified[0].archive_capture_pending);
    assert(g->archive == a && !ds4_gpu_commands_active());
    g->valid = true;
    verifier_state(g, d);

    /* Commit the full verified block: three ordinary IDs join the map. */
    d->verify_start = 10; d->verify_count = 3; d->rewind_end = 0;
    for (unsigned row = 0; row < 3; ++row) d->verified[row].pos = 11 + row;
    d->verified[DS41_VERIFY_ROWS].pos = 10;
    assert(ds41_session_commit(&s, tokens, 3, 3));
    assert(s.checkpoint.len == 13 && g->pos == 13 && d->rewind_end == 13);
    assert(s.checkpoint.v[10] == 200 && s.checkpoint.v[12] == 202);
    assert(s.scr_coordinates.physical_tokens == 13 && s.scr_coordinates.count == 1 &&
        s.scr_coordinates.pads[0] == 3 && s.scr_coordinates.identity == 2 &&
        s.scr_coordinate_serial == 2);

    /* A rejected suffix rewinds the frontier and truncates the map with it. */
    assert(ds4_session_rewind_speculative(&s, 11));
    assert(s.checkpoint.len == 11 && g->pos == 11 && s.checkpoint_valid && g->valid);
    assert(s.scr_coordinates.physical_tokens == 11 && s.scr_coordinates.count == 1 &&
        s.scr_coordinates.identity == 3 && s.scr_coordinate_serial == 3);
    /* Through the public rewind, the speculative restore truncates once. */
    assert(!ds4_session_rewind_physical(&s, 10, error, sizeof(error)));
    assert(s.checkpoint.len == 10 && g->pos == 10 && s.checkpoint_valid);
    assert(s.scr_coordinates.physical_tokens == 10 && s.scr_coordinates.identity == 4 &&
        s.scr_coordinate_serial == 4);

    /* A verified block refuses an archive that is still capturing. */
    a->sealed = false;
    assert(!ds41_dspark_verify(d, g, NULL, NULL, tokens, 3));
    a->sealed = true;

    ds41_session_coordinates_clear(&s);
    assert(!ds4_session_scr_archive_release(&s) && !g->archive);
    for (unsigned i = 0; i < owned_count; ++i) ds4_gpu_tensor_free(owned[i]);
    owned_count = 0;
    free(d->host_verified); free(d); free(s.logits); free(s.checkpoint.v);
}

int main(void) {
    assert(ds4_gpu_init());
    startup_modes(); capture_and_verifier_owners(); dspark_flash41_coexistence();
    ds4_gpu_cleanup();
    puts("upstream SCR fences: V4.0 DSpark/MTP startup refusal, archive admission, verifier/batch no-copy, mapped restore refusal; V4.1 DSpark coexistence passed");
    return 0;
}
