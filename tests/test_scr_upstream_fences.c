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
    assert(!s.scr_capture_batch_chunks && !s.scr_padding_selected);
    e.dspark = false; e.mtp_ready = true;
    assert(!ds41_scr_empty_startup(&s));
    assert(ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(!s.scr_capture_batch_chunks && !s.scr_padding_selected);
    e.mtp_ready = false; e.glm_mtp = true;
    assert(!ds41_scr_empty_startup(&s));
    e.glm_mtp = false;
    assert(!ds4_session_scr_enable_capture_batch_chunks(&s, error, sizeof(error)));
    assert(s.scr_capture_batch_chunks);
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

int main(void) {
    assert(ds4_gpu_init());
    startup_modes(); capture_and_verifier_owners();
    ds4_gpu_cleanup();
    puts("upstream SCR fences: DSpark/MTP startup, archive admission, verifier/batch no-copy and mapped restore refusal passed");
    return 0;
}
