#include <assert.h>
#include <stdio.h>
#define DS4_SCR_ENGINE_COORDINATES
#include "../ds4_scr_engine.inc"

static void artificial_positions_are_not_token_values(void) {
    int physical[] = {10, 201, 20, 201, 30, 201, 201};
    uint32_t pads[] = {1, 5};
    ds41_scr_coordinates live = {0}, archive = {0};
    assert(ds41_coordinates_build(&live, physical, 6, 1000, pads, 2, 10));
    assert(ds41_coordinates_clone(&archive, &live));
    assert(archive.pads != live.pads && archive.pads != pads);
    int logical[7] = {0};
    uint32_t written;
    assert(ds41_coordinates_project(&live, physical, 6, 1000, logical, 7, &written));
    int expected[] = {10, 20, 201, 30};
    assert(written == 4 && !memcmp(logical, expected, sizeof(expected)));
    assert(ds41_coordinates_append(&live, 1, 7, 11));
    assert(live.count == 2);
    assert(ds41_coordinates_project(&live, physical, 7, 1000, logical, 7, &written));
    assert(written == 5 && logical[4] == 201);
    assert(archive.physical_tokens == 6 && archive.identity == 10);
    assert(ds41_coordinates_truncate(&live, 4, 12));
    assert(live.count == 1 && live.capacity == 2);
    assert(ds41_coordinates_project(&live, physical, 4, 1000, logical, 7, &written));
    assert(written == 3 && logical[2] == 201);
    ds41_coordinates_free(&live);
    assert(ds41_coordinates_project(&archive, physical, 6, 1000, logical, 7, &written));
    assert(written == 4 && !memcmp(logical, expected, sizeof(expected)));
    ds41_coordinates_free(&archive);
}

static void boundary_projection_includes_only_preceding_pads(void) {
    int physical[] = {10, 201, 201, 20};
    uint32_t pads[] = {1, 2};
    ds41_scr_coordinates m = {0};
    assert(ds41_coordinates_build(&m, physical, 4, 1000, pads, 2, 3));
    uint32_t expected[] = {0, 1, 1, 1, 2};
    for (uint32_t i = 0; i <= 4; ++i) {
        uint32_t logical = UINT32_MAX;
        assert(ds41_coordinates_logical_boundary(&m, i, &logical));
        assert(logical == expected[i]);
    }
    uint32_t unchanged = 91;
    assert(!ds41_coordinates_logical_boundary(&m, 5, &unchanged));
    assert(unchanged == 91);
    assert(ds41_coordinates_truncate(&m, 0, 4));
    assert(m.count == 0 && m.physical_tokens == 0 && m.capacity == 2);
    ds41_coordinates_free(&m);
}

static void refusal_preserves_existing_owners_and_output(void) {
    int physical[] = {10, 201, 20, 30};
    uint32_t pad = 1;
    ds41_scr_coordinates m = {0};
    assert(ds41_coordinates_build(&m, physical, 4, 1000, &pad, 1, 9));
    ds41_scr_coordinates before = m;
    assert(!ds41_coordinates_build(&m, physical, 4, 1000, &pad, 1, 10));
    assert(!ds41_coordinates_clone(&m, &m));
    assert(!ds41_coordinates_append(&m, 2, 5, 10));
    assert(!ds41_coordinates_append(&m, UINT32_MAX, UINT32_MAX, 10));
    assert(!ds41_coordinates_append(&m, 1, 5, 9));
    assert(!ds41_coordinates_truncate(&m, 5, 10));
    assert(!ds41_coordinates_truncate(&m, 2, 9));
    assert(m.pads == before.pads && m.count == before.count &&
           m.capacity == before.capacity && m.physical_tokens == before.physical_tokens &&
           m.identity == before.identity);
    int logical[] = {91, 92, 93};
    uint32_t written = 99;
    assert(!ds41_coordinates_project(&m, physical, 4, 1000, logical, 2, &written));
    assert(written == 0 && logical[0] == 91 && logical[2] == 93);
    physical[2] = 1000; /* outside the vocabulary */
    assert(!ds41_coordinates_project(&m, physical, 4, 1000, logical, 3, &written));
    assert(written == 0 && logical[0] == 91 && logical[2] == 93);
    ds41_coordinates_free(&m);
}

static void malformed_provenance_is_not_accepted(void) {
    int physical[] = {10, 201, 201, 20};
    uint32_t cases[][2] = {{1, 1}, {2, 1}, {1, 4}, {4, 5}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ds41_scr_coordinates m = {0};
        assert(!ds41_coordinates_build(&m, physical, 4, 1000, cases[i], 2, 3));
        assert(!m.pads && !m.identity && !m.count && !m.capacity && !m.physical_tokens);
    }
    ds41_scr_coordinates m = {0};
    assert(!ds41_coordinates_build(&m, physical, 4, 1000, NULL, 1, 3));
    assert(!ds41_coordinates_build(&m, physical, 4, 1000, NULL, 0, 0));
    assert(ds41_coordinates_build(&m, physical, 4, 1000, NULL, 0, UINT64_MAX));
    assert(!ds41_coordinates_append(&m, 1, 5, 0));
    int logical[4];
    uint32_t written;
    assert(ds41_coordinates_project(&m, physical, 4, 1000, logical, 4, &written));
    assert(written == 4 && !memcmp(logical, physical, sizeof(physical)));
    size_t bytes = 0;
    assert(ds41_coordinates_heap_bytes(2, &bytes) && bytes == 2 * sizeof(uint32_t));
    ds41_coordinates_free(&m);
    ds41_coordinates_free(&m);
}

static void archive_interval_owns_relative_provenance(void) {
    int physical[] = {10, 201, 20, 201, 30, 201, 40, 50};
    uint32_t pads[] = {1, 5};
    ds41_scr_coordinates live = {0}, interval = {0};
    assert(ds41_coordinates_build(&live, physical, 8, 1000, pads, 2, 10));
    uint32_t origin = 99;
    assert(!ds41_coordinates_slice(&live, physical, 8, 1000, 7, 2, &interval, &origin));
    assert(origin == 99 && !interval.identity && !interval.pads);
    assert(ds41_coordinates_slice(&live, physical, 8, 1000, 3, 4, &interval, &origin));
    assert(origin == 2 && interval.physical_tokens == 4 && interval.count == 1);
    assert(interval.pads != live.pads && interval.pads[0] == 2);
    assert(interval.identity == live.identity);
    ds41_coordinates_free(&live);
    int logical[4];
    uint32_t written;
    assert(ds41_coordinates_project(&interval, physical + 3, 4, 1000, logical, 4, &written));
    assert(written == 3 && logical[0] == 201 && logical[1] == 30 && logical[2] == 40);
    ds41_coordinates_free(&interval);
    assert(ds41_coordinates_build(&live, physical, 8, 1000, pads, 2, 10));
    assert(ds41_coordinates_slice(&live, physical, 8, 1000, 1, 4, &interval, &origin));
    assert(origin == 1 && interval.count == 1 && interval.pads[0] == 0);
    ds41_coordinates_free(&interval);
    ds41_coordinates_free(&live);
}

static void charged_transfer_is_atomic_and_consumes_pending(void) {
    int physical[] = {10, 201, 20, 201};
    uint32_t pads[] = {1, 3};
    ds41_scr_coordinates live = {0}, pending = {0};
    assert(ds41_coordinates_build(&live, physical, 4, 1000, pads, 1, 1));
    assert(ds41_coordinates_build(&pending, physical, 4, 1000, pads, 2, 2));
    uint32_t *old = live.pads, *next = pending.pads;
    uint64_t graph = 1000, engine = 0, archive = 500;
    assert(!ds41_coordinates_replace_charged(&live, &pending, physical, 4, 1000,
                                            &graph, &engine, &archive));
    assert(graph == 1000 && engine == 0 && archive == 500);
    assert(live.pads == old && pending.pads == next);
    engine = UINT64_MAX;
    assert(!ds41_coordinates_replace_charged(&live, &pending, physical, 4, 1000,
                                            &graph, &engine, &archive));
    assert(graph == 1000 && engine == UINT64_MAX && archive == 500);
    assert(live.pads == old && pending.pads == next);
    engine = 2000;
    assert(!ds41_coordinates_replace_charged(&live, &pending, physical, 4, 1000,
                                            &graph, &graph, &archive));
    assert(ds41_coordinates_replace_charged(&live, &pending, physical, 4, 1000,
                                           &graph, &engine, &archive));
    assert(graph == 1000 + sizeof(uint32_t) && engine == 2000 + sizeof(uint32_t));
    assert(archive == 500 + sizeof(uint32_t));
    assert(live.pads == next && live.identity == 2 && live.count == 2);
    assert(!pending.pads && !pending.identity && !pending.count);
    assert(ds41_coordinates_build(&pending, physical, 4, 1000, NULL, 0, 3));
    assert(ds41_coordinates_replace_charged(&live, &pending, physical, 4, 1000,
                                           &graph, &engine, &archive));
    assert(graph == 1000 - sizeof(uint32_t) && engine == 2000 - sizeof(uint32_t));
    assert(archive == 500 - sizeof(uint32_t));
    assert(!live.pads && live.count == 0 && live.identity == 3);
    ds41_coordinates_free(&live);
    ds41_coordinates_free(&pending);
}

int main(void) {
    artificial_positions_are_not_token_values();
    boundary_projection_includes_only_preceding_pads();
    refusal_preserves_existing_owners_and_output();
    malformed_provenance_is_not_accepted();
    archive_interval_owns_relative_provenance();
    charged_transfer_is_atomic_and_consumes_pending();
    puts("PASS: coordinate ownership, natural/generated pad IDs, rewind transcript, refusal and provenance");
    return 0;
}
