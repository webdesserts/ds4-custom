/* V4.1 engine-open memory admission charges the support model (DSpark/MTP)
 * before it is mapped: ds4_engine_open admits the graphs before it opens
 * --mtp-model, so the drafter's file bytes stand in until mtp_model.size is
 * known. Strict host: no model file is opened. */
#include <stdbool.h>
#include <stdint.h>
#include "../ds4.c"
#include <assert.h>

/* Size rule of ds4_metal.m (no Metal object is linked here). */
uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    const uint64_t flags = (((uint64_t)rows + ((uint64_t)source_rows + 63u) / 64u) * 4u + 255u) &
        ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u +
        (((uint64_t)source_rows + 63u) / 64u) * 64u * 128u * 2u;
}

int main(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    /* The live shape: 151.76 GiB target, an 8.1 GiB drafter, 15.1 GiB graph. */
    const uint64_t target = (uint64_t)(151.76 * (double)gib), drafter = (uint64_t)(8.1 * (double)gib);
    const uint64_t graph = (uint64_t)(15.11 * (double)gib);
    static ds4_engine e;
    e.backend = DS4_BACKEND_METAL;
    e.model.size = target;

    uint64_t weights = 0;
    assert(ds41_memory_weights(&e, &weights) && weights == target);
    e.mtp_reserve_bytes = drafter;
    assert(ds41_memory_weights(&e, &weights) && weights == target + drafter);
    /* Once mapped, the mapped size counts, not the reserve twice. */
    e.mtp_model.size = drafter + 4096;
    assert(ds41_memory_weights(&e, &weights) && weights == target + drafter + 4096);
    e.mtp_model.size = 0;

    /* A budget that fits target + graph + 2 GiB headroom but not the drafter
     * as well: admission refuses only when the drafter is charged. */
    const uint64_t fixed = target + graph + 2u * gib;
    const uint64_t recommended = fixed + drafter / 2u, host = recommended / 7u * 8u + 8u * gib;
    e.mtp_reserve_bytes = 0;
    assert(ds41_memory_admit_for_host(&e, graph, false, host, recommended));
    e.mtp_reserve_bytes = drafter;
    assert(!ds41_memory_admit_for_host(&e, graph, false, host, recommended));
    assert(ds41_memory_admit_for_host(&e, graph, false, host + 2u * drafter, recommended + 2u * drafter));
    printf("V4.1 admission charges the %.2f GiB drafter before it is mapped: PASS\n",
           (double)drafter / (double)gib);
    return 0;
}
