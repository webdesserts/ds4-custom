/* Timing and exactness for the V4.1 routed-MoE verify path (IQ2_XXS gate/up,
 * Q2_K down, 6 of 384 experts) at 1-8 rows, on synthetic weights:
 *
 *   --check            the compile-time geometry kernels are
 *                      bitwise equal to the generic kernels for mid and the
 *                      summed output, batched (1-8 rows) and single. The
 *                      fused kernels' gate/up stores are diagnostic only: all
 *                      six experts of a row write the same span, so the last
 *                      writer varies run to run and they are not compared.
 *
 *   --bench [sel.bin]  real router selections (see load_selections) beside
 *                      two controls: no-share (6 * rows distinct experts) and
 *                      all-share (every row picks the same 6 experts).
 *
 * all-share reads 1/rows of the no-share weight bytes, so the gap between the
 * two bounds what any expert-grouping of weight reads can save.
 */
#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

enum {
    N_EXPERT = 384, N_USED = 6, IN = 5120, MID = 2304, MAX_ROWS = 8,
    IQ2_BLOCK = 66, Q2K_BLOCK = 84,
    GATE_ROW = IN / 256 * IQ2_BLOCK, DOWN_ROW = MID / 256 * Q2K_BLOCK,
    TYPE_Q2_K = 10, TYPE_IQ2_XXS = 16,
};
static const uint64_t GATE_EXPERT = (uint64_t)MID * GATE_ROW;
static const uint64_t DOWN_EXPERT = (uint64_t)IN * DOWN_ROW;

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_u64(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}

typedef struct {
    uint8_t *map;
    uint64_t size, gate_off, up_off, down_off;
} moe_model;

/* Random quantized blocks with finite, moderate scales. */
static int model_init(moe_model *m) {
    m->gate_off = 0;
    m->up_off = (uint64_t)N_EXPERT * GATE_EXPERT;
    m->down_off = 2 * m->up_off;
    m->size = m->down_off + (uint64_t)N_EXPERT * DOWN_EXPERT;
    const size_t page = (size_t)getpagesize();
    m->size = (m->size + page - 1) / page * page;
    CHECK(posix_memalign((void **)&m->map, page, m->size) == 0);
    uint64_t *w = (uint64_t *)m->map;
    for (uint64_t i = 0; i < m->size / 8; i++) w[i] = next_u64();
    for (uint64_t b = 0; b < m->down_off / IQ2_BLOCK; b++) {
        const uint16_t d = (uint16_t)(0x1c00u | (next_u64() & 0x3ffu));
        memcpy(m->map + b * IQ2_BLOCK, &d, 2);
    }
    uint8_t *down = m->map + m->down_off;
    for (uint64_t b = 0; b < (uint64_t)N_EXPERT * DOWN_EXPERT / Q2K_BLOCK; b++) {
        const uint16_t d = (uint16_t)(0x1c00u | (next_u64() & 0x3ffu));
        const uint16_t dmin = (uint16_t)(0x1800u | (next_u64() & 0x3ffu));
        memcpy(down + b * Q2K_BLOCK + 80, &d, 2);
        memcpy(down + b * Q2K_BLOCK + 82, &dmin, 2);
    }
    CHECK(ds4_gpu_set_model_map(m->map, m->size));
    return 1;
}

typedef struct {
    ds4_gpu_tensor *x, *w, *gate, *up, *mid, *experts, *out;
} moe_io;

static int io_init(moe_io *io) {
    float x[MAX_ROWS * IN], w[MAX_ROWS * N_USED];
    for (int i = 0; i < MAX_ROWS * IN; i++)
        x[i] = (float)((int64_t)(next_u64() % 65537u) - 32768) / 16384.0f;
    for (int i = 0; i < MAX_ROWS * N_USED; i++)
        w[i] = 0.05f + (float)(next_u64() % 1000u) / 4000.0f;
    io->x = ds4_gpu_tensor_alloc(sizeof(x));
    io->w = ds4_gpu_tensor_alloc(sizeof(w));
    io->gate = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * N_USED * MID * 4);
    io->up = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * N_USED * MID * 4);
    io->mid = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * N_USED * MID * 4);
    io->experts = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * N_USED * IN * 4);
    io->out = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * IN * 4 + 4);
    CHECK(io->x && io->w && io->gate && io->up && io->mid && io->experts && io->out);
    CHECK(ds4_gpu_tensor_write(io->x, 0, x, sizeof(x)));
    CHECK(ds4_gpu_tensor_write(io->w, 0, w, sizeof(w)));
    return 1;
}

static int moe_call(const moe_model *m, const moe_io *io, ds4_gpu_tensor *sel,
                    uint32_t rows, uint32_t layer) {
    bool mid_f16 = false;
    return ds4_gpu_routed_moe_batch_tensor(io->out, io->gate, io->up, io->mid, io->experts,
        m->map, m->size, m->gate_off, m->up_off, m->down_off, TYPE_IQ2_XXS, TYPE_Q2_K,
        GATE_EXPERT, GATE_ROW, DOWN_EXPERT, DOWN_ROW, IN, MID, IN, sel, io->w,
        N_EXPERT, N_USED, 10.0f, io->x, layer, rows, &mid_f16, false) && !mid_f16;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* Selection sets: count layers of rows x 6 expert ids. */
typedef struct { uint32_t rows, count; int32_t *ids; } sel_set;

/* route-sel.bin: for rows 1..8, u32 rows, u32 count, count*rows*6 i32 ids
 * (one entry per traced step and layer). */
static int load_selections(const char *path, sel_set sets[MAX_ROWS + 1]) {
    FILE *f = fopen(path, "rb");
    CHECK(f);
    for (uint32_t r = 1; r <= MAX_ROWS; r++) {
        uint32_t hdr[2];
        CHECK(fread(hdr, 4, 2, f) == 2 && hdr[0] == r);
        sets[r].rows = r;
        sets[r].count = hdr[1];
        sets[r].ids = malloc((size_t)hdr[1] * r * N_USED * 4 + 4);
        CHECK(sets[r].ids);
        CHECK(fread(sets[r].ids, 4, (size_t)hdr[1] * r * N_USED, f) == (size_t)hdr[1] * r * N_USED);
    }
    fclose(f);
    return 1;
}

static void pick_distinct(int32_t *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (;;) {
            const int32_t e = (int32_t)(next_u64() % N_EXPERT);
            uint32_t j = 0;
            while (j < i && out[j] != e) j++;
            if (j == i) { out[i] = e; break; }
        }
    }
}

/* Synthetic control: kind 0 no-share (6*rows distinct), 1 all-share. */
static void synth(int32_t *ids, uint32_t rows, uint32_t count, int kind) {
    for (uint32_t c = 0; c < count; c++) {
        int32_t *s = ids + (size_t)c * rows * N_USED;
        if (kind == 0) {
            pick_distinct(s, rows * N_USED);
        } else {
            pick_distinct(s, N_USED);
            for (uint32_t r = 1; r < rows; r++) memcpy(s + r * N_USED, s, N_USED * 4);
        }
    }
}

/* Mean microseconds per call over count selections encoded in one batch. */
static double time_set(const moe_model *m, const moe_io *io, const int32_t *ids,
                       uint32_t rows, uint32_t count) {
    const size_t stride = (size_t)rows * N_USED * 4;
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(stride * count);
    if (!sel || !ds4_gpu_tensor_write(sel, 0, ids, stride * count)) return -1;
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        if (!ds4_gpu_begin_commands()) return -1;
        for (uint32_t c = 0; c < count; c++) {
            ds4_gpu_tensor *v = ds4_gpu_tensor_view(sel, c * stride, stride);
            const int ok = v && moe_call(m, io, v, rows, c % 40u);
            ds4_gpu_tensor_free(v);
            if (!ok) return -1;
        }
        if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return -1;
        const double us = (now_s() - t0) * 1e6 / count;
        if (us < best) best = us;
    }
    ds4_gpu_tensor_free(sel);
    return best;
}

static int bench(const char *path) {
    sel_set sets[MAX_ROWS + 1] = {{0}};
    CHECK(load_selections(path, sets));
    moe_model m;
    moe_io io;
    CHECK(model_init(&m) && io_init(&io));
    fprintf(stderr, "us/call (gate/up pair + down sum6); real = traced selections, "
                    "noshare/allshare = synthetic controls\n");
    for (uint32_t r = 1; r <= 6; r++) {
        const uint32_t count = sets[r].count < 400u ? sets[r].count : 400u;
        if (!count) continue;
        int32_t *ns = malloc((size_t)count * r * N_USED * 4);
        int32_t *as = malloc((size_t)count * r * N_USED * 4);
        CHECK(ns && as);
        synth(ns, r, count, 0);
        synth(as, r, count, 1);
        CHECK(time_set(&m, &io, sets[r].ids, r, count < 8u ? count : 8u) > 0); /* warm-up */
        const double real = time_set(&m, &io, sets[r].ids, r, count);
        const double noshare = time_set(&m, &io, ns, r, count);
        const double allshare = time_set(&m, &io, as, r, count);
        CHECK(real > 0 && noshare > 0 && allshare > 0);
        fprintf(stderr, "rows=%u n=%u real=%.1f noshare=%.1f allshare=%.1f\n",
                r, count, real, noshare, allshare);
        free(ns); free(as);
    }
    return 1;
}

static int read_all(ds4_gpu_tensor *t, void *dst, size_t bytes) {
    return ds4_gpu_tensor_read(t, 0, dst, bytes) != 0;
}

/* One call with the geometry kernels on or off; outputs copied to host. */
/* Geometry kernels chosen over the whole check: the reference runs must
 * choose none, the geometry runs some, or the comparison proves nothing. */
static uint64_t geometry_chosen;

static int run_geometry(const moe_model *m, const moe_io *io, ds4_gpu_tensor *sel,
                        uint32_t rows, uint32_t layer, bool single, bool geometry,
                        float *out, float *gate, float *up, float *mid) {
    ds4_gpu_test_set_flags(geometry ? 0u : DS4_GPU_TEST_V41_GENERIC_MOE);
    const uint64_t chosen = ds4_gpu_test_v41_specialized_count();
    const size_t pair = (size_t)rows * N_USED * MID * 4;
    /* Distinct fills per run, so an unwritten element cannot match. */
    const float fill = geometry ? -2.0f : -1.0f;
    CHECK(ds4_gpu_tensor_fill_f32(io->out, fill, (uint64_t)rows * IN) &&
          ds4_gpu_tensor_fill_f32(io->gate, fill, pair / 4) &&
          ds4_gpu_tensor_fill_f32(io->up, fill, pair / 4) &&
          ds4_gpu_tensor_fill_f32(io->mid, fill, pair / 4));
    CHECK(ds4_gpu_begin_commands());
    if (single) {
        CHECK(ds4_gpu_routed_moe_one_tensor(io->out, io->gate, io->up, io->mid, io->experts,
            m->map, m->size, m->gate_off, m->up_off, m->down_off, TYPE_IQ2_XXS, TYPE_Q2_K,
            GATE_EXPERT, GATE_ROW, DOWN_EXPERT, DOWN_ROW, IN, MID, IN, sel, io->w,
            N_EXPERT, N_USED, 10.0f, io->x, NULL, layer, true));
    } else {
        CHECK(moe_call(m, io, sel, rows, layer));
    }
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    const uint64_t delta = ds4_gpu_test_v41_specialized_count() - chosen;
    CHECK(geometry || delta == 0);
    geometry_chosen += delta;
    return read_all(io->out, out, (size_t)rows * IN * 4) && read_all(io->gate, gate, pair) &&
        read_all(io->up, up, pair) && read_all(io->mid, mid, pair);
}

static int check(void) {
    moe_model m;
    moe_io io;
    CHECK(model_init(&m) && io_init(&io));
    const size_t pair = (size_t)MAX_ROWS * N_USED * MID;
    float *buf[8];
    for (int i = 0; i < 8; i++) {
        buf[i] = malloc((i % 4 ? pair : (size_t)MAX_ROWS * IN) * 4);
        CHECK(buf[i]);
    }
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((size_t)MAX_ROWS * N_USED * 4);
    CHECK(sel);
    int passed = 0, failed = 0;
    for (int single = 0; single < 2; single++) {
        for (uint32_t rows = 1; rows <= (single ? 1u : MAX_ROWS); rows++) {
            /* 0 distinct, 1 all rows equal, 2 drawn from 9 experts (shared). */
            for (int kind = 0; kind < 3; kind++) {
                int32_t ids[MAX_ROWS * N_USED];
                if (kind < 2) synth(ids, rows, 1, kind);
                else for (uint32_t r = 0; r < rows; r++) {
                    int32_t pool[9];
                    pick_distinct(pool, 9);
                    for (int k = 0; k < N_USED; k++) ids[r * N_USED + k] = pool[k];
                }
                CHECK(ds4_gpu_tensor_write(sel, 0, ids, (size_t)rows * N_USED * 4));
                const uint32_t layer = (uint32_t)(kind * 7 + rows) % 40u;
                CHECK(run_geometry(&m, &io, sel, rows, layer, single, false,
                                   buf[0], buf[1], buf[2], buf[3]));
                CHECK(run_geometry(&m, &io, sel, rows, layer, single, true,
                                   buf[4], buf[5], buf[6], buf[7]));
                const size_t pb = (size_t)rows * N_USED * MID * 4;
                const bool same = !memcmp(buf[0], buf[4], (size_t)rows * IN * 4) &&
                    !memcmp(buf[3], buf[7], pb);
                fprintf(stderr, "%s %s rows=%u kind=%d\n", same ? "PASS" : "FAIL",
                        single ? "single" : "batch", rows, kind);
                if (same) passed++; else failed++;
            }
        }
    }
    ds4_gpu_test_set_flags(0);
    fprintf(stderr, "geometry check: %d passed, %d failed, geometry kernels chosen %llu times\n",
            passed, failed, (unsigned long long)geometry_chosen);
    CHECK(geometry_chosen > 0);
    ds4_gpu_tensor_free(sel);
    for (int i = 0; i < 8; i++) free(buf[i]);
    return failed == 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--check")) {
        const int ok = ds4_gpu_init() && check();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc >= 2 && !strcmp(argv[1], "--bench")) {
        const int ok = ds4_gpu_init() &&
            bench(argc >= 3 ? argv[2] : "/tmp/ds4_ab/route-sel.bin");
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    return 2;
}
