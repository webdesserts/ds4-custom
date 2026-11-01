#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t seed = 7919;
static float random_value(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((int)(seed % 65537) - 32768) / 8192.0f;
}

static float bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, 4);
    return value;
}

static float nearest(float value, int fp4) {
    const float fp4_values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float best_value = 0, best_error = INFINITY;
    int best = 0;
    for (int i = 0; i < (fp4 ? 8 : 127); i++) {
        const float v = fp4 ? fp4_values[i] : i < 8 ? ldexpf(i, -9) :
                        ldexpf(1.0f + (i & 7) / 8.0f, (i >> 3) - 7);
        const float error = fabsf(fabsf(value) - v);
        if (error < best_error || (error == best_error && !(i & 1) && (best & 1))) {
            best = i; best_value = v; best_error = error;
        }
    }
    return copysignf(best_value, value);
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc_managed(bytes);
    if (t && data && !ds4_gpu_tensor_write(t, 0, data, bytes)) {
        ds4_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

static int check_router(void) {
    enum { ROWS = 129, MAX_EXPERTS = 384, USED = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(bias_bytes >= MAX_EXPERTS * sizeof(float));
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    float *logits = malloc(ROWS * MAX_EXPERTS * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(bias && logits);
    for (unsigned e = 0; e < MAX_EXPERTS; e++) bias[e] = (e % 7) * 0.125f;
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    ds4_gpu_tensor *x = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *p = upload(NULL, ROWS * MAX_EXPERTS * sizeof(float));
    ds4_gpu_tensor *ids = upload(NULL, ROWS * USED * sizeof(int32_t));
    ds4_gpu_tensor *weights = upload(NULL, ROWS * USED * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    CHECK(x && p && ids && weights && tok);
    const unsigned counts[] = {1, 3, 17, ROWS};
    for (unsigned n = 128; n <= MAX_EXPERTS; n += 128) {
        const unsigned USED = n == 128 ? 3 : 6;
        for (unsigned mode = 0; mode < 3; mode++) {
            for (unsigned i = 0; i < ROWS * n; i++) logits[i] = mode == 1 ? 0 :
                mode == 2 ? (i % 2 ? -80 : 40) : random_value() * 8;
            CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * n * sizeof(float)));
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const unsigned rows = counts[ci];
                CHECK(ds4_gpu_router_select_batch_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, 0, true, false,
                    x, tok, n, USED, 1.5f, rows));
                CHECK(ds4_gpu_synchronize());
                const int32_t *selected = ds4_gpu_tensor_contents(ids);
                const float *actual = ds4_gpu_tensor_contents(weights);
                const float *probs = ds4_gpu_tensor_contents(p);
                CHECK(selected && actual && probs);
                for (unsigned t = 0; t < rows; t++) {
                    double ref[MAX_EXPERTS];
                    int best[USED];
                    for (unsigned k = 0; k < USED; k++) best[k] = -1;
                    for (unsigned e = 0; e < n; e++) {
                        const double v = logits[t * n + e];
                        ref[e] = sqrt(v > 20 ? v : v < -20 ? exp(v) : log1p(exp(v)));
                        if (!(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e])))
                            fprintf(stderr, "router n=%u mode=%u rows=%u row=%u expert=%u logit=%.9g actual=%.9g ref=%.9g\n",
                                    n, mode, rows, t, e, v, probs[t * n + e], ref[e]);
                        CHECK(fabs(probs[t * n + e] - ref[e]) <= 3e-6 * (1 + ref[e]));
                        const double score = ref[e] + bias[e];
                        for (unsigned k = 0; k < USED; k++) {
                            if (best[k] < 0 || score > ref[best[k]] + bias[best[k]]) {
                                for (unsigned j = USED - 1; j > k; j--) best[j] = best[j - 1];
                                best[k] = (int)e;
                                break;
                            }
                        }
                    }
                    double sum = 0;
                    for (unsigned k = 0; k < USED; k++) {
                        const int id = selected[t * USED + k];
                        CHECK(id >= 0 && id < (int)n);
                        for (unsigned j = 0; j < k; j++) CHECK(id != selected[t * USED + j]);
#ifdef __APPLE__
                        /* Bitonic selection does not promise CUDA's lower-ID
                         * tie break. Still require the correct top-k scores. */
                        CHECK(fabs((ref[id] + bias[id]) - (ref[best[k]] + bias[best[k]])) < 3e-6);
#else
                        CHECK(id == best[k]);
#endif
                        sum += ref[id];
                    }
                    for (unsigned k = 0; k < USED; k++) {
                        CHECK(fabs(actual[t * USED + k] - 1.5 * ref[selected[t * USED + k]] /
                              fmax(sum, 0x1p-14)) < 3e-6);
                    }
                }
                int32_t first_ids[USED]; float first_weights[USED];
                memcpy(first_ids, selected, sizeof(first_ids));
                memcpy(first_weights, actual, sizeof(first_weights));
                CHECK(ds4_gpu_router_select_tensor(ids, weights, p, bias,
                    bias_bytes, 0, 0, 0, 0, n, USED, 1.5f,
                    0, 0, true, false, x));
                CHECK(ds4_gpu_synchronize());
                const int32_t *scalar_ids = ds4_gpu_tensor_contents(ids);
                const float *scalar_weights = ds4_gpu_tensor_contents(weights);
#ifdef __APPLE__
                for (unsigned k = 0; k < USED; k++) {
                    CHECK(scalar_ids[k] >= 0 && scalar_ids[k] < (int32_t)n);
                    CHECK(probs[first_ids[k]] + bias[first_ids[k]] ==
                          probs[scalar_ids[k]] + bias[scalar_ids[k]]);
                    CHECK(fabsf(first_weights[k] - scalar_weights[k]) < 3e-6f);
                }
#else
                CHECK(!memcmp(first_ids, scalar_ids, sizeof(first_ids)));
                CHECK(!memcmp(first_weights, scalar_weights, sizeof(first_weights)));
#endif
            }
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(p); ds4_gpu_tensor_free(ids);
    ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(tok);
    ds4_gpu_cleanup();
    free(bias); free(logits);
    CHECK(ds4_gpu_init());
    fprintf(stderr, "256/384-expert routing, ties/extremes and scalar/batch oracle: PASS\n");
    return 1;
}

static int check_quantization(void) {
    enum { WIDTH = 512, ROWS = 33, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    for (int mode = 0; mode < 4; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i++) source[i] = random_value() * (1u << ((i / block) % 4));
        for (int i = 0; i < WIDTH; i++) source[i] = copysignf(0.0f, i & 1 ? -1.0f : 1.0f);
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale = 1;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP8_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f))));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            if (mode == DS4_V41_FP4_E4M3) scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                float expected = bf16(source[start + i]);
                if (mode) expected = bf16(nearest(expected / scale, mode != 1) * scale);
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "quantization mode=%d index=%d: %.9g != %.9g\n",
                            mode, start + i, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    CHECK(ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 24, 1, DS4_V41_FP8_E8M0));
    CHECK(!ds4_gpu_dsv41_quantize(t, UINT32_MAX, UINT32_MAX, DS4_V41_BF16));
    CHECK(!ds4_gpu_dsv41_quantize(t, 32, 1, (ds4_v41_activation_format)4));
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 BF16/FP8/FP4 round trips: exact\n");
    return 1;
}

/* FP4 activation rounding at its decision points: every block holds bf16
 * values on, and one bf16 step either side of, each E2M1 midpoint times the
 * block's scale, for many scales, so a kernel that rounds a midpoint the
 * wrong way or misses a tie fails. Reference as in check_quantization. */
static float bf16_step(float x, int up) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    bits = (bits & 0xffff0000u) + (up ? 0x10000u : (uint32_t)-0x10000);
    memcpy(&x, &bits, 4);
    return x;
}

static int check_fp4_rounding(void) {
    static const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    enum { WIDTH = 4096, ROWS = 64, N = WIDTH * ROWS };
    float *source = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    CHECK(source && actual);
    ds4_gpu_tensor *t = upload(NULL, N * sizeof(float));
    CHECK(t);
    long checked = 0;
    for (int mode = DS4_V41_FP4_E8M0; mode <= DS4_V41_FP4_E4M3; mode++) {
        const int block = mode == DS4_V41_FP4_E4M3 ? 16 : 32;
        for (int i = 0; i < N; i += block) {
            /* A block amax spread over ~2^-12..2^12, then values that probe
             * the midpoints of the scale the reference derives from it. */
            const float amax = bf16(ldexpf(1.0f + (float)((i / block) % 61) / 64.0f, (i / block) % 25 - 12));
            float scale;
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            else
                scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            source[i] = (i / block) & 1 ? -amax : amax;
            for (int j = 1; j < block; j++) {
                const float m = bf16(mids[(j - 1) % 7] * scale);
                const int k = (j - 1) / 7 % 3;
                float v = k == 0 ? m : bf16_step(m, k == 1);
                if (fabsf(v) > amax) v = amax;
                source[i + j] = (j & 2) ? -v : v;
            }
        }
        CHECK(ds4_gpu_tensor_write(t, 0, source, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_quantize(t, WIDTH, ROWS, (ds4_v41_activation_format)mode));
        CHECK(ds4_gpu_tensor_read(t, 0, actual, N * sizeof(float)));
        for (int start = 0; start < N; start += block) {
            float amax = 0, scale;
            for (int i = 0; i < block; i++) amax = fmaxf(amax, fabsf(bf16(source[start + i])));
            if (mode == DS4_V41_FP4_E8M0)
                scale = exp2f(ceilf(log2f(fmaxf(amax, 0x1.8p-124f) * (1.0f / 6.0f))));
            else
                scale = nearest(fmaxf(amax, 6.0f / 512.0f) / 6.0f, 0);
            for (int i = 0; i < block; i++) {
                const float expected = bf16(nearest(bf16(source[start + i]) / scale, 1) * scale);
                checked++;
                if (memcmp(&expected, actual + start + i, 4)) {
                    fprintf(stderr, "fp4 rounding mode=%d index=%d value=%.9g scale=%.9g: %.9g != %.9g\n",
                            mode, start + i, source[start + i], scale, actual[start + i], expected);
                    return 0;
                }
            }
        }
    }
    ds4_gpu_tensor_free(t);
    free(source); free(actual);
    fprintf(stderr, "V4.1 FP4 rounding at E2M1 midpoints: %ld values exact: PASS\n", checked);
    return 1;
}

static double monotonic_seconds(void);

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t bf16_input(size_t i) {
    const uint32_t low[] = {0, 0x7fffu, 0x8000u, 0x8001u, 0xffffu};
    if (i < 65536u * 5u) return ((uint32_t)(i / 5u) << 16u) | low[i % 5u];
    return (uint32_t)i * 2654435761u;
}

static int check_bf16_linear(void) {
    const struct { uint32_t width, rows, offset; } shapes[] = {
        {1, 1, 0}, {3, 1, 0}, {5, 1, 0}, {24, 1, 0}, {31, 3, 0},
        {32, 7, 1}, {129, 3, 4}, {1023, 1, 0}, {1024, 1, 0}, {1025, 1, 0},
        {65536u * 5u, 1, 0}, {5120, 1, 0}, {20480, 512, 0},
        {32768, 4096, 0}, {32768, 8192, 0}
    };
    const unsigned modes[] = {0, 1, 1, 0};
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const size_t count = (size_t)shapes[s].width * shapes[s].rows;
        const size_t offset = shapes[s].offset;
        ds4_gpu_tensor *storage = upload(NULL, (count + 8) * sizeof(uint32_t));
        CHECK(storage);
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(storage,
            offset * sizeof(uint32_t), count * sizeof(uint32_t));
        CHECK(view);
        uint32_t *bits = ds4_gpu_tensor_contents(storage);
        CHECK(bits);
        for (unsigned pass = 0; pass < 4; pass++) {
            const unsigned mode = modes[pass];
            if (mode) unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
            else setenv("DS4_METAL_DISABLE_V41_LINEAR_BF16", "1", 1);
            for (size_t i = 0; i < count + 8; i++) bits[i] = 0x12345678;
            for (size_t i = 0; i < count; i++) bits[offset + i] = bf16_input(i);
            double elapsed = 0;
            for (unsigned repeat = 0; repeat < 5; repeat++) {
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_dsv41_quantize(view, shapes[s].width, shapes[s].rows, DS4_V41_BF16));
                CHECK(ds4_gpu_synchronize());
                if (repeat) elapsed += (monotonic_seconds() - begin) * 250;
            }
            for (size_t i = 0; i < count; i++) CHECK(bits[offset + i] == bf16_bits(bf16_input(i)));
            for (size_t i = 0; i < offset; i++) CHECK(bits[i] == 0x12345678);
            for (size_t i = offset + count; i < count + 8; i++) CHECK(bits[i] == 0x12345678);
            fprintf(stderr, "BF16 width=%u rows=%u offset=%zu mode=%u pass=%u %.3f ms: exact\n",
                shapes[s].width, shapes[s].rows, offset, mode, pass, elapsed);
        }
        ds4_gpu_tensor_free(view);
        ds4_gpu_tensor_free(storage);
    }
    unsetenv("DS4_METAL_DISABLE_V41_LINEAR_BF16");
    return 1;
}

#ifdef __APPLE__
static int check_hc_scaled(void) {
    enum { WIDTH = 20480, OUT = 24, ROWS = 8192 };
    const size_t weight_bytes = WIDTH * OUT * sizeof(_Float16);
    const size_t page = (size_t)getpagesize();
    const size_t mapped_bytes = (weight_bytes + page - 1) / page * page;
    void *model = NULL;
    CHECK(!posix_memalign(&model, page, mapped_bytes));
    _Float16 *weights = model;
    for (size_t i = 0; i < WIDTH * OUT; i++) weights[i] = (_Float16)(random_value() / 64);
    CHECK(ds4_gpu_set_model_map(model, mapped_bytes));
    ds4_gpu_tensor *x = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *norm = upload(NULL, (size_t)WIDTH * ROWS * 4);
    ds4_gpu_tensor *scales = upload(NULL, (ROWS + 1) * 4);
    ds4_gpu_tensor *out = upload(NULL, (OUT * ROWS + 1) * 4);
    ds4_gpu_tensor *ref = upload(NULL, (OUT * ROWS + 1) * 4);
    CHECK(x && norm && scales && out && ref);
    float *input = ds4_gpu_tensor_contents(x), *y = ds4_gpu_tensor_contents(out);
    float *expected = ds4_gpu_tensor_contents(ref), *s = ds4_gpu_tensor_contents(scales);
    for (size_t i = 0; i < (size_t)WIDTH * ROWS; i++)
        input[i] = bf16(random_value() * ((i / WIDTH) % 7 ? 1 : 0x1p-16f));
    const unsigned rows[] = {9, 31, 32, 33, 127, 512, 2048, 4096, 8192};
    for (size_t shape = 0; shape < sizeof(rows) / sizeof(*rows); shape++) {
        const uint32_t n = rows[shape];
        double elapsed[2] = {0};
        for (unsigned repeat = 0; repeat < 7; repeat++) {
            y[n * OUT] = expected[n * OUT] = s[n] = 12345;
            for (unsigned j = 0; j < 2; j++) {
                const unsigned mode = j ^ (repeat & 1);
                const double begin = monotonic_seconds();
                CHECK(ds4_gpu_begin_commands());
                if (mode) {
                    CHECK(ds4_gpu_hc_rms_scale_project_f16_tensor(out, n >= 512u ? scales : norm,
                        model, mapped_bytes, 0, WIDTH, OUT, x, n, 1e-6f));
                } else {
                    CHECK(ds4_gpu_rms_norm_plain_rows_tensor(norm, x, WIDTH, n, 1e-6f));
                    CHECK(ds4_gpu_matmul_f16_tensor(ref, model, mapped_bytes, 0,
                        WIDTH, OUT, norm, n));
                }
                CHECK(ds4_gpu_end_commands());
                if (repeat) elapsed[mode] += (monotonic_seconds() - begin) * (1000.0 / 6);
            }
            CHECK(!memcmp(y, expected, n * OUT * 4));
            CHECK(y[n * OUT] == 12345 && expected[n * OUT] == 12345 && s[n] == 12345);
        }
        fprintf(stderr, "HC scaled rows=%u: exact %.3f -> %.3f ms\n", n, elapsed[0], elapsed[1]);
    }
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(norm); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(model);
    return 1;
}

#endif

static int check_engram(void) {
    enum { D = 5120, ROWS = 5, N = ROWS * 4 * D };
    float *x = malloc(N * sizeof(float)), *actual = malloc(N * sizeof(float));
    float *kv = malloc(ROWS * 5 * D * sizeof(float));
    float *qw = malloc(4 * D * sizeof(float)), *kw = malloc(4 * D * sizeof(float));
    uint8_t mask[] = {1, 1, 0, 1, 0};
    CHECK(x && actual && kv && qw && kw);
    for (int i = 0; i < N; i++) x[i] = bf16(random_value());
    for (int i = 0; i < ROWS * 5 * D; i++) kv[i] = bf16(random_value());
    for (int i = 0; i < 4 * D; i++) { qw[i] = random_value(); kw[i] = random_value(); }
    memset(x, 0, D * sizeof(float));
    memset(kv + D, 0, D * sizeof(float));
    ds4_gpu_tensor *xt = upload(x, N * sizeof(float));
    ds4_gpu_tensor *kt = upload(kv, ROWS * 5 * D * sizeof(float));
    ds4_gpu_tensor *qwt = upload(qw, 4 * D * sizeof(float));
    ds4_gpu_tensor *kwt = upload(kw, 4 * D * sizeof(float));
    ds4_gpu_tensor *mt = upload(mask, sizeof(mask));
    CHECK(xt && kt && qwt && kwt && mt);
    size_t rounded_differently = 0;
    double error2 = 0, norm2 = 0;
    for (int masked = 0; masked < 2; masked++) {
        CHECK(ds4_gpu_tensor_write(xt, 0, x, N * sizeof(float)));
        CHECK(ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, masked ? mt : NULL, D, ROWS, 1e-20f));
        CHECK(ds4_gpu_tensor_read(xt, 0, actual, N * sizeof(float)));
        for (int row = 0; row < ROWS; row++) for (int h = 0; h < 4; h++) {
            double dot = 0, h2 = 0, k2 = 0;
            for (int i = 0; i < D; i++) {
                const double a = x[(row * 4 + h) * D + i], b = kv[(row * 5 + h) * D + i];
                h2 += a * a; k2 += b * b;
                dot += a * (float)(qw[h * D + i] * kw[h * D + i]) * b;
            }
            dot /= sqrt(h2 / D + 1e-20) * sqrt(k2 / D + 1e-20) * sqrt(D);
            double gate = 1 / (1 + exp(-copysign(sqrt(fmax(fabs(dot), 1e-6)), dot)));
            if (masked && !mask[row]) gate = 0;
            for (int i = 0; i < D; i++) {
                const int off = (row * 4 + h) * D + i;
                const float expected = bf16(x[off] + (float)gate * kv[(row * 5 + 4) * D + i]);
                const double error = actual[off] - expected;
                CHECK(isfinite(actual[off]));
                CHECK(fabs(error) <= fmax(1e-6, fabs(expected) / 128));
                if (masked && !mask[row]) CHECK(actual[off] == x[off]);
                rounded_differently += actual[off] != expected;
                error2 += error * error; norm2 += (double)expected * expected;
            }
        }
    }
    CHECK(rounded_differently < N / 1000);
    CHECK(sqrt(error2 / norm2) < 1e-4);
    fprintf(stderr, "V4.1 Engram gate: %zu BF16 boundary differences, relative RMS %.8g\n",
            rounded_differently, sqrt(error2 / norm2));
    CHECK(!ds4_gpu_dsv41_engram_add(xt, kt, qwt, kwt, mt, D, ROWS + 1, 1e-20f));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qwt);
    ds4_gpu_tensor_free(kwt); ds4_gpu_tensor_free(mt);
    free(x); free(actual); free(kv); free(qw); free(kw);
    return 1;
}

static int check_rope_stride(void) {
    enum { WIDTH = 512, HEADS = 2, ROWS = 129, N = WIDTH * HEADS * ROWS };
    ds4_gpu_tensor *a = upload(NULL, (N + 1u) * sizeof(float));
    ds4_gpu_tensor *b = upload(NULL, (N + 1u) * sizeof(float));
    CHECK(a && b);
    float *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    const uint32_t starts[] = {0, 126, 32766, 1048318};
    for (uint32_t kind = 0; kind < 2; kind++) for (uint32_t inverse = 0; inverse < 2; inverse++) {
        for (size_t i = 0; i < sizeof(starts) / sizeof(*starts); i++) {
            for (size_t j = 0; j < N; j++) x[j] = y[j] = bf16(random_value());
            x[N] = y[N] = 12345;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, starts[i], 2, kind, inverse));
            for (uint32_t row = 0; row < ROWS; row++) {
                ds4_gpu_tensor *v = ds4_gpu_tensor_view(b, (uint64_t)row * WIDTH * HEADS * 4u,
                                                       WIDTH * HEADS * 4u);
                CHECK(v && ds4_gpu_dsv41_rope(v, WIDTH, HEADS, 1, starts[i] + row * 2u, kind, inverse));
                ds4_gpu_tensor_free(v);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(!memcmp(x, y, (N + 1u) * sizeof(float)) && x[N] == 12345);
        }
    }
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, 0, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 1048320, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS + 1, 0, 2, true, false));
    CHECK(!ds4_gpu_dsv41_rope_stride(a, WIDTH, HEADS, ROWS, 0, UINT32_MAX, true, false));
    ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(a);
    fprintf(stderr, "V4.1 strided RoPE: exact row parity through 1M positions, guards PASS\n");
    return 1;
}

static int check_pool(void) {
    enum { D = 512, ROWS = 257, PAIRS = ROWS / 2 };
    float *kv = malloc(ROWS * D * sizeof(float)), *scores = malloc(ROWS * D * sizeof(float));
    float *got = malloc(PAIRS * D * sizeof(float)), *reference = malloc(PAIRS * D * sizeof(float));
    CHECK(kv && scores && got && reference);
    for (int i = 0; i < ROWS * D; i++) { kv[i] = random_value(); scores[i] = random_value() * 25; }
    ds4_gpu_tensor *kt = upload(kv, ROWS * D * sizeof(float));
    ds4_gpu_tensor *st = upload(scores, ROWS * D * sizeof(float));
    ds4_gpu_tensor *pk = upload(NULL, D * sizeof(float)), *ps = upload(NULL, D * sizeof(float));
    ds4_gpu_tensor *out = upload(NULL, PAIRS * D * sizeof(float));
    CHECK(kt && st && pk && ps && out);
    const uint32_t chunks[] = {257, 1, 2, 3, 17, 127, 128, 129};
    for (size_t c = 0; c < sizeof(chunks) / sizeof(*chunks); c++) {
        CHECK(ds4_gpu_tensor_fill_f32(pk, NAN, D));
        CHECK(ds4_gpu_tensor_fill_f32(ps, NAN, D));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t start = 0; start < ROWS;) {
            uint32_t n = chunks[c] < ROWS - start ? chunks[c] : ROWS - start;
            uint32_t pairs = (n + (start & 1u)) / 2;
            ds4_gpu_tensor *k = ds4_gpu_tensor_view(kt, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(st, (uint64_t)start * D * 4, (uint64_t)n * D * 4);
            ds4_gpu_tensor *o = pairs ? ds4_gpu_tensor_view(out, (uint64_t)(start / 2) * D * 4,
                                                         (uint64_t)pairs * D * 4) : NULL;
            CHECK(k && s && (!pairs || o));
            CHECK(ds4_gpu_dsv41_pool2(o, k, s, pk, ps, D, n, start));
            ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(o);
            start += n;
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(out, 0, got, PAIRS * D * sizeof(float)));
        if (!c) {
            memcpy(reference, got, PAIRS * D * sizeof(float));
            for (int p = 0; p < PAIRS; p++) for (int i = 0; i < D; i++) {
                const int a = 2 * p * D + i, b = a + D;
                const double gate = 1 / (1 + exp((double)scores[b] - scores[a]));
                const float expected = bf16((float)(kv[a] * gate + kv[b] * (1 - gate)));
                CHECK(isfinite(got[p * D + i]));
                CHECK(fabsf(got[p * D + i] - expected) <= fmaxf(1e-6f, fabsf(expected) / 128));
            }
        } else CHECK(!memcmp(reference, got, PAIRS * D * sizeof(float)));
        float tail[D];
        CHECK(ds4_gpu_tensor_read(pk, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, kv + (ROWS - 1) * D, sizeof(tail)));
        CHECK(ds4_gpu_tensor_read(ps, 0, tail, sizeof(tail)));
        CHECK(!memcmp(tail, scores + (ROWS - 1) * D, sizeof(tail)));
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_dsv41_pool2(out, kt, st, pk, ps, D, ROWS - 1, 0));
    CHECK(ds4_gpu_end_commands());
    CHECK(!memcmp(ds4_gpu_tensor_contents(pk), kv + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(ps), scores + (ROWS - 3) * D, D * sizeof(float)));
    CHECK(!memcmp(ds4_gpu_tensor_contents(out), reference, PAIRS * D * sizeof(float)));
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(pk);
    ds4_gpu_tensor_free(ps); ds4_gpu_tensor_free(out);
    free(kv); free(scores); free(got); free(reference);
    fprintf(stderr, "V4.1 pair pooling: exact across eight chunk sizes\n");
    return 1;
}

typedef struct { float score; uint32_t index; } candidate;
static int candidate_desc(const void *a, const void *b) {
    const candidate *x = a, *y = b;
    return x->score > y->score ? -1 : x->score < y->score ? 1 :
           x->index < y->index ? -1 : x->index > y->index;
}

static int check_candidates(void) {
    const uint32_t widths[] = {1, 7, 8, 9, 127, 16385, 17017};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi], blocks = (n + 7) / 8, rows = 17;
        const uint32_t top = blocks < 2048 ? blocks : 2048;
        float *scores = malloc((size_t)n * rows * 4), *got = malloc((size_t)n * rows * 4);
        float *maxima = malloc((size_t)blocks * rows * 4);
        candidate *sorted = malloc(blocks * sizeof(candidate));
        uint8_t *kept = malloc(blocks);
        CHECK(scores && got && maxima && sorted && kept);
        /* Unique finite values avoid unspecified top-k tie ordering. */
        for (uint32_t r = 0; r < rows; r++) for (uint32_t i = 0; i < n; i++)
            scores[(size_t)r * n + i] = (float)((i * 7919u + r * 1009u) % 104729u) - 50000;
        ds4_gpu_tensor *s = upload(NULL, (size_t)n * rows * 4);
        ds4_gpu_tensor *b = upload(NULL, (size_t)blocks * rows * 4);
        ds4_gpu_tensor *t = upload(NULL, (size_t)top * rows * 4);
        ds4_gpu_tensor *m = upload(NULL, (size_t)blocks * rows * 4);
        CHECK(s && b && t && m);
        for (uint32_t ratio = 1; ratio <= 2; ratio++) for (int late = 0; late < 2; late++) {
            const uint32_t start = late ? n * ratio - 1 : 0;
            CHECK(ds4_gpu_tensor_write(s, 0, scores, (size_t)n * rows * 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, start, ratio));
            CHECK(ds4_gpu_indexer_topk_tensor(t, b, blocks, rows, top));
            CHECK(ds4_gpu_dsv4_topk_mask_tensor(m, t, blocks, rows, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(s, m, n, rows, start, ratio));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(b, 0, maxima, (size_t)blocks * rows * 4));
            CHECK(ds4_gpu_tensor_read(s, 0, got, (size_t)n * rows * 4));
            for (uint32_t r = 0; r < rows; r++) {
                uint32_t visible = (start + r + 1) / ratio;
                if (visible > n) visible = n;
                for (uint32_t j = 0; j < blocks; j++) {
                    float best = -INFINITY;
                    for (uint32_t i = j * 8; i < (j + 1) * 8 && i < visible; i++)
                        best = fmaxf(best, scores[(size_t)r * n + i]);
                    if (visible && j == (visible - 1) / 8) best = INFINITY;
                    CHECK(maxima[(size_t)r * blocks + j] == best);
                    sorted[j] = (candidate){best, j};
                }
                qsort(sorted, blocks, sizeof(*sorted), candidate_desc);
                memset(kept, 0, blocks);
                for (uint32_t j = 0; j < top; j++)
                    if (sorted[j].score > -INFINITY) kept[sorted[j].index] = 1;
                for (uint32_t i = 0; i < n; i++) {
                    float expected = i < visible && kept[i / 8] ? scores[(size_t)r * n + i] : -INFINITY;
                    CHECK(got[(size_t)r * n + i] == expected);
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, UINT32_MAX, 1));
        CHECK(!ds4_gpu_dsv41_candidate_blocks(b, s, n, rows, 0, 0));
        CHECK(!ds4_gpu_dsv41_candidate_filter(s, m, n, rows + 1, 0, 1));
        ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(t); ds4_gpu_tensor_free(m);
        free(scores); free(got); free(maxima); free(sorted); free(kept);
    }
    fprintf(stderr, "V4.1 causal candidate blocks and filtering: exact\n");
    return 1;
}

static int check_sparse_gather(void) {
    const uint32_t sizes[] = {1, 3, 511, 512, 513, 8193, 17017};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++) {
        const uint32_t rows = sizes[i], selected = rows < 512 ? rows : 512;
        ds4_gpu_tensor *source = upload(NULL, (uint64_t)rows * 512 * 4);
        ds4_gpu_tensor *ids = upload(NULL, selected * 4);
        ds4_gpu_tensor *out = upload(NULL, (uint64_t)selected * 512 * 4);
        CHECK(source && ids && out);
        float *input = ds4_gpu_tensor_contents(source);
        int32_t *indices = ds4_gpu_tensor_contents(ids);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t c = 0; c < 512; c++)
            input[(size_t)r * 512 + c] = (float)r + (float)c / 512.0f;
        for (uint32_t r = 0; r < selected; r++) indices[r] = (int32_t)(rows - 1u - r);
        CHECK(ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected));
        const float *result = ds4_gpu_tensor_contents(out);
        for (uint32_t r = 0; r < selected; r++)
            CHECK(!memcmp(result + r * 512, input + (size_t)indices[r] * 512, 512 * 4));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows + 1, selected));
        CHECK(!ds4_gpu_dsv41_gather_kv(out, source, ids, rows, selected + 1));
        ds4_gpu_tensor_free(source); ds4_gpu_tensor_free(ids); ds4_gpu_tensor_free(out);
    }
    fprintf(stderr, "V4.1 bounded sparse KV gather: exact\n");
    return 1;
}

/* The decode-row low projection streams each group's weights once per token
 * tile. With random scales and inputs, any change to the per-token K walk or
 * reduction order shows up in the bits, so compare against the single-token
 * kernel for every tile size and the per-token (DS4_METAL_V41_LOW_ROWS=0)
 * grid. */
static int check_low_rows(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120, ROWS = 8 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        const int value = (int)(random_value() * 8192);
        w[i].d = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u));
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 128);
    }
    float *x = malloc(xb), *reference = malloc(lb);
    CHECK(x && reference);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        CHECK(xr && lr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(low, 0, reference, lb));
    for (int tiled = 1; tiled >= 0; tiled--) {
        if (tiled) unsetenv("DS4_METAL_V41_LOW_ROWS");
        else setenv("DS4_METAL_V41_LOW_ROWS", "0", 1);
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
                0, a_bytes, xt, rows));
            const float *got = ds4_gpu_tensor_contents(low);
            if (memcmp(got, reference, (uint64_t)rows * GROUPS * RANK * 4))
                fprintf(stderr, "low rows mismatch tiled=%d rows=%u\n", tiled, rows);
            CHECK(!memcmp(got, reference, (uint64_t)rows * GROUPS * RANK * 4));
            if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * GROUPS * RANK]));
            fprintf(stderr, "V4.1 decode-row low projection tiled=%d rows=%u: PASS\n", tiled, rows);
        }
    }
    unsetenv("DS4_METAL_V41_LOW_ROWS");
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    free(x); free(reference); free(model);
    return 1;
}

#ifdef __APPLE__
/* The DSpark draft attention-output low projection runs one strided
 * multi-row F16 dispatch per group instead of a matvec per (row, group).
 * Random F16 weights and inputs expose any change in the K walk or
 * reduction, so compare bits with per-row ds4_gpu_matmul_f16_tensor calls
 * for every row count, tiled (auto) and one-row (off) plans. Untouched
 * output (other groups' slots, rows past the count) must stay NaN. */
static int check_draft_low_rows(void) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, ROWS = 6 };
    const uint64_t wb = (uint64_t)GROUPS * RANK * GROUP * 2;
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4, lb = (uint64_t)ROWS * GROUPS * RANK * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
    uint16_t *w = model;
    for (uint64_t i = 0; i < wb / 2; i++) {
        /* Finite halves with varied exponents: sign, exponent 1..29, mantissa. */
        const uint32_t v = (uint32_t)(random_value() * 8192) * 2654435761u + (uint32_t)i;
        w[i] = (uint16_t)((v & 0x8000u) | ((1u + (v >> 3) % 29u) << 10) | (v & 0x3ffu));
    }
    float *x = malloc(xb), *reference = malloc(lb), *got = malloc(lb);
    CHECK(x && reference && got);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb);
    CHECK(xt && low && ds4_gpu_set_model_map(model, wb));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) for (uint32_t g = 0; g < GROUPS; g++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, ((uint64_t)r * GROUPS + g) * GROUP * 4, GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, ((uint64_t)r * GROUPS + g) * RANK * 4, RANK * 4);
        CHECK(xr && lr && ds4_gpu_matmul_f16_tensor(lr, model, wb,
            (uint64_t)g * RANK * GROUP * 2, GROUP, RANK, xr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(low, 0, reference, lb));
    const char *modes[] = {"auto", "off"};
    for (int mode = 0; mode < 2; mode++) {
        setenv("DS4_METAL_DECODE_ROWS_MODE", modes[mode], 1);
        for (uint32_t rows = 1; rows <= ROWS; rows++) for (uint32_t only = 0; only <= GROUPS; only++) {
            /* only < GROUPS: just that group, checking the strides leave the
             * other groups alone; only == GROUPS: all groups. */
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t g = 0; g < GROUPS; g++) {
                if (only < GROUPS && g != only) continue;
                ds4_gpu_tensor *xg = ds4_gpu_tensor_view(xt, (uint64_t)g * GROUP * 4,
                    ((uint64_t)(rows - 1) * GROUPS * GROUP + GROUP) * 4);
                ds4_gpu_tensor *lg = ds4_gpu_tensor_view(low, (uint64_t)g * RANK * 4,
                    ((uint64_t)(rows - 1) * GROUPS * RANK + RANK) * 4);
                CHECK(xg && lg && ds4_gpu_dsv41_projection_rows_strided(lg, model, wb,
                    (uint64_t)g * RANK * GROUP * 2, GROUP, RANK, rows, xg,
                    (uint64_t)GROUPS * GROUP, (uint64_t)GROUPS * RANK));
                ds4_gpu_tensor_free(xg); ds4_gpu_tensor_free(lg);
            }
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(low, 0, got, lb));
            for (uint32_t r = 0; r < ROWS; r++) for (uint32_t g = 0; g < GROUPS; g++) {
                const uint64_t at = ((uint64_t)r * GROUPS + g) * RANK;
                const bool written = r < rows && (only == GROUPS || g == only);
                if (written) {
                    if (memcmp(got + at, reference + at, RANK * 4))
                        fprintf(stderr, "draft low rows mismatch mode=%s rows=%u only=%u r=%u g=%u\n",
                                modes[mode], rows, only, r, g);
                    CHECK(!memcmp(got + at, reference + at, RANK * 4));
                } else {
                    for (uint32_t i = 0; i < RANK; i++) CHECK(isnan(got[at + i]));
                }
            }
        }
        fprintf(stderr, "V4.1 draft strided F16 low projection mode=%s rows 1..%u: PASS\n", modes[mode], ROWS);
    }
    unsetenv("DS4_METAL_DECODE_ROWS_MODE");
    /* Strides narrower than a row, or strides on the non-exact path, refuse. */
    CHECK(!ds4_gpu_dsv41_projection_rows_strided(low, model, wb, 0, GROUP, RANK, 2, xt, GROUP - 1, 0));
    CHECK(!ds4_gpu_dsv41_projection_rows_strided(low, model, wb, 0, GROUP, RANK, 2, xt, 0, RANK - 1));
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low);
    free(x); free(reference); free(got); free(model);
    return 1;
}
#endif

#ifdef __APPLE__
/* F32 decode-row matvecs (routers) against per-row matvecs: random weights
 * and inputs, shapes covering the one-row-per-SIMD-group router (384
 * outputs), a K tail (5124), a narrow grid that keeps one row per
 * threadgroup (24 outputs) and a two-row tile (2048), every row count,
 * auto and off plans. Rows past the count stay NaN. */
static int check_f32_rows(void) {
    enum { ROWS = 6 };
    const struct { uint32_t in, out; } shapes[] = {{5120, 384}, {5124, 384}, {4096, 24}, {4096, 2048}};
    const char *modes[] = {"auto", "off"};
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++) {
        const uint32_t in = shapes[sh].in, out_dim = shapes[sh].out;
        const uint64_t wb = (uint64_t)in * out_dim * 4, xb = (uint64_t)ROWS * in * 4, ob = (uint64_t)ROWS * out_dim * 4;
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
        float *w = model;
        for (uint64_t i = 0; i < wb / 4; i++) w[i] = random_value();
        float *x = malloc(xb), *reference = malloc(ob), *got = malloc(ob);
        CHECK(x && reference && got);
        for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
        ds4_gpu_tensor *xt = upload(x, xb), *ot = upload(NULL, ob);
        CHECK(xt && ot && ds4_gpu_set_model_map(model, wb));
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t r = 0; r < ROWS; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * in * 4, (uint64_t)in * 4);
            ds4_gpu_tensor *orow = ds4_gpu_tensor_view(ot, (uint64_t)r * out_dim * 4, (uint64_t)out_dim * 4);
            CHECK(xr && orow && ds4_gpu_matmul_f32_tensor(orow, model, wb, 0, in, out_dim, xr, 1));
            ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(orow);
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(ot, 0, reference, ob));
        for (int mode = 0; mode < 2; mode++) {
            setenv("DS4_METAL_DECODE_ROWS_MODE", modes[mode], 1);
            for (uint32_t rows = 1; rows <= ROWS; rows++) {
                CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, ob / 4));
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_dsv41_projection_rows_f32(ot, model, wb, 0, in, out_dim, rows, xt));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_tensor_read(ot, 0, got, ob));
                if (memcmp(got, reference, (uint64_t)rows * out_dim * 4))
                    fprintf(stderr, "f32 rows mismatch %ux%u mode=%s rows=%u\n", in, out_dim, modes[mode], rows);
                CHECK(!memcmp(got, reference, (uint64_t)rows * out_dim * 4));
                for (uint64_t i = (uint64_t)rows * out_dim; i < ob / 4; i++) CHECK(isnan(got[i]));
            }
            fprintf(stderr, "V4.1 F32 decode rows %ux%u mode=%s rows 1..%u: PASS\n", in, out_dim, modes[mode], ROWS);
        }
        unsetenv("DS4_METAL_DECODE_ROWS_MODE");
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
        free(x); free(reference); free(got); free(model);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Fused BF16 rounding in the Q8_0 decode-row writer against the unfused
 * matvec plus the separate BF16 pass: random scales and inputs, a NaN row
 * and an overflowing row (non-finite values pass the rounding unchanged),
 * rows 1..6, auto and off plans. When the plan has no row tile the call
 * reports *rounded == 0 and must leave outputs unrounded. */
/* DS4_METAL_V41_FUSED_ROUTER: the one-dispatch 384-expert router against
 * the generic nine-dispatch chain (=0), bitwise over probs, selected ids and
 * weights, for the batch and single-row entry points. Patterns include all
 * scores equal and few distinct values, where only an identical sorting
 * network keeps the tie order. */
static int check_fused_router(void) {
    enum { ROWS = 129, N = 384, USED = 6, PATTERNS = 6 };
    const size_t bias_bytes = (size_t)getpagesize();
    float *bias = NULL;
    CHECK(posix_memalign((void **)&bias, bias_bytes, bias_bytes) == 0);
    memset(bias, 0, bias_bytes);
    CHECK(ds4_gpu_set_model_map(bias, bias_bytes));
    float *logits = malloc((size_t)ROWS * N * sizeof(float));
    int32_t tokens[ROWS] = {0};
    CHECK(logits);
    ds4_gpu_tensor *x = upload(NULL, ROWS * N * sizeof(float));
    ds4_gpu_tensor *tok = upload(tokens, sizeof(tokens));
    ds4_gpu_tensor *p[2], *ids[2], *w[2];
    for (int v = 0; v < 2; v++) {
        p[v] = upload(NULL, ROWS * N * sizeof(float));
        ids[v] = upload(NULL, ROWS * USED * sizeof(int32_t));
        w[v] = upload(NULL, ROWS * USED * sizeof(float));
        CHECK(p[v] && ids[v] && w[v]);
    }
    CHECK(x && tok);
    static const unsigned counts[] = {1, 2, 3, 5, 6, 8, 17, ROWS};
    static const char *const names[PATTERNS] = {
        "random", "all-equal", "four-values", "saturating", "tiny", "zero-bias-random"};
    int passes = 0;
    for (unsigned mode = 0; mode < PATTERNS; mode++) {
        for (unsigned e = 0; e < N; e++)
            bias[e] = mode == 5 ? 0.0f : mode == 1 ? 0.25f : (float)(e % 7) * 0.125f;
        for (unsigned i = 0; i < ROWS * N; i++)
            logits[i] = mode == 1 ? 0.0f : mode == 2 ? (float)(i % 4) - 1.5f :
                mode == 3 ? (i % 2 ? -80.0f : 40.0f) : mode == 4 ? random_value() * 1e-3f - 30.0f :
                random_value() * 8;
        CHECK(ds4_gpu_tensor_write(x, 0, logits, ROWS * N * sizeof(float)));
        for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const unsigned rows = counts[ci];
            for (int v = 0; v < 2; v++) {
                if (v) setenv("DS4_METAL_V41_FUSED_ROUTER", "0", 1);
                memset(ds4_gpu_tensor_contents(p[v]), 0xa5, ROWS * N * sizeof(float));
                memset(ds4_gpu_tensor_contents(ids[v]), 0xa5, ROWS * USED * sizeof(int32_t));
                memset(ds4_gpu_tensor_contents(w[v]), 0xa5, ROWS * USED * sizeof(float));
                const int ok = rows == 1 ?
                    ds4_gpu_router_select_tensor(ids[v], w[v], p[v], bias, bias_bytes,
                        0, 0, 0, 0, N, USED, 1.5f, 0, 0, true, false, x) :
                    ds4_gpu_router_select_batch_tensor(ids[v], w[v], p[v], bias, bias_bytes,
                        0, 0, 0, 0, 0, true, false, x, tok, N, USED, 1.5f, rows);
                unsetenv("DS4_METAL_V41_FUSED_ROUTER");
                CHECK(ok && ds4_gpu_synchronize());
            }
            const size_t pb = (size_t)rows * N * sizeof(float), sb = (size_t)rows * USED * 4u;
            const int same_p = memcmp(ds4_gpu_tensor_contents(p[0]), ds4_gpu_tensor_contents(p[1]), pb) == 0;
            const int same_i = memcmp(ds4_gpu_tensor_contents(ids[0]), ds4_gpu_tensor_contents(ids[1]), sb) == 0;
            const int same_w = memcmp(ds4_gpu_tensor_contents(w[0]), ds4_gpu_tensor_contents(w[1]), sb) == 0;
            if (!same_p || !same_i || !same_w) {
                const int32_t *a = ds4_gpu_tensor_contents(ids[0]), *b = ds4_gpu_tensor_contents(ids[1]);
                const float *wa = ds4_gpu_tensor_contents(w[0]), *wb = ds4_gpu_tensor_contents(w[1]);
                fprintf(stderr, "fused router %s rows=%u: probs %s ids %s weights %s; row0 ids %d,%d,%d vs %d,%d,%d w %.9g vs %.9g\n",
                    names[mode], rows, same_p ? "same" : "DIFF", same_i ? "same" : "DIFF",
                    same_w ? "same" : "DIFF", a[0], a[1], a[2], b[0], b[1], b[2], wa[0], wb[0]);
                CHECK(0);
            }
            passes++;
        }
        fprintf(stderr, "V4.1 fused router %s rows=1-8,17,129: bitwise PASS\n", names[mode]);
    }
    for (int v = 0; v < 2; v++) {
        ds4_gpu_tensor_free(p[v]); ds4_gpu_tensor_free(ids[v]); ds4_gpu_tensor_free(w[v]);
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(tok);
    free(logits);
    fprintf(stderr, "V4.1 fused router: %d bitwise cases PASS\n", passes);
    return 1;
}

static int check_fused_bf16(void) {
    enum { ROWS = 6 };
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const struct { uint32_t in, out; } shapes[] = {{5120, 1280}, {1280, 4096}, {2304, 5120}, {4096, 64}};
    const char *modes[] = {"auto", "off"};
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++) {
        const uint32_t in = shapes[sh].in, out_dim = shapes[sh].out;
        const uint64_t wb = (uint64_t)out_dim * in / 32 * sizeof(q8_block);
        const uint64_t xb = (uint64_t)ROWS * in * 4, ob = (uint64_t)ROWS * out_dim * 4;
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
        q8_block *w = model;
        for (uint64_t i = 0; i < wb / sizeof(*w); i++) {
            const int value = (int)(random_value() * 8192);
            w[i].d = (uint16_t)(0x3000u | ((unsigned)abs(value) % 1024u));
            for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 128);
        }
        float *x = malloc(xb), *plain = malloc(ob), *fused = malloc(ob);
        CHECK(x && plain && fused);
        for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
        x[(uint64_t)1 * in + 7] = NAN;
        for (uint32_t i = 0; i < in; i++) x[(uint64_t)3 * in + i] = 3.0e38f;
        ds4_gpu_tensor *xt = upload(x, xb), *ot = upload(NULL, ob);
        CHECK(xt && ot && ds4_gpu_set_model_map(model, wb));
        for (int mode = 0; mode < 2; mode++) {
            setenv("DS4_METAL_DECODE_ROWS_MODE", modes[mode], 1);
            for (uint32_t rows = 1; rows <= ROWS; rows++) {
                CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, ob / 4));
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(ot, model, wb, 0, in, out_dim, xt, rows));
                CHECK(ds4_gpu_dsv41_quantize(ot, out_dim, rows, DS4_V41_BF16));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_tensor_read(ot, 0, plain, ob));
                CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, ob / 4));
                int rounded = -1;
                CHECK(ds4_gpu_begin_commands());
                CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_bf16_tensor(ot, model, wb, 0, in, out_dim, xt, rows, &rounded));
                if (!rounded) CHECK(ds4_gpu_dsv41_quantize(ot, out_dim, rows, DS4_V41_BF16));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_tensor_read(ot, 0, fused, ob));
                CHECK(rounded == 0 || rounded == 1);
                if (memcmp(fused, plain, (uint64_t)rows * out_dim * 4))
                    fprintf(stderr, "fused bf16 mismatch %ux%u mode=%s rows=%u rounded=%d\n",
                            in, out_dim, modes[mode], rows, rounded);
                CHECK(!memcmp(fused, plain, (uint64_t)rows * out_dim * 4));
                for (uint64_t i = (uint64_t)rows * out_dim; i < ob / 4; i++) CHECK(isnan(fused[i]));
                /* Row tiles exist only for auto with several rows and a wide grid. */
                const bool tiled = mode == 0 && rows > 1 && out_dim >= 128 * 2;
                if (tiled != (rounded == 1))
                    fprintf(stderr, "fused bf16 %ux%u mode=%s rows=%u rounded=%d\n", in, out_dim, modes[mode], rows, rounded);
                CHECK(tiled == (rounded == 1));
                if (rows > 3) {
                    /* The NaN and overflow rows really reach non-finite outputs. */
                    bool nan_seen = false, nonfinite_seen = false;
                    for (uint32_t i = 0; i < out_dim; i++) {
                        nan_seen |= isnan(fused[(uint64_t)1 * out_dim + i]);
                        nonfinite_seen |= !isfinite(fused[(uint64_t)3 * out_dim + i]);
                    }
                    CHECK(nan_seen && nonfinite_seen);
                }
            }
            fprintf(stderr, "V4.1 fused BF16 Q8_0 decode rows %ux%u mode=%s rows 1..%u: PASS\n",
                    in, out_dim, modes[mode], ROWS);
        }
        unsetenv("DS4_METAL_DECODE_ROWS_MODE");
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
        free(x); free(plain); free(fused); free(model);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Weighted RMS norm with fused BF16 rounding against the norm plus the
 * separate BF16 pass: random weights and rows, widths of the V4.1 norms,
 * rows 1..6, a NaN row and a row of huge values (overflow to inf). */
static uint32_t as_bits(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static int check_fused_norm(void) {
    enum { ROWS = 6 };
    const uint32_t widths[] = {5120, 1280, 512, 4096, 128};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t n = widths[wi];
        const uint64_t xb = (uint64_t)ROWS * n * 4;
        /* Whole pages: the model map rejects a sub-page mapping. */
        const uint64_t wb = ((uint64_t)n * 4 + 16383u) & ~(uint64_t)16383u;
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
        float *w = model;
        for (uint32_t i = 0; i < n; i++) w[i] = random_value() * 4.0f;
        w[0] = 3.0e38f;
        float *x = malloc(xb), *plain = malloc(xb), *fused = malloc(xb);
        CHECK(x && plain && fused);
        for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
        x[(uint64_t)2 * n + 5] = NAN;
        x[0] = 50.0f; /* times w[0] = 3e38 overflows */
        ds4_gpu_tensor *xt = upload(x, xb), *ot = upload(NULL, xb);
        CHECK(xt && ot && ds4_gpu_set_model_map(model, wb));
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, xb / 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_rms_norm_weight_rows_tensor(ot, xt, model, wb, 0, n, rows, 1e-6f));
            CHECK(ds4_gpu_dsv41_quantize(ot, n, rows, DS4_V41_BF16));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, plain, xb));
            CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, xb / 4));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_rms_norm_weight_rows_bf16_tensor(ot, xt, model, wb, 0, n, rows, 1e-6f));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, fused, xb));
            if (memcmp(fused, plain, (uint64_t)rows * n * 4))
                fprintf(stderr, "fused norm mismatch n=%u rows=%u\n", n, rows);
            CHECK(!memcmp(fused, plain, (uint64_t)rows * n * 4));
            for (uint64_t i = (uint64_t)rows * n; i < xb / 4; i++) CHECK(isnan(fused[i]));
            /* Rounding happened, and the special rows stay non-finite. */
            bool changed = false;
            for (uint64_t i = 0; i < (uint64_t)rows * n; i++)
                changed |= isfinite(fused[i]) && (as_bits(fused[i]) & 0xffffu) != 0;
            CHECK(!changed);
            if (rows > 2) CHECK(isnan(fused[(uint64_t)2 * n]));
            CHECK(isinf(fused[0]));
        }
        fprintf(stderr, "V4.1 fused BF16 weighted RMS norm n=%u rows 1..%u: PASS\n", n, ROWS);
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
        free(x); free(plain); free(fused); free(model);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Fused BF16 rounding in the HC weighted sum (pre and split weights) and
 * the HC=4 split expand against the plain kernels plus the separate pass:
 * random activations and mixers with some huge values, rows 1..6. */
static int check_fused_hc(void) {
    enum { ROWS = 6, E = 5120, HC = 4, MIX = 2 * HC + HC * HC };
    const uint64_t hcb = (uint64_t)ROWS * HC * E * 4, eb = (uint64_t)ROWS * E * 4, mb = (uint64_t)ROWS * MIX * 4;
    float *hc = malloc(hcb), *blk = malloc(eb), *mix = malloc(mb), *plain = malloc(hcb), *fused = malloc(hcb);
    CHECK(hc && blk && mix && plain && fused);
    for (uint64_t i = 0; i < hcb / 4; i++) hc[i] = random_value() * 8.0f;
    for (uint64_t i = 0; i < eb / 4; i++) blk[i] = random_value() * 8.0f;
    for (uint64_t i = 0; i < mb / 4; i++) mix[i] = random_value();
    hc[3] = 3.0e38f; mix[0] = 4.0f; /* overflow in row 0 */
    ds4_gpu_tensor *hct = upload(hc, hcb), *bt = upload(blk, eb), *mt = upload(mix, mb);
    ds4_gpu_tensor *ot = upload(NULL, hcb);
    CHECK(hct && bt && mt && ot);
    for (int kind = 0; kind < 3; kind++) {
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            const uint64_t width = kind == 2 ? (uint64_t)HC * E : E;
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(ot, 0, rows * width * 4);
            CHECK(o);
            for (int f = 0; f < 2; f++) {
                CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, hcb / 4));
                CHECK(ds4_gpu_begin_commands());
                if (kind < 2 && !f) {
                    CHECK(kind ? ds4_gpu_hc_weighted_sum_split_tensor(o, hct, mt, E, HC) :
                                 ds4_gpu_hc_weighted_sum_tensor(o, hct, mt, E, HC));
                    CHECK(ds4_gpu_dsv41_quantize(o, E, rows, DS4_V41_BF16));
                } else if (kind < 2) {
                    CHECK(ds4_gpu_hc_weighted_sum_bf16_tensor(o, hct, mt, E, HC, kind == 1));
                } else if (!f) {
                    CHECK(ds4_gpu_hc_expand_split_tensor(o, bt, hct, mt, E, HC));
                    CHECK(ds4_gpu_dsv41_quantize(o, HC * E, rows, DS4_V41_BF16));
                } else {
                    CHECK(ds4_gpu_hc_expand_split_bf16_tensor(o, bt, hct, mt, E, HC));
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_tensor_read(ot, 0, f ? fused : plain, hcb));
            }
            if (memcmp(fused, plain, rows * width * 4))
                fprintf(stderr, "fused hc mismatch kind=%d rows=%u\n", kind, rows);
            CHECK(!memcmp(fused, plain, rows * width * 4));
            for (uint64_t i = rows * width; i < hcb / 4; i++) CHECK(isnan(fused[i]));
            bool unrounded = false, nonfinite = false;
            for (uint64_t i = 0; i < rows * width; i++) {
                uint32_t bits; memcpy(&bits, &fused[i], 4);
                unrounded |= isfinite(fused[i]) && (bits & 0xffffu);
                nonfinite |= !isfinite(fused[i]);
            }
            CHECK(!unrounded && nonfinite);
            ds4_gpu_tensor_free(o);
        }
        fprintf(stderr, "V4.1 fused BF16 HC %s rows 1..%u: PASS\n",
                kind == 0 ? "weighted sum" : kind == 1 ? "weighted sum split" : "expand split", ROWS);
    }
    CHECK(!ds4_gpu_hc_expand_split_bf16_tensor(ot, bt, hct, mt, E, 3));
    ds4_gpu_tensor_free(hct); ds4_gpu_tensor_free(bt); ds4_gpu_tensor_free(mt); ds4_gpu_tensor_free(ot);
    free(hc); free(blk); free(mix); free(plain); free(fused);
    return 1;
}
#endif

#ifdef __APPLE__
/* Fused BF16 rounding in F16/F32 matvec writers against the plain matvec
 * plus the separate pass: single row and decode-row plans, shapes covering
 * nr0 1/2/4 grids, a narrow grid, a K tail, F32 routers and a width the
 * float4 kernel cannot take (rounded must then be 0), rows 1..6. */
static int check_fused_f16(void) {
    enum { ROWS = 6 };
    const struct { uint32_t in, out; bool f32; } shapes[] = {
        {4096, 1024, false}, {5120, 24, false}, {1024, 5120, false}, {4100, 2048, false},
        {4098, 512, false}, {5120, 384, true}, {5124, 384, true}};
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++) {
        const uint32_t in = shapes[sh].in, out_dim = shapes[sh].out;
        const bool f32 = shapes[sh].f32;
        const uint64_t wbytes = (uint64_t)in * out_dim * (f32 ? 4 : 2);
        const uint64_t wb = (wbytes + 16383u) & ~(uint64_t)16383u;
        const uint64_t xb = (uint64_t)ROWS * in * 4, ob = (uint64_t)ROWS * out_dim * 4;
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
        if (f32) { float *w = model; for (uint64_t i = 0; i < wbytes / 4; i++) w[i] = random_value(); }
        else {
            uint16_t *w = model;
            for (uint64_t i = 0; i < wbytes / 2; i++) {
                const uint32_t v = (uint32_t)(random_value() * 8192) * 2654435761u + (uint32_t)i;
                w[i] = (uint16_t)((v & 0x8000u) | ((1u + (v >> 3) % 29u) << 10) | (v & 0x3ffu));
            }
        }
        float *x = malloc(xb), *plain = malloc(ob), *fused = malloc(ob);
        CHECK(x && plain && fused);
        for (uint64_t i = 0; i < xb / 4; i++) x[i] = random_value();
        ds4_gpu_tensor *xt = upload(x, xb), *ot = upload(NULL, ob);
        CHECK(xt && ot && ds4_gpu_set_model_map(model, wb));
        for (uint32_t rows = 1; rows <= ROWS; rows++) {
            ds4_gpu_tensor *o = ds4_gpu_tensor_view(ot, 0, (uint64_t)rows * out_dim * 4);
            CHECK(o);
            CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, ob / 4));
            CHECK(ds4_gpu_begin_commands());
            if (rows == 1) CHECK(f32 ? ds4_gpu_matmul_f32_tensor(o, model, wb, 0, in, out_dim, xt, 1) :
                                       ds4_gpu_matmul_f16_tensor(o, model, wb, 0, in, out_dim, xt, 1));
            else CHECK(f32 ? ds4_gpu_dsv41_projection_rows_f32(o, model, wb, 0, in, out_dim, rows, xt) :
                             ds4_gpu_dsv41_projection_rows(o, model, wb, 0, in, out_dim, rows, xt));
            CHECK(ds4_gpu_dsv41_quantize(o, out_dim, rows, DS4_V41_BF16));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, plain, ob));
            CHECK(ds4_gpu_tensor_fill_f32(ot, NAN, ob / 4));
            int rounded = -1;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_dsv41_projection_bf16(o, model, wb, 0, f32, in, out_dim, rows, xt, &rounded));
            if (!rounded) CHECK(ds4_gpu_dsv41_quantize(o, out_dim, rows, DS4_V41_BF16));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, fused, ob));
            if (memcmp(fused, plain, (uint64_t)rows * out_dim * 4) || rounded != (in % 4 == 0))
                fprintf(stderr, "fused f16 mismatch %ux%u f32=%d rows=%u rounded=%d\n", in, out_dim, f32, rows, rounded);
            CHECK(!memcmp(fused, plain, (uint64_t)rows * out_dim * 4));
            CHECK(rounded == (in % 4 == 0));
            for (uint64_t i = (uint64_t)rows * out_dim; i < ob / 4; i++) CHECK(isnan(fused[i]));
            ds4_gpu_tensor_free(o);
        }
        fprintf(stderr, "V4.1 fused BF16 %s matvec %ux%u rows 1..%u: PASS\n", f32 ? "F32" : "F16", in, out_dim, ROWS);
        ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
        free(x); free(plain); free(fused); free(model);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Fused BF16 rounding in the single-row Q8_0 matvec writer against the
 * plain matvec plus the separate pass, including the nsg=8 wide grid and
 * NaN / overflowing inputs. */
static int check_fused_q8_single(void) {
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const struct { uint32_t in, out; } shapes[] = {{5120, 1280}, {1280, 32768}, {2304, 5120}, {4096, 64}, {128, 70000}};
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(*shapes); sh++) {
        const uint32_t in = shapes[sh].in, out_dim = shapes[sh].out;
        const uint64_t wbytes = (uint64_t)out_dim * in / 32 * sizeof(q8_block);
        const uint64_t wb = (wbytes + 16383u) & ~(uint64_t)16383u, ob = (uint64_t)out_dim * 4;
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), wb) == 0);
        q8_block *w = model;
        for (uint64_t i = 0; i < wbytes / sizeof(*w); i++) {
            const int value = (int)(random_value() * 8192);
            w[i].d = (uint16_t)(0x3000u | ((unsigned)abs(value) % 1024u));
            for (int j = 0; j < 32; j++) w[i].qs[j] = (int8_t)((int)(random_value() * 8192) % 128);
        }
        for (int special = 0; special < 3; special++) {
            float *x = malloc((uint64_t)in * 4), *plain = malloc(ob), *fused = malloc(ob);
            CHECK(x && plain && fused);
            for (uint32_t i = 0; i < in; i++) x[i] = special == 2 ? 3.0e38f : random_value();
            if (special == 1) x[7] = NAN;
            ds4_gpu_tensor *xt = upload(x, (uint64_t)in * 4), *ot = upload(NULL, ob);
            CHECK(xt && ot && ds4_gpu_set_model_map(model, wb));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_matmul_q8_0_tensor(ot, model, wb, 0, in, out_dim, xt, 1));
            CHECK(ds4_gpu_dsv41_quantize(ot, out_dim, 1, DS4_V41_BF16));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, plain, ob));
            CHECK(ds4_gpu_tensor_fill_f32(ot, 0, ob / 4));
            int rounded = -1;
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_matmul_q8_0_bf16_tensor(ot, model, wb, 0, in, out_dim, xt, &rounded));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(ot, 0, fused, ob));
            if (memcmp(fused, plain, ob) || rounded != 1)
                fprintf(stderr, "fused q8 single mismatch %ux%u special=%d rounded=%d\n", in, out_dim, special, rounded);
            CHECK(rounded == 1 && !memcmp(fused, plain, ob));
            bool nonfinite = false;
            for (uint32_t i = 0; i < out_dim; i++) nonfinite |= !isfinite(fused[i]);
            CHECK(nonfinite == (special != 0));
            ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(ot);
            free(x); free(plain); free(fused);
        }
        fprintf(stderr, "V4.1 fused BF16 single-row Q8_0 %ux%u: PASS\n", in, out_dim);
        free(model);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Fused BF16 rounding in swiglu (with and without the clamp) and in the
 * two-input add, against the plain kernels plus the separate pass; inputs
 * span the clamp, NaN and overflow. */
static int check_fused_ew(void) {
    const uint32_t sizes[] = {2304, 5 * 2304, 5120, 6 * 5120, 37};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(*sizes); si++) {
        const uint32_t n = sizes[si];
        const uint64_t b = (uint64_t)n * 4;
        float *a = malloc(b), *c = malloc(b), *plain = malloc(b), *fused = malloc(b);
        CHECK(a && c && plain && fused);
        for (uint32_t i = 0; i < n; i++) { a[i] = random_value() * 40.0f; c[i] = random_value() * 40.0f; }
        a[1] = NAN; a[2] = 3.0e38f; c[2] = 3.0e38f; c[3] = -INFINITY;
        ds4_gpu_tensor *at = upload(a, b), *ct = upload(c, b), *ot = upload(NULL, b);
        CHECK(at && ct && ot);
        for (int kind = 0; kind < 3; kind++) {
            const float limit = kind == 1 ? 10.0f : 0.0f;
            for (int f = 0; f < 2; f++) {
                CHECK(ds4_gpu_tensor_fill_f32(ot, 0, n));
                CHECK(ds4_gpu_begin_commands());
                if (kind < 2) CHECK(f ? ds4_gpu_swiglu_bf16_tensor(ot, at, ct, n, limit, 1.0f) :
                                        ds4_gpu_swiglu_tensor(ot, at, ct, n, limit, 1.0f));
                else CHECK(f ? ds4_gpu_add_bf16_tensor(ot, at, ct, n) : ds4_gpu_add_tensor(ot, at, ct, n));
                if (!f) CHECK(ds4_gpu_dsv41_quantize(ot, n, 1, DS4_V41_BF16));
                CHECK(ds4_gpu_end_commands());
                CHECK(ds4_gpu_tensor_read(ot, 0, f ? fused : plain, b));
            }
            if (memcmp(fused, plain, b)) fprintf(stderr, "fused ew mismatch kind=%d n=%u\n", kind, n);
            CHECK(!memcmp(fused, plain, b));
            bool nonfinite = false;
            for (uint32_t i = 0; i < n; i++) nonfinite |= !isfinite(fused[i]);
            /* Fast-math swiglu need not propagate NaN; the add must. */
            CHECK(kind < 2 || nonfinite);
        }
        ds4_gpu_tensor_free(at); ds4_gpu_tensor_free(ct); ds4_gpu_tensor_free(ot);
        free(a); free(c); free(plain); free(fused);
        fprintf(stderr, "V4.1 fused BF16 swiglu/clamped swiglu/add n=%u: PASS\n", n);
    }
    return 1;
}
#endif

#ifdef __APPLE__
/* Sessions sharing one workspace inherit each other's batch block masks. The
 * layer-20 candidate step (fill or top-k mask) must write every mask word the
 * later candidate filter reads for that row and position, so stale words
 * from another session (masked blocks of a longer prompt) cannot change the
 * filtered scores. Mirrors ds41_attention_candidates at ratio 1. */
static int check_stale_mask(void) {
    const uint32_t ctx = 65536, stride = (ctx + 7u) / 8u;
    const uint32_t positions[] = {0, 7, 100, 5000, 16382, 16383, 16500, 40000, 65535};
    float *scores = malloc((size_t)ctx * 4), *clean = malloc((size_t)ctx * 4),
          *dirty = malloc((size_t)ctx * 4), *fill = malloc((size_t)stride * 4);
    CHECK(scores && clean && dirty && fill);
    ds4_gpu_tensor *mask = upload(NULL, (uint64_t)stride * 4),
        *block_scores = upload(NULL, (uint64_t)stride * 4),
        *selected = upload(NULL, 2048u * 4u), *row = upload(NULL, (uint64_t)ctx * 4);
    CHECK(mask && block_scores && selected && row);
    for (size_t pi = 0; pi < sizeof(positions) / sizeof(*positions); pi++) {
        const uint32_t pos = positions[pi], n_comp = pos + 1u;
        const uint32_t blocks = (n_comp + 7u) / 8u, top = blocks < 2048u ? blocks : 2048u;
        for (uint32_t i = 0; i < n_comp; i++) scores[i] = random_value();
        /* Stale patterns: all blocks masked, and a mix of kept and masked. */
        for (int stale = 0; stale < 3; stale++) {
            for (uint32_t i = 0; i < stride; i++)
                fill[i] = stale == 1 || (stale == 2 && i % 3u) ? -INFINITY : 0.0f;
            CHECK(ds4_gpu_tensor_write(mask, 0, fill, (uint64_t)stride * 4));
            CHECK(ds4_gpu_tensor_write(row, 0, scores, (uint64_t)n_comp * 4));
            CHECK(ds4_gpu_begin_commands());
            if (top == blocks) CHECK(ds4_gpu_tensor_fill_f32(mask, 0, blocks));
            else CHECK(ds4_gpu_dsv41_candidate_blocks(block_scores, row, n_comp, 1, pos, 1) &&
                       ds4_gpu_indexer_topk_tensor(selected, block_scores, blocks, 1, top) &&
                       ds4_gpu_dsv4_topk_mask_tensor(mask, selected, blocks, 1, top));
            CHECK(ds4_gpu_dsv41_candidate_filter(row, mask, n_comp, 1, pos, 1));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(row, 0, stale ? dirty : clean, (uint64_t)n_comp * 4));
            if (stale && memcmp(clean, dirty, (size_t)n_comp * 4))
                fprintf(stderr, "stale block mask changed candidates at pos=%u pattern=%d\n", pos, stale);
            CHECK(!stale || !memcmp(clean, dirty, (size_t)n_comp * 4));
        }
        uint32_t kept = 0;
        for (uint32_t i = 0; i < n_comp; i++) kept += isfinite(clean[i]) != 0;
        CHECK(top < blocks ? kept < n_comp : kept == n_comp);
        fprintf(stderr, "V4.1 stale block mask pos=%u (%s, %u kept): PASS\n", pos,
                top == blocks ? "fill" : "top-k", kept);
    }
    ds4_gpu_tensor_free(mask); ds4_gpu_tensor_free(block_scores);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(row);
    free(scores); free(clean); free(dirty); free(fill);
    return 1;
}
#endif

static int check_attention_output(bool large) {
    enum { GROUP = 4096, RANK = 1024, GROUPS = 8, OUT = 5120 };
    const uint32_t ROWS = large ? 8192u : 513u;
    typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
    const uint64_t a_bytes = (uint64_t)GROUPS * RANK * GROUP / 32 * sizeof(q8_block);
    const uint64_t b_bytes = (uint64_t)OUT * GROUPS * RANK / 32 * sizeof(q8_block);
    const uint64_t xb = (uint64_t)ROWS * GROUPS * GROUP * 4;
    const uint64_t lb = (uint64_t)ROWS * GROUPS * RANK * 4, ob = (uint64_t)ROWS * OUT * 4;
    void *model = NULL;
    CHECK(posix_memalign(&model, getpagesize(), a_bytes + b_bytes) == 0);
    q8_block *w = model;
    for (uint64_t i = 0; i < (a_bytes + b_bytes) / sizeof(*w); i++) {
        w[i].d = 0x2000; /* Exact 1/128 scale and binary-fraction inputs. */
        for (int j = 0; j < 32; j++) w[i].qs[j] = (int)(random_value() * 8192) % 8;
    }
    float *x = malloc(xb), *reference = malloc(ob), *reference_low = malloc(lb);
    CHECK(x && reference && reference_low);
    for (uint64_t i = 0; i < xb / 4; i++) x[i] = (int)(random_value() * 8192) % 51 / 256.0f;
    ds4_gpu_tensor *xt = upload(x, xb), *low = upload(NULL, lb), *out = upload(NULL, ob);
    CHECK(xt && low && out && ds4_gpu_set_model_map(model, a_bytes + b_bytes));
    CHECK(ds4_gpu_begin_commands());
    for (uint32_t r = 0; r < ROWS; r++) {
        ds4_gpu_tensor *xr = ds4_gpu_tensor_view(xt, (uint64_t)r * GROUPS * GROUP * 4,
                                                GROUPS * GROUP * 4);
        ds4_gpu_tensor *lr = ds4_gpu_tensor_view(low, (uint64_t)r * GROUPS * RANK * 4,
                                                GROUPS * RANK * 4);
        ds4_gpu_tensor *yr = ds4_gpu_tensor_view(out, (uint64_t)r * OUT * 4, OUT * 4);
        CHECK(xr && lr && yr);
        CHECK(ds4_gpu_attention_output_low_q8_tensor(lr, model, a_bytes + b_bytes,
            0, GROUP, RANK, GROUPS, xr));
        CHECK(ds4_gpu_dsv41_quantize(lr, GROUPS * RANK, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(yr, model, a_bytes + b_bytes,
            a_bytes, GROUPS * RANK, OUT, lr, 1));
        ds4_gpu_tensor_free(xr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(yr);
    }
    CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(out, 0, reference, ob));
    CHECK(ds4_gpu_tensor_read(low, 0, reference_low, lb));
    const uint32_t sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 63, 64, 65, 257, 512, 513, 8191, 8192};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        const uint32_t rows = sizes[n];
        if (rows > ROWS) break;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, ob / 4));
        CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
        CHECK(ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
            0, a_bytes, xt, rows));
        const float *got = ds4_gpu_tensor_contents(out);
        const float *got_low = ds4_gpu_tensor_contents(low);
        CHECK(!memcmp(got_low, reference_low, (uint64_t)rows * GROUPS * RANK * 4));
        for (uint64_t i = 0; i < (uint64_t)rows * OUT; i++) {
            if (!isfinite(got[i]) || fabsf(got[i] - reference[i]) > 2e-5f * (1 + fabsf(reference[i])))
                fprintf(stderr, "attention output rows=%u index=%llu actual=%.9g reference=%.9g\n",
                    rows, (unsigned long long)i, got[i], reference[i]);
            CHECK(isfinite(got[i]) && fabsf(got[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        }
        if (rows < ROWS) CHECK(isnan(got[(uint64_t)rows * OUT]));
        fprintf(stderr, "V4.1 batched Q8 output, BF16 boundary, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, xt, ROWS));
    CHECK(!ds4_gpu_dsv41_attention_output_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, xt, ROWS + 1));
#ifndef DS4_ROCM_BUILD
    ds4_gpu_tensor *packed = upload(NULL, xb / 2);
    ds4_gpu_tensor *partial[2] = {upload(NULL, ob), upload(NULL, ob)};
    CHECK(packed && partial[0] && partial[1]);
    const uint32_t tp_sizes[] = {1, 2, 3, 5, 6, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 512, 513};
    const uint32_t q_half = GROUPS / 2 * GROUP, low_half = GROUPS / 2 * RANK;
    for (size_t n = 0; n < sizeof(tp_sizes) / sizeof(*tp_sizes); n++) {
        const uint32_t rows = tp_sizes[n];
        for (uint32_t rank = 0; rank < 2; rank++) {
            float *px = ds4_gpu_tensor_contents(packed);
            for (uint32_t t = 0; t < rows; t++)
                memcpy(px + (size_t)t * q_half, x + (size_t)t * 2 * q_half + rank * q_half,
                       q_half * sizeof(float));
            CHECK(ds4_gpu_tensor_fill_f32(partial[rank], NAN, ob / 4));
            CHECK(ds4_gpu_tensor_fill_f32(low, NAN, lb / 4));
            CHECK(ds4_gpu_dsv41_attention_output_tp_batch(partial[rank], low,
                model, a_bytes + b_bytes, 0, a_bytes, packed, rows, rank));
            const float *got_low = ds4_gpu_tensor_contents(low);
            const float *got = ds4_gpu_tensor_contents(partial[rank]);
            for (uint32_t t = 0; t < rows; t++) {
                CHECK(!memcmp(got_low + (size_t)t * low_half,
                    reference_low + (size_t)t * 2 * low_half + rank * low_half,
                    low_half * sizeof(float)));
                /* Independent double sums sample both ends and interior output
                 * rows, including rank one's strided input slice. */
                const uint32_t columns[] = {0, 1, 31, 64, 997, 4095, OUT - 1};
                for (size_t c = 0; c < sizeof(columns) / sizeof(*columns); c++) {
                    const uint32_t col = columns[c];
                    const q8_block *bw = (const q8_block *)((const char *)model + a_bytes) +
                        (size_t)col * 2 * low_half / 32 + rank * low_half / 32;
                    double sum = 0;
                    double rounding_bound = 0;
#ifndef __APPLE__
                    /* CUDA's established Q8 dot first rounds each activation
                     * block to signed bytes. TP must keep that same boundary. */
                    for (uint32_t k = 0; k < low_half; k += 32) {
                        const float *v = reference_low + (size_t)t * 2 * low_half + rank * low_half + k;
                        float amax = 0;
                        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
                        const float scale = amax / 127.0f;
                        const float inv = scale ? 1.0f / scale : 0;
                        int dot = 0;
                        for (uint32_t j = 0; j < 32; j++) {
                            dot += bw[k / 32].qs[j] * (int)lrintf(v[j] * inv);
                            /* The fast GPU reciprocal can approach an exact
                             * half-integer from the other side. Bound only
                             * those ambiguous bins, not arbitrary dot error. */
                            const double q = amax ? fabs((double)v[j] * 127.0 / amax) : 0;
                            if (q - floor(q) == 0.5)
                                rounding_bound += abs(bw[k / 32].qs[j]) * (double)scale / 128.0;
                        }
                        sum += dot * (double)scale / 128.0;
                    }
#else
                    for (uint32_t k = 0; k < low_half; k++)
                        sum += bw[k / 32].qs[k % 32] / 128.0 *
                            reference_low[(size_t)t * 2 * low_half + rank * low_half + k];
#endif
                    if (!isfinite(got[(size_t)t * OUT + col]) ||
                        fabs(got[(size_t)t * OUT + col] - sum) > rounding_bound + 2e-5 * (1 + fabs(sum)))
                        fprintf(stderr, "TP output rank=%u rows=%u row=%u col=%u actual=%.9g oracle=%.9g tie_bound=%.9g\n",
                            rank, rows, t, col, got[(size_t)t * OUT + col], sum, rounding_bound);
                    CHECK(isfinite(got[(size_t)t * OUT + col]) &&
                        fabs(got[(size_t)t * OUT + col] - sum) <= rounding_bound + 2e-5 * (1 + fabs(sum)));
                }
            }
            CHECK(isnan(got_low[(size_t)rows * low_half]));
            if (rows < ROWS) CHECK(isnan(got[(size_t)rows * OUT]));
        }
        const float *a = ds4_gpu_tensor_contents(partial[0]);
        const float *b = ds4_gpu_tensor_contents(partial[1]);
        for (size_t i = 0; i < (size_t)rows * OUT; i++)
            CHECK(isfinite(a[i]) && isfinite(b[i]) &&
                fabsf(a[i] + b[i] - reference[i]) <= 2e-5f * (1 + fabsf(reference[i])));
        fprintf(stderr, "V4.1 TP Q8 output, both ranks, BF16 and double oracle, rows=%u: PASS\n", rows);
    }
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, 1, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes - 1,
        0, a_bytes, packed, 1, 1));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(out, low, model, a_bytes + b_bytes,
        0, a_bytes, packed, ROWS + 1, 0));
    ds4_gpu_tensor_free(partial[0]); ds4_gpu_tensor_free(partial[1]); ds4_gpu_tensor_free(packed);
#else
    fprintf(stderr, "V4.1 TP output projection: skipped (ROCm has no network TP)\n");
#endif
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();
    free(x); free(reference); free(reference_low); free(model);
    return 1;
}

static int check_indexer_batch(void) {
    enum { KEYS = 1025, ROWS = 33, HEADS = 32, DIM = 128 };
    ds4_gpu_tensor *qt = upload(NULL, (size_t)ROWS * HEADS * DIM * sizeof(float));
    ds4_gpu_tensor *kt = upload(NULL, (size_t)KEYS * DIM * sizeof(float));
    ds4_gpu_tensor *wt = upload(NULL, (size_t)ROWS * HEADS * sizeof(float));
    ds4_gpu_tensor *st = upload(NULL, (size_t)(ROWS * KEYS + 1) * sizeof(float));
    const uint64_t packed_bytes = ds4_gpu_dsv41_indexer_packed_bytes(KEYS, ROWS);
    ds4_gpu_tensor *pt = upload(NULL, (size_t)packed_bytes + 4);
    CHECK(qt && kt && wt && st && pt);
#ifndef __APPLE__
    ds4_gpu_tensor *reference = upload(NULL, (size_t)KEYS * sizeof(float));
    CHECK(reference);
#endif
    float *q = ds4_gpu_tensor_contents(qt), *k = ds4_gpu_tensor_contents(kt);
    float *w = ds4_gpu_tensor_contents(wt), *s = ds4_gpu_tensor_contents(st);
    CHECK(q && k && w && s);
    const bool packed_available = ds4_gpu_dsv41_tensor_ops_available();
    for (uint32_t range = 0; range < 3; range++) {
        for (size_t i = 0; i < (size_t)ROWS * HEADS * DIM; i++)
            q[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)KEYS * DIM; i++)
            k[i] = ldexpf(random_value(), range ? (int)((i / 32) % 41) - 20 : 0);
        for (size_t i = 0; i < (size_t)ROWS * HEADS; i++) w[i] = bf16(random_value());
        CHECK(ds4_gpu_dsv41_quantize(qt, DIM, ROWS * HEADS, DS4_V41_FP4_E8M0));
        CHECK(ds4_gpu_dsv41_quantize(kt, DIM, KEYS, DS4_V41_FP4_E8M0));
        if (range == 2) { q[0] = 1.0001f; k[67u * DIM] = 1.0003f; }
        uint32_t *packed_guard = (uint32_t *)((char *)ds4_gpu_tensor_contents(pt) + packed_bytes);
        *packed_guard = 0xabcdef01;
        if (packed_available) CHECK(ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS));
        CHECK(*packed_guard == 0xabcdef01);
        const uint32_t counts[] = {1, 7, 8, 9, 31, 32, 33};
        for (uint32_t ratio = 1; ratio <= 2; ratio++) {
            for (uint32_t early = 0; early <= 1; early++) {
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    const uint32_t start = early ? 0 : KEYS * ratio - rows;
                    for (uint32_t mode = 0; mode < (packed_available ? 3u : 1u); mode++) {
                        const uint32_t offset = mode == 2 && rows < ROWS ? 1u : 0u;
                        ds4_gpu_tensor *qv = ds4_gpu_tensor_view(qt, (uint64_t)offset * HEADS * DIM * 4u,
                            (uint64_t)rows * HEADS * DIM * 4u);
                        ds4_gpu_tensor *wv = ds4_gpu_tensor_view(wt, (uint64_t)offset * HEADS * 4u,
                            (uint64_t)rows * HEADS * 4u);
                        CHECK(qv && wv);
                        s[(size_t)rows * KEYS] = 12345;
                        CHECK(mode ? ds4_gpu_dsv41_indexer_scores_packed(st, qv, wv, kt, pt,
                            KEYS, rows, start, ratio, ROWS, offset) :
                            ds4_gpu_dsv41_indexer_scores_batch(st, qv, wv, kt, KEYS, rows, start, ratio));
                        CHECK(ds4_gpu_synchronize());
                        ds4_gpu_tensor_free(qv); ds4_gpu_tensor_free(wv);
                        CHECK(s[(size_t)rows * KEYS] == 12345);
                        CHECK(*packed_guard == 0xabcdef01);
#ifndef __APPLE__
                        /* Compare the CUDA warp scorer with the original
                         * shared-memory reduction, not only a loose oracle. */
                        ds4_gpu_set_quality(true);
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            if (!visible) continue;
                            ds4_gpu_tensor *qr = ds4_gpu_tensor_view(qt,
                                (uint64_t)t * HEADS * DIM * 4u, HEADS * DIM * 4u);
                            ds4_gpu_tensor *wr = ds4_gpu_tensor_view(wt,
                                (uint64_t)t * HEADS * 4u, HEADS * 4u);
                            CHECK(qr && wr && ds4_gpu_glm_indexer_score_one_tensor(
                                reference, qr, wr, kt, visible, HEADS, DIM, 1.0f / 64.0f, false));
                            CHECK(ds4_gpu_synchronize());
                            CHECK(!memcmp(s + (size_t)t * KEYS,
                                ds4_gpu_tensor_contents(reference), visible * sizeof(float)));
                            ds4_gpu_tensor_free(qr); ds4_gpu_tensor_free(wr);
                        }
                        ds4_gpu_set_quality(false);
#endif
                        double worst = 0;
                        for (uint32_t t = 0; t < rows; t++) {
                            const uint32_t visible = (start + t + 1u) / ratio;
                            for (uint32_t j = 0; j < KEYS; j++) {
                                const float actual = s[(size_t)t * KEYS + j];
                                if (j >= visible) { CHECK(actual == -INFINITY); continue; }
                                double expected = 0, magnitude = 0;
                                for (uint32_t h = 0; h < HEADS; h++) {
                                    double dot = 0;
                                    for (uint32_t d = 0; d < DIM; d++)
                                        dot += (double)q[((size_t)(t + offset) * HEADS + h) * DIM + d] *
                                               k[(size_t)j * DIM + d];
                                    const double term = fmax(dot / 64.0, 0) * w[(t + offset) * HEADS + h];
                                    expected += term;
                                    magnitude += fabs(term);
                                }
                                CHECK(isfinite(actual));
                                const double error = fabs(actual - expected) / fmax(magnitude, 1);
                                worst = fmax(worst, error);
                                CHECK(error < 0.00001);
                            }
                        }
                        fprintf(stderr, "V4.1 index scores mode=%u range=%u ratio=%u start=%u rows=%u error=%.9g\n",
                                mode, range, ratio, start, rows, worst);
                    }
                }
            }
        }
    }
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, 0, 4));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, UINT32_MAX, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS + 1, 0, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_batch(st, qt, wt, kt, KEYS, ROWS, KEYS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_pack(st, qt, kt, KEYS, ROWS));
    CHECK(!ds4_gpu_dsv41_indexer_pack(pt, qt, kt, KEYS, ROWS + 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 0, ROWS, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, ROWS, 1));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, pt,
        KEYS, ROWS, 0, 1, UINT32_MAX, 0));
    CHECK(!ds4_gpu_dsv41_indexer_scores_packed(st, qt, wt, kt, st,
        KEYS, ROWS, 0, 1, ROWS, 0));
    ds4_gpu_tensor_free(pt);
#ifndef __APPLE__
    ds4_gpu_tensor_free(reference);
#endif
    ds4_gpu_tensor_free(st); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(kt); ds4_gpu_tensor_free(qt);
    fprintf(stderr, "V4.1 causal batched index scores: double-precision oracle PASS\n");
    return 1;
}

static double monotonic_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int check_embedding(void) {
    enum { VOCAB = 19, WIDTH = 5120, HC = 4, ROWS = 257 };
    const size_t row_bytes = WIDTH * HC * sizeof(float);
    const size_t weight_bytes = VOCAB * WIDTH * sizeof(uint16_t);
    uint16_t *model = NULL;
    CHECK(posix_memalign((void **)&model, getpagesize(), weight_bytes) == 0);
    for (unsigned i = 0; i < VOCAB * WIDTH; i++)
        model[i] = (uint16_t)(0x2000u | (i % 1024u) | ((i & 1u) ? 0x8000u : 0));
    int32_t ids[ROWS];
    for (unsigned i = 0; i < ROWS; i++) ids[i] = (int32_t)((i * 7u) % VOCAB);
    CHECK(ds4_gpu_set_model_map(model, weight_bytes));
    ds4_gpu_tensor *tokens = upload(ids, sizeof(ids));
    ds4_gpu_tensor *out = upload(NULL, ROWS * row_bytes + sizeof(float));
    ds4_gpu_tensor *ref = upload(NULL, VOCAB * row_bytes);
    CHECK(tokens && out && ref);
    for (unsigned token = 0; token < VOCAB; token++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(ref, token * row_bytes, row_bytes);
        CHECK(row && ds4_gpu_embed_token_hc_tensor(row, model, weight_bytes, 0,
                                                   VOCAB, token, WIDTH, HC));
        ds4_gpu_tensor_free(row);
    }
    const unsigned counts[] = {1, 7, 127, 128, ROWS};
    const float marker = 12345;
    float *actual = malloc(ROWS * row_bytes + sizeof(float));
    float *expected = malloc(VOCAB * row_bytes);
    CHECK(actual && expected && ds4_gpu_tensor_read(ref, 0, expected, VOCAB * row_bytes));
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) {
        const unsigned rows = counts[i];
        CHECK(ds4_gpu_tensor_write(out, rows * row_bytes, &marker, sizeof(marker)));
        CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tokens, model, weight_bytes, 0,
                                             VOCAB, rows, WIDTH, HC));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, rows * row_bytes + sizeof(float)));
        CHECK(actual[rows * WIDTH * HC] == marker);
        for (unsigned row = 0; row < rows; row++) {
            CHECK(!memcmp(actual + row * WIDTH * HC, expected + ids[row] * WIDTH * HC, row_bytes));
            for (unsigned hc = 0; hc < HC; hc++) for (unsigned d = 0; d < WIDTH; d++) {
                const unsigned w = (unsigned)ids[row] * WIDTH + d;
                const float value = (1.0f + (w % 1024u) / 1024.0f) / 128.0f * ((w & 1u) ? -1 : 1);
                CHECK(actual[(row * HC + hc) * WIDTH + d] == value);
            }
        }
        fprintf(stderr, "V4.1 batched embedding rows=%u: exact scalar and independent oracle PASS\n", rows);
    }
    free(expected); free(actual);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(tokens);
    ds4_gpu_cleanup(); free(model);
    CHECK(ds4_gpu_init());
    return 1;
}

static int check_index_projection(void) {
    enum { MAX_ROWS = 4096 };
    const uint32_t widths[] = {1280, 5120, 5120, 512}, outputs[] = {4096, 32, 512, 128};
    const uint32_t counts[] = {1, 31, 64, 65, 513, 2048, MAX_ROWS};
    for (uint32_t shape = 0; shape < 4; shape++) {
        const uint32_t width = widths[shape], output = outputs[shape];
        const size_t weight_bytes = (size_t)width * output * sizeof(uint16_t);
        void *model = NULL;
        CHECK(posix_memalign(&model, getpagesize(), weight_bytes) == 0);
        uint16_t *weights = model;
        for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
            const int value = (int)(random_value() * 8192);
            weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                    (value < 0 ? 0x8000u : 0));
        }
        CHECK(ds4_gpu_set_model_map(model, weight_bytes));
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());
        for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t rows = counts[ci];
            y[(size_t)rows * output] = 12345;
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < rows; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr && ds4_gpu_matmul_f16_tensor(yr, model, weight_bytes, 0, width, output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands());
            const double start = monotonic_seconds();
            CHECK(ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, rows, in));
            CHECK(ds4_gpu_synchronize());
            const double seconds = monotonic_seconds() - start;
            CHECK(y[(size_t)rows * output] == 12345);
            double error = 0, norm = 0;
            for (size_t i = 0; i < (size_t)rows * output; i++) {
                CHECK(isfinite(y[i]) && isfinite(expected[i]));
                const double d = (double)y[i] - expected[i];
                error += d * d; norm += (double)expected[i] * expected[i];
            }
            const double relative = sqrt(error / fmax(norm, 1e-30));
            fprintf(stderr, "V4.1 exact F16 projection width=%u output=%u rows=%u RMS=%.9g %.3f ms\n",
                width, output, rows, relative, seconds * 1000);
            CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            for (uint32_t j = 0; j < 8; j++) {
                const uint32_t o = j * (output / 8), t = rows - 1;
                double sum = 0, magnitude = 0;
                for (uint32_t k = 0; k < width; k++) {
                    const uint16_t w = weights[(size_t)o * width + k];
                    const float value = (1.0f + (w & 1023) / 1024.0f) / 128.0f *
                                        (w & 0x8000 ? -1 : 1);
                    const double term = (double)value * x[(size_t)t * width + k];
                    sum += term; magnitude += fabs(term);
                }
                CHECK(fabs(y[(size_t)t * output + o] - sum) < 0.000001 * fmax(magnitude, 1));
            }
        }
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes - 1, 0, width, output, 1, in));
        CHECK(!ds4_gpu_dsv41_projection_rows(out, model, weight_bytes, 0, width, output, MAX_ROWS + 1, in));
        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

/* Exact decode-row Q8_0 and F16 matvecs have several grid plans
 * (DS4_METAL_DECODE_ROWS_MODE and the row-tile caps); every plan must equal
 * the single-row kernel on each row bit for bit.  Bench mode times the plans
 * with weights cold and warm.  Cold rotates the calls through about 1 GiB of
 * weight copies, so each call reads its weights from DRAM as in decode, where
 * gigabytes of other weights stream between two uses of a matrix.  Warm
 * reuses one copy, which stays in the system-level cache. */
typedef struct { uint32_t width, outputs; int q8; } decode_rows_shape;
typedef struct { const char *mode, *cap, *label; } decode_rows_mode;

static size_t decode_rows_weight_bytes(const decode_rows_shape *s) {
    return s->q8 ? (size_t)s->outputs * (s->width / 32u) * 34u
                 : (size_t)s->outputs * s->width * sizeof(uint16_t);
}

static void decode_rows_set_mode(const decode_rows_shape *s, const decode_rows_mode *m) {
    const char *knob = s->q8 ? "DS4_METAL_Q8_ROWS_NR1" : "DS4_METAL_F16_ROWS_NR1";
    if (m) setenv("DS4_METAL_DECODE_ROWS_MODE", m->mode, 1);
    else unsetenv("DS4_METAL_DECODE_ROWS_MODE");
    if (m && m->cap) setenv(knob, m->cap, 1);
    else unsetenv(knob);
}

static int decode_rows_call(ds4_gpu_tensor *out, const void *model, size_t model_bytes,
                            size_t offset, const decode_rows_shape *s,
                            ds4_gpu_tensor *in, uint32_t rows) {
    if (s->q8)
        return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model, model_bytes, offset,
            s->width, s->outputs, in, rows);
    return ds4_gpu_dsv41_projection_rows(out, model, model_bytes, offset, s->width,
        s->outputs, rows, in);
}

/* Best of three batched submissions of iters calls, in ms per call. */
static double decode_rows_time(ds4_gpu_tensor *out, const void *model, size_t model_bytes,
                               size_t stride, uint32_t copies, const decode_rows_shape *s,
                               ds4_gpu_tensor *in, uint32_t rows, uint32_t iters) {
    double best = INFINITY;
    for (int rep = 0; rep < 3; rep++) {
        const double start = monotonic_seconds();
        if (!ds4_gpu_begin_commands()) return -1;
        for (uint32_t i = 0; i < iters; i++) {
            if (!decode_rows_call(out, model, model_bytes, (size_t)(i % copies) * stride,
                                  s, in, rows))
                return -1;
        }
        if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return -1;
        const double ms = (monotonic_seconds() - start) * 1000.0 / iters;
        if (ms < best) best = ms;
    }
    return best;
}

static int check_decode_rows(int bench) {
    enum { MAX_ROWS = 16 };
    static const decode_rows_shape shapes[] = {
        /* Q8_0: V4.1 attn_kv, q_a, shared experts, attn_output_a/b, q_b. */
        {5120, 512, 1}, {5120, 1024, 1}, {5120, 1280, 1}, {5120, 2304, 1},
        {2304, 5120, 1}, {4096, 4096, 1}, {4096, 8192, 1}, {8192, 5120, 1},
        {1280, 32768, 1},
        /* F16: engram_kv, hc_*_fn, indexer q_b/proj/k, compressor; 4100 runs
         * the scalar tail loop. */
        {6144, 25600, 0}, {20480, 24, 0}, {1280, 4096, 0}, {5120, 32, 0},
        {512, 128, 0}, {5120, 512, 0}, {4096, 1024, 0}, {4096, 4096, 0},
        {4100, 1024, 0},
    };
    static const uint32_t counts[] = {1, 2, 3, 5, 6, 7, 8, 13, MAX_ROWS};
    /* Every compiled kernel: row tiles 2-8 through the caps, 2-4 interleaved. */
    static const decode_rows_mode modes[] = {
        {"auto", NULL, "auto"}, {"off", NULL, "off"}, {"rows", "4", "rows4"},
        {"rows", "8", "rows8"}, {"interleave", NULL, "interleave"},
        {"interleave-rows", "4", "irows4"}, {"interleave-rows", "2", "irows2"},
    };
    static const decode_rows_mode bench_modes[] = {
        {"off", NULL, "off"}, {"rows", "8", "rows8"}, {"interleave", NULL, "interleave"},
        {"interleave-rows", "2", "irows2"}, {"auto", NULL, "auto"},
    };
    enum { BENCH_MODES = sizeof(bench_modes) / sizeof(*bench_modes) };
    static const uint32_t bench_counts[] = {2, 3, 5, 6, 8};
    /* DS4_METAL_PLAIN_MV_NR0=1 selects the one-output-row F16 tile; =2 the
     * two-row tile, or four rows for 512/1024 outputs from >= 4096 inputs. */
    static const char *const f16_nr0[] = {"1", "2"};

    for (size_t si = 0; si < sizeof(shapes) / sizeof(*shapes); si++) {
        const decode_rows_shape *s = &shapes[si];
        const size_t weight_bytes = decode_rows_weight_bytes(s);
        const size_t page = (size_t)getpagesize();
        const size_t stride = (weight_bytes + page - 1) / page * page;
        uint32_t copies = 1;
        if (bench) {
            const size_t target = (size_t)1 << 30;
            copies = (uint32_t)((target + stride - 1) / stride);
            if (copies > 1024u) copies = 1024u;
        }
        const size_t model_bytes = stride * copies;
        void *model = NULL;
        CHECK(posix_memalign(&model, page, model_bytes) == 0);
        if (s->q8) {
            uint8_t *blocks = model;
            for (size_t b = 0; b < weight_bytes / 34u; b++) {
                const int value = (int)(random_value() * 8192);
                const uint16_t scale = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u));
                memcpy(blocks + b * 34u, &scale, sizeof(scale));
                for (uint32_t q = 0; q < 32; q++)
                    blocks[b * 34u + 2u + q] = (uint8_t)(int8_t)((int)(random_value() * 31) % 128);
            }
        } else {
            uint16_t *weights = model;
            for (size_t i = 0; i < weight_bytes / sizeof(*weights); i++) {
                const int value = (int)(random_value() * 8192);
                weights[i] = (uint16_t)(0x2000u | ((unsigned)abs(value) % 1024u) |
                                        (value < 0 ? 0x8000u : 0));
            }
        }
        for (uint32_t c = 1; c < copies; c++)
            memcpy((char *)model + (size_t)c * stride, model, weight_bytes);
        CHECK(ds4_gpu_set_model_map(model, model_bytes));

        const uint32_t width = s->width, output = s->outputs;
        const char *kind = s->q8 ? "Q8_0" : "F16";
        ds4_gpu_tensor *in = upload(NULL, (size_t)MAX_ROWS * width * sizeof(float));
        ds4_gpu_tensor *out = upload(NULL, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
        ds4_gpu_tensor *ref = upload(NULL, (size_t)MAX_ROWS * output * sizeof(float));
        CHECK(in && out && ref);
        float *x = ds4_gpu_tensor_contents(in), *y = ds4_gpu_tensor_contents(out);
        const float *expected = ds4_gpu_tensor_contents(ref);
        for (size_t i = 0; i < (size_t)MAX_ROWS * width; i++) x[i] = bf16(random_value());

        const size_t variants = s->q8 ? 1 : sizeof(f16_nr0) / sizeof(*f16_nr0);
        for (size_t vi = 0; vi < variants; vi++) {
            const char *nr0_env = s->q8 ? "-" : f16_nr0[vi];
            if (!s->q8) setenv("DS4_METAL_PLAIN_MV_NR0", nr0_env, 1);
            decode_rows_set_mode(s, NULL);
            CHECK(ds4_gpu_begin_commands());
            for (uint32_t t = 0; t < MAX_ROWS; t++) {
                ds4_gpu_tensor *xr = ds4_gpu_tensor_view(in, (size_t)t * width * 4, width * 4);
                ds4_gpu_tensor *yr = ds4_gpu_tensor_view(ref, (size_t)t * output * 4, output * 4);
                CHECK(xr && yr);
                CHECK(s->q8 ? ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(yr, model, model_bytes,
                                  0, width, output, xr, 1)
                            : ds4_gpu_matmul_f16_tensor(yr, model, model_bytes, 0, width,
                                  output, xr, 1));
                ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
            }
            CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
            for (size_t mi = 0; mi < sizeof(modes) / sizeof(*modes); mi++) {
                decode_rows_set_mode(s, &modes[mi]);
                double ms6 = 0;
                for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                    const uint32_t rows = counts[ci];
                    memset(y, 0xff, ((size_t)MAX_ROWS * output + 1) * sizeof(float));
                    y[(size_t)rows * output] = 12345;
                    const double start = monotonic_seconds();
                    CHECK(decode_rows_call(out, model, model_bytes, 0, s, in, rows));
                    CHECK(ds4_gpu_synchronize());
                    if (rows == 6) ms6 = (monotonic_seconds() - start) * 1000;
                    CHECK(y[(size_t)rows * output] == 12345);
                    if (memcmp(y, expected, (size_t)rows * output * sizeof(float)) != 0) {
                        size_t i = 0;
                        while (y[i] == expected[i]) i++;
                        fprintf(stderr, "%s width=%u output=%u nr0_env=%s mode=%s rows=%u: first mismatch row=%zu col=%zu got %.9g want %.9g\n",
                            kind, width, output, nr0_env, modes[mi].label, rows,
                            i / output, i % output, y[i], expected[i]);
                        CHECK(0);
                    }
                }
                fprintf(stderr, "V4.1 exact %s decode rows width=%u output=%u nr0_env=%s mode=%s rows=1-8,13,16: bitwise PASS (rows=6 %.3f ms)\n",
                    kind, width, output, nr0_env, modes[mi].label, ms6);
            }
        }
        if (!s->q8) unsetenv("DS4_METAL_PLAIN_MV_NR0");

        for (size_t ci = 0; bench && ci < sizeof(bench_counts) / sizeof(*bench_counts); ci++) {
            const uint32_t rows = bench_counts[ci];
            const uint32_t iters = copies > 64u ? copies : 64u;
            double cold[BENCH_MODES], warm[BENCH_MODES];
            for (size_t mi = 0; mi < BENCH_MODES; mi++) {
                decode_rows_set_mode(s, &bench_modes[mi]);
                cold[mi] = decode_rows_time(out, model, model_bytes, stride, copies, s, in, rows, iters);
                warm[mi] = decode_rows_time(out, model, model_bytes, stride, 1, s, in, rows, 64);
                CHECK(cold[mi] > 0 && warm[mi] > 0);
                CHECK(memcmp(y, expected, (size_t)rows * output * sizeof(float)) == 0);
            }
            fprintf(stderr, "decode-rows bench %s width=%u output=%u rows=%u ms/call cold:",
                kind, width, output, rows);
            for (size_t mi = 0; mi < BENCH_MODES; mi++)
                fprintf(stderr, " %s=%.4f", bench_modes[mi].label, cold[mi]);
            fprintf(stderr, " warm:");
            for (size_t mi = 0; mi < BENCH_MODES; mi++)
                fprintf(stderr, " %s=%.4f", bench_modes[mi].label, warm[mi]);
            fprintf(stderr, "\n");
        }
        decode_rows_set_mode(s, NULL);

        ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(in);
        ds4_gpu_cleanup(); free(model);
        CHECK(ds4_gpu_init());
    }
    return 1;
}

typedef struct { float value; uint32_t id; } topk_entry;

static int compare_topk_entry(const void *a, const void *b) {
    const topk_entry *x = a, *y = b;
    if (x->value > y->value) return -1;
    if (x->value < y->value) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int check_general_topk(void) {
    const uint32_t widths[] = {1, 2, 31, 127, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        const uint32_t width = widths[wi], rows = 3;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * rows * 4);
        ds4_gpu_tensor *selected = upload(NULL, ((size_t)width * rows + 1) * 4);
        topk_entry *reference = malloc((size_t)width * rows * sizeof(*reference));
        bool *seen = calloc(width, sizeof(*seen));
        CHECK(scores && selected && reference && seen);
        float *s = ds4_gpu_tensor_contents(scores);
        uint32_t *ids = ds4_gpu_tensor_contents(selected);
        const uint32_t counts[] = {1, 2, 31, 128, 511, 512, 513, 2048, width};
        for (uint32_t pattern = 0; pattern < 4; pattern++) {
            for (uint32_t t = 0; t < rows; t++) {
                for (uint32_t i = 0; i < width; i++) {
                    const float value = pattern == 0 ? random_value() :
                        pattern == 1 ? (float)(i % 7) :
                        pattern == 2 ? -INFINITY : (i & 1 ? -0.0f : 0.0f);
                    const float v = pattern == 1 && i % 17 == 0 ? INFINITY : value;
                    s[t * width + i] = v;
                    reference[t * width + i] = (topk_entry){v, i};
                }
                qsort(reference + t * width, width, sizeof(*reference), compare_topk_entry);
            }
            for (size_t ki = 0; ki < sizeof(counts) / sizeof(*counts); ki++) {
                const uint32_t k = counts[ki];
                if (k > width) continue;
                ids[rows * k] = 0xabcdef01;
                CHECK(ds4_gpu_indexer_topk_tensor(selected, scores, width, rows, k));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * k] == 0xabcdef01);
                for (uint32_t t = 0; t < rows; t++) {
                    memset(seen, 0, width * sizeof(*seen));
                    for (uint32_t i = 0; i < k; i++) {
                        const uint32_t id = ids[t * k + i];
                        CHECK(id < width && !seen[id]);
                        seen[id] = true;
#ifdef __APPLE__
                        /* Metal's bitonic order may permute exactly tied keys. */
                        CHECK(s[t * width + id] == reference[t * width + i].value);
#else
                        if (ids[t * k + i] != reference[t * width + i].id) {
                            fprintf(stderr, "top-k width=%u k=%u pattern=%u row=%u rank=%u: %u vs %u\n",
                                width, k, pattern, t, i, ids[t * k + i], reference[t * width + i].id);
                            CHECK(0);
                        }
#endif
                    }
                }
            }
        }
        fprintf(stderr, "V4.1 variable-k sort width=%u: independent sorted-value/unique-ID oracle PASS\n", width);
        ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores); free(reference); free(seen);
    }
    return 1;
}

static int check_causal_topk(void) {
    const uint32_t frontiers[] = {1024, 1025, 1535, 2047, 2048, 4095, 4096, 8191, 8192, 16383, 32767, 65535};
    const uint32_t counts[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (uint32_t ratio = 1; ratio <= 2; ratio++) {
    for (size_t fi = 0; fi < sizeof(frontiers) / sizeof(*frontiers); fi++) {
        const uint32_t start = frontiers[fi] * ratio - 1u;
        const uint32_t width = (start + 129u) / ratio + 1024u;
        ds4_gpu_tensor *scores = upload(NULL, (size_t)width * 129u * sizeof(float));
        ds4_gpu_tensor *selected = upload(NULL, (129u * 512u + 1u) * sizeof(int32_t));
        ds4_gpu_tensor *reference = upload(NULL, 129u * 512u * sizeof(int32_t));
        CHECK(scores && selected && reference);
        float *s = ds4_gpu_tensor_contents(scores);
        int32_t *ids = ds4_gpu_tensor_contents(selected);
        const int32_t *ref = ds4_gpu_tensor_contents(reference);
        for (uint32_t pattern = 0; pattern < 3; pattern++) {
            for (uint32_t t = 0; t < 129u; t++) {
                const uint32_t visible = (start + t + 1u) / ratio;
                for (uint32_t j = 0; j < width; j++) {
                    /* Future entries are deliberately attractive. Causality
                     * must come from each row's bounds, not score contents. */
                    float v = pattern == 0 ? random_value() : pattern == 1 ? (float)(j % 7u) : -INFINITY;
                    s[(size_t)t * width + j] = j < visible ? v : 12345.0f;
                }
            }
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
                const uint32_t rows = counts[ci];
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t t = 0; t < rows; t++) {
                    const uint32_t visible = (start + t + 1u) / ratio;
                    ds4_gpu_tensor *sr = ds4_gpu_tensor_view(scores, (size_t)t * width * 4u, visible * 4u);
                    ds4_gpu_tensor *out = ds4_gpu_tensor_view(reference, (size_t)t * 512u * 4u, 512u * 4u);
                    CHECK(sr && out && ds4_gpu_indexer_topk_tensor(out, sr, visible, 1, 512));
                    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(sr);
                }
                CHECK(ds4_gpu_end_commands());
                ids[rows * 512u] = 123456;
                CHECK(ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, rows, start, ratio));
                CHECK(ds4_gpu_synchronize());
                CHECK(ids[rows * 512u] == 123456);
                for (uint32_t t = 0; t < rows; t++) {
                    CHECK(s[(size_t)t * width + width - 1u] == 12345.0f);
                    for (uint32_t k = 0; k < 512u; k++) {
                        if (ids[t * 512u + k] != ref[t * 512u + k]) {
                            fprintf(stderr, "top-k mismatch ratio=%u visible=%u rows=%u pattern=%u row=%u rank=%u: %d vs %d\n",
                                ratio, frontiers[fi], rows, pattern, t, k, ids[t * 512u + k], ref[t * 512u + k]);
                            CHECK(0);
                        }
                        CHECK(ids[t * 512u + k] >= 0 && (uint32_t)ids[t * 512u + k] < (start + t + 1u) / ratio);
                    }
                }
            }
        }
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 130, start, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, UINT32_MAX, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, 0, ratio));
        CHECK(!ds4_gpu_dsv41_indexer_topk_batch(selected, scores, width, 1, start, 0));
        fprintf(stderr, "V4.1 causal top-k ratio=%u visible=%u: exact IDs including ties PASS\n", ratio, frontiers[fi]);
        ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(scores);
    }
    }
    return 1;
}


static int check_compact_carry(void) {
    const uint32_t widths[] = {1, 31, 32, 33, 127, 128, 129, 20480};
    const uint32_t rows = 129, offset = 2;
    for (uint32_t format = 0; format <= DS4_V41_CARRY_MASK; format++) {
        for (size_t wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
            const uint32_t width = widths[wi];
            const uint32_t words = format == DS4_V41_CARRY_BF16 ? (width + 1) / 2 : (width + 31) / 32;
            const size_t count = (size_t)width * rows;
            const size_t packed_bytes = (size_t)(rows + 4) * words * sizeof(uint32_t);
            ds4_gpu_tensor *plain = upload(NULL, (count + 1) * sizeof(float));
            ds4_gpu_tensor *packed = upload(NULL, packed_bytes);
            uint32_t *expected = malloc(count * sizeof(uint32_t));
            CHECK(plain && packed && expected);
            uint32_t *bits = ds4_gpu_tensor_contents(plain);
            unsigned char *storage = ds4_gpu_tensor_contents(packed);
            for (size_t i = 0; i < count; i++) {
                const uint32_t b = (uint32_t)(i * 40503u + 32768u) & 0xffffu;
                expected[i] = format == DS4_V41_CARRY_BF16 ? b << 16 : (b & 1u ? 0xff800000u : 0u);
            }
            memcpy(bits, expected, count * sizeof(uint32_t));
            bits[count] = 0x12345678u;
            memset(storage, 0xa5, packed_bytes);
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, true));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)));
            memset(bits, 0, count * sizeof(uint32_t));
            CHECK(ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, format, false));
            CHECK(ds4_gpu_synchronize());
            CHECK(!memcmp(bits, expected, count * sizeof(uint32_t)) && bits[count] == 0x12345678u);
            for (size_t i = 0; i < (size_t)offset * words * 4u; i++) CHECK(storage[i] == 0xa5);
            for (size_t i = (size_t)(offset + rows) * words * 4u; i < packed_bytes; i++)
                CHECK(storage[i] == 0xa5);
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, UINT32_MAX, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset + 3, plain, width, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, 0, rows, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, 0, format, true));
            CHECK(!ds4_gpu_dsv41_carry_copy(packed, offset, plain, width, rows, DS4_V41_CARRY_F32, true));
            fprintf(stderr, "V4.1 compact carry format=%u width=%u: exact round-trip and bounds PASS\n", format, width);
            free(expected);
            ds4_gpu_tensor_free(plain);
            ds4_gpu_tensor_free(packed);
        }
    }
    return 1;
}

/* Exercise the full-head and compact TP layouts with the same causal keys.
 * Selected rows are shuffled; include masked future rows at odd frontiers. */
/* Batched verify runs RoPE/FP8 before and BF16/inverse RoPE after the rows'
 * window writes and attention, one dispatch over all rows. Mirror both
 * orders and require identical bits, with the window write and attention
 * still per row: at a full window, row r's write replaces the entry every
 * earlier row attends to, which the hoisted-write control must expose. */
typedef struct {
    ds4_gpu_tensor *q, *kv, *heads, *window, *comp;
} merge_rows_state;

static int merge_rows_attend(merge_rows_state *t, const float *sinks, uint32_t row,
                             uint32_t pos, uint32_t attended, bool write) {
    enum { D = 512, H = 64 };
    const uint64_t head_bytes = (uint64_t)H * D * 4;
    ds4_gpu_tensor *q = ds4_gpu_tensor_view(t->q, row * head_bytes, head_bytes);
    ds4_gpu_tensor *kv = ds4_gpu_tensor_view(t->kv, (uint64_t)row * D * 4, D * 4);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_view(t->heads, row * head_bytes, head_bytes);
    const uint32_t n_raw = pos + 1u < 128u ? pos + 1u : 128u;
    const int ok = q && kv && heads &&
        (!write || ds4_gpu_tensor_copy(t->window, (uint64_t)(pos % 128u) * D * 4, kv, 0, D * 4)) &&
        ds4_gpu_attention_decode_heads_tensor(heads, sinks, getpagesize(), 0, q, t->window,
            n_raw, 128, (pos + 1u - n_raw) % 128u, t->comp, 0, attended, NULL, 0, H, D);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(heads);
    return ok;
}

static int check_merge_rows(void) {
    enum { D = 512, H = 64, W = 128, C = 64, MAX_ROWS = 7 };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    const size_t nq = (size_t)MAX_ROWS * H * D;
    float *q = malloc(nq * 4), *kv = malloc(MAX_ROWS * D * 4);
    float *window = malloc(W * D * 4), *comp = malloc(C * D * 4);
    float *expected = malloc(nq * 4), *expected_window = malloc(W * D * 4);
    CHECK(q && kv && window && comp && expected && expected_window);
    /* Partial window, wrap while filling, and a full ring at long context. */
    static const uint32_t starts[] = {5, 124, 1000, 4093};
    for (unsigned si = 0; si < sizeof(starts) / sizeof(*starts); si++) {
        for (uint32_t rows = 1; rows <= MAX_ROWS; rows++) {
            const uint32_t pos0 = starts[si], attended = si % 2 ? C : 0;
            const bool compressed = (rows + si) % 2 != 0;
            for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
            for (uint32_t i = 0; i < MAX_ROWS * D; i++) kv[i] = bf16(random_value() / 4);
            for (uint32_t i = 0; i < W * D; i++) window[i] = bf16(random_value() / 4);
            for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
            for (int variant = 0; variant < 3; variant++) {
                merge_rows_state t = {upload(q, nq * 4), upload(kv, MAX_ROWS * D * 4),
                    upload(NULL, nq * 4), upload(window, W * D * 4), upload(comp, C * D * 4)};
                CHECK(t.q && t.kv && t.heads && t.window && t.comp);
                /* Tensor copies encode only into an open batch, as in decode. */
                CHECK(ds4_gpu_begin_commands());
                if (variant == 0) {
                    /* Today's per-row order. */
                    for (uint32_t r = 0; r < rows; r++) {
                        const uint64_t hb = (uint64_t)H * D * 4;
                        ds4_gpu_tensor *qr = ds4_gpu_tensor_view(t.q, r * hb, hb);
                        ds4_gpu_tensor *kr = ds4_gpu_tensor_view(t.kv, (uint64_t)r * D * 4, D * 4);
                        ds4_gpu_tensor *hr = ds4_gpu_tensor_view(t.heads, r * hb, hb);
                        CHECK(qr && kr && hr);
                        CHECK(ds4_gpu_dsv41_rope(qr, D, H, 1, pos0 + r, compressed, false));
                        CHECK(ds4_gpu_dsv41_rope(kr, D, 1, 1, pos0 + r, compressed, false));
                        CHECK(ds4_gpu_dsv41_quantize(kr, D, 1, DS4_V41_FP8_E8M0));
                        CHECK(merge_rows_attend(&t, sinks, r, pos0 + r, attended, true));
                        CHECK(ds4_gpu_dsv41_quantize(hr, H * D, 1, DS4_V41_BF16));
                        CHECK(ds4_gpu_dsv41_rope(hr, D, H, 1, pos0 + r, compressed, true));
                        ds4_gpu_tensor_free(qr);
                        ds4_gpu_tensor_free(kr);
                        ds4_gpu_tensor_free(hr);
                    }
                } else {
                    /* Merged prologue/epilogue; variant 2 also hoists the
                     * window writes, which must break causality. */
                    CHECK(ds4_gpu_dsv41_rope(t.q, D, H, rows, pos0, compressed, false));
                    CHECK(ds4_gpu_dsv41_rope(t.kv, D, 1, rows, pos0, compressed, false));
                    CHECK(ds4_gpu_dsv41_quantize(t.kv, D, rows, DS4_V41_FP8_E8M0));
                    for (uint32_t r = 0; variant == 2 && r < rows; r++) {
                        ds4_gpu_tensor *kr = ds4_gpu_tensor_view(t.kv, (uint64_t)r * D * 4, D * 4);
                        CHECK(kr && ds4_gpu_tensor_copy(t.window,
                            (uint64_t)((pos0 + r) % W) * D * 4, kr, 0, D * 4));
                        ds4_gpu_tensor_free(kr);
                    }
                    for (uint32_t r = 0; r < rows; r++)
                        CHECK(merge_rows_attend(&t, sinks, r, pos0 + r, attended, variant == 1));
                    CHECK(ds4_gpu_dsv41_quantize(t.heads, H * D, rows, DS4_V41_BF16));
                    CHECK(ds4_gpu_dsv41_rope(t.heads, D, H, rows, pos0, compressed, true));
                }
                CHECK(ds4_gpu_end_commands());
                const size_t bytes = (size_t)rows * H * D * 4;
                if (variant == 0) {
                    CHECK(ds4_gpu_tensor_read(t.heads, 0, expected, bytes));
                    CHECK(ds4_gpu_tensor_read(t.window, 0, expected_window, W * D * 4));
                } else {
                    const float *heads = ds4_gpu_tensor_contents(t.heads);
                    const int same = !memcmp(expected, heads, bytes) &&
                        !memcmp(expected_window, ds4_gpu_tensor_contents(t.window), W * D * 4);
                    if (variant == 1) {
                        if (!same) fprintf(stderr, "merge-rows mismatch pos=%u rows=%u\n", pos0, rows);
                        CHECK(same);
                    } else if (pos0 + 1u >= W && rows > 1) {
                        /* Control: a hoisted write is visible to earlier rows. */
                        CHECK(memcmp(expected, heads, bytes) != 0);
                    }
                }
                ds4_gpu_tensor_free(t.q);
                ds4_gpu_tensor_free(t.kv);
                ds4_gpu_tensor_free(t.heads);
                ds4_gpu_tensor_free(t.window);
                ds4_gpu_tensor_free(t.comp);
            }
            fprintf(stderr, "merge-rows pos=%u rows=%u compressed=%d comp=%u: PASS\n",
                    pos0, rows, compressed, attended);
        }
    }
    free(q); free(kv); free(window); free(comp); free(expected); free(expected_window);
    free(sinks);
    return 1;
}

static int check_tp_attention(void) {
    enum { D = 512, H = 64, K = 512, C = 2048 };
    const struct { uint32_t rows, selected; } cases[] = {
        {1, K}, {31, K}, {32, K}, {33, K}, {129, K}, {257, K}, {2048, K},
        {1, 0}, {1, 1}, {2, K}, {6, K}, {8, K},
        {2, 0}, {2, 1}, {6, 0}
    };
    float *sinks = NULL;
    CHECK(posix_memalign((void **)&sinks, getpagesize(), getpagesize()) == 0);
    for (int h = 0; h < H; h++) sinks[h] = random_value();
    CHECK(ds4_gpu_set_model_map(sinks, getpagesize()));
    ds4_gpu_set_quality(false);
    for (unsigned s = 0; s < sizeof(cases) / sizeof(*cases); s++) {
        const uint32_t n = cases[s].rows, nr = n + 127, start = 3073;
        const uint32_t ratio = 1u + s % 2u;
        const size_t nq = (size_t)n * H * D;
        float *q = malloc(nq * 4), *actual = malloc(nq * 4);
        float *compact = malloc(nq * 2), *part = malloc(nq * 2);
        float *raw = malloc((size_t)nr * D * 4), *comp = malloc(C * D * 4);
        int32_t *ids = malloc((size_t)n * K * 4);
        CHECK(q && actual && compact && part && raw && comp && ids);
        for (size_t i = 0; i < nq; i++) q[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < nr * D; i++) raw[i] = bf16(random_value() / 4);
        for (uint32_t i = 0; i < C * D; i++) comp[i] = bf16(random_value() / 4);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t j = 0; j < K; j++)
                ids[t * K + j] = j < cases[s].selected &&
                    (cases[s].selected != K || j % 29)
                    ? (int32_t)((j * 127u + t * 17u) % C) : -1;
        ds4_gpu_tensor *qt = upload(q, nq * 4), *rt = upload(raw, (size_t)nr * D * 4);
        ds4_gpu_tensor *ct = upload(comp, C * D * 4), *it = upload(ids, (size_t)n * K * 4);
        ds4_gpu_tensor *out = upload(NULL, nq * 4), *qp = upload(NULL, nq * 2);
        ds4_gpu_tensor *op = upload(NULL, nq * 2);
        CHECK(qt && rt && ct && it && out && qp && op);
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(out, sinks, getpagesize(),
            0, qt, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H, D));
        CHECK(ds4_gpu_tensor_read(out, 0, actual, nq * 4));
        double max_split = 0, max_oracle = 0;
        for (uint32_t rank = 0; rank < 2; rank++) {
            for (uint32_t t = 0; t < n; t++)
                memcpy(compact + (size_t)t * H/2 * D,
                    q + ((size_t)t * H + rank * H/2) * D, H/2 * D * 4);
            CHECK(ds4_gpu_tensor_write(qp, 0, compact, nq * 2));
            CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
            CHECK(ds4_gpu_tensor_read(op, 0, part, nq * 2));
#ifdef __APPLE__
            if (n > 1) {
                ds4_gpu_set_quality(true); /* Existing eight-head kernel. */
                CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op, sinks, getpagesize(),
                    rank * H/2 * 4, qp, rt, ct, 0, it, n, start, nr, nr, 0, C, K, 128, ratio, H/2, D));
                CHECK(!memcmp(part, ds4_gpu_tensor_contents(op), nq * 2));
                ds4_gpu_set_quality(false);
            }
#endif
            if (n == 2048 && rank == 0) {
                const bool controls[] = {true, false, false, true};
                for (unsigned pass = 0; pass < 4; pass++) {
                    ds4_gpu_set_quality(controls[pass]);
                    const double begin = monotonic_seconds();
                    for (int repeat = 0; repeat < 4; repeat++)
                        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(op,
                            sinks, getpagesize(), 0, qp, rt, ct, 0, it, n, start,
                            nr, nr, 0, C, K, 128, ratio, H/2, D));
                    CHECK(ds4_gpu_synchronize());
                    fprintf(stderr, "TP indexed attention control=%u %.3f ms\n",
                        controls[pass], (monotonic_seconds() - begin) * 250);
                }
                ds4_gpu_set_quality(false);
            }
            for (uint32_t t = 0; t < n; t++) {
                for (uint32_t h = 0; h < H/2; h++) {
                    const float *got = part + ((size_t)t * H/2 + h) * D;
                    const float *ref = actual + ((size_t)t * H + rank * H/2 + h) * D;
                    for (int d = 0; d < D; d++) {
                        const double err = fabs((double)got[d] - ref[d]);
                        max_split = fmax(max_split, err);
                        CHECK(isfinite(got[d]) && err < 3e-5 * (1 + fabs(ref[d])));
                    }
                    if ((t != 0 && t != n/2 && t + 1 != n) || h % 15) continue;
                    const float *query = compact + ((size_t)t * H/2 + h) * D;
                    double logits[128 + K + 1], values[128 + K + 1];
                    const uint32_t col = (t * 71u + h * 19u) % D;
                    unsigned count = 0;
                    for (uint32_t j = 0; j < 128 + K; j++) {
                        const int32_t id = j < 128 ? (int32_t)(t + j) : ids[t * K + j - 128];
                        if (j >= 128 && (uint32_t)id >= (start + t + 1) / ratio) continue;
                        const float *key = (j < 128 ? raw : comp) + (size_t)id * D;
                        double dot = 0;
                        for (int d = 0; d < D; d++) dot += (double)query[d] * key[d];
                        logits[count] = dot / sqrt(512.0);
                        values[count++] = key[col];
                    }
                    logits[count] = sinks[rank * H/2 + h]; values[count++] = 0;
                    double max = -INFINITY, den = 0, num = 0;
                    for (unsigned j = 0; j < count; j++) max = fmax(max, logits[j]);
                    for (unsigned j = 0; j < count; j++) {
                        const double p = exp(logits[j] - max);
                        den += p; num += p * values[j];
                    }
                    const double err = fabs(got[col] - num / den);
                    max_oracle = fmax(max_oracle, err);
                    CHECK(err < 3e-5 * (1 + fabs(num / den)));
                }
            }
        }
        fprintf(stderr, "V4.1 TP attention rows=%u selected=%u ratio=%u max split=%g oracle=%g: PASS\n",
            n, cases[s].selected, ratio, max_split, max_oracle);
        ds4_gpu_tensor_free(qt); ds4_gpu_tensor_free(rt); ds4_gpu_tensor_free(ct);
        ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(qp);
        ds4_gpu_tensor_free(op);
        free(q); free(actual); free(compact); free(part); free(raw); free(comp); free(ids);
    }
    ds4_gpu_cleanup();
    free(sinks);
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--embedding")) {
        const int ok = ds4_gpu_init() && check_embedding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--router")) {
        const int ok = ds4_gpu_init() && check_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--tp-attention")) {
        const int ok = ds4_gpu_init() && check_tp_attention();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && (!strcmp(argv[1], "--attention-output") ||
                      !strcmp(argv[1], "--attention-output-large"))) {
        const int ok = ds4_gpu_init() && check_attention_output(
            !strcmp(argv[1], "--attention-output-large"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--hc-scaled")) {
        const int ok = ds4_gpu_init() && check_hc_scaled();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--bf16-linear")) {
        const int ok = ds4_gpu_init() && check_bf16_linear();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--compact-carry")) {
        const int ok = ds4_gpu_init() && check_compact_carry();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-scores")) {
        const int ok = ds4_gpu_init() && check_indexer_batch();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-topk")) {
        const int ok = ds4_gpu_init() && check_causal_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--general-topk")) {
        const int ok = ds4_gpu_init() && check_general_topk();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && (!strcmp(argv[1], "--decode-rows") ||
                      !strcmp(argv[1], "--decode-rows-bench"))) {
        const int ok = ds4_gpu_init() &&
            check_decode_rows(!strcmp(argv[1], "--decode-rows-bench"));
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--low-rows")) {
        const int ok = ds4_gpu_init() && check_low_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#ifdef __APPLE__
    if (argc == 2 && !strcmp(argv[1], "--stale-mask")) {
        const int ok = ds4_gpu_init() && check_stale_mask();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-ew")) {
        const int ok = ds4_gpu_init() && check_fused_ew();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-q8")) {
        const int ok = ds4_gpu_init() && check_fused_q8_single();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-f16")) {
        const int ok = ds4_gpu_init() && check_fused_f16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-hc")) {
        const int ok = ds4_gpu_init() && check_fused_hc();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-norm")) {
        const int ok = ds4_gpu_init() && check_fused_norm();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-router")) {
        const int ok = ds4_gpu_init() && check_fused_router();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fused-bf16")) {
        const int ok = ds4_gpu_init() && check_fused_bf16();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--f32-rows")) {
        const int ok = ds4_gpu_init() && check_f32_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--draft-low-rows")) {
        const int ok = ds4_gpu_init() && check_draft_low_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
#endif
    if (argc == 2 && !strcmp(argv[1], "--merge-rows")) {
        const int ok = ds4_gpu_init() && check_merge_rows();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--fp4-rounding")) {
        const int ok = ds4_gpu_init() && check_fp4_rounding();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc == 2 && !strcmp(argv[1], "--index-projection")) {
        const int ok = ds4_gpu_init() && check_index_projection();
        ds4_gpu_cleanup();
        return ok ? 0 : 1;
    }
    if (argc != 1) return 2;
    int ok = ds4_gpu_init() && check_router() && check_quantization() && check_fp4_rounding() && check_engram() && check_rope_stride() && check_pool() &&
             check_candidates() && check_sparse_gather() && check_indexer_batch() &&
             check_embedding() && check_index_projection() && check_decode_rows(0) && check_general_topk() && check_causal_topk() && check_compact_carry() && check_attention_output(false) &&
             check_tp_attention() && check_merge_rows() && check_low_rows();
    ds4_gpu_cleanup();
    return ok ? 0 : 1;
}
