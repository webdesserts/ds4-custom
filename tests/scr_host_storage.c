/* Host-only tensor storage for SCR ownership fixtures. Never link Metal here.
 * RoPE/quantization are identity test doubles: these tests prove ownership,
 * capture coverage and publication, not numerical equivalence or GPU behavior. */
#include "../ds4_gpu.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { unsigned char *data; uint64_t bytes, refs; } host_buffer;
struct ds4_gpu_tensor { host_buffer *buffer; uint64_t offset, bytes; };
static uint64_t allocated;
static bool active;
static uint64_t command_begins;
static bool fail_next_begin;
uint64_t scr_host_command_begins(void) { return command_begins; }
void scr_host_fail_next_begin(void) { fail_next_begin = true; }
/* Two archives with full 2048-token capture arenas (160 MiB each) must fit. */
static const uint64_t storage_limit = UINT64_C(2048) * 1024u * 1024u;

int ds4_gpu_init(void) { assert(!allocated && !active); return 1; }
void ds4_gpu_cleanup(void) {
    assert(!active && !allocated);
    puts("host SCR storage drained; no Metal/model backend linked");
}
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!bytes || bytes > storage_limit - allocated) return NULL;
    host_buffer *b = calloc(1, sizeof(*b));
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!b || !t) { free(b); free(t); return NULL; }
    b->data = calloc(1, (size_t)bytes);
    if (!b->data) { free(b); free(t); return NULL; }
    b->bytes = bytes; b->refs = 1; allocated += bytes;
    t->buffer = b; t->bytes = bytes;
    return t;
}
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) { return ds4_gpu_tensor_alloc(bytes); }
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->buffer = base->buffer; ++t->buffer->refs;
    t->offset = base->offset + offset; t->bytes = bytes;
    return t;
}
void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    host_buffer *b = t->buffer;
    assert(b && b->refs);
    if (!--b->refs) { allocated -= b->bytes; free(b->data); free(b); }
    free(t);
}
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *t) { return t ? t->buffer->data + t->offset : NULL; }
static int bounds(const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) {
    return t && offset <= t->bytes && bytes <= t->bytes - offset;
}
int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t offset, const void *data, uint64_t bytes) {
    if (!bounds(t, offset, bytes) || (bytes && !data)) return 0;
    memcpy(t->buffer->data + t->offset + offset, data, (size_t)bytes); return 1;
}
int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t offset, void *data, uint64_t bytes) {
    if (!bounds(t, offset, bytes) || (bytes && !data)) return 0;
    memcpy(data, t->buffer->data + t->offset + offset, (size_t)bytes); return 1;
}
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                        const ds4_gpu_tensor *src, uint64_t src_offset, uint64_t bytes) {
    if (!bounds(dst, dst_offset, bytes) || !bounds(src, src_offset, bytes)) return 0;
    memmove(dst->buffer->data + dst->offset + dst_offset,
            src->buffer->data + src->offset + src_offset, (size_t)bytes); return 1;
}
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *t, float value, uint64_t count) {
    if (!bounds(t, 0, count * sizeof(float))) return 0;
    float *data = ds4_gpu_tensor_contents(t);
    for (uint64_t i = 0; i < count; ++i) data[i] = value;
    return 1;
}
int ds4_gpu_begin_commands(void) {
    ++command_begins;
    if (fail_next_begin) { fail_next_begin = false; return 0; }
    if (active) return 0;
    active = true; return 1;
}
int ds4_gpu_commands_active(void) { return active; }
int ds4_gpu_end_commands(void) { if (!active) return 0; active = false; return 1; }
int ds4_gpu_synchronize(void) { return !active; }
/* Stage profiling labels are diagnostics only. */
void ds4_gpu_profile_stage(const char *stage) { (void)stage; }
uint64_t ds4_gpu_recommended_working_set_size(void) { return UINT64_C(32) << 30; }
/* Streaming fixture has no resident expert cache, forcing its scalar route. */
uint32_t ds4_gpu_stream_expert_cache_configured_count(void) { return 0; }
int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                          ds4_v41_activation_format format) {
    (void)format; return bounds(x, 0, (uint64_t)width * rows * sizeof(float));
}
int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                             uint32_t rows, uint32_t start, uint32_t stride,
                             bool compressed, bool inverse) {
    (void)start; (void)stride; (void)compressed; (void)inverse;
    return bounds(x, 0, (uint64_t)width * heads * rows * sizeof(float));
}
