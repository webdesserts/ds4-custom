#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ds4_scr.h"

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_ROW_BYTES = 264,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
} ds4_engram_table;

/* A separate uncached file descriptor, never an mmap or Metal model view.
 * Each GGUF I8 row is 256 E4M3 bytes followed by 8 original E8M0 scales. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage for the reference's BF16-rounded values. No whole
 * table allocation; caller owns count * DIM floats. Failure invalidates output. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. The request
 * array is bounded to 384 KiB, independent of table/prefix size; observations
 * add fixed scalar workspace for the existing16 readers. On macOS, large
 * batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);

/* Optional caller-owned content-free per-call result. Partial successful bytes
 * remain counted on failure. Parallel row sums overlap call-wall time; neither
 * is GPU latency. NULL preserves the unobserved path without clock calls. */
bool ds4_engram_read_observed(const ds4_engram_table *table, const uint32_t *rows,
    size_t count, float *out, ds4_engram_observation *observation);
bool ds4_engram_read_batch_observed(const ds4_engram_table *table, const uint32_t *rows,
    size_t tokens, size_t stride, float *out, ds4_engram_observation *observation);

/* One reusable reader for a token's two tables. Tables/IDs are copied at
 * start; descriptors and output must remain valid until wait(reader, 1) or
 * free, even if wait(reader, 0) reports an error. No table data is cached.
 * One caller owns the reader. Creation failure permits synchronous fallback. */
typedef struct ds4_engram_reader ds4_engram_reader;
ds4_engram_reader *ds4_engram_reader_create(void);
bool ds4_engram_reader_start(ds4_engram_reader *reader,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out);
/* Result, like output/descriptors, stays alive through wait(reader,1) or free.
 * Workers accumulate privately; only the owning wait1/free (or next start after
 * completion) publishes the result. Repeated wait1 never adds cumulative work.
 * Call-wall time spans accepted start to both tables' I/O/conversion completion,
 * not time spent overlapping later GPU work before the owner waits. Free can
 * cancel workers not yet started, as upstream; report partial work/failed call. */
bool ds4_engram_reader_start_observed(ds4_engram_reader *reader,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out,
    ds4_engram_observation *observation);
bool ds4_engram_reader_wait(ds4_engram_reader *reader, unsigned table);
void ds4_engram_reader_free(ds4_engram_reader *reader);

#endif
