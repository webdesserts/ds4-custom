/* C ABI for kv-archive-store. Hand-maintained: keep in sync with ffi/src. */
#ifndef KV_ARCHIVE_STORE_H
#define KV_ARCHIVE_STORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Every function returns one of these. Errors write a NUL-terminated
 * message to the optional error_out buffer of error_capacity bytes. */
#define KV_ARCHIVE_OK 0
#define KV_ARCHIVE_ERR_INVALID 10 /* arguments, shape, range or state */
#define KV_ARCHIVE_ERR_IO 11
#define KV_ARCHIVE_ERR_BUSY 12 /* budget or key held by leases or a write */
#define KV_ARCHIVE_ERR_NOT_FOUND 13
#define KV_ARCHIVE_ERR_PANIC 14

typedef struct kv_archive_raw_store kv_archive_raw_store;
typedef struct kv_archive_prefetch kv_archive_prefetch;
typedef struct kv_archive_pool kv_archive_pool;
typedef struct kv_archive_snapshot kv_archive_snapshot;
typedef struct kv_archive_lease kv_archive_lease;

/* A borrowed view of a checkpoint header, valid until the snapshot or lease
 * that produced it is freed. key is not NUL-terminated. */
typedef struct kv_archive_checkpoint_view {
    const uint8_t *key;
    size_t key_len;
    const uint32_t *tokens;
    size_t token_count;
    const uint32_t *pad_positions;
    size_t pad_count;
    const uint64_t *payload_sizes;
    size_t payload_count;
    uint64_t total_bytes;
    uint64_t last_used_seq; /* larger is more recent; 0 for a lease view */
    uint64_t active_leases; /* at snapshot time; 0 for a lease view */
} kv_archive_checkpoint_view;

/* Record-major window [record][layer][row] -> layer-major out
 * [layer][record][row]. threads 0 or 1 uses the calling thread only. */
int32_t kv_archive_gather_layers(const uint8_t *window, size_t window_len,
                                 uint32_t record_count, size_t num_layers,
                                 size_t layer_row_bytes, uint8_t *out,
                                 size_t out_len, size_t threads,
                                 char *error_out, size_t error_capacity);

/* Raw window store: fixed-size records with an explicit commit frontier. */
int32_t kv_archive_raw_store_create(const char *path, size_t record_bytes,
                                    uint32_t capacity_records, bool ephemeral,
                                    kv_archive_raw_store **out,
                                    char *error_out, size_t error_capacity);
int32_t kv_archive_raw_store_open_committed(const char *path,
                                            size_t record_bytes,
                                            uint32_t capacity_records,
                                            uint32_t committed_records,
                                            kv_archive_raw_store **out,
                                            char *error_out,
                                            size_t error_capacity);
void kv_archive_raw_store_free(kv_archive_raw_store *store);
int32_t kv_archive_raw_store_append(const kv_archive_raw_store *store,
                                    const uint8_t *data, size_t data_len,
                                    uint32_t record_count, char *error_out,
                                    size_t error_capacity);
int32_t kv_archive_raw_store_commit(const kv_archive_raw_store *store,
                                    uint32_t expected_total_records,
                                    uint32_t *committed_out, char *error_out,
                                    size_t error_capacity);
int32_t kv_archive_raw_store_discard(const kv_archive_raw_store *store,
                                     char *error_out, size_t error_capacity);
int32_t kv_archive_raw_store_sync(const kv_archive_raw_store *store,
                                  char *error_out, size_t error_capacity);
int32_t kv_archive_raw_store_counts(const kv_archive_raw_store *store,
                                    uint32_t *committed_out,
                                    uint32_t *pending_out, char *error_out,
                                    size_t error_capacity);
int32_t kv_archive_raw_store_read(const kv_archive_raw_store *store,
                                  uint32_t first, uint32_t count,
                                  uint8_t *out, size_t out_len,
                                  char *error_out, size_t error_capacity);

/* Background reads of committed records. wait always consumes the job. */
int32_t kv_archive_raw_store_prefetch(const kv_archive_raw_store *store,
                                      uint32_t first, uint32_t count,
                                      kv_archive_prefetch **out,
                                      char *error_out, size_t error_capacity);
int32_t kv_archive_prefetch_wait(kv_archive_prefetch *job, uint8_t *out,
                                 size_t out_len, char *error_out,
                                 size_t error_capacity);
void kv_archive_prefetch_free(kv_archive_prefetch *job);

/* Checkpoint pool: byte-budgeted, crash-safe, LRU with read leases. */
int32_t kv_archive_pool_open(const char *dir, uint64_t byte_budget,
                             kv_archive_pool **out, char *error_out,
                             size_t error_capacity);
void kv_archive_pool_free(kv_archive_pool *pool);
int32_t kv_archive_pool_write(const kv_archive_pool *pool, const char *key,
                              const uint32_t *tokens, size_t token_count,
                              const uint32_t *pad_positions, size_t pad_count,
                              const uint8_t *const *payloads,
                              const size_t *payload_lens, size_t payload_count,
                              char *error_out, size_t error_capacity);
int32_t kv_archive_pool_remove(const kv_archive_pool *pool, const char *key,
                               char *error_out, size_t error_capacity);
int32_t kv_archive_pool_usage(const kv_archive_pool *pool,
                              uint64_t *used_bytes_out,
                              uint64_t *byte_budget_out, char *error_out,
                              size_t error_capacity);
int32_t kv_archive_pool_snapshot(const kv_archive_pool *pool,
                                 kv_archive_snapshot **out,
                                 size_t *count_out, char *error_out,
                                 size_t error_capacity);
int32_t kv_archive_snapshot_entry(const kv_archive_snapshot *snapshot,
                                  size_t index,
                                  kv_archive_checkpoint_view *out,
                                  char *error_out, size_t error_capacity);
void kv_archive_snapshot_free(kv_archive_snapshot *snapshot);
int32_t kv_archive_pool_lease(const kv_archive_pool *pool, const char *key,
                              kv_archive_lease **out,
                              kv_archive_checkpoint_view *view_out,
                              char *error_out, size_t error_capacity);
int32_t kv_archive_lease_read(const kv_archive_lease *lease, size_t index,
                              uint64_t offset, uint8_t *out, size_t out_len,
                              char *error_out, size_t error_capacity);
void kv_archive_lease_release(kv_archive_lease *lease);

#ifdef __cplusplus
}
#endif

#endif /* KV_ARCHIVE_STORE_H */
