#ifndef CACHE_REUSE_UTILS_H
#define CACHE_REUSE_UTILS_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define CACHE_REUSE_COMPLETE 0

#define CACHE_REUSE_NO_FEASIBLE 1

#define CACHE_REUSE_DEADLINE_PARTIAL 2

#define CACHE_REUSE_VARIANT_LIMIT 3

#define CACHE_REUSE_STATE_LIMIT 4

#define CACHE_REUSE_ERR_INVALID 10

#define CACHE_REUSE_ERR_BUDGET 11

#define CACHE_REUSE_ERR_CANCELLED 12

#define CACHE_REUSE_ERR_ADMISSION 13

#define CACHE_REUSE_ERR_PANIC 14

#define CACHE_REUSE_MAX_RANGES 32

#define CACHE_REUSE_MAX_INSERTIONS 32

#define CACHE_REUSE_ACTION_FRESH 0

#define CACHE_REUSE_ACTION_KEEP_HOT 1

#define CACHE_REUSE_ACTION_RESTORE_POOL 2

#define CACHE_REUSE_OP_ACQUIRE_ENCODER 0

#define CACHE_REUSE_OP_RELEASE_ENCODER 1

#define CACHE_REUSE_OP_PREFETCH_WINDOW 2

#define CACHE_REUSE_OP_STAGE_SPAN 3

#define CACHE_REUSE_OP_IMPORT_CHUNK 4

#define CACHE_REUSE_OP_PREFILL_CHUNK 5

#define CACHE_REUSE_OP_COVERED_HEAD 6

#define CACHE_REUSE_OP_FINAL_HEAD 7

typedef struct {
  uint32_t max_ranges;
  uint32_t minimum_match_tokens;
  uint32_t minimum_physical_reuse_tokens;
  uint32_t source_start_alignment;
  uint32_t prepared_target_start_alignment;
  uint32_t reuse_count_alignment;
  uint32_t minimum_prepared_reuse_start;
  uint32_t protected_prefix_tokens;
  uint32_t fresh_suffix_tokens;
  uint32_t max_trim_left;
  uint32_t max_trim_right;
  int32_t pad_token;
  uint32_t loss_basis_points;
  uint32_t max_new_padding;
  uint32_t max_prepared_tokens;
  uint32_t variant_limit;
  uint32_t budget_ms;
  bool allow_all_insertions;
  const uint32_t *allowed_insertions;
  uint32_t allowed_insertion_count;
  uint32_t pad_penalty_tokens;
  uint32_t range_penalty_tokens;
} cache_reuse_config;

typedef struct {
  uint32_t source_start;
  uint32_t destination_start;
  uint32_t count;
  uint32_t prepared_start;
} cache_reuse_range;

typedef struct {
  uint32_t before_target_index;
  uint32_t prepared_start;
  uint32_t token;
  uint32_t count;
} cache_reuse_insertion;

typedef struct {
  uint32_t status;
  uint32_t covered_tokens;
  uint32_t inserted_padding;
  uint32_t prepared_tokens;
  uint32_t best_useful;
  uint32_t admissible_min;
  uint32_t range_count;
  uint32_t insertion_count;
  uint32_t discovered_match_count;
  uint32_t longest_match_tokens;
  uint64_t elapsed_ns;
  uint64_t preparation_ns;
  uint64_t discovery_ns;
  uint64_t search_ns;
  bool discovery_entered;
  bool search_entered;
  bool timing_known;
  cache_reuse_range ranges[CACHE_REUSE_MAX_RANGES];
  cache_reuse_insertion insertions[CACHE_REUSE_MAX_INSERTIONS];
} cache_reuse_result;

typedef struct {
  uint32_t common_logical_tokens;
  uint32_t physical_prefix_tokens;
  uint32_t carried_pads;
  uint32_t live_logical_tokens;
} cache_reuse_common_prefix_result;

typedef struct {
  const int32_t *tokens;
  uint32_t token_len;
  const uint32_t *pad_positions;
  uint32_t pad_len;
  uint32_t planned_reusable_tokens;
} cache_reuse_candidate;

typedef struct {
  uint32_t minimum_match_tokens;
  uint32_t alignment;
  uint32_t restore_penalty_tokens;
} cache_reuse_candidate_policy;

typedef struct {
  uint32_t action;
  uint32_t pool_index;
  uint32_t reusable_tokens;
} cache_reuse_candidate_selection;

typedef struct {
  uint32_t reused_tokens;
  uint32_t gap_tokens;
  uint32_t suffix_tokens;
  uint32_t fresh_tokens;
  uint32_t first_reused_start;
  uint32_t covered_end;
} cache_reuse_execution_summary;

typedef struct {
  uint32_t raw_arena_tokens;
  uint32_t max_import_chunk;
  uint32_t encoder_acquire_threshold;
  uint32_t encoder_release_tail;
  uint32_t head_alignment;
} cache_reuse_archive_geometry;

typedef struct {
  uint32_t kind;
  uint32_t source_start;
  uint32_t source_end;
  uint32_t position;
  uint32_t count;
  bool flush_arena;
  bool defer_decoder;
} cache_reuse_archive_op;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int32_t cache_reuse_plan(const int32_t *target,
                         uint32_t target_len,
                         const int32_t *source,
                         uint32_t source_len,
                         uint64_t cache_identity,
                         const cache_reuse_config *config,
                         cache_reuse_result *result,
                         char *error,
                         size_t error_len,
                         bool (*cancel)(void*),
                         void *cancel_ud);

uint32_t cache_reuse_materialize(const int32_t *target_tokens,
                                 uint32_t target_len,
                                 const int32_t *source_tokens,
                                 uint32_t source_len,
                                 const cache_reuse_result *result,
                                 int32_t *out_prepared_tokens,
                                 uint32_t out_prepared_len,
                                 uint32_t *out_pad_positions,
                                 uint32_t out_pad_len,
                                 char *error_out,
                                 size_t error_len);

uint32_t cache_reuse_project_logical(const int32_t *physical_tokens,
                                     uint32_t physical_len,
                                     const uint32_t *pad_positions,
                                     uint32_t pad_len,
                                     int32_t *out_logical_tokens,
                                     uint32_t out_logical_capacity,
                                     uint32_t *out_logical_len,
                                     char *error_out,
                                     size_t error_len);

uint32_t cache_reuse_logical_boundary(uint32_t physical_boundary,
                                      uint32_t physical_len,
                                      const uint32_t *pad_positions,
                                      uint32_t pad_len,
                                      uint32_t *out_logical_boundary,
                                      char *error_out,
                                      size_t error_len);

uint32_t cache_reuse_common_prefix(const int32_t *logical_tokens,
                                   uint32_t logical_len,
                                   const int32_t *physical_tokens,
                                   uint32_t physical_len,
                                   const uint32_t *pad_positions,
                                   uint32_t pad_len,
                                   cache_reuse_common_prefix_result *out,
                                   char *error_out,
                                   size_t error_len);

uint32_t cache_reuse_verify_matches_logical(const int32_t *physical_tokens,
                                            uint32_t physical_len,
                                            const uint32_t *pad_positions,
                                            uint32_t pad_len,
                                            const int32_t *logical_tokens,
                                            uint32_t logical_len,
                                            bool *out_matches,
                                            char *error_out,
                                            size_t error_len);

uint32_t cache_reuse_slice_pad_positions(const uint32_t *pad_positions,
                                         uint32_t pad_len,
                                         uint32_t physical_len,
                                         uint32_t slice_start,
                                         uint32_t slice_len,
                                         uint32_t *out_pads,
                                         uint32_t out_pad_capacity,
                                         uint32_t *out_pad_len,
                                         char *error_out,
                                         size_t error_len);

uint32_t cache_reuse_append_unpadded(uint32_t physical_len,
                                     const uint32_t *pad_positions,
                                     uint32_t pad_len,
                                     uint32_t appended_tokens,
                                     uint32_t *out_physical_len,
                                     char *error_out,
                                     size_t error_len);

uint32_t cache_reuse_truncate_pad_positions(const uint32_t *pad_positions,
                                            uint32_t pad_len,
                                            uint32_t physical_len,
                                            uint32_t new_physical_len,
                                            uint32_t *out_pad_len,
                                            char *error_out,
                                            size_t error_len);

uint32_t cache_reuse_verify_span_correspondence(const uint32_t *source_pads,
                                                uint32_t source_pad_len,
                                                uint32_t source_len,
                                                const uint32_t *destination_pads,
                                                uint32_t destination_pad_len,
                                                uint32_t destination_len,
                                                const cache_reuse_range *range,
                                                bool *out_corresponds,
                                                char *error_out,
                                                size_t error_len);

uint32_t cache_reuse_collect_separator_boundaries(const int32_t *logical_tokens,
                                                  uint32_t logical_len,
                                                  uint32_t start_index,
                                                  const int32_t *separator_tokens,
                                                  uint32_t separator_len,
                                                  uint32_t *out_boundaries,
                                                  uint32_t out_boundary_capacity,
                                                  uint32_t *out_boundary_len,
                                                  char *error_out,
                                                  size_t error_len);

uint32_t cache_reuse_verify_separator_map(const int32_t *logical_tokens,
                                          uint32_t logical_len,
                                          const int32_t *physical_tokens,
                                          uint32_t physical_len,
                                          const uint32_t *pad_positions,
                                          uint32_t pad_len,
                                          uint32_t carried_physical_prefix,
                                          const uint32_t *carried_pads,
                                          uint32_t carried_pad_len,
                                          int32_t pad_token,
                                          const int32_t *separator_tokens,
                                          uint32_t separator_len,
                                          bool *out_authorized,
                                          char *error_out,
                                          size_t error_len);

uint32_t cache_reuse_select_candidate(const int32_t *target_tokens,
                                      uint32_t target_len,
                                      const cache_reuse_candidate *hot,
                                      const cache_reuse_candidate *pool,
                                      uint32_t pool_len,
                                      const cache_reuse_candidate_policy *policy,
                                      cache_reuse_candidate_selection *out,
                                      char *error_out,
                                      size_t error_len);

uint32_t cache_reuse_clip_ranges_to_archive(const cache_reuse_range *ranges,
                                            uint32_t range_len,
                                            uint32_t archive_start,
                                            uint32_t archive_end,
                                            uint32_t target_cap,
                                            uint32_t alignment,
                                            cache_reuse_range *out_ranges,
                                            uint32_t out_range_capacity,
                                            uint32_t *out_range_len,
                                            char *error_out,
                                            size_t error_len);

uint32_t cache_reuse_reconcile_executed_ranges(const cache_reuse_range *planned,
                                               uint32_t planned_len,
                                               const cache_reuse_range *executed,
                                               uint32_t executed_len,
                                               uint32_t physical_len,
                                               uint32_t alignment,
                                               cache_reuse_execution_summary *out,
                                               char *error_out,
                                               size_t error_len);

uint32_t cache_reuse_compile_archive_schedule(uint32_t physical_len,
                                              const cache_reuse_range *ranges,
                                              uint32_t range_len,
                                              const cache_reuse_archive_geometry *geometry,
                                              cache_reuse_archive_op *out_ops,
                                              uint32_t out_op_capacity,
                                              uint32_t *out_op_len,
                                              char *error_out,
                                              size_t error_len);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* CACHE_REUSE_UTILS_H */
