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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* CACHE_REUSE_UTILS_H */
