//! Link shell: keeps the pinned cache-reuse-utils and kv-archive-store C
//! symbols in the static archive, plus the server's slot router and the
//! DSpark draft-length policy.
pub mod slot_router;
pub use dspark_policy::{ds4_dspark_gain_continue, ds4_dspark_gain_model_default};
pub use cache_reuse_utils_ffi::{
    cache_reuse_append_unpadded, cache_reuse_clip_ranges_to_archive,
    cache_reuse_collect_separator_boundaries, cache_reuse_common_prefix,
    cache_reuse_compile_archive_schedule, cache_reuse_logical_boundary, cache_reuse_materialize,
    cache_reuse_plan, cache_reuse_project_logical, cache_reuse_reconcile_executed_ranges,
    cache_reuse_select_candidate, cache_reuse_slice_pad_positions,
    cache_reuse_truncate_pad_positions, cache_reuse_verify_matches_logical,
    cache_reuse_verify_separator_map, cache_reuse_verify_span_correspondence,
};
pub use kv_archive_store_ffi::{
    kv_archive_gather_layers, kv_archive_lease_read, kv_archive_lease_release,
    kv_archive_pool_free, kv_archive_pool_lease, kv_archive_pool_open, kv_archive_pool_remove,
    kv_archive_pool_snapshot, kv_archive_pool_usage, kv_archive_pool_write,
    kv_archive_prefetch_free, kv_archive_prefetch_wait, kv_archive_raw_store_append,
    kv_archive_raw_store_commit, kv_archive_raw_store_counts, kv_archive_raw_store_create,
    kv_archive_raw_store_discard, kv_archive_raw_store_free, kv_archive_raw_store_open_committed,
    kv_archive_raw_store_prefetch, kv_archive_raw_store_read, kv_archive_raw_store_sync,
    kv_archive_snapshot_entry, kv_archive_snapshot_free,
};
