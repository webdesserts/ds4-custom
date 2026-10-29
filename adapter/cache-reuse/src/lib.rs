//! Link shell: keeps the pinned cache-reuse-utils C symbols in the static archive.
pub use cache_reuse_utils_ffi::{
    cache_reuse_common_prefix, cache_reuse_logical_boundary, cache_reuse_materialize,
    cache_reuse_plan, cache_reuse_project_logical,
};
