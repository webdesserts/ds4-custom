#!/bin/sh
# Build the pinned planner archive and run the host-only C ABI smoke test.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
make -C "$REPO" "$REPO/adapter/cache-reuse/target/release/libds4_cache_reuse.a" >/dev/null
cc -std=c11 -Wall -Wextra -I"$REPO/adapter/include" \
  "$REPO/adapter/tests/cache_reuse_ffi_test.c" \
  "$REPO/adapter/cache-reuse/target/release/libds4_cache_reuse.a" \
  -lpthread -ldl -lm -o "$REPO/adapter/tests/cache_reuse_ffi_test"
"$REPO/adapter/tests/cache_reuse_ffi_test"
