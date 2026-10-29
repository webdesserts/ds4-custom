#!/bin/sh
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# cache-reuse-utils: the Cargo lock must resolve the pinned Git rev, and the
# vendored C header must be byte-identical to that rev's generated header.
CRU_REV=0a290dce62fbcbb7c434fa60057d246934865645
CRU_MANIFEST="$REPO/adapter/cache-reuse/Cargo.toml"
grep -q "rev = \"$CRU_REV\"" "$CRU_MANIFEST"
grep -q "cache-reuse-utils.git?rev=$CRU_REV#$CRU_REV" "$REPO/adapter/cache-reuse/Cargo.lock"
CRU_FFI_MANIFEST=$(cargo metadata --locked --format-version 1 --manifest-path "$CRU_MANIFEST" |
  python3 -c 'import json,sys; m=json.load(sys.stdin); print(next(p["manifest_path"] for p in m["packages"] if p["name"]=="cache-reuse-utils-ffi"))')
cmp "$(dirname "$CRU_FFI_MANIFEST")/include/cache_reuse_utils.h" "$REPO/adapter/include/cache_reuse_utils.h"
echo "cache-reuse-utils pin verified: $CRU_REV"
