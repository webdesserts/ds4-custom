#!/bin/sh
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# cache-reuse-utils: the Cargo lock must resolve the pinned Git rev, and the
# vendored C header must be byte-identical to that rev's generated header.
CRU_REV=b5b6e6d07887a0fc7ad32a1de1f048dddc7b9360
CRU_MANIFEST="$REPO/adapter/cache-reuse/Cargo.toml"
grep -q "rev = \"$CRU_REV\"" "$CRU_MANIFEST"
grep -q "cache-reuse-utils.git?rev=$CRU_REV#$CRU_REV" "$REPO/adapter/cache-reuse/Cargo.lock"
METADATA=$(cargo metadata --locked --format-version 1 --manifest-path "$CRU_MANIFEST")
ffi_dir() {
  printf '%s' "$METADATA" | python3 -c 'import json,os,sys; m=json.load(sys.stdin); print(os.path.dirname(next(p["manifest_path"] for p in m["packages"] if p["name"]==sys.argv[1])))' "$1"
}
cmp "$(ffi_dir cache-reuse-utils-ffi)/include/cache_reuse_utils.h" "$REPO/adapter/include/cache_reuse_utils.h"
echo "cache-reuse-utils pin verified: $CRU_REV"
# kv-archive-store has no remote: it is a path dependency, so the checkout
# Cargo builds must sit clean at the pinned commit.
KV_REV=9e62ff39e455422ad21ebb3eb0324413ca1e3579
KV_FFI_DIR=$(ffi_dir kv-archive-store-ffi)
test "$(git -C "$KV_FFI_DIR" rev-parse HEAD)" = "$KV_REV"
test -z "$(git -C "$KV_FFI_DIR/.." status --porcelain)"
cmp "$KV_FFI_DIR/include/kv_archive_store.h" "$REPO/adapter/include/kv_archive_store.h"
echo "kv-archive-store pin verified: $KV_REV"
