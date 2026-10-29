#!/bin/sh
# Strict-host SCR verification: ownership fixtures, serving producers, help and
# planner ABI. Every executed binary links tests/scr_host_storage.c and
# tests/scr_host_forbidden.c (or nothing backend-related): no Metal object, no
# GPU framework and no model file is ever linked or opened.
#
# usage: tests/run-scr-host.sh [build-dir]   (invoked by `make check-scr-host`)
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT=${1:-"$REPO/build/scr-host"}
CC=${CC:-cc}
PLANNER_LIB="$REPO/adapter/cache-reuse/target/release/libds4_cache_reuse.a"
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$REPO"

FLAGS="-std=c99 -O1 -Wall -Wextra -Wno-unused-function -ffunction-sections -fdata-sections -I$REPO"
HOST="tests/scr_host_storage.c tests/scr_host_forbidden.c"
ENGINE_AUX="ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_engram.o"
SERVER_AUX="ds4.o $ENGINE_AUX ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o"

for obj in $SERVER_AUX; do
    case "$obj" in ds4_metal.o) echo "refusing Metal object" >&2; exit 2 ;; esac
    [ -f "$obj" ] || { echo "missing $obj (run make check-scr-host)" >&2; exit 2; }
done

run() {
    name=$1; shift
    "$CC" $FLAGS "$@" -Wl,-dead_strip -lm -pthread -o "$OUT/$name" > "$OUT/$name.compile.log" 2>&1 || {
        cat "$OUT/$name.compile.log" >&2; echo "FAIL compile $name" >&2; exit 1; }
    if ! (cd "$OUT" && TMPDIR="$OUT" "./$name") > "$OUT/$name.log" 2>&1; then
        tail -20 "$OUT/$name.log" >&2; echo "FAIL $name (log $OUT/$name.log)" >&2; exit 1
    fi
    echo "ok   $name"
}

# Pure coordinate/raw-store fixtures (no engine). Coordinates call the planner helpers.
run test_scr_coordinates tests/test_scr_coordinates.c "$PLANNER_LIB"
for name in test_scr_raw_store test_scr_raw_close; do
    run "$name" "tests/$name.c"
done
# Engine ownership fixtures: ds4.c is included textually, GPU calls hit host storage.
for name in test_scr_append test_scr_rebase_owned test_scr_rebase_failures test_scr_dispose \
            test_capture_chunks test_scr_actual_planner test_scr_upstream_fences test_scr_public_batch_admission \
            test_scr_encoder_residency test_scr_checkpoint_pool; do
    run "$name" "tests/$name.c" $HOST $ENGINE_AUX "$PLANNER_LIB"
done
# Serving: CLI, HTTP worker behaviour and ds4.reuse.v1 telemetry.
run test_engram_reader_observation tests/test_engram_reader_observation.c
for name in test_scr_server_cli test_server_contract test_server_progress test_server_upstream_reuse; do
    run "$name" "tests/$name.c" $HOST $SERVER_AUX "$PLANNER_LIB"
done
run test_help_scr tests/test_help_scr.c ds4_help.c
run budget_parse_test adapter/tests/budget_parse_test.c
run native_range_limits adapter/tests/native_range_limits.c
FLAGS="-std=c11 -O1 -Wall -Wextra -I$REPO/adapter/include"
run cache_reuse_ffi_test adapter/tests/cache_reuse_ffi_test.c "$PLANNER_LIB" -ldl
echo "check-scr-host: all strict-host fixtures passed ($OUT)"
