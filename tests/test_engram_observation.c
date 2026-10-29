/* Real tiny owned files; syscall and clock faults never touch a model. */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
static ssize_t observed_test_pread(int, void *, size_t, off_t);
static int observed_test_clock(clockid_t, struct timespec *);
#define pread observed_test_pread
#define clock_gettime observed_test_clock
#ifndef DS4_ENGRAM_SOURCE
#define DS4_ENGRAM_SOURCE "../ds4_engram.c"
#endif
#include DS4_ENGRAM_SOURCE
#undef pread
#undef clock_gettime

static unsigned read_calls, clock_calls;
static int short_reads, clock_failure;
static ssize_t observed_test_pread(int fd, void *data, size_t count, off_t at) {
    unsigned call = __atomic_add_fetch(&read_calls, 1, __ATOMIC_RELAXED);
    if (short_reads && call == 1) { errno = EINTR; return -1; }
    if (short_reads && call == 2 && count > 17) count = 17;
    return pread(fd, data, count, at);
}
static int observed_test_clock(clockid_t id, struct timespec *t) {
    __atomic_add_fetch(&clock_calls, 1, __ATOMIC_RELAXED);
    if (clock_failure) { errno = ENOTSUP; return -1; }
    return clock_gettime(id, t);
}
int main(void) {
    char path[] = "/tmp/ds4-engram-observation-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    uint8_t raw[2][DS4_ENGRAM_ROW_BYTES];
    memset(raw, 0x38, sizeof(raw));
    for (int r = 0; r < 2; ++r) memset(raw[r] + 256, 127 + r, 8);
    assert(write(fd, raw, sizeof(raw)) == sizeof(raw));
    ds4_engram_table table; assert(ds4_engram_table_open(&table, path, 0, 2));
    assert(unlink(path) == 0);
    ds4_engram_table before = table;
    uint32_t row = 0;
    float expected[256], actual[256];
    assert(ds4_engram_read(&table, &row, 1, expected));
    short_reads = 1; read_calls = 0;
    ds4_engram_observation o;
    assert(ds4_engram_read_observed(&table, &row, 1, actual, &o));
    assert(!memcmp(actual, expected, sizeof(actual)) && !memcmp(&before, &table, sizeof(table)));
    assert(o.enabled && o.calls == 1 && !o.batch_calls && !o.failed_calls && o.row_attempts == 1);
    assert(o.pread_calls == 3 && o.read_bytes == 264 && o.rows_completed == 1 && !o.clock_failed);
    short_reads = 0;
    enum { TOKENS = 32, COUNT = TOKENS * DS4_ENGRAM_COLS, WIDTH = COUNT * DS4_ENGRAM_DIM };
    uint32_t ids[COUNT];
    float *batch = malloc(WIDTH * sizeof(float)), *reference = malloc(WIDTH * sizeof(float));
    assert(batch && reference);
    for (unsigned i = 0; i < COUNT; ++i) ids[i] = i % 2;
    assert(ds4_engram_read_batch(&table, ids, TOKENS, DS4_ENGRAM_COLS, reference));
    read_calls = 0;
    assert(ds4_engram_read_batch_observed(&table, ids, TOKENS, DS4_ENGRAM_COLS, batch, &o));
    assert(!memcmp(batch, reference, WIDTH * sizeof(float)));
    /* Existing16 partitions deduplicate independently: eight parts per ID. */
    assert(o.calls == 1 && o.batch_calls == 1 && o.rows_completed == 16 && o.row_attempts == 16);
    assert(o.pread_calls == 16 && o.read_bytes == 16 * 264 && read_calls == 16 && !o.failed_calls);
    assert(!o.clock_failed && !o.time_overflow && !o.counters_overflow);
    clock_failure = 1; clock_calls = 0;
    errno = EDOM;
    assert(ds4_engram_read(&table, &row, 1, actual) && errno == EDOM && !clock_calls);
    assert(ds4_engram_read_observed(&table, &row, 1, actual, NULL) && errno == EDOM && !clock_calls);
    assert(ds4_engram_read_observed(&table, &row, 1, actual, &o) && errno == EDOM);
    assert(o.clock_failed && o.read_bytes == 264 && o.rows_completed == 1 && !o.failed_calls);
    assert(!memcmp(actual, expected, sizeof(actual)));
    /* A real short file produces13 successful bytes then EOF/EIO, not zero I/O. */
    assert(ftruncate(fd, 13) == 0); read_calls = 0;
    errno = 0;
    assert(!ds4_engram_read_observed(&table, &row, 1, actual, &o) && errno == EIO);
    assert(o.clock_failed && o.pread_calls == 2 && o.read_bytes == 13 && !o.rows_completed && o.failed_calls == 1);
    clock_failure = 0;
    for (unsigned i = 0; i < COUNT; ++i) ids[i] = 0;
    assert(!ds4_engram_read_batch_observed(&table, ids, TOKENS, DS4_ENGRAM_COLS, batch, &o) && errno == EIO);
    assert(o.pread_calls == 32 && o.row_attempts == 16 && o.read_bytes == 16 * 13 && !o.rows_completed);
    assert(o.failed_calls == 1 && !o.clock_failed);
    uint32_t invalid = 2;
    assert(!ds4_engram_read_observed(&table, &invalid, 1, actual, &o) && errno == EINVAL);
    assert(o.calls == 1 && o.failed_calls == 1 && !o.pread_calls && !o.read_bytes);
    ds4_engram_observation total = {.enabled = true, .read_bytes = UINT64_MAX,
        .row_read_nanoseconds = UINT64_MAX};
    o = (ds4_engram_observation){.enabled = true, .read_bytes = 1, .row_read_nanoseconds = 1};
    ds4_engram_observation_add(&total, &o);
    assert(total.counters_overflow && total.time_overflow && total.read_bytes == UINT64_MAX &&
        total.row_read_nanoseconds == UINT64_MAX);
    free(reference); free(batch); ds4_engram_table_close(&table); assert(close(fd) == 0);
    puts("Engram observations: exact outputs/dedup16-reader joins, partial/error bytes, errno/clock isolation, optional/overflow: PASS");
    return 0;
}
