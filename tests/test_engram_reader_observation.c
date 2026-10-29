/* Extend the real syscall/clock fixture over upstream's persistent four-reader
 * path. All files/output are owned here; no model/backend/service is involved. */
#define main synchronous_observations
#include "test_engram_observation.c"
#undef main

int main(void) {
    assert(!synchronous_observations());
    clock_failure = short_reads = 0;
    char path[] = "/tmp/ds4-engram-reader-observation-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    uint8_t raw[2][DS4_ENGRAM_ROW_BYTES]; memset(raw, 0x38, sizeof(raw));
    for (unsigned r = 0; r < 2; ++r) memset(raw[r] + 256, 127 + r, 8);
    assert(write(fd, raw, sizeof(raw)) == sizeof(raw));
    ds4_engram_table tables[2];
    for (unsigned t = 0; t < 2; ++t) assert(ds4_engram_table_open(&tables[t], path, 0, 2));
    assert(!unlink(path));
    uint32_t ids[2][DS4_ENGRAM_COLS];
    float output[2][DS4_ENGRAM_COLS * DS4_ENGRAM_DIM], reference[2][DS4_ENGRAM_COLS * DS4_ENGRAM_DIM];
    for (unsigned t = 0; t < 2; ++t) {
        for (unsigned c = 0; c < DS4_ENGRAM_COLS; ++c) ids[t][c] = c % 2;
        assert(ds4_engram_read(&tables[t], ids[t], DS4_ENGRAM_COLS, reference[t]));
    }
    ds4_engram_reader *reader = ds4_engram_reader_create(); assert(reader);
    ds4_engram_observation o;
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(ds4_engram_reader_wait(reader, 0));
    assert(o.read_bytes == 0); /* wait0 cannot publish while table1 may run. */
    assert(ds4_engram_reader_wait(reader, 1));
    assert(!memcmp(output, reference, sizeof(output)));
    assert(o.enabled && o.calls == 1 && !o.batch_calls && !o.failed_calls);
    assert(o.rows_completed == 48 && o.row_attempts == 48 && o.pread_calls == 48 && o.read_bytes == 48 * 264);
    ds4_engram_observation same = o;
    assert(ds4_engram_reader_wait(reader, 1) && !memcmp(&o, &same, sizeof(o)));
    o.read_bytes = 777; ds4_engram_reader_free(reader); assert(o.read_bytes == 777);

    /* A completed generation may be restarted before wait1, as upstream allows.
     * Reusing the same result object must not seed the next call with old work. */
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    pthread_mutex_lock(&reader->mutex);
    while (reader->active) pthread_cond_wait(&reader->done, &reader->mutex);
    pthread_mutex_unlock(&reader->mutex);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(!o.read_bytes);
    assert(ds4_engram_reader_wait(reader, 1));
    assert(o.calls == 1 && o.rows_completed == 48 && o.read_bytes == 48 * 264);
    ds4_engram_reader_free(reader);

    /* Both-table failure, partial byte accounting and free as the drain boundary. */
    assert(!ftruncate(fd, 13));
    for (unsigned t = 0; t < 2; ++t) for (unsigned c = 0; c < DS4_ENGRAM_COLS; ++c) ids[t][c] = 0;
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(!ds4_engram_reader_wait(reader, 0) && errno == EIO);
    assert(!o.read_bytes); /* Descriptors/output/result still alive. */
    ds4_engram_reader_free(reader);
    assert(o.failed_calls == 1 && o.row_attempts == 8 && !o.rows_completed && o.read_bytes == 8 * 13 && o.pread_calls == 16);

    assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
    clock_failure = 1; clock_calls = 0;
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start(reader, tables, ids, output[0]));
    assert(ds4_engram_reader_wait(reader, 1)); ds4_engram_reader_free(reader);
    assert(!clock_calls); /* Original API remains completely uninstrumented. */
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(ds4_engram_reader_wait(reader, 0));
    ds4_engram_reader_free(reader);
    assert(o.clock_failed && !o.failed_calls && o.rows_completed == 48 && o.read_bytes == 48 * 264);
    /* Immediate free preserves upstream cancellation of not-yet-started workers.
     * Whatever work won the race is measured exactly, never fabricated complete. */
    unsigned reads_before = __atomic_load_n(&read_calls, __ATOMIC_RELAXED);
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    ds4_engram_reader_free(reader);
    unsigned actual_reads = __atomic_load_n(&read_calls, __ATOMIC_RELAXED) - reads_before;
    assert(o.rows_completed == actual_reads && o.read_bytes == (uint64_t)actual_reads * 264);
    assert(o.failed_calls == (actual_reads != 48));

    clock_failure = 0;
    /* Conversion failure counts completed reads, not imaginary successful output. */
    raw[0][0] = 127;
    assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(!ds4_engram_reader_wait(reader, 1) && errno == EDOM);
    assert(o.failed_calls == 1 && o.rows_completed == 8 && o.read_bytes == 8 * 264);
    ds4_engram_reader_free(reader);
    /* Failure of table0 must not shorten table1's descriptor/output lifetime. */
    tables[1].offset = DS4_ENGRAM_ROW_BYTES; tables[1].rows = 1;
    reader = ds4_engram_reader_create(); assert(reader);
    assert(ds4_engram_reader_start_observed(reader, tables, ids, output[0], &o));
    assert(!ds4_engram_reader_wait(reader, 0) && errno == EDOM);
    assert(!o.read_bytes);
    assert(ds4_engram_reader_wait(reader, 1));
    assert(o.failed_calls == 1 && o.rows_completed == 28 && o.read_bytes == 28 * 264);
    assert(output[1][0] == 2.0f);
    ds4_engram_reader_free(reader);
    for (unsigned t = 0; t < 2; ++t) ds4_engram_table_close(&tables[t]);
    assert(!close(fd));
    puts("persistent Engram observations: wait0 lifetime, wait1/free publication, partial/conversion/clock failures and no double counting: PASS");
    return 0;
}
