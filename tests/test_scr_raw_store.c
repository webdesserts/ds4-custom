/* Real private file I/O with syscall-boundary faults; no GPU or model. */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
static ssize_t test_write(int, const void *, size_t, off_t);
static ssize_t test_read(int, void *, size_t, off_t);
static int test_close(int);
static int test_stat(int, struct stat *);
static int test_unlink(const char *);
#define fstat test_stat
#define unlink test_unlink
#define pwrite test_write
#define pread test_read
#define close test_close
#define DS4_SCR_ENGINE_RAW_STORE
#include "../ds4_scr_engine.inc"
#undef pwrite
#undef pread
#undef close
#undef fstat
#undef unlink
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s errno=%d\n", __LINE__, #x, errno); exit(1); } } while (0)
static int write_fault, read_fault, close_fault, write_calls, stat_fault, unlink_fault;
static int test_stat(int fd, struct stat *st) {
    if (stat_fault) { if (stat_fault == 1) stat_fault = 0; errno = EIO; return -1; }
    return fstat(fd, st);
}
static int test_unlink(const char *path) {
    if (unlink_fault) { unlink_fault = 0; errno = EACCES; return -1; }
    return unlink(path);
}
static ssize_t test_write(int fd, const void *p, size_t n, off_t at) {
    ++write_calls;
    if (write_fault == 1) { write_fault = 2; errno = EINTR; return -1; }
    if (write_fault == 3 && write_calls > 1) { errno = ENOSPC; return -1; }
    if (write_fault == 4) return 0;
    if ((write_fault == 2 || write_fault == 3) && n > 8191u) n = 8191u;
    return pwrite(fd, p, n, at);
}
static ssize_t test_read(int fd, void *p, size_t n, off_t at) {
    if (read_fault == 1) { read_fault = 2; errno = EINTR; return -1; }
    if (read_fault == 3) return 0;
    if (read_fault == 2 && n > 65521u) n = 65521u;
    return pread(fd, p, n, at);
}
static int test_close(int fd) {
    if (close_fault == 2) {
        close_fault = 0;
        if (close(fd)) return -1;
        errno = EIO; return -1; /* Real close happened before the reported error. */
    }
    if (close_fault == 3) {
        close_fault = 0; stat_fault = 1; errno = EIO; return -1;
    }
    if (close_fault) { close_fault = 0; errno = EINTR; return -1; }
    return close(fd);
}
static bool cancelled(void *ud) { return *(bool *)ud; }
static uint32_t value(uint32_t token, uint32_t layer, uint32_t column) {
    return token * 2654435761u ^ layer * 2246822519u ^ column;
}
static void fill(void *data, uint32_t start, uint32_t count) {
    uint32_t *p = data;
    for (uint32_t r = 0; r < count / 8u; ++r)
        for (uint32_t l = 0; l < 40u; ++l)
            for (uint32_t t = 0; t < 8u; ++t)
                for (uint32_t c = 0; c < 512u; ++c)
                    *p++ = value(start + r * 8u + t, l, c);
}
int main(void) {
    char directory[] = "/tmp/ds4-raw-test.XXXXXX";
    CHECK(mkdtemp(directory));
    size_t arena_bytes = (size_t)ds41_raw_bytes(128);
    size_t read_bytes = (size_t)(17u * DS41_RAW_RECORD_BYTES);
    void *arena = malloc(arena_bytes), *readback = malloc(read_bytes);
    uint32_t *layer = malloc(128u * DS41_RAW_ROW_BYTES);
    CHECK(arena && readback && layer);
    for (uint32_t l = 0; l < 40; ++l)
        for (uint32_t t = 0; t < 128; ++t) {
            uint64_t offset;
            CHECK(ds41_raw_arena_offset(16, 16 + t, l, &offset));
            CHECK(offset == ((uint64_t)(t / 8u) * 40u * 8u + l * 8u + t % 8u) * 2048u);
            CHECK(offset + 2048u <= arena_bytes);
        }
    uint64_t offset;
    CHECK(!ds41_raw_arena_offset(1, 2, 0, &offset));
    CHECK(!ds41_raw_arena_offset(16, 15, 0, &offset));
    CHECK(!ds41_raw_arena_offset(16, 16 + DS41_RAW_ARENA_TOKENS, 0, &offset));
    CHECK(ds41_raw_arena_offset(16, 15 + DS41_RAW_ARENA_TOKENS, 0, &offset));
    CHECK(!ds41_raw_arena_offset(16, 16, 40, &offset));
    ds41_raw_store s = DS41_RAW_STORE_INIT;
    CHECK(!ds41_raw_open(&s, directory, 256, ds41_raw_bytes(128)) && s.fd < 0);
    stat_fault = 2;
    CHECK(!ds41_raw_open(&s, directory, 256, ds41_raw_bytes(256)));
    CHECK(s.fd >= 0 && s.cleanup_path[0] && !s.identity_known);
    stat_fault = 0;
    CHECK(ds41_raw_close(&s));
    unlink_fault = 1;
    CHECK(!ds41_raw_open(&s, directory, 256, ds41_raw_bytes(256)));
    CHECK(s.fd >= 0 && s.cleanup_path[0] && !s.writable);
    CHECK(ds41_raw_close(&s));
    stat_fault = 1; close_fault = 1;
    CHECK(!ds41_raw_open(&s, directory, 256, ds41_raw_bytes(256)));
    CHECK(s.fd >= 0);
    CHECK(ds41_raw_close(&s));
    CHECK(ds41_raw_open(&s, directory, 256, ds41_raw_bytes(256)));
    struct stat st;
    CHECK(!fstat(s.fd, &st) && !st.st_nlink && (st.st_mode & 0777) == 0600);
    CHECK(fcntl(s.fd, F_GETFD) & FD_CLOEXEC);
    fill(arena, 0, 128);
    write_fault = 1; write_calls = 0;
    CHECK(ds41_raw_append(&s, arena, 128, NULL, NULL) && write_calls > 2);
    CHECK(s.written_tokens == 128 && !s.committed_tokens);
    CHECK(!ds41_raw_read(&s, 0, 1, readback, read_bytes, NULL, NULL));
    CHECK(!ds41_raw_commit(&s, 136));
    CHECK(ds41_raw_commit(&s, 128));
    CHECK(s.allocation_known && s.allocation_time_known && s.allocation_checked_nanoseconds);
    CHECK(!fstat(s.fd, &st) && s.file_bytes == (uint64_t)st.st_size &&
        s.allocated_bytes == (uint64_t)st.st_blocks * 512u);
    CHECK(s.written_bytes == ds41_raw_bytes(128) && !s.read_bytes && !s.counters_overflow);
    fill(arena, 128, 128);
    write_fault = 3; write_calls = 0;
    CHECK(!ds41_raw_append(&s, arena, 8, NULL, NULL) && errno == ENOSPC);
    CHECK(s.committed_tokens == 128 && s.written_tokens == 128 && !s.writable && s.readable);
    CHECK(s.file_bytes == ds41_raw_bytes(128) + 8191u);
    CHECK(s.written_bytes == ds41_raw_bytes(128) + 8191u);
    read_fault = 1;
    CHECK(ds41_raw_read(&s, 0, 1, readback, read_bytes, NULL, NULL));
    CHECK(((uint32_t *)readback)[0] == value(0, 0, 0));
    CHECK(s.read_bytes == DS41_RAW_RECORD_BYTES);
    CHECK(ds41_raw_discard_pending(&s) && s.file_bytes == ds41_raw_bytes(128));
    write_fault = 4;
    CHECK(!ds41_raw_append(&s, arena, 8, NULL, NULL) && errno == EIO);
    CHECK(ds41_raw_discard_pending(&s));
    bool stop = true;
    write_fault = 0;
    CHECK(!ds41_raw_append(&s, arena, 8, cancelled, &stop) && errno == ECANCELED);
    CHECK(ds41_raw_discard_pending(&s));
    CHECK(ds41_raw_append(&s, arena, 8, NULL, NULL));
    CHECK(!ds41_raw_read(&s, 16, 1, readback, read_bytes, NULL, NULL));
    /* Owning metadata/cancellation failure after the complete write. */
    CHECK(ds41_raw_discard_pending(&s) && s.committed_tokens == 128);
    CHECK(ds41_raw_append(&s, arena, 128, NULL, NULL));
    CHECK(ds41_raw_commit(&s, 256));
    CHECK(!ds41_raw_append(&s, arena, 8, NULL, NULL));
    const uint32_t ends[] = {128, 130, 132, 134, 136, 254, 256};
    for (size_t i = 0; i < sizeof(ends) / sizeof(ends[0]); ++i) {
        uint32_t first, records;
        CHECK(ds41_raw_window_records(256, ends[i], &first, &records));
        CHECK(records == (ends[i] % 8u ? 17u : 16u));
        CHECK(ds41_raw_read(&s, first, records, readback, read_bytes, NULL, NULL));
        for (uint32_t l = 0; l < 40u; ++l) {
            CHECK(ds41_raw_gather_layer(readback, read_bytes, first, records,
                                       ends[i] - 128u, l, layer, 128u * DS41_RAW_ROW_BYTES));
            for (uint32_t t = 0; t < 128u; ++t)
                for (uint32_t c = 0; c < 512u; ++c)
                    CHECK(layer[t * 512u + c] == value(ends[i] - 128u + t, l, c));
        }
    }
    uint32_t first, records;
    CHECK(!ds41_raw_window_records(256, 129, &first, &records));
    CHECK(!ds41_raw_window_records(256, 258, &first, &records));
    CHECK(!ds41_raw_window_records(UINT32_MAX, UINT32_MAX - 1u, &first, &records));
    CHECK(ds41_raw_window_records(200000, 200000, &first, &records));
    CHECK(ds41_raw_bytes(200000) == UINT64_C(16384000000));
    CHECK(!ds41_raw_read(&s, 0, 1, readback, read_bytes, cancelled, &stop) && s.readable);
    read_fault = 3;
    CHECK(!ds41_raw_read(&s, 0, 1, readback, read_bytes, NULL, NULL) && !s.readable);
    close_fault = 1;
    CHECK(!ds41_raw_close(&s) && s.fd >= 0);
    CHECK(s.close_error == EINTR && s.cleanup_state == DS41_RAW_CLEANUP_PENDING);
    CHECK(ds41_raw_close(&s) && s.fd < 0);
    CHECK(ds41_raw_close(&s) && s.cleanup_state == DS41_RAW_CLOSED_WITH_ERROR);
    read_fault = 0;
    CHECK(ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)));
    fill(arena, 0, 128);
    CHECK(ds41_raw_append(&s, arena, 128, NULL, NULL) && ds41_raw_commit(&s, 128));
    CHECK(!ftruncate(s.fd, 0));
    CHECK(!ds41_raw_read(&s, 0, 1, readback, read_bytes, NULL, NULL) && !s.readable);
    CHECK(ds41_raw_close(&s));
    CHECK(ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)));
    close_fault = 2;
    CHECK(ds41_raw_close(&s) && s.fd < 0 && s.close_error == EIO);
    CHECK(s.cleanup_state == DS41_RAW_CLOSED_WITH_ERROR);
    CHECK(ds41_raw_close(&s) && s.close_error == EIO);
    CHECK(ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)));
    close_fault = 3;
    CHECK(!ds41_raw_close(&s) && s.fd >= 0 && s.close_error == EIO);
    CHECK(ds41_raw_close(&s) && s.cleanup_state == DS41_RAW_CLOSED_WITH_ERROR);
    CHECK(ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)));
    int old_fd = s.fd;
    CHECK(!close(old_fd));
    char other_path[PATH_MAX];
    CHECK(snprintf(other_path, sizeof(other_path), "%s/other", directory) > 0);
    int other = open(other_path, O_CREAT | O_EXCL | O_RDWR, 0600);
    CHECK(other >= 0);
    if (other != old_fd) { CHECK(dup2(other, old_fd) == old_fd); CHECK(!close(other)); }
    CHECK(ds41_raw_close(&s) && s.fd < 0 && s.identity_lost);
    CHECK(s.cleanup_state == DS41_RAW_CLOSED_WITH_ERROR && s.cleanup_error == ESTALE);
    CHECK(ds41_raw_close(&s) && s.identity_lost && s.cleanup_error == ESTALE);
    CHECK(fcntl(old_fd, F_GETFD) >= 0);
    CHECK(!close(old_fd) && !unlink(other_path));
    CHECK(ds41_raw_open(&s, directory, 128, ds41_raw_bytes(128)));
    s.written_bytes = UINT64_MAX - 4; /* Owned counter-overflow fault, not real usage. */
    CHECK(ds41_raw_append(&s, arena, 8, NULL, NULL) && ds41_raw_commit(&s, 8));
    CHECK(s.counters_overflow && s.written_bytes == UINT64_MAX);
    s.read_bytes = UINT64_MAX - 4;
    CHECK(ds41_raw_read(&s, 0, 1, readback, read_bytes, NULL, NULL));
    CHECK(s.read_bytes == UINT64_MAX && s.counters_overflow && ds41_raw_close(&s));
    free(layer); free(readback); free(arena);
    CHECK(!rmdir(directory));
    puts("raw-store: exact record/layer bytes; even seams; unpublished/read bounds; short I/O/EINTR/ENOSPC/EOF/cancellation; prior-prefix preservation; close retry; no GPU/model");
    return 0;
}
