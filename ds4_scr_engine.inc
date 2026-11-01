/* Suffix cache reuse (SCR) engine for DeepSeek V4.1 Flash, textually included
 * by ds4.c. Each part is selected by a section macro at the point in ds4.c
 * where its dependencies are complete, and emits at most once:
 *
 *   DS4_SCR_ENGINE_COORDINATES  pure physical/logical coordinate maps (also
 *                               included standalone by host tests).
 *   DS4_SCR_ENGINE_RAW_STORE    cold raw-row file store (standalone-capable).
 *   DS4_SCR_ENGINE_ARCHIVE      GPU archive, raw capture, append and rebase
 *                               import; inside the Flash GPU graph section,
 *                               after COORDINATES. Implies RAW_STORE.
 *   DS4_SCR_ENGINE_GUARDS       session guards, after struct ds4_session.
 *   DS4_SCR_ENGINE_SESSION      the owning-session SCR policy: prefix, rebase
 *                               and append transactions, multi-range
 *                               execution, refresh, planner adapter, the
 *                               logical policy entry point and the persistent
 *                               checkpoint pool.
 *
 * Banners group the code by concern; a banner may hold more than one part. */

/* ==== 1. Archive & Raw Store ============================================= */
#if (defined(DS4_SCR_ENGINE_RAW_STORE) || defined(DS4_SCR_ENGINE_ARCHIVE)) && !defined(DS4_SCR_ENGINE_RAW_STORE_DONE)
#define DS4_SCR_ENGINE_RAW_STORE_DONE
/* Session-local canonical raw rows. File completion is not crash durability;
 * the owning archive separately certifies tokens, tags, maps and headed state. */
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/clonefile.h>
#endif
#define DS4_SCR_OBSERVATION_CLOCK
#include "ds4_scr.h"

#define DS41_RAW_RECORD_TOKENS 8u
#define DS41_RAW_LAYERS 40u
#define DS41_RAW_ROW_BYTES 2048u
#define DS41_RAW_RECORD_BYTES UINT64_C(655360)
/* Widest capture arena: one session prefill chunk (DS4_SESSION_PREFILL_CHUNK).
 * The 128-row SWA window read back per reused range is a separate bound. */
#define DS41_RAW_ARENA_TOKENS 2048u
#define DS41_RAW_READ_RECORDS 17u
#define DS41_RAW_MAX_TOKENS 1048576u
#define DS41_RAW_STORE_INIT {.fd = -1}

typedef bool (*ds41_raw_cancel_fn)(void *);
typedef enum {
    DS41_RAW_OPEN, DS41_RAW_CLEANUP_PENDING, DS41_RAW_CLOSED,
    DS41_RAW_CLOSED_WITH_ERROR
} ds41_raw_cleanup_state;
typedef struct {
    int fd, close_error, cleanup_error;
    ds41_raw_cleanup_state cleanup_state;
    bool identity_lost;
    dev_t device;
    ino_t inode;
    uint32_t capacity, written_tokens, committed_tokens;
    uint64_t quota_bytes, file_bytes, allocated_bytes, read_bytes, written_bytes;
    uint64_t allocation_checked_nanoseconds;
    bool allocation_known, allocation_time_known, counters_overflow;
    bool readable, writable, identity_known;
    char cleanup_path[PATH_MAX]; /* Only a failed initial unlink can retain this. */
} ds41_raw_store;

static inline uint64_t ds41_raw_bytes(uint32_t tokens) {
    return (uint64_t)tokens * DS41_RAW_LAYERS * DS41_RAW_ROW_BYTES;
}
static inline bool ds41_raw_identity(ds41_raw_store *s, struct stat *st) {
    if (!s || s->fd < 0) { errno = EBADF; return false; }
    if (fstat(s->fd, st)) return false;
    if (!s->identity_known || !S_ISREG(st->st_mode) || st->st_dev != s->device || st->st_ino != s->inode ||
        st->st_nlink || st->st_size < 0 || st->st_blocks < 0 ||
        (uint64_t)st->st_blocks > UINT64_MAX / 512u) {
        s->readable = s->writable = false;
        errno = ESTALE; return false;
    }
    s->file_bytes = (uint64_t)st->st_size;
    s->allocated_bytes = (uint64_t)st->st_blocks * 512u;
    s->allocation_known = true;
    s->allocation_time_known = ds4_observation_clock(&s->allocation_checked_nanoseconds);
    if (s->file_bytes < ds41_raw_bytes(s->committed_tokens)) {
        s->readable = s->writable = false;
        errno = EIO; return false;
    }
    return true;
}
static inline bool ds41_raw_cleanup_pending(ds41_raw_store *s, int error) {
    s->cleanup_state = DS41_RAW_CLEANUP_PENDING;
    s->cleanup_error = error;
    errno = error;
    return false;
}
/* True means no owned fd/path obligation remains, not that every syscall was
 * clean. Diagnostics persist across later attempts; owners must report them.
 * Exclusive ownership excludes outside dup/close/rename during disposal. */
static inline bool ds41_raw_close(ds41_raw_store *s) {
    if (!s) { errno = EINVAL; return false; }
    s->readable = s->writable = false;
    struct stat st;
    if (s->fd >= 0) {
        if (fstat(s->fd, &st)) {
            if (errno != EBADF) return ds41_raw_cleanup_pending(s, errno);
            s->fd = -1;
            s->cleanup_error = EBADF;
        } else {
            if (!s->identity_known) {
                struct stat named;
                if (!s->cleanup_path[0]) return ds41_raw_cleanup_pending(s, ESTALE);
                if (lstat(s->cleanup_path, &named)) return ds41_raw_cleanup_pending(s, errno);
                if (!S_ISREG(st.st_mode) || named.st_dev != st.st_dev || named.st_ino != st.st_ino)
                    return ds41_raw_cleanup_pending(s, ESTALE);
                s->device = st.st_dev; s->inode = st.st_ino; s->identity_known = true;
            }
            if (st.st_dev != s->device || st.st_ino != s->inode) {
                /* This number now names a foreign descriptor: never close it. */
                s->fd = -1; s->identity_lost = true; s->cleanup_error = ESTALE;
            }
        }
    }
    if (s->cleanup_path[0]) {
        struct stat named;
        if (!s->identity_known) return ds41_raw_cleanup_pending(s, ESTALE);
        if (lstat(s->cleanup_path, &named)) {
            if (errno != ENOENT) return ds41_raw_cleanup_pending(s, errno);
        } else {
            if (named.st_dev != s->device || named.st_ino != s->inode)
                return ds41_raw_cleanup_pending(s, ESTALE);
            if (unlink(s->cleanup_path)) return ds41_raw_cleanup_pending(s, errno);
        }
        s->cleanup_path[0] = '\0';
    }
    if (s->fd >= 0) {
        if (!close(s->fd)) s->fd = -1;
        else {
            int saved = errno;
            if (!s->close_error) s->close_error = saved;
            /* Revalidation, not errno alone, decides whether ownership remains. */
            if (fstat(s->fd, &st)) {
                if (errno != EBADF) return ds41_raw_cleanup_pending(s, errno);
                s->fd = -1;
            } else if (st.st_dev != s->device || st.st_ino != s->inode) {
                s->fd = -1; s->identity_lost = true;
            } else return ds41_raw_cleanup_pending(s, saved);
        }
    }
    s->cleanup_state = s->close_error || s->cleanup_error || s->identity_lost ?
        DS41_RAW_CLOSED_WITH_ERROR : DS41_RAW_CLOSED;
    return true;
}
static inline bool ds41_raw_open(ds41_raw_store *s, const char *directory,
                                 uint32_t capacity, uint64_t quota_bytes) {
    if (!s || s->fd >= 0 || s->cleanup_path[0] || !directory || !*directory || capacity < 128u ||
        capacity > DS41_RAW_MAX_TOKENS || capacity % DS41_RAW_RECORD_TOKENS ||
        quota_bytes < ds41_raw_bytes(capacity)) { errno = EINVAL; return false; }
#ifndef __APPLE__
    /* This first implementation requires the checked Darwin no-cache contract. */
    errno = ENOTSUP; return false;
#else
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%s/ds4-scr-raw.XXXXXX", directory);
    if (length < 0 || (size_t)length >= sizeof(path)) { errno = ENAMETOOLONG; return false; }
    int fd = mkstemp(path);
    if (fd < 0) return false;
    *s = (ds41_raw_store){.fd = fd, .capacity = capacity, .quota_bytes = quota_bytes};
    memcpy(s->cleanup_path, path, (size_t)length + 1u);
    struct stat st;
    if (fstat(fd, &st)) {
        int saved = errno;
        ds41_raw_close(s); /* Failed cleanup remains represented by fd/path. */
        errno = saved; return false;
    }
    s->device = st.st_dev; s->inode = st.st_ino; s->identity_known = true;
    if (unlink(path)) return false; /* Caller retains cleanup ownership on failure. */
    s->cleanup_path[0] = '\0';
    int saved = 0;
    if (fchmod(fd, S_IRUSR | S_IWUSR) || fcntl(fd, F_SETFD, FD_CLOEXEC) ||
        fcntl(fd, F_NOCACHE, 1) || fcntl(fd, F_RDAHEAD, 0)) saved = errno;
    struct statvfs space;
    if (!saved && fstatvfs(fd, &space)) saved = errno;
    if (!saved) {
        uint64_t unit = space.f_frsize ? space.f_frsize : space.f_bsize;
        uint64_t available = unit && space.f_bavail <= UINT64_MAX / unit ?
            (uint64_t)space.f_bavail * unit : 0;
        if (!unit || available < ds41_raw_bytes(capacity)) saved = ENOSPC;
    }
    if (saved) { ds41_raw_close(s); errno = saved; return false; }
    s->readable = s->writable = true;
    return true;
#endif
}
/* Adopt `tokens` committed records persisted at `source` as a new store: an
 * APFS clone in `directory`, unlinked once opened like ds41_raw_open's file.
 * Only a same-volume clone is attempted; on failure the store stays closed
 * and the caller copies records through ds41_raw_open/append instead. */
static inline bool ds41_raw_adopt(ds41_raw_store *s, const char *directory,
                                  const char *source, uint32_t capacity,
                                  uint32_t tokens) {
    if (!s || s->fd >= 0 || s->cleanup_path[0] || !directory || !*directory || !source ||
        capacity < 128u || capacity > DS41_RAW_MAX_TOKENS || capacity % DS41_RAW_RECORD_TOKENS ||
        tokens > capacity || tokens % DS41_RAW_RECORD_TOKENS) { errno = EINVAL; return false; }
#ifndef __APPLE__
    errno = ENOTSUP; return false;
#else
    char path[PATH_MAX];
    int length = snprintf(path, sizeof(path), "%s/ds4-scr-raw.XXXXXX", directory);
    if (length < 0 || (size_t)length >= sizeof(path)) { errno = ENAMETOOLONG; return false; }
    /* Reserve a unique name, then let the clone take it: clonefile never
     * replaces an existing path. */
    int fd = mkstemp(path);
    if (fd < 0) return false;
    close(fd);
    if (unlink(path) || clonefile(source, path, CLONE_NOFOLLOW)) return false;
    fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) { int saved = errno; unlink(path); errno = saved; return false; }
    *s = (ds41_raw_store){.fd = fd, .capacity = capacity, .quota_bytes = ds41_raw_bytes(capacity)};
    memcpy(s->cleanup_path, path, (size_t)length + 1u);
    struct stat st;
    const int bad = fstat(fd, &st) ? errno : !S_ISREG(st.st_mode) ||
        (uint64_t)st.st_size != ds41_raw_bytes(tokens) ? EIO : 0;
    if (bad) { ds41_raw_close(s); errno = bad; return false; }
    s->device = st.st_dev; s->inode = st.st_ino; s->identity_known = true;
    if (unlink(path)) return false; /* Caller retains cleanup ownership on failure. */
    s->cleanup_path[0] = '\0';
    int saved = 0;
    if (fchmod(fd, S_IRUSR | S_IWUSR) || fcntl(fd, F_NOCACHE, 1) || fcntl(fd, F_RDAHEAD, 0))
        saved = errno;
    struct statvfs space;
    if (!saved && fstatvfs(fd, &space)) saved = errno;
    if (!saved) {
        /* Cloned records share blocks; only later appends need free space. */
        uint64_t unit = space.f_frsize ? space.f_frsize : space.f_bsize;
        uint64_t available = unit && space.f_bavail <= UINT64_MAX / unit ?
            (uint64_t)space.f_bavail * unit : 0;
        if (!unit || available < ds41_raw_bytes(capacity - tokens)) saved = ENOSPC;
    }
    if (saved) { ds41_raw_close(s); errno = saved; return false; }
    s->written_tokens = s->committed_tokens = tokens;
    s->file_bytes = ds41_raw_bytes(tokens);
    s->readable = s->writable = true;
    return true;
#endif
}
/* Append only complete records. written_tokens is readiness, never publication.
 * Quota/available-space admission is not an exclusive filesystem reservation. */
static inline bool ds41_raw_append(ds41_raw_store *s, const void *data, uint32_t tokens,
                                   ds41_raw_cancel_fn cancel, void *ud) {
    struct stat st;
    if (!s || !data || !s->writable || !tokens || tokens % DS41_RAW_RECORD_TOKENS ||
        tokens > DS41_RAW_ARENA_TOKENS || s->written_tokens != s->committed_tokens ||
        s->written_tokens > s->capacity || tokens > s->capacity - s->written_tokens) {
        errno = EINVAL; return false;
    }
    if (!ds41_raw_identity(s, &st)) { s->readable = s->writable = false; return false; }
    uint64_t offset = ds41_raw_bytes(s->written_tokens), bytes = ds41_raw_bytes(tokens);
    if (offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset ||
        offset > s->quota_bytes || bytes > s->quota_bytes - offset) { errno = EFBIG; return false; }
    uint64_t done = 0;
    while (done < bytes) {
        if (cancel && cancel(ud)) { errno = ECANCELED; break; }
        ssize_t n = pwrite(s->fd, (const unsigned char *)data + done,
                           (size_t)(bytes - done), (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; break; }
        done += (uint64_t)n;
        ds4_observation_add_u64(&s->written_bytes, (uint64_t)n, &s->counters_overflow);
    }
    int saved = done == bytes ? 0 : errno;
    bool valid = ds41_raw_identity(s, &st);
    if (!valid || saved) {
        s->writable = false;
        if (saved) errno = saved;
        return false;
    }
    s->written_tokens += tokens;
    return true;
}
static inline bool ds41_raw_commit(ds41_raw_store *s, uint32_t tokens) {
    struct stat st;
    if (!s || !s->writable || !s->readable || tokens != s->written_tokens ||
        tokens < s->committed_tokens || tokens % DS41_RAW_RECORD_TOKENS) {
        errno = EINVAL; return false;
    }
    if (!ds41_raw_identity(s, &st)) return false;
    if (s->file_bytes < ds41_raw_bytes(tokens)) { errno = EIO; return false; }
    s->committed_tokens = tokens;
    return true;
}
static inline bool ds41_raw_discard_pending(ds41_raw_store *s) {
    struct stat st;
    if (!s || !s->readable || !ds41_raw_identity(s, &st)) return false;
    if (ftruncate(s->fd, (off_t)ds41_raw_bytes(s->committed_tokens))) {
        s->writable = false; return false;
    }
    if (!ds41_raw_identity(s, &st)) { s->writable = false; return false; }
    s->written_tokens = s->committed_tokens;
    s->writable = true;
    return true;
}
static inline bool ds41_raw_read(ds41_raw_store *s, uint32_t first_record,
                                 uint32_t records, void *data, size_t data_bytes,
                                 ds41_raw_cancel_fn cancel, void *ud) {
    struct stat st;
    if (!s || !data || !s->readable || !records || records > DS41_RAW_READ_RECORDS ||
        first_record > s->committed_tokens / 8u ||
        records > s->committed_tokens / 8u - first_record ||
        data_bytes < (uint64_t)records * DS41_RAW_RECORD_BYTES) { errno = EINVAL; return false; }
    if (!ds41_raw_identity(s, &st)) { s->readable = s->writable = false; return false; }
    uint64_t offset = (uint64_t)first_record * DS41_RAW_RECORD_BYTES;
    uint64_t bytes = (uint64_t)records * DS41_RAW_RECORD_BYTES, done = 0;
    while (done < bytes) {
        if (cancel && cancel(ud)) { errno = ECANCELED; return false; }
        ssize_t n = pread(s->fd, (unsigned char *)data + done,
                          (size_t)(bytes - done), (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (!n) errno = EIO;
            s->readable = s->writable = false;
            return false;
        }
        done += (uint64_t)n;
        ds4_observation_add_u64(&s->read_bytes, (uint64_t)n, &s->counters_overflow);
    }
    return true;
}
/* Asks the kernel to start reading committed records in the background. An
 * evicted store reads at device speed; advising every range a rebase will
 * import before the prefix runs overlaps that I/O with the prefix, stages
 * and import copies. Advisory only: errors are ignored, reads stay checked. */
static inline void ds41_raw_advise(ds41_raw_store *s, uint32_t first_record, uint32_t records) {
#ifdef __APPLE__
    if (!s || s->fd < 0 || !s->readable || first_record > s->committed_tokens / 8u) return;
    if (records > s->committed_tokens / 8u - first_record)
        records = s->committed_tokens / 8u - first_record;
    uint64_t offset = (uint64_t)first_record * DS41_RAW_RECORD_BYTES;
    uint64_t left = (uint64_t)records * DS41_RAW_RECORD_BYTES;
    while (left) {
        const uint64_t n = left < (UINT64_C(1) << 30) ? left : (UINT64_C(1) << 30);
        struct radvisory advice = {.ra_offset = (off_t)offset, .ra_count = (int)n};
        if (fcntl(s->fd, F_RDADVISE, &advice) == -1) return;
        offset += n; left -= n;
    }
#else
    (void)s; (void)first_record; (void)records;
#endif
}
static inline bool ds41_raw_arena_offset(uint32_t origin, uint32_t token,
                                         uint32_t layer, uint64_t *offset) {
    if (!offset || origin % 8u || token < origin || token - origin >= DS41_RAW_ARENA_TOKENS || layer >= 40u)
        return false;
    uint32_t row = token - origin;
    *offset = (uint64_t)(row / 8u) * DS41_RAW_RECORD_BYTES +
        (uint64_t)(layer * 8u + row % 8u) * DS41_RAW_ROW_BYTES;
    return true;
}
static inline bool ds41_raw_window_records(uint32_t covered, uint32_t end,
                                           uint32_t *first, uint32_t *records) {
    if (!first || !records || covered > DS41_RAW_MAX_TOKENS || covered % 8u ||
        end < 128u || end > covered || end % 2u) return false;
    *first = (end - 128u) / 8u;
    *records = (end + 7u) / 8u - *first;
    return *records <= DS41_RAW_READ_RECORDS;
}
/* Read-ahead of one range window on a helper thread while the owner keeps the
 * GPU busy. The helper only preads committed records into a private buffer;
 * readability, identity and byte counters stay with the owning thread, which
 * must finish (join) the prefetch before the store can be closed or fenced. */
typedef struct {
    ds41_raw_store *store;
    unsigned char *data;
    uint64_t offset, bytes, done;
    uint32_t first, records;
    int fd, error;
    pthread_t thread;
    bool running, ready;
} ds41_raw_prefetch;

static void *ds41_raw_prefetch_main(void *arg) {
    ds41_raw_prefetch *p = arg;
    while (p->done < p->bytes) {
        ssize_t n = pread(p->fd, p->data + p->done, (size_t)(p->bytes - p->done),
                          (off_t)(p->offset + p->done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { p->error = n ? errno : EIO; break; }
        p->done += (uint64_t)n;
    }
    return NULL;
}
/* False means no read-ahead started; the caller reads synchronously later. */
static inline bool ds41_raw_prefetch_start(ds41_raw_prefetch *p, ds41_raw_store *s,
                                           uint32_t first_record, uint32_t records,
                                           unsigned char *data) {
    struct stat st;
    if (!p || p->running || !s || !data || !s->readable || !records ||
        records > DS41_RAW_READ_RECORDS || first_record > s->committed_tokens / 8u ||
        records > s->committed_tokens / 8u - first_record ||
        !ds41_raw_identity(s, &st)) return false;
    *p = (ds41_raw_prefetch){.store = s, .data = data, .first = first_record,
        .records = records, .fd = s->fd,
        .offset = (uint64_t)first_record * DS41_RAW_RECORD_BYTES,
        .bytes = (uint64_t)records * DS41_RAW_RECORD_BYTES};
    p->running = !pthread_create(&p->thread, NULL, ds41_raw_prefetch_main, p);
    return p->running;
}
/* Join and apply the read on the owning thread. A failed read fences the
 * store exactly like ds41_raw_read; success requires unchanged identity. */
static inline bool ds41_raw_prefetch_finish(ds41_raw_prefetch *p) {
    struct stat st;
    if (!p || !p->running) return false;
    pthread_join(p->thread, NULL);
    p->running = false;
    ds41_raw_store *s = p->store;
    ds4_observation_add_u64(&s->read_bytes, p->done, &s->counters_overflow);
    if (p->error || p->done != p->bytes) {
        s->readable = s->writable = false;
        errno = p->error ? p->error : EIO;
        return false;
    }
    p->ready = s->fd == p->fd && s->readable && ds41_raw_identity(s, &st);
    return p->ready;
}

/* Gather a bounded canonical subrange without copying neighboring record rows. */
static inline bool ds41_raw_gather_rows(const void *data, size_t bytes, uint32_t first,
                                        uint32_t records, uint32_t start, uint32_t count,
                                        uint32_t layer, void *output, size_t output_bytes) {
    if (!data || !output || !records || records > DS41_RAW_READ_RECORDS || layer >= 40u ||
        !count || count > DS41_RAW_ARENA_TOKENS ||
        first > DS41_RAW_MAX_TOKENS / 8u || start < first * 8u ||
        (uint64_t)start + count > (uint64_t)(first + records) * 8u ||
        bytes < (uint64_t)records * DS41_RAW_RECORD_BYTES ||
        output_bytes < (uint64_t)count * DS41_RAW_ROW_BYTES) return false;
    for (uint32_t done = 0; done < count;) {
        uint32_t row = start + done - first * 8u, n = 8u - row % 8u;
        if (n > count - done) n = count - done;
        uint64_t offset = (uint64_t)(row / 8u) * DS41_RAW_RECORD_BYTES +
            (uint64_t)(layer * 8u + row % 8u) * DS41_RAW_ROW_BYTES;
        memcpy((unsigned char *)output + (uint64_t)done * DS41_RAW_ROW_BYTES,
               (const unsigned char *)data + offset, (size_t)n * DS41_RAW_ROW_BYTES);
        done += n;
    }
    return true;
}
static inline bool ds41_raw_gather_layer(const void *data, size_t bytes, uint32_t first,
                                         uint32_t records, uint32_t start, uint32_t layer,
                                         void *output, size_t output_bytes) {
    return ds41_raw_gather_rows(data, bytes, first, records, start, 128u,
                                layer, output, output_bytes);
}
#endif

#if defined(DS4_SCR_ENGINE_ARCHIVE) && !defined(DS4_SCR_ENGINE_ARCHIVE_DONE)
#define DS4_SCR_ENGINE_ARCHIVE_DONE
#ifndef DS4_SCR_ENGINE_COORDINATES_DONE
#error "the SCR archive section requires the coordinates section first"
#endif
/* Record-major GPU copies only. The owner drains before readback/publication;
 * tags certify scheduled rows, not completion of an outstanding command batch. */
#include "ds4_gpu.h"
#include "adapter/include/kv_archive_store.h"

/* Capture arena of an archive with capacity `cap`: the widest arena, clipped
 * to the capacity so small archives do not reserve an unusable arena. */
static inline uint32_t ds41_archive_arena_tokens(uint32_t cap) {
    return cap < DS41_RAW_ARENA_TOKENS ? cap : DS41_RAW_ARENA_TOKENS;
}
static inline bool ds41_raw_capture_rows(ds4_gpu_tensor *arena, uint32_t arena_tokens,
                                          uint32_t origin, uint32_t layer, uint32_t start,
                                          uint32_t rows, ds4_gpu_tensor *input,
                                          uint32_t tags[40][DS41_RAW_ARENA_TOKENS]) {
    if (!arena || !input || !tags || layer >= 40u || origin % 8u ||
        !arena_tokens || arena_tokens > DS41_RAW_ARENA_TOKENS ||
        origin > DS41_RAW_MAX_TOKENS || start > DS41_RAW_MAX_TOKENS ||
        rows > DS41_RAW_MAX_TOKENS - start ||
        ds4_gpu_tensor_bytes(arena) < ds41_raw_bytes(arena_tokens) ||
        ds4_gpu_tensor_bytes(input) < (uint64_t)rows * DS41_RAW_ROW_BYTES) return false;
    uint32_t end = start + rows, pos = start > origin ? start : origin;
    if (end > origin + arena_tokens) return false;
    while (pos < end) {
        uint32_t n = 8u - (pos - origin) % 8u;
        if (n > end - pos) n = end - pos;
        uint64_t offset;
        if (!ds41_raw_arena_offset(origin, pos, layer, &offset) ||
            !ds4_gpu_tensor_copy(arena, offset, input,
                (uint64_t)(pos - start) * DS41_RAW_ROW_BYTES,
                (uint64_t)n * DS41_RAW_ROW_BYTES)) return false;
        for (uint32_t i = 0; i < n; ++i) tags[layer][pos - origin + i] = pos + i;
        pos += n;
    }
    return true;
}
static inline bool ds41_raw_capture_complete(const uint32_t tags[40][DS41_RAW_ARENA_TOKENS],
                                              uint32_t origin, uint32_t rows) {
    if (!tags || origin % 8u || rows > DS41_RAW_ARENA_TOKENS ||
        origin > DS41_RAW_MAX_TOKENS - rows)
        return false;
    for (uint32_t l = 0; l < 40u; ++l)
        for (uint32_t t = 0; t < rows; ++t)
            if (tags[l][t] != origin + t) return false;
    return true;
}
/* Complete records have already been read back; retain the partial record for
 * the next bounded capture interval. A later drain failure fences the owner. */
static inline bool ds41_raw_capture_shift(ds4_gpu_tensor *arena, uint32_t arena_tokens,
                                           uint32_t tags[40][DS41_RAW_ARENA_TOKENS],
                                           uint32_t origin, uint32_t complete,
                                           uint32_t pending) {
    if (!arena || complete % 8u || pending > 7u ||
        !arena_tokens || arena_tokens > DS41_RAW_ARENA_TOKENS ||
        complete > arena_tokens - pending ||
        ds4_gpu_tensor_bytes(arena) < ds41_raw_bytes(arena_tokens) ||
        !ds41_raw_capture_complete(tags, origin, complete + pending)) return false;
    if (complete && pending && !ds4_gpu_tensor_copy(arena, 0, arena,
            ds41_raw_bytes(complete), DS41_RAW_RECORD_BYTES)) return false;
    for (uint32_t l = 0; l < 40u; ++l) {
        memmove(tags[l], tags[l] + complete, (size_t)pending * sizeof(uint32_t));
        memset(tags[l] + pending, 0xff, (arena_tokens - pending) * sizeof(uint32_t));
    }
    return true;
}

/* Canonical owners stay hot; historical raw rows are session-local cold state.
 * Reservations cover owned host/GPU buffers, not driver overhead or file cache. */

#define DS41_ARCHIVE_READ_BYTES (DS41_RAW_READ_RECORDS * DS41_RAW_RECORD_BYTES)
#define DS41_ARCHIVE_IO_BYTES (DS41_ARCHIVE_READ_BYTES + 128u * DS41_RAW_ROW_BYTES)
/* Threads for the layer-major gather of a prefetched window (11 MB). */
#define DS41_ARCHIVE_GATHER_THREADS 4u
/* One host buffer serves capture readback and range window reads + gathers. */
static inline uint64_t ds41_archive_io_bytes(uint32_t cap) {
    const uint64_t readback = ds41_raw_bytes(ds41_archive_arena_tokens(cap));
    return readback > DS41_ARCHIVE_IO_BYTES ? readback : DS41_ARCHIVE_IO_BYTES;
}

#define DS41_SCR_WINDOW_BYTES (40ull * 128u * 512u * sizeof(float))

/* Last materialized source->destination stage; valid only until the live
 * frontier or either generation moves. */
typedef struct {
    uint32_t start, end, source_start, source_end;
    uint64_t source_generation, destination_generation;
    bool valid;
} ds41_archive_stage_range;

typedef enum {
    DS41_ARCHIVE_INITIAL, DS41_ARCHIVE_ARMED, DS41_ARCHIVE_INCOMPATIBLE,
    DS41_ARCHIVE_BLOCKED, DS41_ARCHIVE_SATURATED
} ds41_archive_writer_state;

typedef struct {
    uint32_t cap, start, end, limit;
    ds41_archive_writer_state writer_state;
    uint32_t captured_end, writer_known_end, writer_future_pads;
    uint64_t writer_generation, writer_coordinate_identity;
    bool history_hole;
    uint64_t generation, source_generation, allocation_bytes, observation_id;
    const void *session, *model;
    bool lost, sealed, retiring;
    ds4_gpu_tensor *source, *stage, *scratch, *raw_arena;
    ds41_raw_store *raw;
    ds41_raw_cancel_fn cancelled;
    unsigned char *raw_io;
    uint32_t raw_origin, raw_arena_tokens, raw_tags[40][DS41_RAW_ARENA_TOKENS];
    double raw_readback_seconds, raw_write_seconds, raw_read_seconds, raw_gather_seconds;
    bool raw_timing_unknown;
    uint32_t *tags;
    int *tokens;
    ds41_scr_coordinates coordinates;
    uint32_t logical_start, stage_alignment;
    /* Direct predecessor is borrowed only until checked source transfer. */
    const void *rebase_predecessor;
    uint64_t predecessor_generation, predecessor_coordinate_identity;
    uint32_t predecessor_start, predecessor_end, predecessor_capacity;
    uint32_t rebase_head_end, imported_tokens;
    bool rebased, retirement_refused;
    ds41_archive_stage_range range;
} ds41_scr_archive;

/* Included after the private archive type. No file/tensor reads or ownership
 * effects: only copying actual owner state and reducing discharged counters. */

static uint64_t ds41_archive_observation_sequence;
static uint64_t ds41_archive_observation_next_id(void) {
    uint64_t prior = __atomic_load_n(&ds41_archive_observation_sequence, __ATOMIC_RELAXED);
    while (prior != UINT64_MAX) {
        if (__atomic_compare_exchange_n(&ds41_archive_observation_sequence, &prior, prior + 1,
                                          false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return prior + 1;
    }
    return 0; /* Exhaustion is unknown identity, never a reused/wrapped ID. */
}
static double ds41_archive_observation_time(ds41_scr_archive *a) {
    uint64_t ns;
    if (!ds4_observation_clock(&ns)) { a->raw_timing_unknown = true; return 0; }
    return (double)ns / 1e9;
}
static void ds41_archive_observation_cleanup(ds4_scr_owner_observation *o, const ds41_raw_store *r) {
    if (!o || !r) return;
    o->descriptor_present = r->fd >= 0;
    o->identity_known = r->identity_known;
    o->identity_lost = r->identity_lost;
    o->readable = r->readable;
    o->writable = r->writable;
    o->cleanup_state = (uint32_t)r->cleanup_state;
    o->close_error = r->close_error;
    o->cleanup_error = r->cleanup_error;
}
static ds4_scr_owner_observation ds41_archive_observe(const ds41_scr_archive *a) {
    ds4_scr_owner_observation o = {0};
    if (!a) return o;
    o.known = true;
    o.observation_time_known = ds4_observation_clock(&o.observed_nanoseconds);
    o.owner_id = a->observation_id;
    o.accounted_bytes = a->allocation_bytes;
    o.retiring = a->retiring;
    o.raw_timing_unknown = a->raw_timing_unknown;
    o.raw_readback_seconds = a->raw_readback_seconds;
    o.raw_write_seconds = a->raw_write_seconds;
    o.raw_read_seconds = a->raw_read_seconds;
    o.raw_gather_seconds = a->raw_gather_seconds;
    if (a->raw) {
        const ds41_raw_store *r = a->raw;
        o.raw_present = true;
        o.allocation_known = r->allocation_known;
        o.allocation_time_known = r->allocation_time_known;
        o.allocation_checked_nanoseconds = r->allocation_checked_nanoseconds;
        o.counters_overflow = r->counters_overflow;
        o.file_bytes = r->file_bytes;
        o.allocated_bytes = r->allocated_bytes;
        o.read_bytes = r->read_bytes;
        o.written_bytes = r->written_bytes;
        ds41_archive_observation_cleanup(&o, r);
    }
    return o;
}
static void ds41_archive_observation_closed(ds4_scr_closed_observation *sum,
                                            const ds4_scr_owner_observation *o) {
    if (!o->known || !o->retirement_complete) return;
    sum->known = true;
    sum->counters_overflow |= o->counters_overflow;
    sum->timing_unknown |= o->raw_timing_unknown;
    ds4_observation_add_u64(&sum->owners, 1, &sum->counters_overflow);
    ds4_observation_add_u64(&sum->read_bytes, o->read_bytes, &sum->counters_overflow);
    ds4_observation_add_u64(&sum->written_bytes, o->written_bytes, &sum->counters_overflow);
#define DS41_CLOSED_TIME(field) do { \
    if (!ds4_observation_seconds_valid(o->field)) sum->timing_unknown = true; \
    else { sum->field += o->field; if (!ds4_observation_seconds_valid(sum->field)) sum->timing_unknown = true; } \
} while (0)
    DS41_CLOSED_TIME(raw_readback_seconds); DS41_CLOSED_TIME(raw_write_seconds);
    DS41_CLOSED_TIME(raw_read_seconds); DS41_CLOSED_TIME(raw_gather_seconds);
#undef DS41_CLOSED_TIME
}

static uint64_t ds41_archive_hot_bytes(uint32_t cap) {
    return cap >= 128u && cap <= DS4_SCR_ARCHIVE_MAX_CAPACITY && !(cap % 8u) ?
        6400ull * cap : 0;
}
static uint64_t ds41_archive_data_bytes(uint32_t cap) {
    const uint64_t hot = ds41_archive_hot_bytes(cap);
    return hot ? hot + DS41_SCR_WINDOW_BYTES : 0;
}
static uint64_t ds41_archive_stage_bytes(uint32_t cap) {
    const uint64_t data = ds41_archive_data_bytes(cap);
    return data ? data + (uint64_t)cap * 512u * sizeof(float) : 0;
}
static uint64_t ds41_archive_bytes(uint32_t cap) {
    const uint64_t hot = ds41_archive_hot_bytes(cap);
    return hot ? hot + ((uint64_t)cap * 5u / 2u + cap) * sizeof(uint32_t) +
        ds41_archive_stage_bytes(cap) + ds41_raw_bytes(ds41_archive_arena_tokens(cap)) +
        ds41_archive_io_bytes(cap) +
        sizeof(ds41_raw_store) + sizeof(ds41_scr_archive) : 0;
}
/* Capacity determines strides; only the sealed interval is reusable source. */
static uint32_t ds41_archive_count(const ds41_scr_archive *a) {
    return a && a->sealed && a->end >= a->start && a->end - a->start <= a->cap ?
        a->end - a->start : 0;
}
static bool ds41_archive_shape(const ds41_scr_archive *a, uint32_t context) {
    if (!a || !ds41_archive_bytes(a->cap) || a->start % 8u ||
        a->start > 1048576u - a->cap || a->limit != a->start + a->cap ||
        a->limit > context || a->end < a->start || a->end > a->limit ||
        (a->end - a->start) % 8u) return false;
    return a->sealed ? ds41_archive_count(a) >= 128u : a->end == a->start;
}
static uint32_t ds41_archive_tag(uint32_t cap, uint32_t owner) {
    return owner < 3u ? owner * (cap / 2u) : 3u * (cap / 2u);
}
static uint64_t ds41_archive_owner(uint32_t cap, uint32_t owner) {
    return (uint64_t)ds41_archive_tag(cap, owner) * 640u * sizeof(float);
}
static uint64_t ds41_archive_ring(uint32_t cap, uint32_t layer) {
    return (uint64_t)cap * 6400u + (uint64_t)layer * 128u * 512u * sizeof(float);
}
static void ds41_archive_invalidate(ds41_scr_archive *a) {
    if (!a) return;
    a->range.valid = false;
    const uint32_t published = ds41_archive_count(a);
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        memset(a->tags + ds41_archive_tag(a->cap, owner) + published / ratio, 0xff,
               (size_t)((a->cap - published) / ratio) * sizeof(uint32_t));
    }
    memset(a->raw_tags, 0xff, sizeof(a->raw_tags));
    if (a->sealed && a->writer_state != DS41_ARCHIVE_BLOCKED)
        a->writer_state = DS41_ARCHIVE_INCOMPATIBLE;
    if (a->generation == UINT64_MAX) a->lost = true;
    else {
        ++a->generation;
        /* A reset neither relabels pending history nor re-arms a blocked writer. */
        a->lost = a->writer_state == DS41_ARCHIVE_BLOCKED || (!a->sealed && a->raw &&
            (a->raw->committed_tokens || a->raw->written_tokens || a->raw->file_bytes));
    }
}
/* Fallible raw cleanup precedes every allocation release. The checked caller
 * retains this object and its charge on failure; it has already drained GPU use. */
static bool ds41_archive_free_observed(ds41_scr_archive *a, ds4_scr_owner_observation *o) {
    if (o) *o = ds41_archive_observe(a);
    if (!a) return true;
    if (o) o->retirement_attempted = true;
    if (a->raw) {
        bool closed = ds41_raw_close(a->raw);
        if (o) {
            ds41_archive_observation_cleanup(o, a->raw);
            o->retirement_complete = closed;
        }
        if (!closed) {
            fprintf(stderr, "ds4: SCR raw cleanup retained fd=%d cleanup_path=%s close=%d cleanup=%d archive_charge=%llu\n",
                    a->raw->fd, a->raw->cleanup_path, a->raw->close_error,
                    a->raw->cleanup_error, (unsigned long long)a->allocation_bytes);
            return false;
        }
        if (a->raw->cleanup_state == DS41_RAW_CLOSED_WITH_ERROR)
            fprintf(stderr, "ds4: SCR raw ownership discharged with diagnostics: close=%d cleanup=%d identity_lost=%d\n",
                    a->raw->close_error, a->raw->cleanup_error, a->raw->identity_lost);
        free(a->raw);
    }
    ds4_gpu_tensor_free(a->source);
    ds4_gpu_tensor_free(a->stage);
    ds4_gpu_tensor_free(a->scratch);
    ds4_gpu_tensor_free(a->raw_arena);
    ds41_coordinates_free(&a->coordinates);
    free(a->raw_io); free(a->tokens); free(a->tags); free(a);
    if (o) o->retirement_complete = true;
    return true;
}
static bool ds41_archive_free(ds41_scr_archive *a) {
    return ds41_archive_free_observed(a, NULL);
}
/* No file effect here: prepare must attach/charge the owner before raw_open. */
static ds41_scr_archive *ds41_archive_alloc(uint32_t start, uint32_t cap,
                                           const void *session, const void *model) {
    if (!ds41_archive_bytes(cap) || start % 8u || start > 1048576u - cap ||
        !session || !model) return NULL;
    ds41_scr_archive *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->cap = cap; a->start = a->end = start; a->limit = start + cap;
    a->logical_start = a->raw_origin = start;
    a->session = session; a->model = model;
    a->allocation_bytes = ds41_archive_bytes(cap);
    a->observation_id = ds41_archive_observation_next_id();
    a->raw = malloc(sizeof(*a->raw));
    if (a->raw) *a->raw = (ds41_raw_store)DS41_RAW_STORE_INIT;
    a->raw_arena_tokens = ds41_archive_arena_tokens(cap);
    a->raw_io = malloc((size_t)ds41_archive_io_bytes(cap));
    a->tags = malloc((size_t)cap * 5u / 2u * sizeof(uint32_t));
    a->tokens = malloc((size_t)cap * sizeof(int));
    a->source = ds4_gpu_tensor_alloc(ds41_archive_hot_bytes(cap));
    a->stage = ds4_gpu_tensor_alloc(ds41_archive_data_bytes(cap));
    a->scratch = ds4_gpu_tensor_alloc((uint64_t)cap * 512u * sizeof(float));
    a->raw_arena = ds4_gpu_tensor_alloc(ds41_raw_bytes(a->raw_arena_tokens));
    if (!a->raw || !a->raw_io || !a->tags || !a->tokens || !a->source ||
        !a->stage || !a->scratch || !a->raw_arena) {
        ds41_archive_free(a); /* No descriptor/path has been created yet. */
        return NULL;
    }
    ds41_archive_invalidate(a);
    return a;
}
static void ds41_archive_block_writer(ds41_scr_archive *a, bool history_hole) {
    if (!a) return;
    a->range.valid = false;
    a->lost = true;
    a->history_hole |= history_hole;
    if (a->sealed) a->writer_state = DS41_ARCHIVE_BLOCKED;
}
static bool ds41_archive_capturing(const ds41_scr_archive *a) {
    return a && !a->lost && !a->retiring &&
        (!a->sealed || a->writer_state == DS41_ARCHIVE_ARMED);
}
static bool ds41_archive_capture_owner(ds41_scr_archive *a, uint32_t owner,
                                       uint32_t start, uint32_t rows,
                                       ds4_gpu_tensor *main, ds4_gpu_tensor *index) {
    if (!ds41_archive_capturing(a)) return true;
    if (owner >= 4u) { ds41_archive_block_writer(a, true); return false; }
    const uint32_t ratio = owner < 3u ? 2u : 1u;
    if (start % ratio || rows > (UINT32_MAX - start) / ratio) {
        ds41_archive_block_writer(a, true); return false;
    }
    const uint64_t base = ds41_archive_owner(a->cap, owner);
    for (uint32_t t = 0; t < rows; ++t) {
        const uint32_t pos = start + t * ratio;
        if (pos < a->start || pos >= a->limit || (a->sealed && pos < a->end)) continue;
        const uint32_t slot = (pos - a->start) / ratio;
        if (!ds4_gpu_tensor_copy(a->source, base + (uint64_t)slot * 512u * 4u,
                main, (uint64_t)t * 512u * 4u, 512u * 4u) ||
            !ds4_gpu_tensor_copy(a->source, base + (uint64_t)(a->cap / ratio) * 512u * 4u +
                (uint64_t)slot * 128u * 4u, index, (uint64_t)t * 128u * 4u, 128u * 4u)) {
            ds41_archive_block_writer(a, true); return false;
        }
        a->tags[ds41_archive_tag(a->cap, owner) + slot] = pos;
    }
    return true;
}
static bool ds41_archive_capture_window(ds41_scr_archive *a, uint32_t layer,
                                        uint32_t start, uint32_t rows, ds4_gpu_tensor *kv) {
    if (!ds41_archive_capturing(a)) return true;
    if (layer >= 40u || rows > UINT32_MAX - start) {
        ds41_archive_block_writer(a, true); return false;
    }
    uint32_t end = start + rows;
    if (end <= a->raw_origin || start >= a->limit) return true;
    if (end > a->limit) end = a->limit;
    if (!ds41_raw_capture_rows(a->raw_arena, a->raw_arena_tokens, a->raw_origin, layer,
                               start, end - start, kv, a->raw_tags)) {
        ds41_archive_block_writer(a, true); return false;
    }
    return true;
}
/* Complete drained capture can advance private backend readiness. Source
 * publication additionally needs its owning exact headed-frontier proof. */
static bool ds41_archive_flush_ready(ds41_scr_archive *a, uint32_t frontier,
                               ds41_raw_cancel_fn cancel, void *ud) {
    if (!ds41_archive_capturing(a) || !a->raw ||
        a->raw_origin != a->start + a->raw->committed_tokens ||
        frontier < a->raw_origin || frontier > a->limit ||
        frontier - a->raw_origin > a->raw_arena_tokens || ds4_gpu_commands_active()) return false;
    uint32_t rows = frontier - a->raw_origin, complete = rows & ~7u, pending = rows - complete;
    if (!ds41_raw_capture_complete(a->raw_tags, a->raw_origin, rows)) return false;
    for (uint32_t o = 0; o < 4u; ++o) {
        uint32_t ratio = o < 3u ? 2u : 1u;
        for (uint32_t pos = a->raw_origin; pos < a->raw_origin + complete; pos += ratio)
            if (a->tags[ds41_archive_tag(a->cap, o) + (pos - a->start) / ratio] != pos) return false;
    }
    if (!complete) return true;
    if (cancel && cancel(ud)) { errno = ECANCELED; return false; }
    double began = ds41_archive_observation_time(a);
    bool ok = ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(a->raw_arena, 0, a->raw_io, ds41_raw_bytes(complete));
    a->raw_readback_seconds += ds41_archive_observation_time(a) - began;
    if (!ok) { a->lost = true; return false; }
    began = ds41_archive_observation_time(a);
    ok = ds41_raw_append(a->raw, a->raw_io, complete, cancel, ud);
    a->raw_write_seconds += ds41_archive_observation_time(a) - began;
    if (!ok || (cancel && cancel(ud))) { a->lost = true; return false; }
    bool owned = ds4_gpu_begin_commands();
    ok = owned && ds41_raw_capture_shift(a->raw_arena, a->raw_arena_tokens, a->raw_tags,
                                              a->raw_origin, complete, pending);
    if (owned) {
        if (!ds4_gpu_end_commands()) ok = false;
        if (!ds4_gpu_synchronize()) ok = false;
    }
    if (!ok || (cancel && cancel(ud)) ||
        !ds41_raw_commit(a->raw, a->raw->committed_tokens + complete)) {
        a->lost = true; return false;
    }
    a->raw_origin += complete;
    return true;
}
/* Public/append callers have certified the actual headed frontier. */
static bool ds41_archive_flush(ds41_scr_archive *a, uint32_t frontier,
                               ds41_raw_cancel_fn cancel, void *ud) {
    return ds41_archive_flush_ready(a, frontier, cancel, ud);
}
static bool ds41_archive_seal(ds41_scr_archive *a, const void *session,
                              const void *model, const int *tokens, uint32_t count) {
    if (!a || a->sealed || a->lost || a->retiring || !tokens || a->session != session ||
        a->model != model || !ds41_archive_shape(a, 1048576u) ||
        count < 128u || count > a->cap || count % 8u || !a->raw || !a->raw->readable ||
        a->raw->committed_tokens != count || a->raw_origin != a->start + count ||
        ds4_gpu_commands_active()) return false;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t ratio = o < 3u ? 2u : 1u;
        for (uint32_t t = 0; t < count / ratio; ++t)
            if (a->tags[ds41_archive_tag(a->cap, o) + t] != a->start + t * ratio) return false;
    }
    struct stat st;
    if (!ds4_gpu_synchronize() || !ds41_raw_identity(a->raw, &st)) { a->lost = true; return false; }
    memcpy(a->tokens, tokens, (size_t)count * sizeof(int));
    a->end = a->captured_end = a->start + count;
    a->source_generation = a->generation;
    a->sealed = true;
    return true;
}
/* The writer may advance only from a certified physical prefix. File-ready
 * records and captured pending rows are not reusable-source publication. */

static bool ds41_archive_published_valid(ds41_scr_archive *a) {
    const uint32_t count = ds41_archive_count(a);
    if (!a || !count || a->retiring || (a->lost && a->generation == UINT64_MAX) ||
        !a->session || !a->model ||
        !ds41_archive_shape(a, DS41_RAW_MAX_TOKENS) || !a->source_generation ||
        a->source_generation > a->generation || !a->source || !a->tokens || !a->tags ||
        !a->raw || !a->raw->readable || a->raw->committed_tokens < count ||
        ds4_gpu_commands_active() || !ds4_gpu_synchronize()) return false;
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t row = 0; row < count / ratio; ++row)
            if (a->tags[ds41_archive_tag(a->cap, owner) + row] !=
                a->start + row * ratio) return false;
    }
    const ds41_scr_coordinates *m = &a->coordinates;
    if (m->identity) {
        if (!ds41_coordinates_match(m, a->tokens, count, DS4_N_VOCAB)) return false;
    } else if (m->pads || m->count || m->capacity || m->physical_tokens) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (a->tokens[i] < 0 || (uint32_t)a->tokens[i] >= DS4_N_VOCAB) return false;
    struct stat st;
    return ds41_raw_identity(a->raw, &st) &&
        a->raw->file_bytes >= ds41_raw_bytes(count);
}

/* Capture hooks only block; this owning boundary may drain/revalidate the
 * immutable prefix. A suffix-only write refusal need not destroy that source. */
static bool ds41_archive_append_failure(ds41_scr_archive *a, bool history_hole,
                                        int error) {
    if (!a) { errno = error; return false; }
    ds41_archive_block_writer(a, history_hole);
    if (!ds41_archive_published_valid(a)) a->retiring = true;
    errno = error;
    return false;
}

/* The owner has admitted and attached an independent interval-map allocation.
 * Records beyond its published count are reserved for the known remainder.
 * No temporary caller pointer or live-map alias is retained by the archive. */
static bool ds41_archive_append_arm(ds41_scr_archive *a, const void *session,
                                    const void *model, const int *tokens,
                                    uint32_t known_count, uint32_t future_pads) {
    const uint32_t count = ds41_archive_count(a);
    if (!a || a->writer_state != DS41_ARCHIVE_INITIAL || a->lost || !tokens ||
        a->session != session || a->model != model || known_count < count ||
        known_count > a->cap || a->generation == UINT64_MAX ||
        !ds41_archive_published_valid(a) || !a->raw->writable ||
        a->raw->committed_tokens != count || a->raw->written_tokens != count ||
        a->raw_origin != a->end || a->raw->file_bytes != ds41_raw_bytes(count) ||
        memcmp(tokens, a->tokens, (size_t)count * sizeof(int))) return false;
    for (uint32_t layer = 0; layer < 40u; ++layer)
        for (uint32_t row = 0; row < a->raw_arena_tokens; ++row)
            if (a->raw_tags[layer][row] != UINT32_MAX) return false;
    ds41_scr_coordinates *m = &a->coordinates;
    if (m->identity) {
        ds41_scr_coordinates full = *m;
        full.count = future_pads; full.physical_tokens = known_count;
        if (!ds41_coordinates_match(&full, tokens, known_count, DS4_N_VOCAB)) return false;
        uint32_t published_pads = 0;
        while (published_pads < future_pads && m->pads[published_pads] < count)
            ++published_pads;
        if (published_pads != m->count) return false;
    } else if (future_pads) return false;
    for (uint32_t i = count; i < known_count; ++i)
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) return false;
    if (known_count > count)
        memcpy(a->tokens + count, tokens + count, (size_t)(known_count - count) * sizeof(int));
    /* Earlier wide owner output cannot certify a later full-model capture. */
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        memset(a->tags + ds41_archive_tag(a->cap, owner) + count / ratio, 0xff,
               (size_t)((a->cap - count) / ratio) * sizeof(uint32_t));
    }
    a->captured_end = a->end;
    a->writer_known_end = a->start + known_count;
    a->writer_future_pads = future_pads;
    a->writer_generation = a->generation;
    a->writer_coordinate_identity = a->coordinates.identity;
    a->writer_state = a->end == a->limit ? DS41_ARCHIVE_SATURATED : DS41_ARCHIVE_ARMED;
    a->range.valid = false;
    return true;
}

/* Called after a real full-model head and physical-map append. Between aligned
 * publications only 0..7 captured rows may remain; a later odd head cannot stand
 * in for a skipped earlier aligned head. Token validation is delta-only. */
static bool ds41_archive_append_observe(ds41_scr_archive *a, const void *session,
                                        const void *model, const int *tokens,
                                        uint32_t frontier, ds41_raw_cancel_fn cancel,
                                        void *ud) {
    if (!a || a->writer_state != DS41_ARCHIVE_ARMED) return false;
    if (a->lost || a->retiring || !tokens || a->session != session || a->model != model ||
        a->writer_generation != a->generation || a->source_generation != a->generation ||
        a->generation == UINT64_MAX || a->captured_end < a->end ||
        frontier < a->captured_end || frontier > a->limit || a->raw_origin < a->end ||
        frontier < a->raw_origin || frontier - a->raw_origin > a->raw_arena_tokens ||
        ds4_gpu_commands_active())
        return ds41_archive_append_failure(a, false, EINVAL);
    if (!ds41_raw_capture_complete(a->raw_tags, a->raw_origin, frontier - a->raw_origin))
        return ds41_archive_append_failure(a, true, EIO);
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t row = (a->captured_end - a->start) / ratio;
             row < (frontier - a->start) / ratio; ++row)
            if (a->tags[ds41_archive_tag(a->cap, owner) + row] != a->start + row * ratio)
                return ds41_archive_append_failure(a, true, EIO);
    }
    if (cancel && cancel(ud))
        return ds41_archive_append_failure(a, false, ECANCELED);
    if (frontier == a->captured_end) return true;
    for (uint32_t pos = a->captured_end; pos < frontier; ++pos) {
        int token = tokens[pos - a->start];
        if (token < 0 || (uint32_t)token >= DS4_N_VOCAB ||
            (pos < a->writer_known_end && token != a->tokens[pos - a->start]))
            return ds41_archive_append_failure(a, false, EINVAL);
    }
    if (frontier > a->captured_end)
        memcpy(a->tokens + a->captured_end - a->start,
               tokens + a->captured_end - a->start,
               (size_t)(frontier - a->captured_end) * sizeof(int));
    a->captured_end = frontier;
    if (frontier % 8u) {
        /* Complete capture is not a history hole. This head simply cannot
         * certify an earlier aligned publication that the caller skipped. */
        if (frontier - a->end >= 8u)
            return ds41_archive_append_failure(a, false, ERANGE);
        return true;
    }
    errno = 0; /* GPU failures do not supply errno; never report a stale cancellation. */
    if (!ds41_archive_flush(a, frontier, cancel, ud))
        return ds41_archive_append_failure(a, false, errno ? errno : EIO);
    const uint32_t count = frontier - a->start;
    ds41_scr_coordinates *m = &a->coordinates;
    uint32_t pads = m->count;
    while (pads < a->writer_future_pads && m->pads[pads] < count) ++pads;
    if (m->identity) {
        ds41_scr_coordinates next = *m;
        next.count = pads; next.physical_tokens = count;
        /* Only new pad records need checking; the published map was certified
         * at the preceding commit and immutable owner bytes are never rewritten. */
        for (uint32_t i = m->count; i < pads; ++i)
            if (m->pads[i] >= count || (i && m->pads[i - 1u] >= m->pads[i]))
                return ds41_archive_append_failure(a, false, EINVAL);
        if (next.count > next.capacity || next.count > next.physical_tokens)
            return ds41_archive_append_failure(a, false, EINVAL);
    }
    if (cancel && cancel(ud))
        return ds41_archive_append_failure(a, false, ECANCELED);
    if (m->identity) { m->count = pads; m->physical_tokens = count; }
    a->range.valid = false;
    ++a->generation;
    a->writer_generation = a->generation;
    a->end = frontier;
    a->source_generation = a->generation;
    if (frontier == a->limit) a->writer_state = DS41_ARCHIVE_SATURATED;
    return true;
}

static bool ds41_archive_range_check_aligned(ds41_scr_archive *a, const void *session,
                                     const void *model, uint32_t source_start,
                                     uint32_t destination, const int *tokens,
                                     uint32_t count, uint32_t context, uint32_t alignment) {
    if (!a) return false;
    a->range.valid = false; a->stage_alignment = 0;
    if ((alignment != 2u && alignment != 8u) ||
        (alignment == 8u && a->coordinates.count)) return false;
    if (!a->sealed || !ds41_archive_shape(a, context) || a->retiring ||
        (a->lost && a->generation == UINT64_MAX) ||
        !a->raw || !a->raw->readable || a->raw->committed_tokens < ds41_archive_count(a) ||
        !tokens || a->session != session || a->model != model ||
        !session || !model || !a->source_generation || a->source_generation > a->generation ||
        ds4_gpu_commands_active() || source_start % alignment || destination % alignment || count % alignment ||
        count < 128u || count > ds41_archive_count(a) || source_start < a->start || source_start > a->end ||
        count > a->end - source_start || a->end > context ||
        destination >= context || count >= context - destination ||
        destination > 1048576u || count > 1048576u - destination ||
        memcmp(tokens, a->tokens + source_start - a->start, (size_t)count * sizeof(int))) return false;
    return true;
}
static bool ds41_archive_range_check(ds41_scr_archive *a, const void *session,
                                     const void *model, uint32_t source_start,
                                     uint32_t destination, const int *tokens,
                                     uint32_t count, uint32_t context) {
    return ds41_archive_range_check_aligned(a, session, model, source_start, destination,
                                            tokens, count, context, 8u);
}
/* Stage one range. `window` may hold a finished read-ahead of this range's raw
 * records; `drained` means the caller synchronized the GPU with no work since. */
static bool ds41_archive_materialize_window(ds41_scr_archive *a, const void *session,
                                     const void *model, uint32_t source_start,
                                     uint32_t destination, const int *tokens,
                                     uint32_t count, uint32_t context, uint32_t alignment,
                                     const ds41_raw_prefetch *window, bool drained) {
    bool checked = alignment == 8u ?
        ds41_archive_range_check(a, session, model, source_start, destination, tokens, count, context) :
        ds41_archive_range_check_aligned(a, session, model, source_start, destination, tokens, count, context, alignment);
    if (!checked || (a->cancelled && a->cancelled((void *)a->session))) return false;
    if (!drained && !ds4_gpu_synchronize()) return false;
    const uint32_t offset = source_start - a->start;
    uint32_t first, records;
    double began = ds41_archive_observation_time(a);
    bool ok = ds41_raw_window_records(ds41_archive_count(a), offset + count, &first, &records);
    const bool prefetched = ok && window && window->ready && window->store == a->raw &&
        window->first == first && window->records == records;
    const unsigned char *data = prefetched ? window->data : a->raw_io;
    if (ok && !prefetched)
        ok = ds41_raw_read(a->raw, first, records, a->raw_io, DS41_ARCHIVE_READ_BYTES,
                           a->cancelled, (void *)a->session);
    a->raw_read_seconds += ds41_archive_observation_time(a) - began;
    if (!ok) {
        /* Cancellation leaves the immutable source readable; an I/O fault does not. */
        if (!a->raw->readable) a->retiring = true;
        return false;
    }
    /* Stage's existing terminal-window region is the canonical read buffer until
     * each layer is transformed through scratch. No second GPU read arena. */
    began = ds41_archive_observation_time(a);
    if (prefetched) {
        /* The read buffer is free: gather every layer of the window at once,
         * layer-major, so each layer's 128 rows are contiguous. */
        const size_t window_bytes = (size_t)records * DS41_RAW_RECORD_BYTES;
        const size_t layer_bytes = (size_t)records * 8u * DS41_RAW_ROW_BYTES;
        const size_t skip = (size_t)(offset + count - 128u - first * 8u) * DS41_RAW_ROW_BYTES;
        ok = kv_archive_gather_layers(data, window_bytes, records, 40u,
                8u * DS41_RAW_ROW_BYTES, a->raw_io, window_bytes, DS41_ARCHIVE_GATHER_THREADS,
                NULL, 0) == KV_ARCHIVE_OK;
        for (uint32_t l = 0; ok && l < 40u; ++l)
            ok = ds4_gpu_tensor_write(a->stage, ds41_archive_ring(a->cap, l),
                                      a->raw_io + l * layer_bytes + skip, 128u * DS41_RAW_ROW_BYTES);
    } else {
        unsigned char *layer = a->raw_io + DS41_ARCHIVE_READ_BYTES;
        for (uint32_t l = 0; ok && l < 40u; ++l)
            ok = ds41_raw_gather_layer(data, DS41_ARCHIVE_READ_BYTES, first, records,
                    offset + count - 128u, l, layer, 128u * DS41_RAW_ROW_BYTES) &&
                 ds4_gpu_tensor_write(a->stage, ds41_archive_ring(a->cap, l), layer,
                                      128u * DS41_RAW_ROW_BYTES);
    }
    a->raw_gather_seconds += ds41_archive_observation_time(a) - began;
    if (!ok || (a->cancelled && a->cancelled((void *)a->session))) return false;
    bool owned = ds4_gpu_begin_commands();
    ok = owned;
    for (uint32_t o = 0; ok && o < 4u; ++o) {
        const uint32_t ratio = o < 3u ? 2u : 1u, rows = count / ratio;
        const uint64_t base = ds41_archive_owner(a->cap, o);
        for (uint32_t idx = 0; ok && idx < 2u; ++idx) {
            const uint32_t width = idx ? 128u : 512u;
            const uint64_t out = base + (idx ? (uint64_t)(a->cap / ratio) * 512u * 4u : 0);
            ok = ds4_gpu_tensor_copy(a->scratch, 0, a->source,
                out + (uint64_t)(offset / ratio) * width * 4u, (uint64_t)rows * width * 4u) &&
                ds4_gpu_dsv41_rope_stride(a->scratch, width, 1, rows, destination, ratio, true, false) &&
                ds4_gpu_dsv41_quantize(a->scratch, width, rows, idx ? DS4_V41_FP4_E8M0 : DS4_V41_FP4_E4M3) &&
                ds4_gpu_tensor_copy(a->stage, out, a->scratch, 0, (uint64_t)rows * width * 4u);
        }
    }
    const uint32_t window_start = destination + count - 128u, slot = window_start % 128u;
    const uint64_t row = 512u * sizeof(float);
    for (uint32_t l = 0; ok && l < 40u; ++l) {
        const uint64_t out = ds41_archive_ring(a->cap, l);
        ok = ds4_gpu_tensor_copy(a->scratch, 0, a->stage, out, 128u * row) &&
            ds4_gpu_dsv41_rope_stride(a->scratch, 512, 1, 128, window_start, 1, l >= 2u, false) &&
            ds4_gpu_dsv41_quantize(a->scratch, 512, 128, DS4_V41_FP8_E8M0) &&
            ds4_gpu_tensor_copy(a->stage, out + slot * row, a->scratch, 0, (128u - slot) * row);
        if (ok && slot) ok = ds4_gpu_tensor_copy(a->stage, out, a->scratch,
                                               (128u - slot) * row, slot * row);
    }
    if (owned) {
        if (!ds4_gpu_end_commands()) ok = false;
        if (!ds4_gpu_synchronize()) ok = false;
    }
    if (!ok) return false;
    a->range.start = destination; a->range.end = destination + count;
    a->range.source_start = source_start; a->range.source_end = source_start + count;
    a->range.source_generation = a->source_generation;
    a->range.destination_generation = a->generation;
    a->stage_alignment = alignment;
    a->range.valid = true;
    return true;
}
static bool ds41_archive_materialize_aligned(ds41_scr_archive *a, const void *session,
                                     const void *model, uint32_t source_start,
                                     uint32_t destination, const int *tokens,
                                     uint32_t count, uint32_t context, uint32_t alignment) {
    return ds41_archive_materialize_window(a, session, model, source_start, destination,
                                           tokens, count, context, alignment, NULL, false);
}
static bool ds41_archive_materialize(ds41_scr_archive *a, const void *session,
                                     const void *model, uint32_t source_start,
                                     uint32_t destination, const int *tokens,
                                     uint32_t count, uint32_t context) {
    return ds41_archive_materialize_aligned(a, session, model, source_start, destination,
                                           tokens, count, context, 8u);
}
/* Rebase copies canonical history into an independently owned private builder.
 * The owning transaction proves token/map correspondence and advances the graph;
 * these copies are imported storage, never forward computation or publication. */

static bool ds41_rebase_import_bounds(ds41_scr_archive *destination,
                                      ds41_scr_archive *source,
                                      uint32_t source_start, uint32_t target_start,
                                      uint32_t count) {
    const uint32_t covered = ds41_archive_count(source);
    return destination && source && destination != source &&
        destination->source != source->source && destination->raw != source->raw &&
        destination->raw_io != source->raw_io && destination->raw_arena != source->raw_arena &&
        destination->session && destination->model &&
        destination->session == source->session && destination->model == source->model &&
        ds41_archive_shape(source, DS41_RAW_MAX_TOKENS) &&
        ds41_archive_shape(destination, DS41_RAW_MAX_TOKENS) &&
        !destination->sealed && !destination->lost && !destination->retiring &&
        source->sealed && !source->retiring && source->source_generation &&
        source->source_generation <= source->generation &&
        !(source->lost && source->generation == UINT64_MAX) &&
        source->raw && source->raw->readable && source->raw->committed_tokens >= covered &&
        destination->raw && destination->raw->readable && destination->raw->writable &&
        destination->source && source->source && destination->raw_arena &&
        destination->tags && source->tags && source->raw_io &&
        count && count <= 128u && !(count % 2u) &&
        !(source_start % 2u) && !(target_start % 2u) &&
        source_start >= source->start && source_start <= source->end &&
        count <= source->end - source_start &&
        destination->raw->committed_tokens <= destination->cap &&
        destination->raw_origin == destination->start + destination->raw->committed_tokens &&
        target_start == destination->captured_end && target_start >= destination->start &&
        target_start >= destination->raw_origin &&
        target_start <= destination->limit && count <= destination->limit - target_start &&
        target_start - destination->raw_origin <= destination->raw_arena_tokens &&
        count <= destination->raw_arena_tokens - (target_start - destination->raw_origin) &&
        !ds4_gpu_commands_active();
}

static bool ds41_rebase_import_owners(ds41_scr_archive *destination,
                                      ds41_scr_archive *source,
                                      uint32_t source_start, uint32_t target_start,
                                      uint32_t count) {
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t row = 0; row < count / ratio; ++row) {
            const uint32_t from = source_start + row * ratio;
            const uint32_t to = target_start + row * ratio;
            if (source->tags[ds41_archive_tag(source->cap, owner) +
                    (from - source->start) / ratio] != from ||
                destination->tags[ds41_archive_tag(destination->cap, owner) +
                    (to - destination->start) / ratio] != UINT32_MAX) return false;
        }
    }
    bool owned = ds4_gpu_begin_commands();
    bool ok = owned;
    for (uint32_t owner = 0; ok && owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        const uint64_t from_base = ds41_archive_owner(source->cap, owner);
        const uint64_t to_base = ds41_archive_owner(destination->cap, owner);
        for (uint32_t index = 0; ok && index < 2u; ++index) {
            const uint32_t width = index ? 128u : 512u;
            const uint64_t from = from_base +
                (index ? (uint64_t)(source->cap / ratio) * 512u * sizeof(float) : 0u) +
                (uint64_t)((source_start - source->start) / ratio) * width * sizeof(float);
            const uint64_t to = to_base +
                (index ? (uint64_t)(destination->cap / ratio) * 512u * sizeof(float) : 0u) +
                (uint64_t)((target_start - destination->start) / ratio) * width * sizeof(float);
            ok = ds4_gpu_tensor_copy(destination->source, to, source->source, from,
                                      (uint64_t)(count / ratio) * width * sizeof(float));
        }
    }
    if (owned) {
        if (!ds4_gpu_end_commands()) ok = false;
        if (!ds4_gpu_synchronize()) ok = false;
    }
    if (!ok || ds4_gpu_commands_active()) return false;
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t row = 0; row < count / ratio; ++row) {
            const uint32_t to = target_start + row * ratio;
            destination->tags[ds41_archive_tag(destination->cap, owner) +
                (to - destination->start) / ratio] = to;
        }
    }
    return true;
}

/* At most 128 rows (one window read), including partial source records and partial target records.
 * Neighboring fresh/imported rows in the destination arena remain untouched. */
static bool ds41_rebase_import_raw(ds41_scr_archive *destination,
                                   ds41_scr_archive *source,
                                   uint32_t source_start, uint32_t target_start,
                                   uint32_t count) {
    const uint32_t offset = source_start - source->start;
    const uint32_t first = offset / DS41_RAW_RECORD_TOKENS;
    const uint32_t records = (offset + count + 7u) / DS41_RAW_RECORD_TOKENS - first;
    double began = ds41_archive_observation_time(source);
    bool ok = ds41_raw_read(source->raw, first, records, source->raw_io,
        DS41_ARCHIVE_READ_BYTES, source->cancelled, (void *)source->session);
    source->raw_read_seconds += ds41_archive_observation_time(source) - began;
    if (!ok) {
        if (!source->raw->readable) source->retiring = true;
        return false;
    }
    unsigned char *gather = source->raw_io + DS41_ARCHIVE_READ_BYTES;
    began = ds41_archive_observation_time(source);
    for (uint32_t layer = 0; ok && layer < 40u; ++layer) {
        for (uint32_t row = 0; row < count; ++row)
            if (destination->raw_tags[layer][target_start - destination->raw_origin + row] !=
                    UINT32_MAX) { ok = false; break; }
        if (!ok) break;
        ok = ds41_raw_gather_rows(source->raw_io, DS41_ARCHIVE_READ_BYTES,
            first, records, offset, count, layer, gather, 128u * DS41_RAW_ROW_BYTES);
        for (uint32_t copied = 0; ok && copied < count;) {
            const uint32_t position = target_start + copied;
            uint32_t rows = 8u - position % 8u;
            if (rows > count - copied) rows = count - copied;
            uint64_t arena_offset;
            ok = ds41_raw_arena_offset(destination->raw_origin, position, layer, &arena_offset) &&
                ds4_gpu_tensor_write(destination->raw_arena, arena_offset,
                    gather + (uint64_t)copied * DS41_RAW_ROW_BYTES,
                    (uint64_t)rows * DS41_RAW_ROW_BYTES);
            copied += rows;
        }
    }
    source->raw_gather_seconds += ds41_archive_observation_time(source) - began;
    if (!ok || !ds4_gpu_synchronize() || ds4_gpu_commands_active() ||
        (destination->cancelled && destination->cancelled((void *)destination->session))) return false;
    for (uint32_t layer = 0; layer < 40u; ++layer)
        for (uint32_t row = 0; row < count; ++row)
            destination->raw_tags[layer][target_start - destination->raw_origin + row] =
                target_start + row;
    return true;
}

static bool ds41_rebase_import_chunk(ds41_scr_archive *destination,
                                     ds41_scr_archive *source,
                                     uint32_t source_start, uint32_t target_start,
                                     uint32_t count) {
    if (!ds41_rebase_import_bounds(destination, source, source_start, target_start, count)) {
        errno = EINVAL; return false;
    }
    if ((destination->cancelled && destination->cancelled((void *)destination->session)) ||
        !ds4_gpu_synchronize() ||
        !ds41_rebase_import_raw(destination, source, source_start, target_start, count)) {
        destination->lost = true;
        return false;
    }
    destination->captured_end = target_start + count;
    return true;
}
/* Owner rows of a whole range in one command buffer (one drain instead of one
 * per raw chunk). The raw chunks that follow keep their own bounds; every
 * owner row is checked against its source tag and an empty destination tag. */
static bool ds41_rebase_import_range_owners(ds41_scr_archive *destination,
                                            ds41_scr_archive *source,
                                            uint32_t source_start, uint32_t target_start,
                                            uint32_t count) {
    const uint32_t first = count < 128u ? count : 128u;
    if (!ds41_rebase_import_bounds(destination, source, source_start, target_start, first) ||
        count % 2u || count > source->end - source_start ||
        count > destination->limit - target_start) {
        errno = EINVAL; return false;
    }
    if ((destination->cancelled && destination->cancelled((void *)destination->session)) ||
        !ds4_gpu_synchronize() ||
        !ds41_rebase_import_owners(destination, source, source_start, target_start, count)) {
        destination->lost = true;
        return false;
    }
    return true;
}
#endif

/* ==== 2. Coordinates & Prefix ============================================ */
#if defined(DS4_SCR_ENGINE_COORDINATES) && !defined(DS4_SCR_ENGINE_COORDINATES_DONE)
#define DS4_SCR_ENGINE_COORDINATES_DONE
/* Physical transcript coordinates. These helpers own only pad metadata, not
 * model state: a known transcript may have no reusable graph/head after rewind.
 * Callers publish or invalidate graph/checkpoint/map together under their owner.
 * Pads are identified by position only; the token at a pad position is the
 * transcript's own ID, never a second copy. Logical projection, boundary
 * mapping and prefix walks are the pinned cache-reuse-utils helpers. */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "adapter/include/cache_reuse_utils.h"

/* pads: strictly increasing physical indices below physical_tokens. */
typedef struct {
    uint32_t *pads;
    uint32_t count, capacity, physical_tokens;
    uint64_t identity;
} ds41_scr_coordinates;

static void ds41_coordinates_free(ds41_scr_coordinates *m) {
    if (!m) return;
    free(m->pads);
    memset(m, 0, sizeof(*m));
}

static bool ds41_coordinates_shape(const ds41_scr_coordinates *m) {
    if (!m || !m->identity || m->count > m->capacity ||
        m->count > m->physical_tokens || (m->capacity && !m->pads) ||
        (!m->capacity && m->pads)) return false;
    for (uint32_t i = 0; i < m->count; ++i)
        if (m->pads[i] >= m->physical_tokens || (i && m->pads[i - 1] >= m->pads[i]))
            return false;
    return true;
}

static bool ds41_coordinates_match(const ds41_scr_coordinates *m,
                                   const int *physical, uint32_t n,
                                   uint32_t vocab) {
    if (!ds41_coordinates_shape(m) || n != m->physical_tokens ||
        (n && !physical) || !vocab) return false;
    for (uint32_t i = 0; i < n; ++i)
        if (physical[i] < 0 || (uint32_t)physical[i] >= vocab) return false;
    return true;
}

static bool ds41_coordinates_heap_bytes(uint32_t capacity, size_t *bytes) {
    if (!bytes || (uint64_t)capacity * sizeof(uint32_t) > SIZE_MAX) return false;
    *bytes = (size_t)capacity * sizeof(uint32_t);
    return true;
}

/* Build into an empty owner. Validate before allocation; failure leaves it
 * unchanged. The owning admission must reserve heap_bytes before calling. */
static bool ds41_coordinates_build(ds41_scr_coordinates *out,
                                   const int *physical, uint32_t n, uint32_t vocab,
                                   const uint32_t *pads, uint32_t count,
                                   uint64_t identity) {
    if (!out || out->pads || out->count || out->capacity ||
        out->physical_tokens || out->identity || (count && !pads)) return false;
    ds41_scr_coordinates view = {(uint32_t *)pads, count, count, n, identity};
    if (!ds41_coordinates_match(&view, physical, n, vocab)) return false;
    size_t bytes;
    if (!ds41_coordinates_heap_bytes(count, &bytes)) return false;
    uint32_t *copy = count ? malloc(bytes) : NULL;
    if (count && !copy) return false;
    if (count) memcpy(copy, pads, bytes);
    *out = (ds41_scr_coordinates){copy, count, count, n, identity};
    return true;
}

/* Archive snapshots must own their metadata, never alias a live map. */
static bool ds41_coordinates_clone(ds41_scr_coordinates *out,
                                   const ds41_scr_coordinates *source) {
    if (!out || out == source || !ds41_coordinates_shape(source) ||
        out->pads || out->count || out->capacity || out->physical_tokens ||
        out->identity) return false;
    size_t bytes;
    if (!ds41_coordinates_heap_bytes(source->count, &bytes)) return false;
    uint32_t *copy = source->count ? malloc(bytes) : NULL;
    if (source->count && !copy) return false;
    if (source->count) memcpy(copy, source->pads, bytes);
    *out = (ds41_scr_coordinates){copy, source->count, source->count,
                               source->physical_tokens, source->identity};
    return true;
}

/* Two maps agree on their first `count` pad positions. */
static bool ds41_coordinates_same_pads(const ds41_scr_coordinates *a,
                                       const ds41_scr_coordinates *b, uint32_t count) {
    return a && b && count <= a->count && count <= b->count &&
        (!count || !memcmp(a->pads, b->pads, (size_t)count * sizeof(uint32_t)));
}

/* Destination is caller-owned separate storage. Only recorded artificial
 * positions disappear; equal token IDs elsewhere remain ordinary content. */
static inline bool ds41_coordinates_project(const ds41_scr_coordinates *m,
                                            const int *physical, uint32_t n, uint32_t vocab,
                                            int *logical, uint32_t capacity,
                                            uint32_t *written) {
    if (!written) return false;
    *written = 0;
    if (!ds41_coordinates_match(m, physical, n, vocab)) return false;
    return cache_reuse_project_logical((const int32_t *)physical, n, m->pads, m->count,
        (int32_t *)logical, capacity, written, NULL, 0) == CACHE_REUSE_COMPLETE;
}

/* Return the logical boundary preceding a physical boundary, including when
 * the latter is inside a run of pads. No graph/frontier validity is implied. */
static inline bool ds41_coordinates_logical_boundary(const ds41_scr_coordinates *m,
                                                     uint32_t physical, uint32_t *logical) {
    if (!logical || !ds41_coordinates_shape(m)) return false;
    uint32_t mapped;
    if (cache_reuse_logical_boundary(physical, m->physical_tokens, m->pads, m->count,
            &mapped, NULL, 0) != CACHE_REUSE_COMPLETE) return false;
    *logical = mapped;
    return true;
}

/* A sealed source owns a relative map for its physical interval. Preserve
 * the interval's logical origin separately; no pointer aliases the live map. */
static bool ds41_coordinates_slice(const ds41_scr_coordinates *source,
                                   const int *physical, uint32_t n, uint32_t vocab,
                                   uint32_t start, uint32_t count,
                                   ds41_scr_coordinates *out, uint32_t *logical_start) {
    if (!out || out == source || !logical_start || out->pads || out->identity ||
        out->count || out->capacity || out->physical_tokens || !count ||
        start > n || count > n - start ||
        !ds41_coordinates_match(source, physical, n, vocab)) return false;
    /* Size the copy first: pads before the end, less pads before the start. */
    uint32_t origin, end, kept = 0;
    if (cache_reuse_logical_boundary(start, n, source->pads, source->count, &origin,
            NULL, 0) != CACHE_REUSE_COMPLETE ||
        cache_reuse_truncate_pad_positions(source->pads, source->count, n, start + count,
            &end, NULL, 0) != CACHE_REUSE_COMPLETE || end < start - origin) return false;
    const uint32_t capacity = end - (start - origin);
    size_t bytes;
    if (!ds41_coordinates_heap_bytes(capacity, &bytes)) return false;
    uint32_t *copy = capacity ? malloc(bytes) : NULL;
    if (capacity && !copy) return false;
    if (cache_reuse_slice_pad_positions(source->pads, source->count, n, start, count,
            copy, capacity, &kept, NULL, 0) != CACHE_REUSE_COMPLETE || kept != capacity) {
        free(copy);
        return false;
    }
    *out = (ds41_scr_coordinates){copy, kept, kept, count, source->identity};
    *logical_start = origin;
    return true;
}

/* Commit already-admitted metadata ownership and its persistent charges.
 * Pending allocation is a transient reservation until this transfer. Validate
 * every counter before changing any; the third counter is the archive owner
 * when present, in addition to graph and engine accounting. */
static bool ds41_coordinates_replace_charged(ds41_scr_coordinates *live,
                                            ds41_scr_coordinates *pending,
                                            const int *physical, uint32_t n,
                                            uint32_t vocab, uint64_t *graph_bytes,
                                            uint64_t *engine_bytes, uint64_t *archive_bytes) {
    if (!live || !pending || live == pending || !graph_bytes || !engine_bytes ||
        graph_bytes == engine_bytes || (archive_bytes &&
        (archive_bytes == graph_bytes || archive_bytes == engine_bytes)) ||
        (live->pads && live->pads == pending->pads) ||
        !ds41_coordinates_match(pending, physical, n, vocab)) return false;
    const bool empty = !live->pads && !live->count && !live->capacity &&
                       !live->physical_tokens && !live->identity;
    if (!empty && !ds41_coordinates_shape(live)) return false;
    size_t old_heap, new_heap;
    if (!ds41_coordinates_heap_bytes(live->capacity, &old_heap) ||
        !ds41_coordinates_heap_bytes(pending->capacity, &new_heap)) return false;
    uint64_t *counters[3] = {graph_bytes, engine_bytes, archive_bytes};
    const unsigned count = archive_bytes ? 3 : 2;
    for (unsigned i = 0; i < count; ++i)
        if (*counters[i] < old_heap || new_heap > UINT64_MAX - (*counters[i] - old_heap))
            return false;
    for (unsigned i = 0; i < count; ++i)
        *counters[i] = *counters[i] - old_heap + new_heap;
    ds41_coordinates_free(live);
    *live = *pending;
    memset(pending, 0, sizeof(*pending));
    return true;
}

/* Metadata-only truncation preserves a known transcript for an explicit
 * rebuild. The session owner must separately invalidate the model frontier. */
static bool ds41_coordinates_truncate(ds41_scr_coordinates *m, uint32_t n,
                                      uint64_t identity) {
    uint32_t count;
    if (!ds41_coordinates_shape(m) || identity <= m->identity ||
        cache_reuse_truncate_pad_positions(m->pads, m->count, m->physical_tokens, n,
            &count, NULL, 0) != CACHE_REUSE_COMPLETE) return false;
    m->count = count;
    m->physical_tokens = n;
    m->identity = identity;
    return true;
}

/* Successful generation appends ordinary IDs, not new artificial pad records.
 * The owner checks actual checkpoint bytes and head before publishing this. */
static inline bool ds41_coordinates_append(ds41_scr_coordinates *m, uint32_t count,
                                           uint32_t physical_limit, uint64_t identity) {
    uint32_t physical;
    if (!ds41_coordinates_shape(m) || identity <= m->identity ||
        cache_reuse_append_unpadded(m->physical_tokens, m->pads, m->count, count,
            &physical, NULL, 0) != CACHE_REUSE_COMPLETE || physical > physical_limit) return false;
    m->physical_tokens = physical;
    m->identity = identity;
    return true;
}
#endif

#ifdef DS4_SCR_ENGINE_SESSION
/* Read-only logical comparisons keep physical checkpoint accessors unchanged. */
bool ds4_session_has_physical_coordinates(const ds4_session *s) {
    return s && s->scr_coordinates.identity != 0;
}

int ds4_session_compare_logical(ds4_session *s, const ds4_tokens *logical,
                                const char *text, size_t text_len,
                                ds4_logical_comparison *out, char *err, size_t errlen) {
    if (out) memset(out, 0, sizeof(*out));
    if (!s || !s->engine || !out || !logical || logical->len < 0 ||
        (logical->len && !logical->v) || (!text && text_len) ||
        !s->checkpoint_valid || s->checkpoint.len < 0 ||
        s->checkpoint.len > s->ctx_size || s->checkpoint.cap < s->checkpoint.len ||
        (s->checkpoint.len && !s->checkpoint.v) || s->scr_policy_active ||
        s->scr_multi_active || s->scr_dense_owner) {
        snprintf(err, errlen, "logical comparison: invalid input or unavailable checkpoint"); return 1;
    }
    const ds41_scr_coordinates *m = &s->scr_coordinates;
    if ((m->identity && !ds41_coordinates_match(m, s->checkpoint.v,
            (uint32_t)s->checkpoint.len, DS4_N_VOCAB)) ||
        (!m->identity && (m->pads || m->count || m->capacity || m->physical_tokens))) {
        snprintf(err, errlen, "logical comparison: invalid provenance"); return 1;
    }
    if (logical->len > s->ctx_size) {
        snprintf(err, errlen, "logical comparison: target exceeds context"); return 1;
    }
    for (int i = 0; i < logical->len; ++i) {
        if (!(i % 4096) && ds4_session_cancelled(s)) goto interrupted;
        if (logical->v[i] < 0 || (uint32_t)logical->v[i] >= DS4_N_VOCAB) {
            snprintf(err, errlen, "logical comparison: invalid target token"); return 1;
        }
    }
    uint64_t piece_bytes = 0;
    for (int i = 0; i < s->checkpoint.len; ++i) {
        if (!(i % 4096) && ds4_session_cancelled(s)) goto interrupted;
        int token = s->checkpoint.v[i];
        if (token < 0 || (uint32_t)token >= DS4_N_VOCAB ||
            (text && (!s->engine->vocab.token || token >= s->engine->vocab.n_vocab))) {
            snprintf(err, errlen, "logical comparison: invalid token vocabulary"); return 1;
        }
        if (text) {
            ds4_str piece = s->engine->vocab.token[token];
            if (piece.len >= SIZE_MAX || (piece.len && !piece.ptr)) {
                snprintf(err, errlen, "logical comparison: invalid token text"); return 1;
            }
            if (piece.len + 1u > piece_bytes) piece_bytes = piece.len + 1u;
        }
    }
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (m->identity && s->engine->backend == DS4_BACKEND_METAL &&
        (piece_bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
         !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + piece_bytes, false))) {
        snprintf(err, errlen, "logical comparison: token text reservation refused"); return 1;
    }
#else
    (void)piece_bytes;
#endif
    ds4_logical_comparison result = {.text_equal = text != NULL};
    uint32_t pad = 0;
    size_t offset = 0;
    bool matching = true;
    for (int i = 0; i < s->checkpoint.len; ++i) {
        if (!(i % 4096) && ds4_session_cancelled(s)) goto interrupted;
        if (pad < m->count && m->pads[pad] == (uint32_t)i) { ++pad; continue; }
        int token = s->checkpoint.v[i];
        if (matching && result.live_tokens < logical->len &&
            logical->v[result.live_tokens] == token) ++result.common_tokens;
        else matching = false;
        ++result.live_tokens;
        if (result.text_equal) {
            size_t len = 0;
            char *piece = ds4_token_text(s->engine, token, &len);
            result.text_equal = len <= text_len - offset && !memcmp(piece, text + offset, len);
            if (result.text_equal) offset += len;
            free(piece);
        }
    }
    result.text_equal = result.text_equal && offset == text_len;
    *out = result;
    return 0;
interrupted:
    snprintf(err, errlen, "interrupted"); return DS4_SESSION_SYNC_INTERRUPTED;
}

/* Bounded planning work accounting and the live-prefix reader shared by the
 * logical policy. Every loop charges the budget; the session cancel callback is
 * polled on each charge. Failure publishes nothing. */
#define DS41_SCR_PLAN_WORK_MAX (128ull * (1048576u + 65536u + DS4_SCR_DENSE_RANGES_MAX))
static bool ds41_scr_poll(ds4_session *s) {
    return s && ds4_session_cancelled(s);
}
typedef struct {
    uint64_t work, limit;
    ds4_session *session;
    bool cancelled;
    uint64_t comparisons, comparison_limit; /* Zero limit selects the existing ceiling. */
} ds41_pad_budget;
static bool ds41_pad_comparison(ds41_pad_budget *b) {
    if (!b) return false;
    uint64_t limit = b->comparison_limit ? b->comparison_limit : DS4_SCR_DENSE_COMPARISONS_MAX;
    if (limit > DS4_SCR_DENSE_COMPARISONS_MAX || b->comparisons >= limit) return false;
    ++b->comparisons;
    return true;
}
typedef enum {
    DS41_PAD_OK, DS41_PAD_REFUSED, DS41_PAD_CANCELLED
} ds41_pad_result;
static bool ds41_pad_tick(ds41_pad_budget *b) {
    if (!b || b->cancelled || b->work >= b->limit) return false;
    /* Poll cancellation on the first tick and every 4096 thereafter, not per token. */
    const bool poll = (b->work & 4095u) == 0;
    ++b->work;
    if (poll && ds41_scr_poll(b->session)) { b->cancelled = true; return false; }
    return true;
}
static ds41_pad_result ds41_pad_limit_result(const ds41_pad_budget *b) {
    return b && b->cancelled ? DS41_PAD_CANCELLED : DS41_PAD_REFUSED;
}
static bool ds41_pad_transcript(const int *ids, uint32_t n,
                                const ds41_scr_coordinates *map, ds41_pad_budget *b) {
    if (!map || n >= 1048576u || (n && !ids) ||
        (map->identity ? !ds41_coordinates_shape(map) || map->physical_tokens != n :
         (map->pads || map->count || map->capacity || map->physical_tokens))) return false;
    for (uint32_t i = 0; i < n; ++i)
        if (!ds41_pad_tick(b) || ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_VOCAB) return false;
    return true;
}
typedef struct {
    uint32_t common, live_logical, carried_pads;
} ds41_pad_prefix_info;
/* Read before discovery: a headed full logical continuation must not acquire
 * new alignment pads for grafts that the ordinary continuation will not use.
 * The pad-skipping walk is cache_reuse_common_prefix; this owner validates the
 * vocabulary and charges the comparisons it performed to the planning budget. */
static ds41_pad_result ds41_pad_prefix(const int *target, uint32_t n,
                                      const int *live, uint32_t live_n,
                                      const ds41_scr_coordinates *map,
                                      ds41_pad_prefix_info *out, ds41_pad_budget *budget) {
    if (out) memset(out, 0, sizeof(*out));
    if (!target || !n || n >= 1048576u || !map || !out || !budget) return DS41_PAD_REFUSED;
    if (!ds41_pad_transcript(live, live_n, map, budget)) return ds41_pad_limit_result(budget);
    for (uint32_t i = 0; i < n; ++i)
        if (!ds41_pad_tick(budget) || target[i] < 0 || (uint32_t)target[i] >= DS4_N_VOCAB)
            return ds41_pad_limit_result(budget);
    cache_reuse_common_prefix_result common;
    if (cache_reuse_common_prefix((const int32_t *)target, n, (const int32_t *)live, live_n,
            map->pads, map->count, &common, NULL, 0) != CACHE_REUSE_COMPLETE ||
        common.common_logical_tokens > n ||
        common.live_logical_tokens != live_n - map->count) return DS41_PAD_REFUSED;
    /* One comparison per matched logical token plus the mismatch, if any. */
    uint64_t compared = common.common_logical_tokens +
        (common.common_logical_tokens < n && common.common_logical_tokens < common.live_logical_tokens);
    uint64_t limit = budget->comparison_limit ? budget->comparison_limit : DS4_SCR_DENSE_COMPARISONS_MAX;
    if (limit > DS4_SCR_DENSE_COMPARISONS_MAX || compared > limit - budget->comparisons)
        return DS41_PAD_REFUSED;
    budget->comparisons += compared;
    *out = (ds41_pad_prefix_info){.common = common.common_logical_tokens,
        .live_logical = common.live_logical_tokens, .carried_pads = common.carried_pads};
    return DS41_PAD_OK;
}
#endif

/* ==== 3. Rebase Transaction ============================================== */
#ifdef DS4_SCR_ENGINE_SESSION
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
/* Common pin prefix: a sealed, non-retiring archive owned by this session and
 * model. Callers add their own generation, extent and storage checks. */
static bool ds41_archive_owned_sealed(const ds41_scr_archive *a, const ds4_session *s) {
    return a && a->session == s && a->model == &s->engine->model &&
        a->sealed && !a->retiring;
}
#endif

/* Private stack scope shared only by the dense consumer and its owned entries.
 * Not a public capability, thread lock, or permission to observe during SCR. */
typedef struct ds41_policy_owner {
    ds4_session *session;
    const struct ds41_policy_owner *identity;
    const ds4_tokens *full;
    const int *bytes;
    int length, capacity;
    uint64_t live_host_bytes;
    uint32_t archive_capacity;
    bool active;
    bool source_pads_are_content; /* Actual-token planner: source pads are ordinary content. */
    const ds41_scr_coordinates *coordinates;
    const uint32_t *coordinate_pads;
    uint32_t coordinate_count, coordinate_capacity;
    uint64_t coordinate_identity, coordinate_reservation;
    bool coordinates_published;
} ds41_policy_owner;
static bool ds41_policy_bound(ds4_session *s, const ds4_tokens *full,
                              const ds41_policy_owner *owner) {
    /* Check session identity before dereferencing any supplied token. A copy of
     * this stack value is not the installed scope; no expired token is retained. */
    return s && owner && s->scr_dense_owner == owner && s->scr_policy_active &&
        owner->active && owner->identity == owner &&
        owner->session == s && owner->full == full && full &&
        owner->bytes == full->v && owner->length == full->len &&
        owner->capacity == full->cap && full->v && full->len > 0 &&
        !owner->coordinates_published &&
        (!owner->coordinates ? owner->coordinate_reservation == 0 :
         (owner->coordinates->pads == owner->coordinate_pads &&
          owner->coordinates->count == owner->coordinate_count &&
          owner->coordinates->capacity == owner->coordinate_capacity &&
          owner->coordinates->identity == owner->coordinate_identity &&
          owner->coordinates->physical_tokens == (uint32_t)full->len));
}
static bool ds41_policy_authorized(ds4_session *s, const ds4_tokens *full,
                                   const ds41_policy_owner *owner) {
    return s && !s->scr_multi_active && ds41_policy_bound(s, full, owner);
}
static bool ds41_policy_reservation(ds4_session *s, const ds4_tokens *full,
                                    const ds41_policy_owner *owner,
                                    uint64_t inner, uint64_t *total) {
    if (!owner) { *total = inner; return true; }
    if (!ds41_policy_authorized(s, full, owner) ||
        owner->live_host_bytes > UINT64_MAX - inner) return false;
    *total = inner + owner->live_host_bytes;
    return true;
}
/* The caller includes this heap in its pre-mutation admission. Binding consumes
 * a session nonce even if later work aborts, avoiding reuse of attempt identity. */
static bool ds41_policy_bind_coordinates(ds41_policy_owner *owner,
                                         const ds41_scr_coordinates *map,
                                         uint64_t reserved_heap) {
    if (!owner || !ds41_policy_authorized(owner->session, owner->full, owner) ||
        owner->coordinates || !map || map == &owner->session->scr_coordinates ||
        (map->pads && map->pads == owner->session->scr_coordinates.pads) ||
        map->identity <= owner->session->scr_coordinate_serial ||
        !ds41_coordinates_match(map, owner->full->v, (uint32_t)owner->full->len, DS4_N_VOCAB))
        return false;
    size_t bytes;
    if (!ds41_coordinates_heap_bytes(map->capacity, &bytes) || bytes != reserved_heap ||
        reserved_heap > owner->live_host_bytes) return false;
    owner->coordinates = map;
    owner->coordinate_pads = map->pads;
    owner->coordinate_count = map->count;
    owner->coordinate_capacity = map->capacity;
    owner->coordinate_identity = map->identity;
    owner->coordinate_reservation = reserved_heap;
    owner->session->scr_coordinate_serial = map->identity;
    return true;
}

/* Final headed publication only. The physical descriptor remains immutable
 * through the last GPU operation; metadata ownership transfers afterwards. */
static bool ds41_policy_publish_coordinates(ds41_policy_owner *owner,
                                            ds41_scr_coordinates *pending) {
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (!owner || !pending || owner->coordinates != pending ||
        !ds41_policy_authorized(owner->session, owner->full, owner) ||
        owner->session->scr_coordinate_serial != owner->coordinate_identity) return false;
    ds4_session *s = owner->session;
    ds41_gpu_graph *g = &s->ds41_graph;
    size_t old_bytes, new_bytes;
    if (!s->engine || s->engine->backend != DS4_BACKEND_METAL || s->distributed ||
        !ds4_session_is_ds41(s) || !s->ds41_graph_ready || !g->valid || !s->checkpoint_valid ||
        !s->logits || g->encoder_resident ||
        ds4_gpu_commands_active() || g->pos != (uint32_t)owner->full->len ||
        s->checkpoint.len != owner->full->len || !s->checkpoint.v ||
        memcmp(s->checkpoint.v, owner->full->v, (size_t)owner->full->len * sizeof(int)) ||
        !ds41_coordinates_heap_bytes(s->scr_coordinates.capacity, &old_bytes) ||
        !ds41_coordinates_heap_bytes(pending->capacity, &new_bytes) ||
        old_bytes != s->scr_coordinate_charge || new_bytes != owner->coordinate_reservation ||
        new_bytes > owner->live_host_bytes) return false;
    if (!ds41_coordinates_replace_charged(&s->scr_coordinates, pending,
            owner->full->v, (uint32_t)owner->full->len, DS4_N_VOCAB,
            &g->allocation_bytes, &s->engine->ds41_session_bytes, NULL)) return false;
    s->scr_coordinate_charge = new_bytes;
    owner->live_host_bytes -= new_bytes;
    owner->coordinate_reservation = 0;
    owner->coordinates_published = true;
    return true;
#else
    (void)owner; (void)pending;
    return false;
#endif
}

/* Private destination ownership and drained capture readiness. The old source
 * stays installed throughout preparation; backend readiness is not publication. */
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
#ifndef DS41_REBASE_OPEN
#define DS41_REBASE_OPEN ds41_raw_open
#endif
static int ds41_session_rebase_prepare(ds4_session *s, const ds4_tokens *full,
        const ds41_policy_owner *owner, uint64_t inner_host, char *err, size_t errlen) {
    if (!ds41_policy_authorized(s, full, owner) || !s->engine || !s->ds41_graph_ready ||
        !ds4_session_is_ds41(s) || s->distributed || s->engine->tp.active ||
        s->engine->backend != DS4_BACKEND_METAL || !s->logits ||
        full->len < 128 || full->cap < full->len || full->len >= s->ctx_size) {
        snprintf(err, errlen, "SCR rebase: invalid preparation owner"); return 1;
    }
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *old = g->archive;
    const uint32_t capacity = owner->archive_capacity, n = (uint32_t)full->len;
    const uint32_t covered = n & ~7u;
    const uint64_t base = ds41_archive_bytes(capacity);
    size_t map_bytes = 0;
    uint64_t extra;
    if (!base || n > capacity || capacity > g->ctx || n >= g->ctx ||
        g->tp_world != 1u || g->image_count || s->sync_image_count ||
        s->checkpoint_image_count || g->imatrix || g->encoder_resident ||
        g->archive_pending || g->archive_capture_pending || ds4_gpu_commands_active() ||
        !old || old->session != s || old->model != &s->engine->model || old->retiring ||
        !old->generation || old->generation == UINT64_MAX ||
        (owner->coordinates && !ds41_coordinates_heap_bytes(owner->coordinates->count, &map_bytes)) ||
        !ds41_policy_reservation(s, full, owner, inner_host, &extra)) {
        snprintf(err, errlen, "SCR rebase: unsupported reservation/source"); return 1;
    }
    if (owner->coordinates &&
        !ds41_coordinates_match(owner->coordinates, full->v, n, DS4_N_VOCAB)) {
        snprintf(err, errlen, "SCR rebase: invalid physical provenance"); return 1;
    }
    for (uint32_t i = 0; i < n; ++i)
        if (full->v[i] < 0 || (uint32_t)full->v[i] >= DS4_N_VOCAB) {
            snprintf(err, errlen, "SCR rebase: invalid physical token"); return 1;
        }
    if (map_bytes > UINT64_MAX - base || base + map_bytes > UINT64_MAX - g->allocation_bytes ||
        base + map_bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
        extra > UINT64_MAX - s->engine->ds41_session_bytes - base - map_bytes ||
        !ds41_memory_admit(s->engine,
            s->engine->ds41_session_bytes + base + map_bytes + extra, false)) {
        snprintf(err, errlen, "SCR rebase: simultaneous old/new reservation refused"); return 1;
    }
    if (!ds41_archive_published_valid(old)) {
        old->retiring = true;
        snprintf(err, errlen, "SCR rebase: published predecessor proof refused"); return 1;
    }
    ds41_scr_archive *next = ds41_archive_alloc(0u, capacity, s, &s->engine->model);
    if (!next) {
        snprintf(err, errlen, "SCR rebase: destination allocation refused"); return 1;
    }
    g->archive_pending = next;
    g->allocation_bytes += base;
    s->engine->ds41_session_bytes += base;
    next->cancelled = ds41_archive_raw_cancel;
    next->rebase_predecessor = old;
    next->predecessor_generation = old->source_generation;
    next->predecessor_coordinate_identity = old->coordinates.identity;
    next->predecessor_start = old->start; next->predecessor_end = old->end;
    next->predecessor_capacity = old->cap;
    memcpy(next->tokens, full->v, (size_t)n * sizeof(int));
    next->writer_known_end = n;
    if (owner->coordinates) {
        if (!ds41_coordinates_clone(&next->coordinates, owner->coordinates)) goto failed;
        next->allocation_bytes += map_bytes;
        g->allocation_bytes += map_bytes;
        s->engine->ds41_session_bytes += map_bytes;
        next->writer_future_pads = next->coordinates.count;
        while (next->coordinates.count &&
               next->coordinates.pads[next->coordinates.count - 1u] >= covered)
            --next->coordinates.count;
        next->coordinates.physical_tokens = covered;
    }
    const char *directory = getenv("TMPDIR");
    if (!directory || !*directory) directory = "/tmp";
    /* The graph already owns both RAM and any descriptor/path created here. */
    if (!DS41_REBASE_OPEN(next->raw, directory, capacity, ds41_raw_bytes(capacity))) goto failed;
    return 0;
failed: {
        const int saved = errno;
        int retained = ds41_session_archive_release_slot(s, true);
        snprintf(err, errlen, "SCR rebase: destination preparation failed%s",
                 retained ? "; cleanup retained/fenced" : "");
        errno = saved;
        return 1;
    }
}
static int ds41_session_rebase_failure(ds4_session *s, char *err, size_t errlen,
                                        const char *reason) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_archive_block_writer(g->archive_pending, false);
    g->archive_capture_pending = false;
    s->checkpoint_valid = g->valid = false;
    s->mtp_draft_valid = false;
    const bool cancelled = ds4_session_cancelled(s);
    snprintf(err, errlen, "SCR rebase: %s%s", reason, cancelled ? " (interrupted)" : "");
    return cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
}
/* Reset withdraws routing. Only the installed private owner can activate a
 * pristine prepared destination at the actual empty full-model frontier. */
static bool ds41_session_rebase_capture_start(ds4_session *s) {
    const ds41_policy_owner *owner = s ? s->scr_dense_owner : NULL;
    if (!owner || !ds41_policy_bound(s, owner->full, owner) || !s->scr_multi_active ||
        !s->ds41_graph_ready || s->checkpoint_valid || s->checkpoint.len) return false;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *a = g->archive_pending;
    if (!a || a->session != s || a->model != &s->engine->model || a->sealed || a->lost ||
        a->retiring || a->start || a->end || a->captured_end || a->raw_origin ||
        !a->raw || !a->raw->readable || !a->raw->writable || a->raw->committed_tokens ||
        a->raw->written_tokens || a->raw->file_bytes || g->pos || !g->valid ||
        g->encoder_resident || ds4_gpu_commands_active() || !g->archive ||
        a->writer_known_end != (uint32_t)owner->full->len ||
        a->coordinates.identity != (owner->coordinates ? owner->coordinates->identity : 0u) ||
        memcmp(a->tokens, owner->full->v, (size_t)owner->full->len * sizeof(int))) return false;
    g->archive_capture_pending = true;
    return true;
}
/* A complete private forward frontier can make file records ready without a
 * synthetic output head. Every retained row and complete canonical pair must
 * have been captured by the full-model call before its frontier is certified. */
static bool ds41_session_rebase_forward_ready(ds4_session *s, uint32_t from, uint32_t end) {
    if (!s || !s->engine || !s->scr_multi_active || !s->ds41_graph_ready) return false;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *a = g->archive_pending;
    if (!g->archive_capture_pending || !a || a->lost || a->retiring || a->sealed ||
        a->session != s || a->model != &s->engine->model || a->captured_end != from ||
        end < from || end > a->writer_known_end || end > a->limit ||
        !g->valid || g->pos != end || s->checkpoint_valid ||
        s->checkpoint.len != (int)end || !s->checkpoint.v ||
        ds4_gpu_commands_active() || !ds4_gpu_synchronize() ||
        memcmp(s->checkpoint.v + from, a->tokens + from, (size_t)(end - from) * sizeof(int)) ||
        end < a->raw_origin || end - a->raw_origin > a->raw_arena_tokens) return false;
    if (!ds41_raw_capture_complete(a->raw_tags, a->raw_origin, end - a->raw_origin)) {
        ds41_archive_block_writer(a, true); return false;
    }
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t pos = from / ratio * ratio; pos + ratio <= end; pos += ratio)
            if (a->tags[ds41_archive_tag(a->cap, owner) + pos / ratio] != pos) {
                ds41_archive_block_writer(a, true); return false;
            }
    }
    if (!ds41_archive_flush_ready(a, end, ds41_archive_raw_cancel, s)) return false;
    a->captured_end = end;
    return true;
}
/* Batched refresh capture: a complete forward frontier inside the unsealed
 * refresh archive's arena makes its records ready without an output head.
 * The source span still ends at one real head, which certifies it before seal. */
static bool ds41_session_refresh_forward_ready(ds4_session *s, uint32_t end) {
    if (!s || !s->engine || !s->scr_multi_active || !s->scr_refresh_capture ||
        !s->ds41_graph_ready) return false;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *a = g->archive;
    if (!a || g->archive_pending || g->archive_capture_pending || a->sealed || a->lost ||
        a->retiring || a->session != s || a->model != &s->engine->model ||
        !g->valid || g->pos != end || s->checkpoint.len != (int)end ||
        end < a->raw_origin || end - a->raw_origin > a->raw_arena_tokens || end > a->limit ||
        ds4_gpu_commands_active() || !ds4_gpu_synchronize()) return false;
    if (!ds41_raw_capture_complete(a->raw_tags, a->raw_origin, end - a->raw_origin)) {
        ds41_archive_block_writer(a, true); return false;
    }
    return ds41_archive_flush_ready(a, end, ds41_archive_raw_cancel, s);
}
#endif

/* Owning Flash adapters. The source map remains independently owned; a live
 * map nonce binds compatible operations without a full prefix scan per token. */
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
#ifndef DS41_EVAL_STEP
#define DS41_EVAL_STEP ds41_graph_step
#endif
static ds41_scr_archive *ds41_session_append_writer(ds4_session *s) {
    ds41_scr_archive *a = s && s->ds41_graph_ready ? s->ds41_graph.archive : NULL;
    return a && a->writer_state == DS41_ARCHIVE_ARMED ? a : NULL;
}
static void ds41_session_append_disarm(ds41_scr_archive *a) {
    if (!a || a->writer_state != DS41_ARCHIVE_ARMED) return;
    a->writer_state = DS41_ARCHIVE_INCOMPATIBLE;
    a->range.valid = false;
}
static bool ds41_session_append_cursor(ds4_session *s, uint32_t pos) {
    if (!s || !s->engine || !s->ds41_graph_ready || s->distributed ||
        s->engine->backend != DS4_BACKEND_METAL || !ds4_session_is_ds41(s) ||
        s->engine->tp.active || !s->checkpoint_valid || !s->logits ||
        s->checkpoint.len < 0 || s->checkpoint.cap < s->checkpoint.len ||
        (pos && !s->checkpoint.v)) return false;
    ds41_gpu_graph *g = &s->ds41_graph;
    return g->valid && g->pos == pos && s->checkpoint.len == (int)pos &&
        pos < g->ctx && pos < (uint32_t)s->ctx_size && g->tp_world == 1u &&
        !g->image_count && !g->encoder_resident && !ds4_gpu_commands_active();
}
static int ds41_session_append_failure(ds4_session *s, char *err, size_t errlen,
                                        const char *reason) {
    ds41_scr_archive *a = s->ds41_graph.archive;
    const bool cancelled = ds4_session_cancelled(s);
    if (a && a->sealed)
        ds41_archive_append_failure(a, a->history_hole, cancelled ? ECANCELED : EIO);
    s->checkpoint_valid = s->ds41_graph.valid = false;
    s->mtp_draft_valid = false;
    if (errlen) snprintf(err, errlen, "SCR append: %s%s", reason,
                        cancelled ? " (interrupted)" : "");
    return cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
}
static int ds41_session_append_head(ds4_session *s, char *err, size_t errlen) {
    ds41_scr_archive *a = ds41_session_append_writer(s);
    if (!a) return 0;
    uint32_t pos = s->checkpoint.len >= 0 ? (uint32_t)s->checkpoint.len : UINT32_MAX;
    if (!ds41_session_append_cursor(s, pos) || a->session != s ||
        a->model != &s->engine->model || pos < a->start || pos > a->limit ||
        !ds41_archive_append_observe(a, s, &s->engine->model,
            s->checkpoint.v + a->start, pos, ds41_archive_raw_cancel, s))
        return ds41_session_append_failure(s, err, errlen, "headed publication refused");
    return 0;
}
/* Establish compatibility once at a request boundary. A new owned physical
 * map may reserve additional future pads, but its captured prefix must retain
 * every old pad position. Genuine changes disarm, never replay. */
static int ds41_session_append_prepare_prefix(ds4_session *s, const ds4_tokens *prompt,
                                              char *err, size_t errlen) {
    ds41_scr_archive *a = ds41_session_append_writer(s);
    if (!a) return 0;
    const uint32_t from = a->captured_end;
    if (!ds41_session_append_cursor(s, from) || !prompt || !prompt->v ||
        prompt->len < (int)from || a->session != s || a->model != &s->engine->model) {
        ds41_session_append_disarm(a); return 0;
    }
    const ds41_policy_owner *owner = s->scr_dense_owner;
    const ds4_tokens *full = prompt;
    const ds41_scr_coordinates *map = &s->scr_coordinates;
    if (owner) {
        if (!ds41_policy_bound(s, owner->full, owner) || prompt->len > owner->full->len) {
            snprintf(err, errlen, "SCR append: invalid owning prefix scope"); return 1;
        }
        full = owner->full;
        map = owner->coordinates ? owner->coordinates : &s->scr_coordinates;
    }
    uint32_t stop = (uint32_t)full->len < a->limit ? (uint32_t)full->len : a->limit;
    bool rebind = map->identity != a->writer_coordinate_identity;
    if (!rebind && stop <= a->writer_known_end) return 0;
    if (memcmp(a->tokens, s->checkpoint.v + a->start,
               (size_t)(from - a->start) * sizeof(int))) {
        ds41_session_append_disarm(a); return 0;
    }
    if (rebind) {
        const ds41_scr_coordinates *live = &s->scr_coordinates;
        if (!map->identity || !a->coordinates.identity ||
            live->identity != a->writer_coordinate_identity ||
            !ds41_coordinates_match(live, s->checkpoint.v, from, DS4_N_VOCAB) ||
            !ds41_coordinates_match(map, full->v, (uint32_t)full->len, DS4_N_VOCAB)) {
            ds41_session_append_disarm(a); return 0;
        }
        uint32_t before = 0;
        while (before < map->count && map->pads[before] < from) ++before;
        if (before != live->count) { ds41_session_append_disarm(a); return 0; }
        if (!ds41_coordinates_same_pads(map, live, before)) {
            ds41_session_append_disarm(a); return 0;
        }
        uint32_t pads = 0;
        for (uint32_t i = 0; i < map->count; ++i)
            if (map->pads[i] >= a->start && map->pads[i] < stop) ++pads;
        size_t heap;
        uint64_t outer = owner ? owner->live_host_bytes : (uint64_t)full->len * sizeof(int);
        if (!ds41_coordinates_heap_bytes(pads, &heap) ||
            outer > UINT64_MAX - heap || outer + heap > UINT64_MAX - s->engine->ds41_session_bytes ||
            !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + outer + heap, false)) {
            snprintf(err, errlen, "SCR append: interval-map reservation refused"); return 1;
        }
        ds41_scr_coordinates pending = {0};
        uint32_t logical_start;
        if (!ds41_coordinates_slice(map, full->v, (uint32_t)full->len, DS4_N_VOCAB,
                a->start, stop - a->start, &pending, &logical_start)) {
            snprintf(err, errlen, "SCR append: interval-map allocation refused"); return 1;
        }
        const uint32_t future_pads = pending.count, count = ds41_archive_count(a);
        uint32_t published = 0;
        while (published < pending.count && pending.pads[published] < count) ++published;
        pending.count = published; pending.physical_tokens = count;
        pending.identity = a->coordinates.identity;
        bool compatible = logical_start == a->logical_start && published == a->coordinates.count &&
            ds41_coordinates_same_pads(&pending, &a->coordinates, published);
        if (!compatible) {
            ds41_coordinates_free(&pending); ds41_session_append_disarm(a); return 0;
        }
        if (!ds41_coordinates_replace_charged(&a->coordinates, &pending, a->tokens, count,
                DS4_N_VOCAB, &s->ds41_graph.allocation_bytes,
                &s->engine->ds41_session_bytes, &a->allocation_bytes)) {
            ds41_coordinates_free(&pending);
            snprintf(err, errlen, "SCR append: interval-map publication refused"); return 1;
        }
        a->writer_future_pads = future_pads;
        a->writer_coordinate_identity = map->identity;
    } else if ((a->coordinates.identity != 0) != (map->identity != 0)) {
        ds41_session_append_disarm(a); return 0;
    }
    for (uint32_t pos = from; pos < stop; ++pos)
        if (full->v[pos] < 0 || (uint32_t)full->v[pos] >= DS4_N_VOCAB) {
            snprintf(err, errlen, "SCR append: invalid future token"); return 1;
        }
    if (stop > from)
        memcpy(a->tokens + from - a->start, full->v + from, (size_t)(stop - from) * sizeof(int));
    a->writer_known_end = stop;
    return 0;
}
/* The existing coordinate append receipt certifies one ordinary generated ID.
 * Source tokens/maps were bound at refresh or prefix; nonce/length checks here
 * are constant time and never infer pads from an equal numeric token ID. */
static int ds41_session_append_before_eval(ds4_session *s, char *err, size_t errlen) {
    if (s && s->ds41_graph_ready && s->ds41_graph.archive_pending) {
        snprintf(err, errlen, "SCR rebase: unresolved private archive ownership"); return 1;
    }
    ds41_scr_archive *a = ds41_session_append_writer(s);
    if (!a) return 0;
    const ds41_scr_coordinates *m = &s->scr_coordinates;
    if (!ds41_session_append_cursor(s, a->captured_end) ||
        m->identity != a->writer_coordinate_identity ||
        (m->identity && m->physical_tokens != a->captured_end) ||
        (!m->identity && (m->pads || m->count || m->capacity || m->physical_tokens)))
        ds41_session_append_disarm(a);
    if (a->writer_state != DS41_ARCHIVE_ARMED) return 0;
    if (s->checkpoint.len == s->checkpoint.cap) {
        uint32_t capacity = s->checkpoint.cap ? (uint32_t)s->checkpoint.cap * 2u : 64u;
        if (capacity > (uint32_t)s->ctx_size) capacity = (uint32_t)s->ctx_size;
        uint64_t bytes = (uint64_t)capacity * sizeof(int);
        if (capacity <= (uint32_t)s->checkpoint.len ||
            bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
            !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + bytes, false)) {
            snprintf(err, errlen, "SCR append: generation token reservation refused"); return 1;
        }
        int *tokens = realloc(s->checkpoint.v, (size_t)bytes);
        if (!tokens) {
            snprintf(err, errlen, "SCR append: generation token allocation refused"); return 1;
        }
        s->checkpoint.v = tokens; s->checkpoint.cap = (int)capacity;
    }
    return 0;
}
static int ds41_session_append_after_eval(ds4_session *s,
                                          const ds41_coordinate_append *coordinates,
                                          char *err, size_t errlen) {
    ds41_scr_archive *a = ds41_session_append_writer(s);
    if (!a) return 0;
    uint32_t pos = (uint32_t)s->checkpoint.len;
    if (pos != a->captured_end + 1u ||
        (coordinates->mapped && (coordinates->before != a->captured_end ||
                                 coordinates->identity != a->writer_coordinate_identity)))
        return ds41_session_append_failure(s, err, errlen, "generation continuity refused");
    int rc = ds41_session_append_head(s, err, errlen);
    if (!rc) a->writer_coordinate_identity = s->scr_coordinates.identity;
    return rc;
}
/* The ordinary Flash scalar route runs even without an armed SCR writer. */
static int ds41_session_eval_scalar(ds4_session *s, int token, char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    /* Preserve upstream recent-restore logits for ordinary, non-SCR sessions. */
    ds41_dspark *d = s->ds41_graph.dspark;
    const bool rewindable = d && d->rewind_end == s->ds41_graph.pos &&
        d->rewind_end && d->rewind_end < d->verify_start + d->verify_count;
    if (ds41_session_append_before_eval(s, err, errlen)) return 1;
    ds41_coordinate_append coordinates;
    if (!ds41_session_coordinates_begin_append(s, token, &coordinates)) {
        if (errlen) snprintf(err, errlen, "V4.1 generation coordinate binding refused");
        return 1;
    }
    ds41_route_trace_begin("decode1", s->route_trace_slot - 1, s->ds41_graph.pos);
    const bool stepped = s->ds41_graph_ready &&
        (s->checkpoint_valid || s->checkpoint.len == 0) &&
        s->ds41_graph.pos == (uint32_t)s->checkpoint.len &&
        DS41_EVAL_STEP(&s->ds41_graph, &e->model, &e->weights, token, s->logits);
    ds41_route_trace_end(-1, stepped);
    if (!stepped) {
        if (s->ds41_graph.archive && s->ds41_graph.archive->sealed &&
            (s->ds41_graph.archive->writer_state == DS41_ARCHIVE_ARMED ||
             s->ds41_graph.archive->writer_state == DS41_ARCHIVE_BLOCKED))
            return ds41_session_append_failure(s, err, errlen, "model step failed");
        s->checkpoint_valid = false;
        if (errlen) snprintf(err, errlen, "V4.1 decode failed at position %d", s->checkpoint.len);
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    if (rewindable) d->rewind_end = s->ds41_graph.pos;
    if ((coordinates.mapped && (!s->ds41_graph.valid ||
        s->ds41_graph.pos != (uint32_t)s->checkpoint.len ||
        s->ds41_graph.encoder_resident || ds4_gpu_commands_active())) ||
        !ds41_session_coordinates_finish_append(s, &coordinates)) {
        if (ds41_session_append_writer(s))
            ds41_session_append_failure(s, err, errlen, "generation map publication failed");
        ds4_session_invalidate(s);
        if (errlen) snprintf(err, errlen, "V4.1 generation coordinate publication failed");
        return 1;
    }
    return ds41_session_append_after_eval(s, &coordinates, err, errlen);
}
#endif
#endif

/* ==== 4. Multi-Range Execution & Sync ==================================== */
#ifdef DS4_SCR_ENGINE_SESSION
/* Shared owning-session Flash orchestration. Private mode never publishes a
 * prefix head; deferred/display rows are not a proven full-model frontier. */
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
#ifndef DS41_SYNC_STEP
#define DS41_SYNC_STEP ds41_graph_step
#define DS41_SYNC_SHORT ds41_graph_short_prefill
#define DS41_SYNC_PREFILL ds41_graph_prefill
#define DS41_SYNC_SWEEP ds41_graph_prefill_sweep
#define DS41_SYNC_HEAD ds41_graph_logits
#define DS41_SYNC_ACQUIRE ds41_encoder_acquire
#define DS41_SYNC_RELEASE ds41_encoder_release
#endif
#ifndef DS41_SYNC_CAPTURE_BATCH_PREFILL
#define DS41_SYNC_CAPTURE_BATCH_PREFILL ds41_graph_prefill
#endif
typedef struct { uint32_t completed, heads; bool unknown, mutated; } ds41_sync_frontier;
/* Encoder residency is held across raw arena flushes and dropped before the
 * prompt's last rows, which then warm the expert cache for decode. */
#define DS41_ENCODER_RELEASE_TAIL 768u
static void ds41_encoder_release_tail(ds41_gpu_graph *g, const ds4_model *m,
                                      ds41_encoder_residency *encoder, uint32_t remaining) {
    if (g->encoder_resident && remaining < DS41_ENCODER_RELEASE_TAIL)
        DS41_SYNC_RELEASE(g, m, encoder);
}
/* A capturing call stops at the writer's raw arena frontier (and its limit),
 * where the arena flushes; refuses a cursor outside the current arena. */
static bool ds41_capture_arena_bound(const ds41_scr_archive *writer, uint32_t pos,
                                     uint32_t end, uint32_t *bound) {
    const uint32_t arena = writer->raw_arena_tokens;
    if (writer->raw_origin > UINT32_MAX - arena || pos < writer->raw_origin ||
        pos >= writer->raw_origin + arena || pos >= writer->limit) return false;
    uint32_t b = end;
    if (b > writer->raw_origin + arena) b = writer->raw_origin + arena;
    if (b > writer->limit) b = writer->limit;
    *bound = b;
    return true;
}
static int ds41_session_prefix_impl(ds4_session *s, const ds4_tokens *prompt,
                               bool private_fresh, ds41_sync_frontier *frontier,
                               char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    ds41_gpu_graph *g = &s->ds41_graph;
    if (frontier) memset(frontier, 0, sizeof(*frontier));
    if (!s->ds41_graph_ready) {
        snprintf(err, errlen, "V4.1 graph is not initialized");
        return 1;
    }
    if (g->archive_pending) {
        const ds41_policy_owner *owner = s->scr_dense_owner;
        if (!private_fresh || !s->scr_multi_active || !owner ||
            !ds41_policy_bound(s, owner->full, owner) || !prompt || !prompt->v ||
            prompt->len <= 0 || prompt->cap < prompt->len ||
            (uint32_t)prompt->len > g->archive_pending->writer_known_end ||
            g->archive_pending->retiring || g->archive_pending->lost ||
            memcmp(prompt->v, g->archive_pending->tokens, (size_t)prompt->len * sizeof(int))) {
            snprintf(err, errlen, "SCR rebase: unresolved private archive ownership"); return 1;
        }
    }
    if (private_fresh || !s->checkpoint_valid || !g->valid ||
        g->pos != (uint32_t)s->checkpoint.len ||
        !ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        if (frontier) frontier->mutated = true;
        ds41_graph_reset(g);
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        ds41_session_coordinates_clear(s);
    }
    if (g->archive_pending && !ds41_session_rebase_capture_start(s))
        return ds41_session_rebase_failure(s, err, errlen, "private capture activation refused");
    if (!private_fresh && ds41_session_append_prepare_prefix(s, prompt, err, errlen)) return 1;
    bool pending_logits = false, interrupted = false, decoder_pending = false;
    ds41_encoder_residency encoder = {0};
    uint32_t acquire_count = (uint32_t)(prompt->len - s->checkpoint.len);
    DS41_SYNC_ACQUIRE(g, &e->model, &e->weights,
        acquire_count, &encoder, s->cancel, s->cancel_ud);
    for (int i = s->checkpoint.len; i < prompt->len;) {
        if (ds4_session_cancelled(s)) { interrupted = true; break; }
        uint32_t remaining = (uint32_t)(prompt->len - i), capture_target = 0;
        const bool rebase_capture = g->archive_capture_pending;
        const bool refresh_capture = !rebase_capture && s->scr_refresh_capture &&
            g->archive && !g->archive->sealed;
        ds41_scr_archive *writer = rebase_capture ? g->archive_pending :
            refresh_capture ? g->archive : ds41_session_append_writer(s);
        if (writer) {
            uint32_t bound;
            if (!ds41_capture_arena_bound(writer, (uint32_t)i, (uint32_t)prompt->len, &bound)) {
                DS41_SYNC_RELEASE(g, &e->model, &encoder);
                if (refresh_capture) {
                    snprintf(err, errlen, "SCR refresh: capture arena frontier refused"); return 1;
                }
                return rebase_capture ?
                    ds41_session_rebase_failure(s, err, errlen, "capture arena frontier refused") :
                    ds41_session_append_failure(s, err, errlen, "capture arena frontier refused");
            }
            if (!rebase_capture && !refresh_capture && bound % 8u && bound - writer->end >= 8u)
                bound &= ~7u;
            if (bound <= (uint32_t)i) {
                DS41_SYNC_RELEASE(g, &e->model, &encoder);
                if (refresh_capture) {
                    snprintf(err, errlen, "SCR refresh: aligned capture frontier refused"); return 1;
                }
                return rebase_capture ?
                    ds41_session_rebase_failure(s, err, errlen, "aligned capture frontier refused") :
                    ds41_session_append_failure(s, err, errlen, "aligned capture frontier refused");
            }
            capture_target = bound; remaining = bound - (uint32_t)i;
        }
        /* Residency follows the prompt remainder, not the capture arena bound;
         * a resident call still stops at the arena frontier. */
        const uint32_t prompt_remaining = (uint32_t)(prompt->len - i);
        ds41_encoder_release_tail(g, &e->model, &encoder, prompt_remaining);
        const uint32_t short_count = decoder_pending || g->encoder_resident ? 0u :
            ds41_short_prefill_count(g, &e->weights, remaining);
        uint32_t resident_count = prompt_remaining - 512u < g->prefill_cap ?
            prompt_remaining - 512u : g->prefill_cap;
        if (writer && resident_count > remaining) resident_count = remaining;
        const uint32_t count = short_count ? short_count : g->encoder_resident ?
            resident_count : ds41_prefill_count(g, remaining);
        const bool defer_decoder = !rebase_capture && !g->encoder_resident && count >= 16384u &&
            remaining - count >= 8192u &&
            !getenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX") &&
            !getenv("DS4_METAL_DISABLE_V41_DEFER_DECODER");
        /* Never add an encoder sweep just to defer the decoder. A short
         * remainder finishes the pending decoder here, then runs normally. */
        const bool layer_major = count > 1u;
        if (frontier) frontier->mutated = true;
        const bool ok = short_count ?
            DS41_SYNC_SHORT(g, &e->model, &e->weights, prompt->v + i, count) :
            layer_major ?
            (defer_decoder || decoder_pending ?
             DS41_SYNC_SWEEP(g, &e->model, &e->weights, prompt->v + i, count,
                private_fresh ? NULL : s->progress, s->progress_ud, prompt->len,
                s->cancel, s->cancel_ud, defer_decoder, decoder_pending) :
             DS41_SYNC_PREFILL(g, &e->model, &e->weights, prompt->v + i, count,
                private_fresh ? NULL : s->progress, s->progress_ud, prompt->len,
                s->cancel, s->cancel_ud)) :
            DS41_SYNC_STEP(g, &e->model, &e->weights, prompt->v[i], NULL);
        if (!ok) {
            DS41_SYNC_RELEASE(g, &e->model, &encoder);
            s->checkpoint_valid = false;
            if (frontier) frontier->unknown = true;
            if (rebase_capture)
                return ds41_session_rebase_failure(s, err, errlen, "prefix model step failed");
            if (g->archive && g->archive->sealed &&
                (g->archive->writer_state == DS41_ARCHIVE_ARMED ||
                 g->archive->writer_state == DS41_ARCHIVE_BLOCKED))
                return ds41_session_append_failure(s, err, errlen, "prefix model step failed");
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            snprintf(err, errlen, "V4.1 prefill failed at position %d", i);
            return 1;
        }
        for (uint32_t t = 0; t < count; t++)
            token_vec_push(&s->checkpoint, prompt->v[i + (int)t]);
        i += (int)count;
        decoder_pending = defer_decoder;
        /* Successful nondeferred calls also finish any pending decoder. */
        if (private_fresh && !decoder_pending && (!g->valid || g->pos != (uint32_t)i)) {
            DS41_SYNC_RELEASE(g, &e->model, &encoder);
            s->checkpoint_valid = false;
            if (frontier) frontier->unknown = true;
            if (rebase_capture)
                return ds41_session_rebase_failure(s, err, errlen, "private chunk frontier incomplete");
            snprintf(err, errlen, "V4.1 private chunk frontier incomplete");
            return 1;
        }
        if (frontier && !decoder_pending && g->valid && g->pos == (uint32_t)i)
            frontier->completed = (uint32_t)i;
        pending_logits = true;
        s->checkpoint_valid = false;
        /* A headless private or refresh flush keeps encoder residency across
         * arena frontiers; an append flush needs a real head, which never runs
         * resident. The refresh span keeps its pending head for the end. */
        if (capture_target && (uint32_t)i == capture_target && refresh_capture) {
            if (!ds41_session_refresh_forward_ready(s, (uint32_t)i)) {
                DS41_SYNC_RELEASE(g, &e->model, &encoder);
                s->checkpoint_valid = false;
                if (frontier) frontier->unknown = true;
                if (g->archive) g->archive->lost = true;
                snprintf(err, errlen, "SCR refresh: headless capture flush refused");
                return ds4_session_cancelled(s) ? DS4_SESSION_SYNC_INTERRUPTED : 1;
            }
        } else if (capture_target && (uint32_t)i == capture_target && rebase_capture) {
            if (!ds41_session_rebase_forward_ready(s, writer->captured_end, (uint32_t)i)) {
                DS41_SYNC_RELEASE(g, &e->model, &encoder);
                return ds41_session_rebase_failure(s, err, errlen, "private prefix capture proof refused");
            }
            pending_logits = false;
        } else if (capture_target && (uint32_t)i == capture_target) {
            DS41_SYNC_RELEASE(g, &e->model, &encoder);
            if (!DS41_SYNC_HEAD(g, &e->model, &e->weights, s->logits))
                return ds41_session_append_failure(s, err, errlen, "prefix output head failed");
            s->checkpoint_valid = true;
            if (frontier && g->valid && g->pos == (uint32_t)s->checkpoint.len &&
                !g->encoder_resident && !ds4_gpu_commands_active()) ++frontier->heads;
            int rc = ds41_session_append_head(s, err, errlen);
            if (rc) return rc;
            pending_logits = false;
        }
        if (!private_fresh && s->progress)
            s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
    }
    DS41_SYNC_RELEASE(g, &e->model, &encoder);
    if (decoder_pending) {
        s->checkpoint_valid = false;
        g->valid = false;
        if (frontier) frontier->unknown = true;
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    /* Prefill needs only its final output head. An interrupted ordinary prefix
     * must still have matching logits before it can be saved or continued. */
    if (pending_logits && !private_fresh) {
        if (!DS41_SYNC_HEAD(g, &e->model, &e->weights, s->logits)) {
            g->valid = false;
            if (ds41_session_append_writer(s))
                return ds41_session_append_failure(s, err, errlen, "prefix output head failed");
            snprintf(err, errlen, "V4.1 prefill output head failed");
            return 1;
        }
        s->checkpoint_valid = true;
        if (frontier && g->valid && g->pos == (uint32_t)s->checkpoint.len &&
            !g->encoder_resident && !ds4_gpu_commands_active()) ++frontier->heads;
        int append_rc = ds41_session_append_head(s, err, errlen);
        if (append_rc) return append_rc;
        if (s->progress)
            s->progress(s->progress_ud, "prefill_chunk", s->checkpoint.len, prompt->len);
    }
    if (interrupted) {
        if (g->archive_pending)
            return ds41_session_rebase_failure(s, err, errlen, "private prefix interrupted");
        if (ds41_session_append_writer(s))
            return ds41_session_append_failure(s, err, errlen, "prefix interrupted");
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (private_fresh && (!g->valid || g->pos != (uint32_t)prompt->len ||
        s->checkpoint.len != prompt->len || g->encoder_resident ||
        ds4_gpu_commands_active() ||
        memcmp(s->checkpoint.v, prompt->v, (size_t)prompt->len * sizeof(int)))) {
        if (frontier) frontier->unknown = true;
        if (g->archive_pending)
            return ds41_session_rebase_failure(s, err, errlen, "private frontier incomplete");
        snprintf(err, errlen, "V4.1 private frontier incomplete");
        return 1;
    }
    return 0;
}
static int ds41_session_prefix(ds4_session *s, const ds4_tokens *prompt,
                               bool private_fresh, ds41_sync_frontier *frontier,
                               char *err, size_t errlen) {
    int *copy = NULL;
    ds4_tokens owned;
    if (!private_fresh && ds41_session_append_writer(s) && !s->scr_dense_owner) {
        if (!prompt || !prompt->v || prompt->len <= 0 || prompt->len >= s->ctx_size ||
            prompt->cap < prompt->len) {
            if (frontier) memset(frontier, 0, sizeof(*frontier));
            snprintf(err, errlen, "SCR append: invalid prefix input"); return 1;
        }
        uint64_t bytes = (uint64_t)prompt->len * sizeof(int) + sizeof(owned);
        if (bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
            !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + bytes, false) ||
            !(copy = malloc((size_t)prompt->len * sizeof(int)))) {
            if (frontier) memset(frontier, 0, sizeof(*frontier));
            snprintf(err, errlen, "SCR append: prefix ownership reservation refused"); return 1;
        }
        memcpy(copy, prompt->v, (size_t)prompt->len * sizeof(int));
        owned = (ds4_tokens){.v = copy, .len = prompt->len, .cap = prompt->len};
        prompt = &owned;
    }
    int rc = ds41_session_prefix_impl(s, prompt, private_fresh, frontier, err, errlen);
    /* A deferred decoder never outlives its prefix call. */
    ds41_workspace_abandon(&s->ds41_graph);
    free(copy);
    return rc;
}
#endif

/* Known physical continuation/rebuild. This entry never discovers or inserts
 * pads: only an unchanged recorded prefix may carry existing provenance. */
int ds4_session_sync_physical(ds4_session *s, const ds4_tokens *request,
                               char *err, size_t errlen) {
    if (!s || s->scr_policy_active || s->scr_multi_active || s->scr_dense_owner) {
        snprintf(err, errlen, "physical sync: session ownership unavailable");
        return 1;
    }
    if (!s->scr_coordinates.identity) {
        if (s->scr_coordinates.pads || s->scr_coordinates.count ||
            s->scr_coordinates.capacity || s->scr_coordinates.physical_tokens) {
            snprintf(err, errlen, "physical sync: invalid empty provenance"); return 1;
        }
        return ds4_session_sync(s, request, err, errlen);
    }
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_coordinates *old = &s->scr_coordinates, pending = {0};
    size_t old_heap, heap;
    if (!ds41_coordinates_eval_supported(s) || !s->ds41_graph_ready || !s->logits ||
        !request || !request->v || request->len <= 0 || request->len >= s->ctx_size ||
        request->len >= 1048576 || (uint32_t)request->len >= g->ctx ||
        s->checkpoint.len < 0 || s->checkpoint.cap < s->checkpoint.len ||
        request->len < s->checkpoint.len || g->tp_world != 1u ||
        g->image_count || s->sync_image_count || g->imatrix ||
        g->encoder_resident || ds4_gpu_commands_active() ||
        (g->archive && (!g->archive->sealed || g->archive->retiring)) ||
        old->identity > s->scr_coordinate_serial || s->scr_coordinate_serial == UINT64_MAX ||
        !ds41_coordinates_match(old, s->checkpoint.v, (uint32_t)s->checkpoint.len, DS4_N_VOCAB) ||
        !ds41_coordinates_heap_bytes(old->capacity, &old_heap) || old_heap != s->scr_coordinate_charge ||
        old_heap > g->allocation_bytes || old_heap > s->engine->ds41_session_bytes ||
        !ds41_coordinates_heap_bytes(old->count, &heap)) {
        snprintf(err, errlen, "physical sync: unsupported runtime or transcript binding");
        return 1;
    }
    for (int i = 0; i < request->len; ++i) {
        if (!(i % 4096) && ds4_session_cancelled(s)) {
            snprintf(err, errlen, "interrupted"); return DS4_SESSION_SYNC_INTERRUPTED;
        }
        if (request->v[i] < 0 || (uint32_t)request->v[i] >= DS4_N_VOCAB ||
            (i < s->checkpoint.len && request->v[i] != s->checkpoint.v[i])) {
            snprintf(err, errlen, "physical sync: changed known prefix or invalid token");
            return 1;
        }
    }
    ds4_tokens full = {.len = request->len, .cap = request->len};
    ds41_policy_owner owner = {0};
    ds41_sync_frontier frontier = {0};
    bool grow = s->checkpoint.cap < full.len;
    uint64_t host = (uint64_t)full.len * sizeof(int) * (grow ? 2u : 1u) + heap +
        sizeof(full) + sizeof(owner) + sizeof(pending) + sizeof(frontier);
    if (host > UINT64_MAX - s->engine->ds41_session_bytes ||
        !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + host, false)) {
        snprintf(err, errlen, "physical sync: reservation refused"); return 1;
    }
    int *checkpoint = grow ? malloc((size_t)full.len * sizeof(int)) : NULL;
    full.v = malloc((size_t)full.len * sizeof(int));
    if (!full.v || (grow && !checkpoint) || !ds41_coordinates_clone(&pending, old)) {
        free(checkpoint); free(full.v); ds41_coordinates_free(&pending);
        snprintf(err, errlen, "physical sync: allocation failed"); return 1;
    }
    /* The input may borrow checkpoint storage. Own it before replacing that
     * storage or letting a rewind rebuild reset the live map. */
    memcpy(full.v, request->v, (size_t)full.len * sizeof(int));
    if (grow && s->checkpoint.len)
        memcpy(checkpoint, s->checkpoint.v, (size_t)s->checkpoint.len * sizeof(int));
    pending.physical_tokens = (uint32_t)full.len;
    pending.identity = s->scr_coordinate_serial + 1u;
    owner = (ds41_policy_owner){.session = s, .identity = &owner, .full = &full,
        .bytes = full.v, .length = full.len, .capacity = full.cap,
        .live_host_bytes = host, .active = true};
    ds4_session_progress_fn progress = s->progress, display = s->display_progress;
    s->progress = s->display_progress = NULL;
    s->scr_policy_active = true; s->scr_dense_owner = &owner;
    int rc = 1;
    bool started = false;
    if (!ds41_policy_bind_coordinates(&owner, &pending, heap)) {
        snprintf(err, errlen, "physical sync: coordinate binding refused"); goto done;
    }
    if (ds4_session_cancelled(s)) {
        rc = DS4_SESSION_SYNC_INTERRUPTED; snprintf(err, errlen, "interrupted"); goto done;
    }
    if (grow) {
        free(s->checkpoint.v); s->checkpoint.v = checkpoint; checkpoint = NULL;
        s->checkpoint.cap = full.len;
    }
    started = true;
    s->scr_multi_active = true;
    rc = ds41_session_prefix(s, &full, false, &frontier, err, errlen);
    s->scr_multi_active = false;
    if (ds4_session_cancelled(s)) {
        rc = DS4_SESSION_SYNC_INTERRUPTED; snprintf(err, errlen, "interrupted");
    }
    if (!rc && !ds41_policy_publish_coordinates(&owner, &pending)) {
        rc = 1; snprintf(err, errlen, "physical sync: headed publication failed");
    }
done:
    if (started && rc) {
        if (ds41_session_append_writer(s))
            ds41_session_append_failure(s, err, errlen, "physical destination publication failed");
        ds4_session_invalidate(s);
    }
    owner.active = false; s->scr_dense_owner = NULL; s->scr_policy_active = false;
    s->progress = progress; s->display_progress = display;
    ds41_coordinates_free(&pending); free(checkpoint); free(full.v);
    return rc;
#else
    (void)request;
    snprintf(err, errlen, "physical sync: Flash41 Metal required");
    return 1;
#endif
}

/* Bounded private archive transaction. Included after shared Flash prefix.
 * The archive owns numerical staging; this helper owns live installation and
 * the only publication. Borrowed cache rows never enter compute receipts. */
typedef enum { DS41_SUFFIX_SCALAR, DS41_SUFFIX_BATCH } ds41_suffix_policy;
typedef ds4_scr_suffix_counts ds41_suffix_counts;
#ifndef DS41_SUFFIX_COMMANDS_ACTIVE
#define DS41_SUFFIX_COMMANDS_ACTIVE ds4_gpu_commands_active
#endif
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
static bool ds41_multi_pin(ds4_session *s, ds41_scr_archive *a,
                           uint64_t generation, uint32_t start, uint32_t end) {
    return s->ds41_graph.archive == a && ds41_archive_owned_sealed(a, s) &&
        a->source_generation == generation && a->start == start && a->end == end &&
        ds41_archive_shape(a, s->ds41_graph.ctx) && start <= end &&
        end - start == ds41_archive_count(a) && a->tokens && a->source && a->stage &&
        generation && generation <= a->generation &&
        !(a->lost && a->generation == UINT64_MAX);
}
/* Unheaded private frontier; capture chunks may hold encoder residency. */
static bool ds41_multi_frontier(ds4_session *s, const int *tokens, uint32_t pos) {
    ds41_gpu_graph *g = &s->ds41_graph;
    return !s->checkpoint_valid && g->valid && g->pos == pos &&
        s->checkpoint.len == (int)pos &&
        !memcmp(s->checkpoint.v, tokens, (size_t)pos * sizeof(int)) &&
        !DS41_SUFFIX_COMMANDS_ACTIVE();
}
static bool ds41_multi_cursor(ds4_session *s, const int *tokens, uint32_t pos) {
    return ds41_multi_frontier(s, tokens, pos) && !s->ds41_graph.encoder_resident;
}
static bool ds41_multi_coordinate_shape(const ds41_scr_coordinates *m,
                                         const int *tokens, uint32_t n) {
    if (!m || !tokens || n >= 1048576u) return false;
    if (!m->identity) return !m->pads && !m->count && !m->capacity && !m->physical_tokens;
    return m->physical_tokens == n && ds41_coordinates_shape(m);
}
/* Source pads in the range reappear at the same destination offsets, never
 * on the range's first or last token. A mapped side must contain the range;
 * an unmapped side (identity 0) is all content. */
static bool ds41_multi_pad_correspondence(const ds41_scr_coordinates *source,
                                          const ds41_scr_coordinates *target,
                                          uint32_t source_start, uint32_t destination,
                                          uint32_t count) {
    if (count > UINT32_MAX - source_start || count > UINT32_MAX - destination) return false;
    const cache_reuse_range range = {source_start, destination, count, destination};
    bool corresponds = false;
    return cache_reuse_verify_span_correspondence(source->pads, source->count,
            source->identity ? source->physical_tokens : source_start + count,
            target->pads, target->count,
            target->identity ? target->physical_tokens : destination + count,
            &range, &corresponds, NULL, 0) == CACHE_REUSE_COMPLETE && corresponds;
}
static bool ds41_multi_install_aligned(ds4_session *s, const int *tokens,
                               ds41_scr_archive *a, const ds4_scr_range *r,
                               uint64_t generation, uint32_t start, uint32_t end,
                               uint32_t alignment) {
    ds41_gpu_graph *g = &s->ds41_graph;
    const uint32_t d = r->destination_start, n = r->count;
    if ((alignment != 2u && alignment != 8u) ||
        !ds41_multi_pin(s, a, generation, start, end) || !g->token_map ||
        !ds41_multi_cursor(s, tokens, d) || d % alignment || n % alignment ||
        r->source_start % alignment || n < 128u ||
        d >= g->ctx || n >= g->ctx - d || n > ds41_archive_count(a) ||
        r->source_start < start || r->source_start > end || n > end - r->source_start ||
        !a->range.valid || a->stage_alignment != alignment || a->range.start != d || a->range.end != d + n ||
        a->range.source_start != r->source_start ||
        a->range.source_end != r->source_start + n ||
        a->range.source_generation != generation ||
        a->range.destination_generation != a->generation ||
        memcmp(tokens + d, a->tokens + r->source_start - start, (size_t)n * sizeof(int)) ||
        ds4_gpu_tensor_bytes(a->stage) < ds41_archive_data_bytes(a->cap)) return false;
    for (uint32_t o = 0; o < 4; ++o) {
        const uint32_t ratio = o < 3 ? 2u : 1u;
        if (ds4_gpu_tensor_bytes(g->compressed[o]) < (uint64_t)(d + n) / ratio * 512u * 4u ||
            ds4_gpu_tensor_bytes(g->index_cache[o]) < (uint64_t)(d + n) / ratio * 128u * 4u)
            return false;
    }
    for (uint32_t l = 0; l < 40; ++l)
        if (ds4_gpu_tensor_bytes(g->window[l]) < 128u * 512u * 4u) return false;
    g->valid = false;
    a->range.valid = false; /* single-use even after partial copy failure */
    bool owned = false;
    bool ok = ds4_gpu_synchronize() && (owned = ds4_gpu_begin_commands());
    for (uint32_t o = 0; ok && o < 4; ++o) {
        const uint32_t ratio = o < 3 ? 2u : 1u;
        const uint64_t base = ds41_archive_owner(a->cap, o);
        ok = ds4_gpu_tensor_copy(g->compressed[o], (uint64_t)d / ratio * 512u * 4u,
                 a->stage, base, (uint64_t)n / ratio * 512u * 4u) &&
             ds4_gpu_tensor_copy(g->index_cache[o], (uint64_t)d / ratio * 128u * 4u,
                 a->stage, base + (uint64_t)(a->cap / ratio) * 512u * 4u,
                 (uint64_t)n / ratio * 128u * 4u);
    }
    for (uint32_t l = 0; ok && l < 40; ++l)
        ok = ds4_gpu_tensor_copy(g->window[l], 0, a->stage,
                                 ds41_archive_ring(a->cap, l), 128u * 512u * 4u);
    if (owned) {
        if (!ds4_gpu_end_commands()) ok = false;
        if (!ds4_gpu_synchronize()) ok = false;
    }
    if (!ok || ds4_gpu_commands_active() || g->encoder_resident) return false;
    g->pos = d + n;
    ds4_engram_history_reset(&g->history);
    for (uint32_t i = 0; i < 3; ++i)
        g->history.tail[i] = (int32_t)g->token_map[tokens[g->pos - 1u - i]];
    /* Even graft end leaves no unfinished pair. The next real even token
     * overwrites pair inputs; zero gaps require no dummy decoder token. */
    for (uint32_t i = d; i < d + n; ++i) token_vec_push(&s->checkpoint, tokens[i]);
    g->valid = true;
    return true;
}
static bool ds41_multi_install(ds4_session *s, const int *tokens,
                               ds41_scr_archive *a, const ds4_scr_range *r,
                               uint64_t generation, uint32_t start, uint32_t end) {
    return ds41_multi_install_aligned(s, tokens, a, r, generation, start, end, 8u);
}
static int ds41_multi_steps(ds4_session *s, const int *tokens, uint32_t end,
                            uint32_t *completed, bool *unknown) {
    ds41_gpu_graph *g = &s->ds41_graph;
    while (g->pos < end) {
        if (ds4_session_cancelled(s)) return DS4_SESSION_SYNC_INTERRUPTED;
        const uint32_t pos = g->pos;
        ds41_scr_archive *pending = g->archive_capture_pending ? g->archive_pending : NULL;
        if (pending && (pending->lost || pending->retiring || pos < pending->raw_origin ||
            pos >= pending->raw_origin + pending->raw_arena_tokens ||
            pos >= pending->writer_known_end)) return 1;
        if (!DS41_SYNC_STEP(g, &s->engine->model, &s->engine->weights, tokens[pos], NULL)) {
            *unknown = true;
            return 1;
        }
        if (!g->valid || g->pos != pos + 1u || g->encoder_resident ||
            DS41_SUFFIX_COMMANDS_ACTIVE()) { *unknown = true; return 1; }
        ++*completed; /* known before any subsequent fallible drain/head */
        token_vec_push(&s->checkpoint, tokens[pos]);
        if (pending && !pending->sealed && (g->pos == end || g->pos == pending->raw_origin + pending->raw_arena_tokens) &&
            !ds41_session_rebase_forward_ready(s, pending->captured_end, g->pos)) return 1;
    }
    return 0;
}
/* Only the final unpublished suffix can select batching. No partial chunk is
 * checkpointed: uncertainty belongs to the transaction invalidation path. */
static int ds41_multi_suffix(ds4_session *s, const int *tokens, uint32_t end,
                            uint32_t *completed, bool *unknown,
                            ds41_scr_archive *a, uint64_t generation,
                            uint32_t start, uint32_t archive_end,
                            ds41_suffix_policy policy, ds41_suffix_counts *counts) {
    if (policy == DS41_SUFFIX_SCALAR)
        return ds41_multi_steps(s, tokens, end, completed, unknown);
    ds41_gpu_graph *g = &s->ds41_graph;
    if (policy != DS41_SUFFIX_BATCH || !counts || !tokens || end > g->ctx ||
        g->pos > end || s->checkpoint.cap < (int)end || !g->prefill_cap ||
        !ds41_multi_pin(s, a, generation, start, archive_end)) return 1;
    const uint64_t pin_generation = a->generation;
    const int *pin_tokens = a->tokens;
    const ds4_gpu_tensor *pin_source = a->source, *pin_stage = a->stage;
    while (g->pos < end) {
        if (ds4_session_cancelled(s)) return DS4_SESSION_SYNC_INTERRUPTED;
        const uint32_t pos = g->pos;
        uint32_t n = end - pos;
        if (n > g->prefill_cap) n = g->prefill_cap;
        if (n > 2048u) n = 2048u;
        if (!n || n > ds41_encoder_chunk_cap(g, n) ||
            !ds41_multi_cursor(s, tokens, pos) ||
            !ds41_multi_pin(s, a, generation, start, archive_end) ||
            a->generation != pin_generation || a->tokens != pin_tokens ||
            a->source != pin_source || a->stage != pin_stage) return 1;
        ++counts->attempted;
        if (!DS41_SYNC_PREFILL(g, &s->engine->model, &s->engine->weights,
                tokens + pos, n, NULL, NULL, (int)end, s->cancel, s->cancel_ud)) {
            *unknown = true; return 1;
        }
        if (s->checkpoint_valid || !g->valid || g->pos != pos + n ||
            g->encoder_resident || DS41_SUFFIX_COMMANDS_ACTIVE() ||
            s->checkpoint.len != (int)pos ||
            memcmp(s->checkpoint.v, tokens, (size_t)pos * sizeof(int)) ||
            !ds41_multi_pin(s, a, generation, start, archive_end) ||
            a->generation != pin_generation || a->tokens != pin_tokens ||
            a->source != pin_source || a->stage != pin_stage) {
            *unknown = true; return 1;
        }
        for (uint32_t i = pos; i < pos + n; ++i) token_vec_push(&s->checkpoint, tokens[i]);
        *completed += n; ++counts->completed; counts->certified += n;
        if (!counts->min_completed || n < counts->min_completed) counts->min_completed = n;
        if (n > counts->max_completed) counts->max_completed = n;
    }
    return ds4_session_cancelled(s) ? DS4_SESSION_SYNC_INTERRUPTED : 0;
}
/* Owning splice certificates and checked source transfer. Included after the
 * multi cursor/install helpers; no imported row is a model-compute receipt. */
static bool ds41_session_rebase_pin(ds4_session *s) {
    ds41_scr_archive *next = s->ds41_graph.archive_pending, *old = s->ds41_graph.archive;
    return next && old && next != old && next->rebase_predecessor == old &&
        ds41_archive_owned_sealed(old, s) && old->raw && old->raw->readable && old->source_generation == next->predecessor_generation &&
        old->start == next->predecessor_start && old->end == next->predecessor_end &&
        old->cap == next->predecessor_capacity &&
        old->coordinates.identity == next->predecessor_coordinate_identity;
}
/* Clip only the selected suffix obstructing a genuine covered-frontier head.
 * The caller rechecks exact tokens and pad endpoints; unsafe clipping refuses. */
static bool ds41_rebase_clip(ds4_scr_range *plan, uint32_t *count, uint32_t n,
                             uint32_t alignment, uint32_t archive_start, uint32_t archive_end) {
    if (!plan || !count || *count > DS4_SCR_MULTI_MAX_RANGES || n < 128u ||
        (alignment != 2u && alignment != 8u)) return false;
    cache_reuse_range ranges[DS4_SCR_MULTI_MAX_RANGES], clipped[DS4_SCR_MULTI_MAX_RANGES];
    for (uint32_t i = 0; i < *count; ++i)
        ranges[i] = (cache_reuse_range){plan[i].source_start, plan[i].destination_start,
                                        plan[i].count, plan[i].destination_start};
    uint32_t written = 0, kept = 0;
    if (*count && cache_reuse_clip_ranges_to_archive(ranges, *count, archive_start, archive_end,
            (n & ~7u) - 2u, alignment, clipped, DS4_SCR_MULTI_MAX_RANGES, &written,
            NULL, 0) != CACHE_REUSE_COMPLETE) return false;
    for (uint32_t i = 0; i < written; ++i) {
        if (clipped[i].count < 128u) continue;
        plan[kept++] = (ds4_scr_range){clipped[i].source_start, clipped[i].prepared_start,
                                       clipped[i].count};
    }
    for (uint32_t i = kept; i < DS4_SCR_MULTI_MAX_RANGES; ++i)
        memset(&plan[i], 0, sizeof(plan[i]));
    *count = kept;
    return true;
}
static bool ds41_session_rebase_import(ds4_session *s, const ds4_scr_range *range) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *next = g->archive_pending, *old = g->archive;
    if (!range || !g->archive_capture_pending || !ds41_session_rebase_pin(s) ||
        !ds41_multi_cursor(s, next->tokens, range->destination_start) ||
        next->captured_end != range->destination_start ||
        !ds41_rebase_import_range_owners(next, old, range->source_start,
                                         range->destination_start, range->count)) return false;
    for (uint32_t copied = 0; copied < range->count;) {
        /* One window read per chunk; flush only a full arena or a finished range. */
        uint32_t occupied = next->captured_end - next->raw_origin;
        if (occupied > next->raw_arena_tokens) return false;
        uint32_t chunk = range->count - copied, room = next->raw_arena_tokens - occupied;
        if (chunk > room) chunk = room;
        if (chunk > 128u) chunk = 128u;
        chunk &= ~1u;
        if (!chunk || !ds41_session_rebase_pin(s) ||
            memcmp(next->tokens + range->destination_start + copied,
                   old->tokens + range->source_start - old->start + copied,
                   (size_t)chunk * sizeof(int)) ||
            !ds41_rebase_import_chunk(next, old, range->source_start + copied,
                                      range->destination_start + copied, chunk))
            return false;
        const bool arena_full = next->captured_end - next->raw_origin == next->raw_arena_tokens;
        if ((arena_full || copied + chunk == range->count) &&
            !ds41_archive_flush_ready(next, next->captured_end, ds41_archive_raw_cancel, s))
            return false;
        next->imported_tokens += chunk;
        copied += chunk;
    }
    return true;
}
static int ds41_session_rebase_cover(ds4_session *s, const int *tokens, float *logits,
                                     ds4_scr_multi_receipt *receipt) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *next = g->archive_pending;
    const uint32_t covered = next ? next->writer_known_end & ~7u : 0u;
    if (!covered || !logits || !ds41_session_rebase_pin(s) || !g->archive_capture_pending ||
        !ds41_multi_cursor(s, tokens, covered) || next->captured_end != covered ||
        next->raw_origin != covered || next->raw->committed_tokens != covered ||
        next->imported_tokens > covered) return 1;
    double began = now_sec();
    bool headed = DS41_SYNC_HEAD(g, &s->engine->model, &s->engine->weights, logits);
    receipt->covered_head_seconds += now_sec() - began;
    if (!headed || !ds4_gpu_synchronize() || !ds41_multi_cursor(s, tokens, covered) ||
        !ds41_session_rebase_pin(s)) return 1;
    receipt->covered_head_completed = true;
    receipt->covered_head_end = covered;
    if (ds4_session_cancelled(s)) return DS4_SESSION_SYNC_INTERRUPTED;
    if (!ds41_archive_seal(next, s, &s->engine->model, tokens, covered) ||
        !ds41_archive_append_arm(next, s, &s->engine->model, tokens,
                                 next->writer_known_end, next->writer_future_pads)) return 1;
    next->rebase_head_end = covered;
    return 0;
}
static int ds41_session_rebase_abort(ds4_session *s, ds4_scr_multi_receipt *receipt) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *next = g->archive_pending;
    g->archive_capture_pending = false;
    if (receipt->target_mutated) {
        s->checkpoint_valid = g->valid = false;
        s->mtp_draft_valid = false;
    }
    if (!next) return 0;
    ds41_archive_block_writer(next, false);
    /* A prior cleanup/old-retirement refusal is not an automatic retry. */
    if (next->retiring || next->retirement_refused) {
        receipt->pending_owner_retained = true; return 1;
    }
    if (ds41_session_archive_release_slot(s, true)) {
        receipt->pending_owner_retained = true; return 1;
    }
    ds41_scr_archive *old = g->archive;
    if (old && !ds41_archive_published_valid(old)) old->retiring = true;
    return 0;
}
/* Mapped callers publish their independently owned live map first. Every
 * fallible proof precedes old retirement; successful retirement has no rollback. */
static int ds41_session_rebase_commit(ds4_session *s, const ds41_policy_owner *owner,
                                      ds4_scr_multi_receipt *receipt) {
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *next = g->archive_pending;
    if (!next) return receipt && receipt->rebase_prepared && !receipt->rebase_committed ? 1 : 0;
    const ds4_tokens *full = owner ? owner->full : NULL;
    if (!receipt || !receipt->rebase_prepared || receipt->rebase_committed ||
        !owner || s->scr_dense_owner != owner || !s->scr_policy_active ||
        !owner->active || owner->identity != owner || owner->session != s ||
        s->scr_multi_active || !full || owner->bytes != full->v ||
        owner->length != full->len || owner->capacity != full->cap ||
        !full->v || !next->tokens || full->len <= 0 || (uint32_t)full->len > next->cap ||
        memcmp(next->tokens, full->v, (size_t)full->len * sizeof(int)) ||
        receipt->covered_head_end != next->rebase_head_end ||
        receipt->imported_tokens != next->imported_tokens ||
        !receipt->published || !receipt->head_completed || !receipt->covered_head_completed ||
        !ds41_session_rebase_pin(s) || next->lost || next->retiring ||
        !ds41_session_append_cursor(s, (uint32_t)full->len) ||
        memcmp(s->checkpoint.v, full->v, (size_t)full->len * sizeof(int)) ||
        next->captured_end != (uint32_t)full->len ||
        next->rebase_head_end != ((uint32_t)full->len & ~7u) ||
        next->end != next->rebase_head_end ||
        !ds41_archive_published_valid(next) || !ds41_archive_published_valid(g->archive)) return 1;
    if (owner->coordinates) {
        ds41_scr_coordinates future = next->coordinates;
        future.count = next->writer_future_pads;
        future.physical_tokens = next->writer_known_end;
        if (!owner->coordinates_published ||
            s->scr_coordinates.identity != owner->coordinate_identity ||
            !ds41_coordinates_match(&s->scr_coordinates, full->v, (uint32_t)full->len, DS4_N_VOCAB) ||
            !ds41_coordinates_match(&future, full->v, (uint32_t)full->len, DS4_N_VOCAB) ||
            future.identity != s->scr_coordinates.identity || future.count != s->scr_coordinates.count)
            return 1;
        if (!ds41_coordinates_same_pads(&future, &s->scr_coordinates, future.count)) return 1;
    } else if (s->scr_coordinates.identity || next->coordinates.identity ||
               !ds41_policy_authorized(s, full, owner)) return 1;
    if (ds4_session_cancelled(s)) return DS4_SESSION_SYNC_INTERRUPTED;
    g->archive_capture_pending = false;
    s->checkpoint_valid = false;
    if (ds41_session_archive_release_slot(s, false)) {
        next->retirement_refused = next->retiring = true;
        ds41_archive_block_writer(next, false);
        g->valid = false;
        receipt->pending_owner_retained = true;
        return 1;
    }
    /* Prepared assignments only. Cancellation now reports this actual commit. */
    next->rebase_predecessor = NULL;
    next->rebased = true;
    g->archive = next; g->archive_pending = NULL;
    receipt->rebase_committed = true;
    s->checkpoint_valid = true;
    return 0;
}
/* Capture-preserving private K6 fresh work, not legacy non-rebasing suffix batch.
 * Included after multi cursor/scalar and rebase-pin helpers. No head/publication. */
static bool ds41_capture_chunk_tags(const ds41_scr_archive *pending,
                                     uint32_t from, uint32_t end) {
    if (!pending || !pending->tags || end < from || end < pending->raw_origin ||
        end - pending->raw_origin > pending->raw_arena_tokens ||
        !ds41_raw_capture_complete(pending->raw_tags, pending->raw_origin,
                                  end - pending->raw_origin)) return false;
    for (uint32_t owner = 0; owner < 4u; ++owner) {
        const uint32_t ratio = owner < 3u ? 2u : 1u;
        for (uint32_t pos = from / ratio * ratio; pos + ratio <= end; pos += ratio)
            if (pending->tags[ds41_archive_tag(pending->cap, owner) + pos / ratio] != pos)
                return false;
    }
    return true;
}
static bool ds41_capture_chunk_history(const ds41_gpu_graph *g,
                                        const int *tokens, uint32_t pos) {
    if (!g->token_map || pos < 3u) return false;
    for (uint32_t i = 0; i < 3u; ++i)
        if (g->history.tail[i] != (int32_t)g->token_map[tokens[pos - 1u - i]]) return false;
    return true;
}
/* Arena-bounded capture chunks. Encoder residency, when held, spans arena
 * flushes and is dropped before the release tail or any scalar step. */
static int ds41_multi_capture_chunks(ds4_session *s, const int *tokens, uint32_t end,
                            uint32_t *completed, bool *unknown,
                            ds41_scr_archive *source, uint64_t generation,
                            uint32_t start, uint32_t source_end,
                            ds41_encoder_residency *encoder) {
    ds41_gpu_graph *g = &s->ds41_graph;
    const uint64_t pin_generation = source->generation;
    const int *pin_tokens = source->tokens;
    const ds4_gpu_tensor *pin_source = source->source, *pin_stage = source->stage;
    while (g->pos < end) {
        if (ds4_session_cancelled(s)) return DS4_SESSION_SYNC_INTERRUPTED;
        const uint32_t pos = g->pos;
        ds41_encoder_release_tail(g, &s->engine->model, encoder, end - pos);
        ds41_scr_archive *pending = g->archive_capture_pending ? g->archive_pending : NULL;
        uint32_t bound = end;
        if (!tokens || !pending || pending == source || pending->sealed || pending->lost ||
            pending->retiring || pending->session != s || pending->model != &s->engine->model ||
            !pending->raw || !pending->raw->readable || !pending->raw->writable ||
            !pending->tokens || s->checkpoint.cap < (int)end ||
            end > g->ctx || end > pending->writer_known_end || end > pending->limit ||
            !ds41_capture_arena_bound(pending, pos, end, &bound) ||
            !ds41_multi_frontier(s, tokens, pos) || !ds41_session_rebase_pin(s) ||
            !ds41_multi_pin(s, source, generation, start, source_end) ||
            source->generation != pin_generation || source->tokens != pin_tokens ||
            source->source != pin_source || source->stage != pin_stage) return 1;
        uint32_t count = ds41_prefill_count(g, bound - pos);
        /* Use the existing chooser/primitive contract, never a forced minimum.
         * Scalar work stops at the same actual flush boundary, not each token. */
        if (count <= 1u || count > bound - pos || count > g->prefill_cap ||
            count > ds41_encoder_chunk_cap(g, count)) {
            DS41_SYNC_RELEASE(g, &s->engine->model, encoder);
            int rc = ds41_multi_steps(s, tokens, bound, completed, unknown);
            if (rc) return rc;
            continue;
        }
        const uint32_t origin = pending->raw_origin, captured = pending->captured_end;
        const uint32_t known_end = pending->writer_known_end, capacity = pending->cap;
        const uint32_t limit = pending->limit;
        const uint64_t map_identity = pending->coordinates.identity;
        const uint32_t *tags = pending->tags;
        const int *pending_tokens = pending->tokens;
        const ds41_raw_store *raw = pending->raw;
        const ds4_gpu_tensor *arena = pending->raw_arena;
        const ds4_gpu_tensor *destination = pending->source, *stage = pending->stage;
        if (!ds41_capture_chunk_history(g, tokens, pos) ||
            memcmp(pending_tokens + pos, tokens + pos, (size_t)count * sizeof(int))) return 1;
        bool ok = DS41_SYNC_CAPTURE_BATCH_PREFILL(g, &s->engine->model, &s->engine->weights,
            tokens + pos, count, NULL, NULL, (int)end, s->cancel, s->cancel_ud);
        /* Once dispatched, every uncertain condition goes to owning failure.
         * No checkpoint/row credit or scalar retry precedes the whole proof. */
        if (!ok || s->checkpoint_valid || !g->valid || g->pos != pos + count ||
            DS41_SUFFIX_COMMANDS_ACTIVE() || s->checkpoint.len != (int)pos ||
            memcmp(s->checkpoint.v, tokens, (size_t)pos * sizeof(int)) ||
            !ds41_capture_chunk_history(g, tokens, pos + count) ||
            !ds41_session_rebase_pin(s) ||
            !ds41_multi_pin(s, source, generation, start, source_end) ||
            source->generation != pin_generation || source->tokens != pin_tokens ||
            source->source != pin_source || source->stage != pin_stage ||
            !g->archive_capture_pending || g->archive_pending != pending ||
            pending->sealed || pending->lost || pending->retiring ||
            pending->session != s || pending->model != &s->engine->model ||
            pending->limit != limit || pending->coordinates.identity != map_identity ||
            pending->tags != tags || !raw->readable || !raw->writable ||
            pending->raw != raw || pending->raw_arena != arena ||
            pending->source != destination || pending->stage != stage ||
            pending->tokens != pending_tokens || pending->raw_origin != origin ||
            pending->captured_end != captured || pending->writer_known_end != known_end ||
            pending->cap != capacity ||
            memcmp(pending_tokens + pos, tokens + pos, (size_t)count * sizeof(int)) ||
            !ds41_capture_chunk_tags(pending, pos, pos + count)) {
            *unknown = true; return 1;
        }
        for (uint32_t i = pos; i < pos + count; ++i) token_vec_push(&s->checkpoint, tokens[i]);
        *completed += count;
        if (g->pos == bound &&
            !ds41_session_rebase_forward_ready(s, pending->captured_end, g->pos)) {
            *unknown = true; return 1;
        }
    }
    return ds4_session_cancelled(s) ? DS4_SESSION_SYNC_INTERRUPTED : 0;
}
/* A rebase captures every gap into its pending archive in arena-bounded
 * batches; a non-rebasing graft has no capture owner and steps its gaps. */
static int ds41_multi_capture_steps(ds4_session *s, const int *tokens, uint32_t end,
                            uint32_t *completed, bool *unknown,
                            ds41_scr_archive *source, uint64_t generation,
                            uint32_t start, uint32_t source_end, bool rebasing) {
    if (!rebasing) return ds41_multi_steps(s, tokens, end, completed, unknown);
    ds41_gpu_graph *g = &s->ds41_graph;
    if (!source || !tokens) return 1;
    ds41_encoder_residency encoder = {0};
    if (g->pos < end)
        DS41_SYNC_ACQUIRE(g, &s->engine->model, &s->engine->weights, end - g->pos,
            &encoder, s->cancel, s->cancel_ud);
    int rc = ds41_multi_capture_chunks(s, tokens, end, completed, unknown,
                                       source, generation, start, source_end, &encoder);
    /* Every exit, including failure, leaves the graph ready for a head. */
    DS41_SYNC_RELEASE(g, &s->engine->model, &encoder);
    return rc;
}
#endif
static int ds41_scr_multi_selected(ds4_session *s, const ds4_tokens *request,
                              const ds4_scr_range *ranges, uint32_t range_count,
                              ds4_scr_multi_receipt *out, char *err, size_t errlen,
                              const ds41_policy_owner *owner,
                              ds41_suffix_policy suffix, ds41_suffix_counts *counts) {
    ds4_scr_multi_receipt local = {0}, *r = out ? out : &local;
    memset(r, 0, sizeof(*r));
    r->outcome = DS4_SCR_ERROR; r->reason = DS4_SCR_REASON_INVALID;
    r->input_spans = range_count;
    const double begin = now_sec();
    int rc = 1, *tokens = NULL;
    float *logits = NULL;
    bool active = false, mutated = false;
    ds4_session_progress_fn progress = NULL, display = NULL;
    ds4_scr_range plan[DS4_SCR_MULTI_MAX_RANGES] = {{0}};
    ds41_raw_prefetch prefetch = {0};
    unsigned char *prefetch_data = NULL;
    /* Count precedes all array access, including absent-archive fallback. */
    if (range_count > DS4_SCR_MULTI_MAX_RANGES || (range_count && !ranges) || !s || !s->engine ||
        !request || !request->v || request->len <= 0 || request->len >= s->ctx_size ||
        !s->logits || s->scr_multi_active ||
        (s->scr_coordinates.identity && (!owner || !owner->coordinates)) ||
        (owner ? !ds41_policy_authorized(s, request, owner) : s->scr_policy_active)) {
        snprintf(err, errlen, "SCR multi: invalid request/count/ownership"); goto done;
    }
    r->requested_tokens = (uint32_t)request->len;
    for (int i = 0; i < request->len; ++i)
        if (request->v[i] < 0 || (uint32_t)request->v[i] >= DS4_N_VOCAB) {
            snprintf(err, errlen, "SCR multi: invalid token at %d", i); goto done;
        }
    for (uint32_t i = 0; i < range_count; ++i) {
        plan[i] = ranges[i];
        const ds4_scr_range *p = &plan[i];
        if (!p->count || p->source_start > (uint32_t)s->ctx_size ||
            p->count > (uint32_t)s->ctx_size - p->source_start ||
            p->destination_start > (uint32_t)request->len ||
            p->count > (uint32_t)request->len - p->destination_start ||
            (i && (p->source_start < plan[i-1].source_start + plan[i-1].count ||
                   p->destination_start < plan[i-1].destination_start + plan[i-1].count))) {
            snprintf(err, errlen, "SCR multi: malformed range %u", i); goto done;
        }
    }
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *a = s->ds41_graph_ready ? g->archive : NULL;
    bool rebasing = false, rebase_cleanup_done = false;
    if (g->archive_pending) {
        snprintf(err, errlen, "SCR multi: unresolved private archive ownership"); goto done;
    }
    if (a && (a->session != s || a->model != &s->engine->model || a->retiring)) {
        snprintf(err, errlen, "SCR multi: foreign or retiring archive"); goto done;
    }
    r->reason = DS4_SCR_REASON_UNSUPPORTED;
    if (!s->ds41_graph_ready || !ds4_session_is_ds41(s) || s->distributed ||
        s->engine->backend != DS4_BACKEND_METAL || s->engine->tp.active ||
        g->tp_world != 1u || g->image_count || s->sync_image_count || g->imatrix ||
        (uint32_t)request->len >= g->ctx) {
        snprintf(err, errlen, "SCR multi: unsupported runtime"); goto done;
    }
    if (g->encoder_resident || ds4_gpu_commands_active()) {
        r->reason = DS4_SCR_REASON_INVALID;
        snprintf(err, errlen, "SCR multi: borrowed command/encoder ownership"); goto done;
    }
    for (uint32_t i = 0; i < range_count; ++i)
        if (plan[i].source_start > g->ctx || plan[i].count > g->ctx - plan[i].source_start) {
            r->reason = DS4_SCR_REASON_INVALID;
            snprintf(err, errlen, "SCR multi: source outside graph context"); goto done;
        }
    const bool mapped = owner && owner->coordinates;
    const uint32_t alignment = mapped ? 2u : 8u;
    if ((mapped && suffix != DS41_SUFFIX_SCALAR) || (!mapped && a && a->coordinates.count)) {
        snprintf(err, errlen, "SCR multi: unsupported coordinate/suffix combination"); goto done;
    }
    bool candidate = range_count && a && a->sealed;
    if (mapped && (!ds41_multi_coordinate_shape(owner->coordinates, request->v, (uint32_t)request->len) ||
        (candidate && !ds41_multi_coordinate_shape(&a->coordinates, a->tokens, ds41_archive_count(a))))) {
        snprintf(err, errlen, "SCR multi: invalid coordinate provenance"); goto done;
    }
    r->reason = range_count ? DS4_SCR_REASON_UNSUPPORTED : DS4_SCR_REASON_NO_CANDIDATE;
    const uint64_t generation = a ? a->source_generation : 0;
    const uint32_t start = a ? a->start : 0, end = a ? a->end : 0;
    for (uint32_t i = 0; i < range_count; ++i) {
        ds4_scr_range *p = &plan[i];
        if (p->source_start % alignment || p->destination_start % alignment || p->count % alignment ||
            p->count < 128u || plan[0].destination_start < 8u ||
            p->destination_start + p->count >= (uint32_t)request->len) candidate = false;
        if (candidate && (!ds41_multi_pin(s, a, generation, start, end) ||
            p->count > ds41_archive_count(a) || p->source_start < start || p->source_start > end ||
            p->count > end - p->source_start ||
            memcmp(request->v + p->destination_start, a->tokens + p->source_start - start,
                   (size_t)p->count * sizeof(int)))) {
            candidate = false; r->reason = DS4_SCR_REASON_MISMATCH;
        }
        if (candidate && mapped && !(owner && owner->source_pads_are_content) &&
            !ds41_multi_pad_correspondence(&a->coordinates, owner->coordinates,
                p->source_start - start, p->destination_start, p->count)) {
            candidate = false; r->reason = DS4_SCR_REASON_MISMATCH;
        }
    }
    if (mapped && range_count && !candidate) {
        snprintf(err, errlen, "SCR multi: mapped graft refused before mutation"); goto done;
    }
    /* Stable transitive correspondence compaction. Eligibility above belongs
     * to each input, so joining fragments cannot rescue a refused candidate. */
    uint32_t selected = 0;
    for (uint32_t i = 0; i < range_count; ++i) {
        const ds4_scr_range next = plan[i];
        ds4_scr_range *last = selected ? &plan[selected - 1u] : NULL;
        if (last && last->source_start + last->count == next.source_start &&
            last->destination_start + last->count == next.destination_start)
            last->count += next.count;
        else plan[selected++] = next;
    }
    for (uint32_t i = selected; i < DS4_SCR_MULTI_MAX_RANGES; ++i)
        memset(&plan[i], 0, sizeof(plan[i]));
    r->planned_spans = selected;
    for (uint32_t i = 0; i < selected; ++i) r->planned_tokens += plan[i].count;
    rebasing = candidate && owner && suffix == DS41_SUFFIX_SCALAR &&
        owner->archive_capacity >= (uint32_t)request->len;
    if (rebasing) {
        if (!ds41_rebase_clip(plan, &selected, (uint32_t)request->len, alignment,
                              start, end)) goto done;
        for (uint32_t i = 0; i < selected; ++i) {
            const ds4_scr_range *p = &plan[i];
            if (p->count < 128u || p->count % alignment ||
                memcmp(request->v + p->destination_start, a->tokens + p->source_start - start,
                       (size_t)p->count * sizeof(int)) ||
                (mapped && !(owner && owner->source_pads_are_content) &&
                 !ds41_multi_pad_correspondence(&a->coordinates, owner->coordinates,
                    p->source_start - start, p->destination_start, p->count))) {
                r->reason = DS4_SCR_REASON_MISMATCH;
                snprintf(err, errlen, "SCR rebase: clipped range authority refused"); goto done;
            }
        }
        candidate = selected != 0u;
    }
    r->normalized_spans = selected;
    memcpy(r->ranges, plan, sizeof(plan));
    if (suffix != DS41_SUFFIX_SCALAR &&
        (suffix != DS41_SUFFIX_BATCH || !counts || !candidate || !g->prefill_cap)) {
        r->reason = DS4_SCR_REASON_UNSUPPORTED;
        snprintf(err, errlen, "SCR multi: private suffix has no supported graft/capacity"); goto done;
    }
    r->archive_bytes = ds4_session_scr_archive_bytes(s);
    r->temporary_host_bytes = (uint64_t)request->len * sizeof(int) +
                              (uint64_t)DS4_N_VOCAB * sizeof(float) + sizeof(plan) +
                              sizeof(local) + 2u * sizeof(ds4_tokens) + sizeof(ds41_sync_frontier);
    if (s->checkpoint.cap < request->len)
        r->publication_growth_bytes = (uint64_t)(request->len - s->checkpoint.cap) * sizeof(int);
    uint64_t extra = 0;
    if (!ds41_policy_reservation(s, request, owner,
        r->temporary_host_bytes + r->publication_growth_bytes, &extra)) {
        r->reason = DS4_SCR_REASON_RESOURCE;
        snprintf(err, errlen, "SCR multi: invalid outer reservation"); goto done;
    }
    /* The existing total includes the still-live archive and graph. Host
     * transaction storage is additional, never a replacement device charge. */
    if (extra > UINT64_MAX - s->engine->ds41_session_bytes ||
        !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + extra, false)) {
        r->reason = DS4_SCR_REASON_RESOURCE;
        snprintf(err, errlen, "SCR multi: simultaneous reservation refused"); goto done;
    }
    if (rebasing) {
        double prepare_begin = now_sec();
        int prepare = ds41_session_rebase_prepare(s, request, owner,
            r->temporary_host_bytes + r->publication_growth_bytes, err, errlen);
        r->rebase_prepare_seconds = now_sec() - prepare_begin;
        if (prepare) {
            r->reason = DS4_SCR_REASON_RESOURCE;
            r->pending_owner_retained = g->archive_pending != NULL;
            goto done;
        }
        r->rebase_prepared = true;
        for (uint32_t i = 0; a && i < selected; ++i) {
            const uint32_t offset = plan[i].source_start - a->start;
            ds41_raw_advise(a->raw, offset / DS41_RAW_RECORD_TOKENS,
                            (offset + plan[i].count + 7u) / DS41_RAW_RECORD_TOKENS -
                                offset / DS41_RAW_RECORD_TOKENS);
        }
        r->archive_bytes = ds4_session_scr_archive_bytes(s);
    }
    tokens = malloc((size_t)request->len * sizeof(int));
    logits = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (!tokens || !logits) {
        r->reason = DS4_SCR_REASON_RESOURCE;
        snprintf(err, errlen, "SCR multi: private reservation failed"); goto done;
    }
    memcpy(tokens, request->v, (size_t)request->len * sizeof(int));
    /* The public request may be &s->checkpoint. Own its metadata as well as
     * bytes before growth/reset/install can mutate the caller's descriptor. */
    const ds4_tokens full = {.v = tokens, .len = request->len, .cap = request->len};
    if (s->checkpoint.cap < full.len) {
        int *v = realloc(s->checkpoint.v, (size_t)full.len * sizeof(int));
        if (!v) {
            r->reason = DS4_SCR_REASON_RESOURCE;
            snprintf(err, errlen, "SCR multi: publication reservation failed"); goto done;
        }
        s->checkpoint.v = v; s->checkpoint.cap = full.len;
    }
    progress = s->progress; display = s->display_progress;
    s->progress = s->display_progress = NULL;
    s->scr_multi_active = active = true;
    if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto failed; }
    /* Drain before withdrawing any previously published destination. */
    if (!ds4_gpu_synchronize()) { r->partial_work_unknown = true; goto failed; }
    mutated = r->target_mutated = true;
    s->checkpoint_valid = false; s->mtp_draft_valid = false;
    free(s->checkpoint_images);
    s->checkpoint_images = NULL; s->checkpoint_image_count = 0;
    ds4_session_dspark_capture_invalidate(s);
    const ds4_tokens prefix = {.v = full.v,
        .len = candidate ? (int)plan[0].destination_start :
            rebasing ? (int)((uint32_t)full.len & ~7u) : full.len, .cap = full.cap};
    ds41_sync_frontier frontier = {0};
    r->phase = candidate ? DS4_SCR_MULTI_PREFIX : DS4_SCR_MULTI_FRESH;
    double t = now_sec();
    rc = ds41_session_prefix(s, &prefix, true, &frontier, err, errlen);
    if (candidate) { r->prefix_seconds = now_sec() - t; r->prefix_completed = frontier.completed; }
    else { r->fresh_seconds = now_sec() - t; r->fresh_completed = frontier.completed; }
    r->partial_work_unknown = frontier.unknown;
    if (rc) goto failed;
    if (!ds4_gpu_synchronize() || !ds41_multi_cursor(s, tokens, (uint32_t)prefix.len)) {
        rc = 1; r->partial_work_unknown = true; goto failed;
    }
    /* Several mapped ranges: read range i+1's raw window during gap i. */
    if (candidate && mapped && selected > 1u) prefetch_data = malloc(DS41_ARCHIVE_READ_BYTES);
    for (uint32_t i = 0; candidate && i < selected; ++i) {
        ds4_scr_range *p = &plan[i];
        t = now_sec();
        if (prefetch.running && !ds41_raw_prefetch_finish(&prefetch) && !a->raw->readable) {
            a->retiring = true; rc = 1; r->partial_work_unknown = true; goto failed;
        }
        a->raw_read_seconds += now_sec() - t;
        if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto failed; }
        r->phase = DS4_SCR_MULTI_STAGE;
        ++r->attempted_spans; r->attempted_tokens += p->count;
        t = now_sec();
        bool pinned = ds41_multi_pin(s, a, generation, start, end) &&
            ds41_multi_cursor(s, tokens, p->destination_start);
        const bool prefetched = prefetch.ready;
        rc = !pinned ? 1 : mapped ?
            (ds41_archive_materialize_window(a, s, &s->engine->model, p->source_start,
                p->destination_start, tokens + p->destination_start, p->count, g->ctx, alignment,
                prefetched ? &prefetch : NULL, i == 0u) ? 0 : 1) :
            ds4_session_scr_archive_stage(s, p->source_start,
                p->destination_start, tokens + p->destination_start, p->count);
        r->stage_seconds += now_sec() - t;
        prefetch.ready = false;
        if (rc) { r->partial_work_unknown = true; goto failed; }
        ++r->materialized_spans; r->materialized_tokens += p->count;
        r->prefetched_spans += prefetched;
        if (rebasing) {
            t = now_sec();
            bool imported = ds41_session_rebase_import(s, p);
            r->import_seconds += now_sec() - t;
            if (!imported) { rc = 1; r->partial_work_unknown = true; goto failed; }
            r->imported_tokens = g->archive_pending->imported_tokens;
        }
        r->phase = DS4_SCR_MULTI_INSTALL;
        t = now_sec();
        bool installed = mapped ?
            ds41_multi_install_aligned(s, tokens, a, p, generation, start, end, alignment) :
            ds41_multi_install(s, tokens, a, p, generation, start, end);
        r->install_seconds += now_sec() - t;
        if (!installed) { rc = 1; r->partial_work_unknown = true; goto failed; }
        ++r->installed_spans; r->installed_tokens += p->count;
        if (i + 1u < selected) {
            uint32_t first, records;
            const ds4_scr_range *n = &plan[i + 1u];
            if (prefetch_data &&
                ds41_raw_window_records(ds41_archive_count(a), n->source_start - a->start + n->count,
                                        &first, &records))
                (void)ds41_raw_prefetch_start(&prefetch, a->raw, first, records, prefetch_data);
            r->phase = DS4_SCR_MULTI_GAP;
            t = now_sec();
            rc = ds41_multi_capture_steps(s, tokens, plan[i+1].destination_start,
                                  &r->gap_completed[i], &r->partial_work_unknown,
                                  a, generation, start, end, rebasing);
            r->gap_seconds += now_sec() - t;
            if (rc) goto failed;
        }
    }
    if (candidate) {
        r->phase = DS4_SCR_MULTI_SUFFIX;
        t = now_sec();
        rc = rebasing ?
            ds41_multi_capture_steps(s, full.v, (uint32_t)full.len & ~7u,
                &r->suffix_completed, &r->partial_work_unknown,
                a, generation, start, end, true) :
            ds41_multi_suffix(s, full.v, (uint32_t)full.len,
                &r->suffix_completed, &r->partial_work_unknown,
                a, generation, start, end, suffix, counts);
        r->suffix_seconds = now_sec() - t;
        if (rc) goto failed;
    }
    if (rebasing) {
        r->phase = DS4_SCR_MULTI_HEAD;
        rc = ds41_session_rebase_cover(s, full.v, logits, r);
        if (rc) goto failed;
        r->phase = DS4_SCR_MULTI_SUFFIX;
        t = now_sec();
        rc = ds41_multi_steps(s, full.v, (uint32_t)full.len,
            candidate ? &r->suffix_completed : &r->fresh_completed, &r->partial_work_unknown);
        r->suffix_seconds += now_sec() - t;
        if (rc) goto failed;
    }
    if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto failed; }
    if (!ds41_multi_cursor(s, full.v, (uint32_t)full.len) ||
        (candidate && !ds41_multi_pin(s, a, generation, start, end))) {
        rc = 1; r->partial_work_unknown = true; goto failed;
    }
    r->phase = DS4_SCR_MULTI_HEAD;
    t = now_sec();
    bool headed = DS41_SYNC_HEAD(g, &s->engine->model, &s->engine->weights, logits);
    r->head_seconds = now_sec() - t;
    if (!headed) { rc = 1; r->partial_work_unknown = true; goto failed; }
    r->head_completed = true;
    if (rebasing) {
        ds41_scr_archive *pending = g->archive_pending;
        if (!ds41_multi_cursor(s, full.v, (uint32_t)full.len) ||
            (pending->writer_state == DS41_ARCHIVE_ARMED &&
             !ds41_archive_append_observe(pending, s, &s->engine->model, full.v,
                 (uint32_t)full.len, ds41_archive_raw_cancel, s))) {
            rc = 1; r->partial_work_unknown = true; goto failed;
        }
        g->archive_capture_pending = false;
    }
    r->phase = DS4_SCR_MULTI_PUBLICATION;
    t = now_sec();
    if (!ds4_gpu_synchronize() || !ds41_multi_cursor(s, full.v, (uint32_t)full.len) ||
        (candidate && !ds41_multi_pin(s, a, generation, start, end))) {
        rc = 1; r->partial_work_unknown = true; goto failed;
    }
    if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto failed; }
    memcpy(s->checkpoint.v, full.v, (size_t)full.len * sizeof(int));
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(float));
    s->checkpoint.len = full.len;
    if (a) a->range.valid = false;
    r->outcome = candidate ? DS4_SCR_REUSED : DS4_SCR_FALLBACK;
    if (candidate) {
        r->reason = DS4_SCR_REASON_NONE;
        r->committed_spans = r->installed_spans; r->committed_tokens = r->installed_tokens;
    }
    r->published = true;
    s->checkpoint_valid = true; /* validity LAST: no fallible work follows */
    r->publication_seconds = now_sec() - t;
    rc = 0; goto done;
failed:
    if (prefetch.running) (void)ds41_raw_prefetch_finish(&prefetch);
    if (ds4_session_cancelled(s)) rc = DS4_SESSION_SYNC_INTERRUPTED;
    /* Validated, disjoint target intervals bound this known work by full N.
     * Borrowed/materialized/installed rows are never forward computation. */
    r->discarded_tokens = r->prefix_completed + r->suffix_completed + r->fresh_completed;
    for (uint32_t i = 0; i < DS4_SCR_MULTI_MAX_RANGES; ++i)
        r->discarded_tokens += r->gap_completed[i];
    r->outcome = rc == DS4_SESSION_SYNC_INTERRUPTED ? DS4_SCR_CANCELLED : DS4_SCR_ERROR;
    r->reason = DS4_SCR_REASON_COMPUTE;
    if (mutated) {
        s->checkpoint_valid = false; g->valid = false;
        if (a) a->range.valid = false;
        if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) r->partial_work_unknown = true;
        if (!ds4_gpu_synchronize()) r->partial_work_unknown = true;
    }
    if (r->rebase_prepared) {
        (void)ds41_session_rebase_abort(s, r);
        rebase_cleanup_done = true;
    }
    snprintf(err, errlen, "SCR multi: %s in phase %d",
        rc == DS4_SESSION_SYNC_INTERRUPTED ? "interrupted" : "failure", (int)r->phase);
#else
    r->reason = DS4_SCR_REASON_UNSUPPORTED;
    snprintf(err, errlen, "SCR multi: GPU Flash support not compiled in");
#endif
done:
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (prefetch.running) (void)ds41_raw_prefetch_finish(&prefetch);
    if (r->rebase_prepared && !r->published && !rebase_cleanup_done)
        (void)ds41_session_rebase_abort(s, r);
#endif
    if (active) {
        s->progress = progress; s->display_progress = display; s->scr_multi_active = false;
    }
    free(logits); free(tokens); free(prefetch_data);
    r->total_seconds = now_sec() - begin;
    return rc;
}
static int ds41_scr_multi_owned(ds4_session *s, const ds4_tokens *request,
                              const ds4_scr_range *ranges, uint32_t count,
                              ds4_scr_multi_receipt *out, char *err, size_t errlen,
                              const ds41_policy_owner *owner) {
    return ds41_scr_multi_selected(s, request, ranges, count, out, err, errlen,
                                  owner, DS41_SUFFIX_SCALAR, NULL);
}
int ds4_session_scr_multi_sync(ds4_session *s, const ds4_tokens *request,
                              const ds4_scr_range *ranges, uint32_t count,
                              ds4_scr_multi_receipt *out, char *err, size_t errlen) {
    return ds41_scr_multi_owned(s, request, ranges, count, out, err, errlen, NULL);
}

/* Explicit owning dense-source producer. Normal prefix certifications/heads
 * remain real; exclusive caller use and callback suppression hide observation,
 * not computation or publication. No scheduler or archive guard is bypassed. */
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
static bool ds41_refresh_cursor(ds4_session *s, const ds4_tokens *full, uint32_t pos) {
    ds41_gpu_graph *g = &s->ds41_graph;
    return s->checkpoint_valid && g->valid && g->pos == pos &&
        s->checkpoint.len == (int)pos && s->checkpoint.v &&
        !memcmp(s->checkpoint.v, full->v, (size_t)pos * sizeof(int)) &&
        !g->encoder_resident && !ds4_gpu_commands_active();
}
static int ds41_refresh_call(ds4_session *s, const ds4_tokens *full, uint32_t from,
                            uint32_t end, ds4_scr_refresh_receipt *r,
                            uint32_t *completed, double *seconds,
                            char *err, size_t errlen) {
    /* A lost resume proof must not invoke normal helper's reset/replay path. */
    if (!ds41_refresh_cursor(s, full, from)) {
        r->work_unknown = r->head_unknown = true;
        snprintf(err, errlen, "SCR refresh: lost prior prefix proof");
        return 1;
    }
    const ds4_tokens part = {.v = full->v, .len = (int)end, .cap = full->cap};
    ds41_sync_frontier f = {0};
    ++r->normal_calls;
    const double t = now_sec();
    int rc = ds41_session_prefix(s, &part, false, &f, err, errlen);
    *seconds += now_sec() - t;
    /* Snapshot known work/head BEFORE any drain, seal or invalidation. The
     * frontier is absolute; only the newly completed interval is counted. */
    if (f.completed > from && f.completed <= end) *completed += f.completed - from;
    else if (f.completed && f.completed != from) r->work_unknown = true;
    r->work_unknown |= f.unknown;
    bool headed = s->checkpoint.len > (int)from && s->checkpoint.len <= (int)end &&
        ds41_refresh_cursor(s, full, (uint32_t)s->checkpoint.len);
    r->completed_heads += f.heads;
    if (!headed) r->head_unknown = true;
    if (!rc && (!headed || !ds41_refresh_cursor(s, full, end) || f.completed != end)) {
        r->work_unknown = true;
        snprintf(err, errlen, "SCR refresh: incomplete normal frontier");
        rc = 1;
    }
    ds41_scr_archive *capture = s->ds41_graph.archive;
    if (!rc && capture && !capture->sealed && end >= capture->start) {
        double flush_begin = now_sec();
        rc = ds4_session_scr_archive_flush(s);
        *seconds += now_sec() - flush_begin;
        if (rc) snprintf(err, errlen, "SCR refresh: canonical raw flush refused");
    }
    return rc;
}
#endif
static int ds41_scr_refresh_owned(ds4_session *s, const ds4_tokens *request,
                                  uint32_t start, uint32_t count,
                                  ds4_scr_refresh_receipt *out,
                                  char *err, size_t errlen,
                                  const ds41_policy_owner *owner) {
    ds4_scr_refresh_receipt local = {0}, *r = out ? out : &local;
    memset(r, 0, sizeof(*r));
    const double begin = now_sec();
    int rc = 1, *tokens = NULL;
    ds41_scr_coordinates source_coordinates = {0};
    bool active = false, mutated = false;
    const uint32_t capacity = owner && owner->archive_capacity ? owner->archive_capacity : count;
    uint32_t known_count = 0;
    ds4_session_progress_fn progress = NULL, display = NULL;
    r->source_start = start; r->source_count = count;
    if (!s || !s->engine || !request || !request->v || request->len <= 0 ||
        request->len >= s->ctx_size || !s->logits || s->scr_multi_active ||
        (owner ? !ds41_policy_authorized(s, request, owner) : s->scr_policy_active) ||
        start % 8u || count % 8u || count < 128u ||
        count > DS4_SCR_ARCHIVE_MAX_CAPACITY || capacity < count ||
        capacity > DS4_SCR_ARCHIVE_MAX_CAPACITY || capacity % 8u ||
        start > (uint32_t)request->len ||
        count > (uint32_t)request->len - start || request->len >= 1048576) {
        snprintf(err, errlen, "SCR refresh: invalid input/geometry/ownership"); goto done;
    }
    r->requested_tokens = (uint32_t)request->len;
    r->source_end = start + count;
    r->reserved_capacity = capacity;
    known_count = (uint32_t)request->len - start;
    if (known_count > capacity) known_count = capacity;
    for (int i = 0; i < request->len; ++i)
        if (request->v[i] < 0 || (uint32_t)request->v[i] >= DS4_N_VOCAB) {
            snprintf(err, errlen, "SCR refresh: invalid token at %d", i); goto done;
        }
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    uint32_t future_pads = 0;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *old = s->ds41_graph_ready ? g->archive : NULL;
    if (!s->ds41_graph_ready || !ds4_session_is_ds41(s) || s->distributed ||
        s->engine->backend != DS4_BACKEND_METAL || s->engine->tp.active ||
        g->tp_world != 1u || g->image_count || s->sync_image_count || g->imatrix ||
        (uint32_t)request->len >= g->ctx || start > g->ctx || capacity > g->ctx - start ||
        start > (uint32_t)s->ctx_size || capacity > (uint32_t)s->ctx_size - start ||
        start > DS41_RAW_MAX_TOKENS - capacity || g->archive_pending || g->encoder_resident ||
        ds4_gpu_commands_active() || (old && (old->session != s ||
        old->model != &s->engine->model || old->retiring || old->generation == UINT64_MAX))) {
        r->reason = DS4_SCR_REFRESH_UNSUPPORTED;
        snprintf(err, errlen, "SCR refresh: unsupported runtime/source/ownership"); goto done;
    }
    r->reason = DS4_SCR_REFRESH_RESOURCE;
    r->engine_charged_bytes_before = s->engine->ds41_session_bytes;
    uint64_t source_map_bytes = 0;
    uint32_t source_logical_start = start;
    if (owner && owner->coordinates) {
        uint32_t pads = 0;
        for (uint32_t i = 0; i < owner->coordinates->count; ++i) {
            uint32_t pos = owner->coordinates->pads[i];
            if (pos >= start && pos < start + known_count) ++pads;
        }
        size_t bytes;
        if (!ds41_coordinates_heap_bytes(pads, &bytes)) {
            snprintf(err, errlen, "SCR refresh: coordinate size overflow"); goto done;
        }
        source_map_bytes = bytes;
    }
    r->prospective_archive_bytes = ds41_archive_bytes(capacity);
    if (source_map_bytes > UINT64_MAX - r->prospective_archive_bytes) {
        snprintf(err, errlen, "SCR refresh: archive coordinate size overflow"); goto done;
    }
    r->prospective_archive_bytes += source_map_bytes;
    r->temporary_host_bytes = (uint64_t)request->len * sizeof(int) +
        sizeof(local) + 2u * sizeof(ds4_tokens) + sizeof(ds41_sync_frontier) +
        sizeof(source_coordinates) + source_map_bytes;
    if (s->checkpoint.cap < request->len)
        r->publication_growth_bytes = (uint64_t)(request->len - s->checkpoint.cap) * sizeof(int);
    uint64_t extra = 0;
    if (!ds41_policy_reservation(s, request, owner,
        r->temporary_host_bytes + r->publication_growth_bytes, &extra) || extra < source_map_bytes) {
        snprintf(err, errlen, "SCR refresh: invalid outer reservation"); goto done;
    }
    if (extra > UINT64_MAX - s->engine->ds41_session_bytes ||
        !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + extra, false)) {
        snprintf(err, errlen, "SCR refresh: live host reservation refused"); goto done;
    }
    tokens = malloc((size_t)request->len * sizeof(int));
    if (!tokens) { snprintf(err, errlen, "SCR refresh: target allocation failed"); goto done; }
    memcpy(tokens, request->v, (size_t)request->len * sizeof(int));
    /* Last caller descriptor read: accessor metadata may change on realloc/reset. */
    const ds4_tokens full = {.v = tokens, .len = request->len, .cap = request->len};
    if (owner && owner->coordinates &&
        !ds41_coordinates_slice(owner->coordinates, full.v, (uint32_t)full.len,
            DS4_N_VOCAB, start, known_count, &source_coordinates, &source_logical_start)) {
        snprintf(err, errlen, "SCR refresh: coordinate projection refused"); goto done;
    }
    future_pads = source_coordinates.count;
    if (source_coordinates.identity) {
        uint32_t published_pads = 0;
        while (published_pads < source_coordinates.count &&
               source_coordinates.pads[published_pads] < count) ++published_pads;
        source_coordinates.count = published_pads;
        source_coordinates.physical_tokens = count;
    }
    if (s->checkpoint.cap < full.len) {
        int *v = realloc(s->checkpoint.v, (size_t)full.len * sizeof(int));
        if (!v) { snprintf(err, errlen, "SCR refresh: publication allocation failed"); goto done; }
        s->checkpoint.v = v; s->checkpoint.cap = full.len;
    }
    progress = s->progress; display = s->display_progress;
    s->progress = s->display_progress = NULL;
    /* Reuse existing exclusive SCR transaction ownership; no shared guard change. */
    s->scr_multi_active = active = true;
    if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto done; }
    r->reason = DS4_SCR_REFRESH_RELEASE;
    r->phase = DS4_SCR_REFRESH_RETIRE;
    double t = now_sec();
    if (old) {
        r->old_retirement_attempted = true;
        rc = ds4_session_scr_archive_release(s);
        r->retire_seconds = now_sec() - t;
        if (rc) { snprintf(err, errlen, "SCR refresh: old release failed; retained/fenced"); goto done; }
        r->old_retired = true;
    } else r->retire_seconds = now_sec() - t;
    rc = 1;
    r->reason = DS4_SCR_REFRESH_RESOURCE;
    r->phase = DS4_SCR_REFRESH_PREPARE;
    t = now_sec();
    const uint64_t live = s->engine->ds41_session_bytes;
    r->engine_charged_bytes_after_retirement = live;
    /* The interval map moves from this temporary owner to the archive; it is
     * not allocated twice when the archive is prepared. */
    const uint64_t allocation_extra = extra - source_map_bytes;
    if (allocation_extra > UINT64_MAX - live ||
        r->prospective_archive_bytes > UINT64_MAX - live - allocation_extra ||
        !ds41_memory_admit(s->engine, live + allocation_extra + r->prospective_archive_bytes, false)) {
        snprintf(err, errlen, "SCR refresh: new archive plus host refused; no rollback");
        r->prepare_seconds = now_sec() - t; goto done;
    }
    if (!ds4_gpu_synchronize()) {
        snprintf(err, errlen, "SCR refresh: initial drain failed");
        r->work_unknown = true; r->prepare_seconds = now_sec() - t; goto done;
    }
    mutated = r->target_mutated = true;
    ds41_graph_reset(g);
    s->checkpoint.len = 0; s->checkpoint_valid = false; s->mtp_draft_valid = false;
    ds41_session_coordinates_clear(s);
    free(s->checkpoint_images); s->checkpoint_images = NULL; s->checkpoint_image_count = 0;
    ds4_session_dspark_capture_invalidate(s);
    rc = ds4_session_scr_archive_prepare(s, start, capacity);
    r->prepare_seconds = now_sec() - t;
    if (rc) goto failed;
    r->prepared = true;
    r->reason = DS4_SCR_REFRESH_COMPUTE;
    /* Empty reset frontier is a resume checkpoint, not a computed head. Prevent
     * normal orchestration from resetting the just-prepared coverage. */
    s->checkpoint_valid = true;
    if (start) {
        r->phase = DS4_SCR_REFRESH_PREFIX;
        rc = ds41_refresh_call(s, &full, 0, start, r, &r->prefix_completed,
                              &r->prefix_seconds, err, errlen);
        if (rc) goto failed;
    }
    r->phase = DS4_SCR_REFRESH_SOURCE;
    /* The whole source span is one call: capture flushes at each raw arena
     * frontier without a head, encoder residency can span them, and one head
     * certifies the span. */
    r->source_chunks = 1;
    s->scr_refresh_capture = true;
    rc = ds41_refresh_call(s, &full, start, r->source_end, r, &r->source_completed,
                          &r->source_seconds, err, errlen);
    s->scr_refresh_capture = false;
    if (rc) goto failed;
    r->phase = DS4_SCR_REFRESH_SEAL;
    t = now_sec();
    rc = ds4_session_scr_archive_seal(s, full.v + start, count);
    if (!rc && source_coordinates.identity) {
        ds41_scr_archive *sealed = g->archive;
        if (!sealed || !sealed->sealed || sealed->retiring ||
            !ds41_coordinates_replace_charged(&sealed->coordinates, &source_coordinates,
                sealed->tokens, count, DS4_N_VOCAB, &g->allocation_bytes,
                &s->engine->ds41_session_bytes, &sealed->allocation_bytes)) rc = 1;
        else sealed->logical_start = source_logical_start;
    }
    r->seal_seconds = now_sec() - t;
    if (rc) goto failed;
    r->sealed = true;
    if (!ds41_archive_append_arm(g->archive, s, &s->engine->model,
                                full.v + start, known_count, future_pads)) {
        rc = ds4_session_cancelled(s) ? DS4_SESSION_SYNC_INTERRUPTED : 1;
        goto failed;
    }
    if (r->source_end < (uint32_t)full.len) {
        r->phase = DS4_SCR_REFRESH_REMAINDER;
        rc = ds41_refresh_call(s, &full, r->source_end, (uint32_t)full.len, r,
                              &r->remainder_completed, &r->remainder_seconds, err, errlen);
        if (rc) goto failed;
    }
    r->phase = DS4_SCR_REFRESH_COMPLETE;
    t = now_sec();
    ds41_scr_archive *a = g->archive;
    const uint32_t captured = (uint32_t)full.len < start + capacity ?
        (uint32_t)full.len : start + capacity;
    const uint32_t published_end = captured & ~7u, published_count = published_end - start;
    bool pinned = ds41_archive_owned_sealed(a, s) && !a->lost && a->start == start && a->end == published_end &&
        a->cap == capacity && ds41_archive_count(a) == published_count &&
        ds41_archive_shape(a, g->ctx) && a->captured_end == captured &&
        a->writer_generation == a->generation && a->source_generation == a->generation &&
        a->raw && a->raw_origin == published_end && a->raw->committed_tokens == published_count &&
        a->raw->written_tokens == published_count &&
        !memcmp(a->tokens, full.v + start, (size_t)(captured - start) * sizeof(int));
    r->published_end = a ? a->end : 0;
    r->published_count = a ? ds41_archive_count(a) : 0;
    bool complete = ds4_gpu_synchronize() && ds41_refresh_cursor(s, &full, (uint32_t)full.len);
    bool cancelled = ds4_session_cancelled(s);
    r->complete_seconds = now_sec() - t;
    if (!complete || !pinned || cancelled) {
        rc = cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
        r->work_unknown |= !complete;
        goto failed;
    }
    r->complete_n = true;
    r->reason = DS4_SCR_REFRESH_OK;
    rc = 0; goto done;
failed:
    /* Work/head receipts already snapshotted, including completed interruption.
     * Invalidate destination even if a normal call completed N before seal failed. */
    t = now_sec();
    if (mutated) {
        s->checkpoint_valid = false; g->valid = false; s->mtp_draft_valid = false;
        if (g->archive && r->sealed) {
            ds41_archive_append_failure(g->archive, g->archive->history_hole,
                rc == DS4_SESSION_SYNC_INTERRUPTED ? ECANCELED : EIO);
        } else if (g->archive) {
            r->failed_source_retirement_attempted = true;
            r->failed_source_retired = ds4_session_scr_archive_release(s) == 0;
        }
    }
    r->cleanup_seconds = now_sec() - t;
    snprintf(err, errlen, "SCR refresh: %s phase %d; old_retired=%d new_retired=%d (no rollback)",
        rc == DS4_SESSION_SYNC_INTERRUPTED ? "interrupted" : "failure", (int)r->phase,
        r->old_retired, r->failed_source_retired);
#else
    r->reason = DS4_SCR_REFRESH_UNSUPPORTED;
    snprintf(err, errlen, "SCR refresh: GPU Flash support not compiled in");
#endif
done:
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (s && s->engine) r->engine_charged_bytes_final = s->engine->ds41_session_bytes;
    if (s && s->engine && s->ds41_graph_ready && s->ds41_graph.archive) {
        ds41_scr_archive *a = s->ds41_graph.archive;
        r->source_charged_bytes = a->allocation_bytes;
        r->source_retiring = a->retiring;
        r->source_usable = ds41_archive_owned_sealed(a, s) && a->source_generation && a->raw && a->raw->readable &&
            !(a->lost && a->generation == UINT64_MAX);
        r->published_end = a->end; r->published_count = ds41_archive_count(a);
    }
#endif
    if (active) {
        s->progress = progress; s->display_progress = display; s->scr_multi_active = false;
    }
    if (rc == DS4_SESSION_SYNC_INTERRUPTED && r->phase == DS4_SCR_REFRESH_PREFLIGHT)
        snprintf(err, errlen, "SCR refresh: interrupted before mutation");
    ds41_coordinates_free(&source_coordinates);
    free(tokens);
    if (rc == DS4_SESSION_SYNC_INTERRUPTED) r->reason = DS4_SCR_REFRESH_CANCELLED;
    r->error_code = rc; r->total_seconds = now_sec() - begin;
    return rc;
}
int ds4_session_scr_archive_refresh(ds4_session *s, const ds4_tokens *request,
                                  uint32_t start, uint32_t count,
                                  ds4_scr_refresh_receipt *out,
                                  char *err, size_t errlen) {
    return ds41_scr_refresh_owned(s, request, start, count, out, err, errlen, NULL);
}
#endif

/* ==== 5. Planner Adapter & Logical Policy ================================ */
#ifdef DS4_SCR_ENGINE_SESSION
/* Startup selection is session-owned; no request can reinterpret an old map. */
static bool ds41_padding_ordinary_space(ds4_engine *e, int id) {
    if (!e || !e->vocab.token || id < 0 || id >= e->vocab.n_vocab ||
        !e->vocab.token[id].ptr || !e->vocab.token[id].len ||
        e->vocab.token[id].len > 16u ||
        vocab_token_is_generation_stop(&e->vocab, id) ||
        vocab_token_is_literal_special(e->vocab.token[id])) return false;
    const ds4_vocab *v = &e->vocab;
    const int controls[] = {v->bos_id,v->system_id,v->user_id,v->assistant_id,
        v->observation_id,v->sop_id,v->think_start_id,v->think_end_id,
        v->tool_call_start_id,v->tool_call_end_id,v->tool_response_start_id,
        v->tool_response_end_id,v->arg_key_start_id,v->arg_key_end_id,
        v->arg_value_start_id,v->arg_value_end_id,v->dsml_id,v->im_start_id,v->im_end_id};
    for (size_t i = 0; i < sizeof(controls)/sizeof(controls[0]); ++i)
        if (controls[i] == id) return false;
    size_t length = 0;
    char *piece = ds4_token_text(e, id, &length);
    bool valid = piece && length == 1u && piece[0] == ' ';
    free(piece);
    return valid;
}
static bool ds41_scr_empty_startup(ds4_session *s) {
    if (!ds41_scr_engine_compatible(s) || s->checkpoint.len ||
        s->checkpoint_valid || s->scr_coordinate_serial || s->scr_coordinates.identity ||
        s->scr_coordinates.count || s->scr_coordinates.physical_tokens ||
        s->scr_coordinates.pads || s->scr_coordinates.capacity || s->scr_coordinate_charge ||
        s->sync_image_count || s->checkpoint_image_count ||
        s->scr_policy_active || s->scr_multi_active || s->scr_dense_owner) return false;
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (s->ds41_graph_ready && (s->ds41_graph.pos || s->ds41_graph.image_count || s->ds41_graph.archive ||
        s->ds41_graph.archive_pending)) return false;
#endif
    return true;
}
int ds4_session_scr_set_padding_policy(ds4_session *s, ds4_scr_padding_policy policy,
                                      char *err, size_t errlen) {
    if (!ds41_scr_empty_startup(s) || s->scr_padding_selected ||
        policy != DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE) goto refused;
    int token = -1;
    ds4_tokens encoded = {0};
    ds4_tokenize_text(s->engine, " ", &encoded);
    if (encoded.len == 1) token = encoded.v[0];
    ds4_tokens_free(&encoded);
    if (!ds41_padding_ordinary_space(s->engine, token)) goto refused;
    s->scr_padding_policy = policy;
    s->scr_padding_token = token;
    s->scr_padding_selected = true;
    return 0;
refused:
    if (err && errlen) snprintf(err, errlen, "SCR padding policy requires empty startup state and a verified ordinary space token");
    return 1;
}

/* Compatibility startup check: batched capture is SCR's only capture body,
 * so this selects nothing and only validates an SCR-compatible empty session. */
int ds4_session_scr_enable_capture_batch_chunks(ds4_session *s, char *err, size_t errlen) {
    if (ds41_scr_empty_startup(s)) return 0;
    if (err && errlen) snprintf(err, errlen, "capture batch requires an SCR-compatible empty session");
    return 1;
}

/* Logical SCR policy: the server's only SCR prefill entry. Archive planning is
 * the cache-reuse-utils planner (ds4_deepseek41_scr_planner_adapter.h); every
 * other route materializes the request with its carried live-prefix pads. */
/* Verified physical target: planner ranges in physical coordinates plus the
 * materialized target's pad accounting. */
typedef struct {
    ds4_scr_range ranges[DS4_SCR_MULTI_MAX_RANGES];
    uint32_t range_count, physical_reused, logical_reused, imported_pads;
    uint32_t physical_count, pad_count, new_pads;
    bool planned, verified;
} ds41_even_plan;
typedef struct {
    ds4_scr_dense_policy_receipt execution; /* Physical execution coordinates. */
    ds4_scr_dense_match_receipt logical_match;
    ds41_even_plan plan;
    uint32_t logical_tokens, physical_tokens, new_pads, carried_pads;
    uint32_t imported_pads, useful_committed, boundary_policy;
    /* Rebase execution accounting in physical coordinates (see
     * ds41_even_executed_subset): reused, fresh gaps between executed
     * ranges, and the fresh suffix after the last one. */
    uint32_t executed_reused, executed_gap, executed_suffix;
    uint64_t map_identity, planning_work, planning_comparisons, peak_host_bytes;
    uint32_t planner_status, planner_best_useful, planner_admissible_min;
    uint32_t planner_covered, planner_padding;
    uint32_t planner_discovered_matches, planner_longest_match;
    uint64_t planner_elapsed_ns, planner_preparation_ns, planner_discovery_ns, planner_search_ns;
    bool physical_known, published, target_mutated, planner_status_known;
    bool planner_discovery_known, planner_search_known, planner_timing_known;
} ds41_even_receipt;
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
static bool ds41_even_admit(ds4_session *s, uint64_t bytes, ds41_even_receipt *r) {
    if (bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
        !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + bytes, false)) {
        r->execution.reason = DS4_SCR_DENSE_RESOURCE; return false;
    }
    if (bytes > r->peak_host_bytes) r->peak_host_bytes = bytes;
    return true;
}
/* The engine may stop a selected range early but not move it; the library
 * checks that subset and alignment, the floor and pad count stay here. */
static bool ds41_even_executed_subset(ds41_even_receipt *r,
        const ds41_scr_coordinates *map, ds41_pad_budget *budget, uint32_t *pads) {
    const ds41_even_plan *proposal = &r->plan;
    const ds4_scr_multi_receipt *actual = &r->execution.multi;
    *pads = 0;
    if (!proposal->verified || !actual->published || !actual->covered_head_completed ||
        actual->planned_spans != proposal->range_count ||
        actual->planned_tokens != proposal->physical_reused ||
        proposal->range_count > DS4_SCR_MULTI_MAX_RANGES ||
        actual->normalized_spans > proposal->range_count ||
        actual->committed_spans != actual->normalized_spans ||
        actual->committed_tokens != actual->imported_tokens) return false;
    cache_reuse_range planned[DS4_SCR_MULTI_MAX_RANGES], executed[DS4_SCR_MULTI_MAX_RANGES];
    for (uint32_t i = 0; i < proposal->range_count; ++i) {
        const ds4_scr_range *p = &proposal->ranges[i];
        planned[i] = (cache_reuse_range){p->source_start, p->destination_start, p->count,
                                         p->destination_start};
    }
    for (uint32_t i = 0; i < actual->normalized_spans; ++i) {
        const ds4_scr_range *p = &actual->ranges[i];
        if (p->count < 128u) return false;
        executed[i] = (cache_reuse_range){p->source_start, p->destination_start, p->count,
                                          p->destination_start};
    }
    cache_reuse_execution_summary summary;
    if (cache_reuse_reconcile_executed_ranges(planned, proposal->range_count, executed,
            actual->normalized_spans, map->physical_tokens, 2u, &summary, NULL, 0) !=
        CACHE_REUSE_COMPLETE) return false;
    for (uint32_t i = 0; i < actual->normalized_spans; ++i) {
        const ds4_scr_range *range = &actual->ranges[i];
        for (uint32_t p = 0; p < map->count; ++p) {
            if (!ds41_pad_tick(budget)) return false;
            if (map->pads[p] >= range->destination_start &&
                map->pads[p] - range->destination_start < range->count) ++*pads;
        }
    }
    r->executed_reused = summary.reused_tokens;
    r->executed_gap = summary.gap_tokens;
    r->executed_suffix = summary.suffix_tokens;
    return summary.reused_tokens == actual->committed_tokens && *pads <= summary.reused_tokens;
}
/* No archive plan (no source, logical continuation, or <=128 unmatched
 * tokens): keep the live physical prefix that covers the common logical prefix,
 * including its carried pads, and append the remaining logical tokens. */
static int ds41_even_carry_prefix(const ds4_tokens *logical, const ds4_session *s,
        const ds41_pad_prefix_info *prefix, uint32_t context, uint64_t identity,
        ds4_tokens *physical, ds41_scr_coordinates *pending, ds41_pad_budget *budget) {
    const ds41_scr_coordinates *live_map = &s->scr_coordinates;
    const uint32_t n = (uint32_t)logical->len, carried = prefix->carried_pads;
    const uint32_t head = prefix->common + carried;
    if (!identity || prefix->common > n || carried > live_map->count ||
        head > (uint32_t)s->checkpoint.len || n + carried >= context) return 1;
    physical->v = malloc(((size_t)n + carried) * sizeof(int));
    pending->pads = carried ? malloc((size_t)carried * sizeof(uint32_t)) : NULL;
    if (!physical->v || (carried && !pending->pads)) return 1;
    if (head) memcpy(physical->v, s->checkpoint.v, (size_t)head * sizeof(int));
    memcpy(physical->v + head, logical->v + prefix->common,
        (size_t)(n - prefix->common) * sizeof(int));
    for (uint32_t i = 0; i < carried; ++i) {
        if (!ds41_pad_tick(budget)) return budget->cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
        pending->pads[i] = live_map->pads[i];
    }
    physical->len = physical->cap = (int)(n + carried);
    pending->count = pending->capacity = carried;
    pending->physical_tokens = n + carried;
    pending->identity = identity;
    return 0;
}
/* Included after owning-entry admission helpers, inside the Flash Metal guard. */
/* Insertion authority: startup fresh-gap selection, or none (policy 0). */
typedef struct {
    uint32_t logical_tokens, policy;
    int pad_token;
} ds41_boundary_permissions;

static int ds41_even_prepare_boundaries(ds4_session *s, const ds4_tokens *logical,
        ds41_boundary_permissions *out, ds41_pad_budget *budget) {
    *out = (ds41_boundary_permissions){.logical_tokens = (uint32_t)logical->len};
    if (s->scr_padding_policy != DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE) return 0;
    if (!s->scr_padding_selected || !ds41_padding_ordinary_space(s->engine, s->scr_padding_token)) return 1;
    for (int i = 0; i < logical->len; ++i) {
        if (!ds41_pad_tick(budget)) return budget->cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
        int id = logical->v[i];
        if (id < 0 || id >= s->engine->vocab.n_vocab ||
            !s->engine->vocab.token[id].ptr || !s->engine->vocab.token[id].len ||
            s->engine->vocab.token[id].len >= SIZE_MAX) return 1;
    }
    out->policy = DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1;
    out->pad_token = s->scr_padding_token;
    return 0;
}

/* Without fresh-gap authority a map may carry no pads at all. With `logical`,
 * removing the pads must also leave exactly that request. Pads are checked
 * here rather than by cache_reuse_verify_separator_map: the planner may insert
 * at any boundary (allow_all_insertions), not only beside a separator. */
static bool ds41_boundary_map_authorized(const ds41_scr_coordinates *map, const int *physical,
        const int *logical, const ds41_boundary_permissions *permissions,
        ds41_pad_budget *budget) {
    if (!ds41_coordinates_shape(map) || map->physical_tokens - map->count != permissions->logical_tokens ||
        (map->count && !physical))
        return false;
    bool matches = false;
    if (logical && (cache_reuse_verify_matches_logical((const int32_t *)physical,
            map->physical_tokens, map->pads, map->count, (const int32_t *)logical,
            permissions->logical_tokens, &matches, NULL, 0) != CACHE_REUSE_COMPLETE || !matches))
        return false;
    if (permissions->policy != DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1) return map->count == 0;
    for (uint32_t i = 0; i < map->count; ++i) {
        if (!ds41_pad_tick(budget) || physical[map->pads[i]] != permissions->pad_token) return false;
    }
    return true;
}
#endif
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
static int ds41_scr_planner_prepare_actual(
        ds4_session *s, const ds4_tokens *request, const ds41_scr_archive *a,
        const ds41_boundary_permissions *permissions, uint32_t protected_prefix,
        uint32_t pad_token, uint32_t context, ds41_even_receipt *out,
        ds4_tokens *physical, ds41_scr_coordinates *pending,
        uint64_t *host, ds41_pad_budget *budget, char *err, size_t errlen);
#endif

static int ds41_even_policy_impl(ds4_session *s, const ds4_tokens *request,
        uint32_t cap, uint32_t k, int pad_token, ds41_even_receipt *out,
        char *err, size_t errlen, uint64_t caller_host) {
    ds41_even_receipt local = {0}, *r = out ? out : &local;
    memset(r, 0, sizeof(*r));
    r->execution.configured_cap = cap; r->execution.configured_k = k;
    r->execution.reason = DS4_SCR_DENSE_INVALID;
    if (request && request->len > 0 && request->len < 1048576)
        r->logical_tokens = (uint32_t)request->len;
    bool fresh_space = s && s->scr_padding_policy == DS4_SCR_PADDING_PLANNER_FRESH_GAP_SPACE;
    if (fresh_space) {
        r->boundary_policy = DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1;
        pad_token = s->scr_padding_token;
    }
    int rc = 1;
    double began = now_sec();
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    ds4_tokens logical = {0}, physical = {0};
    ds41_scr_coordinates pending = {0};
    ds41_policy_owner owner = {0};
    ds41_pad_prefix_info prefix = {0};
    ds41_boundary_permissions permissions = {0};
    ds41_pad_budget budget = {.limit = DS41_SCR_PLAN_WORK_MAX, .session = s};
    ds4_session_progress_fn progress = NULL, display = NULL;
    void *progress_ud = NULL, *display_ud = NULL;
    bool active = false, publication_attempted = false;
    bool actual_plan = false;
    uint64_t host = 0, source_identity = 0, generation = 0, pad_bytes = 0;
    uint32_t source_start = 0, source_end = 0;
    uint32_t executed_pads = 0;
    if (!s || !s->engine || !request || !request->v || request->len <= 0 ||
        request->len >= s->ctx_size || s->ctx_size > 1048576 || !s->logits ||
        s->scr_policy_active || s->scr_multi_active || s->scr_dense_owner ||
        s->checkpoint.len < 0 || s->checkpoint.cap < s->checkpoint.len ||
        (s->checkpoint.len && !s->checkpoint.v) ||
        (s->scr_coordinates.count && request->v == s->checkpoint.v) ||
        cap < 128u || cap > DS4_SCR_ARCHIVE_MAX_CAPACITY || cap % 8u || cap > (uint32_t)s->ctx_size ||
        !k || k > DS4_SCR_MULTI_MAX_RANGES || s->scr_coordinate_serial == UINT64_MAX) {
        snprintf(err, errlen, "logical SCR: invalid input or ownership"); goto done;
    }
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *a = s->ds41_graph_ready ? g->archive : NULL;
    size_t old_heap;
    if (!s->ds41_graph_ready || !ds4_session_is_ds41(s) || s->distributed ||
        s->engine->backend != DS4_BACKEND_METAL || s->engine->tp.active ||
        g->tp_world != 1u || g->image_count || s->sync_image_count || s->checkpoint_image_count || g->imatrix ||
        g->encoder_resident || ds4_gpu_commands_active() ||
        (uint32_t)request->len >= g->ctx || cap > g->ctx ||
        s->scr_coordinates.identity > s->scr_coordinate_serial ||
        !ds41_coordinates_heap_bytes(s->scr_coordinates.capacity, &old_heap) ||
        old_heap != s->scr_coordinate_charge || old_heap > g->allocation_bytes ||
        old_heap > s->engine->ds41_session_bytes ||
        (a && (a->session != s || a->model != &s->engine->model || !a->sealed || a->retiring ||
        !a->generation || a->generation == UINT64_MAX || !a->source_generation ||
        a->source_generation > a->generation || !ds41_archive_shape(a, g->ctx) ||
        !a->tokens || !a->source || !a->stage))) {
        snprintf(err, errlen, "logical SCR: unsupported runtime/source"); goto done;
    }
    ds4_vocab *vocab = &s->engine->vocab;
    if (!vocab->token || pad_token < 0 || pad_token >= vocab->n_vocab ||
        vocab_token_is_generation_stop(vocab, pad_token) ||
        !vocab->token[pad_token].ptr || !vocab->token[pad_token].len ||
        vocab->token[pad_token].len > 16u || vocab_token_is_literal_special(vocab->token[pad_token])) {
        snprintf(err, errlen, "logical SCR: unsupported pad token"); goto done;
    }
    logical.len = logical.cap = request->len;
    r->logical_tokens = (uint32_t)logical.len;
    host = (uint64_t)logical.len * sizeof(int) + sizeof(local) + (out ? sizeof(*out) : 0u) +
        sizeof(logical) + sizeof(physical) + sizeof(pending) + sizeof(owner) +
        sizeof(prefix) + sizeof(budget) + sizeof(permissions);
    if (caller_host > UINT64_MAX - host) {
        snprintf(err, errlen, "logical SCR: caller reservation overflow"); goto done;
    }
    host += caller_host;
    if (!ds41_even_admit(s, host + vocab->token[pad_token].len + 1u, r)) {
        snprintf(err, errlen, "logical SCR: input reservation refused"); goto done;
    }
    size_t piece_length = 0;
    char *piece = ds4_token_text(s->engine, pad_token, &piece_length);
    bool valid_pad = piece && piece_length == 1u && piece[0] == (fresh_space ? ' ' : '\n');
    free(piece);
    if (!valid_pad || (fresh_space && (!s->scr_padding_selected || !ds41_padding_ordinary_space(s->engine, pad_token)))) {
        snprintf(err, errlen, "logical SCR: pad does not match selected startup policy"); goto done;
    }
    logical.v = malloc((size_t)logical.len * sizeof(int));
    if (!logical.v) { snprintf(err, errlen, "logical SCR: input allocation failed"); goto done; }
    memcpy(logical.v, request->v, (size_t)logical.len * sizeof(int));
    progress = s->progress; display = s->display_progress;
    progress_ud = s->progress_ud; display_ud = s->display_progress_ud;
    s->progress = s->display_progress = NULL;
    s->scr_policy_active = active = true;
    rc = ds41_even_prepare_boundaries(s, &logical, &permissions, &budget);
    if (rc) {
        snprintf(err, errlen, "logical SCR: fresh-gap padding verification refused");
        goto done;
    }
    r->boundary_policy = permissions.policy;
    rc = 1;
    if (ds41_pad_prefix(logical.v, r->logical_tokens, s->checkpoint.v,
            (uint32_t)s->checkpoint.len, &s->scr_coordinates, &prefix, &budget) != DS41_PAD_OK) {
        snprintf(err, errlen, "logical SCR: prefix validation refused"); goto done;
    }
    bool continuation = s->checkpoint.len > 0 && s->checkpoint_valid && g->valid &&
        g->pos == (uint32_t)s->checkpoint.len && prefix.common == prefix.live_logical;
    /* A short divergent follow-up (an edited or re-rendered last turn) cannot
     * continue the live graph, and the fresh route would recompute the whole
     * prompt from token 0; a sealed archive lets the planner cover it instead. */
    const bool cutoff = r->logical_tokens - prefix.common <= 128u && (!a || continuation);
    if (a) {
        generation = a->source_generation; source_start = a->start; source_end = a->end;
        source_identity = a->coordinates.identity;
        r->execution.source_start = source_start; r->execution.source_end = source_end;
        r->execution.source_generation = generation;
    }
    uint32_t context = g->ctx < (uint32_t)s->ctx_size ? g->ctx : (uint32_t)s->ctx_size;
    if (a && !continuation && !cutoff) {
        /* No protected live prefix: a planned request runs through
         * ds41_scr_multi_owned, which resets the live graph at token 0, so the
         * tokens shared with the live transcript are recomputed unless the
         * planner may cover them from the archive (from token 8, its minimum
         * prepared reuse start). */
        rc = ds41_scr_planner_prepare_actual(s, request, a, &permissions, 0u,
            (uint32_t)pad_token, context, r, &physical, &pending, &host, &budget, err, errlen);
        if (rc) goto done;
        actual_plan = true;
    } else {
        uint64_t bytes = ((uint64_t)r->logical_tokens + prefix.carried_pads) * sizeof(int) +
            (uint64_t)prefix.carried_pads * sizeof(uint32_t);
        if (!ds41_even_admit(s, host + bytes, r)) {
            snprintf(err, errlen, "logical SCR: construction reservation refused"); goto done;
        }
        host += bytes;
        rc = ds41_even_carry_prefix(&logical, s, &prefix, context, s->scr_coordinate_serial + 1u,
            &physical, &pending, &budget);
        if (rc) { snprintf(err, errlen, "logical SCR: physical prefix construction refused"); goto done; }
        r->plan = (ds41_even_plan){.planned = true, .verified = true,
            .physical_count = pending.physical_tokens, .pad_count = pending.count};
        if (!ds41_boundary_map_authorized(&pending, physical.v, logical.v, &permissions, &budget)) {
            rc = budget.cancelled ? DS4_SESSION_SYNC_INTERRUPTED : 1;
            snprintf(err, errlen, "logical SCR: artificial token outside certified separator");
            goto done;
        }
    }
    rc = 1; /* Every later refusal is an error until execution sets rc. */
    pad_bytes = (uint64_t)pending.capacity * sizeof(uint32_t);
    r->physical_known = true; r->physical_tokens = pending.physical_tokens;
    r->new_pads = r->plan.new_pads; r->carried_pads = pending.count - r->new_pads;
    r->map_identity = pending.identity; r->execution.requested_tokens = pending.physical_tokens;
    uint32_t bound = (uint32_t)s->checkpoint.len < pending.physical_tokens ?
        (uint32_t)s->checkpoint.len : pending.physical_tokens;
    while (r->execution.native_lcp < bound) {
        if (!ds41_pad_tick(&budget) || !ds41_pad_comparison(&budget)) {
            snprintf(err, errlen, "logical SCR: physical prefix budget refused"); goto done;
        }
        uint32_t i = r->execution.native_lcp;
        if (physical.v[i] != s->checkpoint.v[i]) break;
        ++r->execution.native_lcp;
    }
    /* A logical continuation must retain the actual physical frontier too. */
    if (continuation && !ds4_tokens_starts_with(&physical, &s->checkpoint)) {
        snprintf(err, errlen, "logical SCR: continuation changed physical prefix"); goto done;
    }
    if (a && (!ds41_multi_pin(s, a, generation, source_start, source_end) ||
        a->coordinates.identity != source_identity)) {
        snprintf(err, errlen, "logical SCR: source binding changed"); goto done;
    }
    free(logical.v); logical.v = NULL; host -= (uint64_t)logical.len * sizeof(int);
    owner = (ds41_policy_owner){.session = s, .identity = &owner, .full = &physical,
        .bytes = physical.v, .length = physical.len, .capacity = physical.cap,
        .live_host_bytes = host, .archive_capacity = cap, .active = true,
        .source_pads_are_content = actual_plan};
    s->scr_dense_owner = &owner;
    r->execution.outer_host_bytes = host;
    if (!ds41_policy_bind_coordinates(&owner, &pending, pad_bytes)) {
        snprintf(err, errlen, "logical SCR: map binding refused"); goto done;
    }
    if (ds4_session_cancelled(s)) { rc = DS4_SESSION_SYNC_INTERRUPTED; goto done; }
    if (continuation) {
        r->execution.route = DS4_SCR_DENSE_CONTINUE;
        r->execution.reason = r->execution.trigger = DS4_SCR_DENSE_LIVE;
        r->execution.retained_tokens = (uint32_t)s->checkpoint.len;
        uint64_t growth = s->checkpoint.cap < physical.len ?
            (uint64_t)(physical.len - s->checkpoint.cap) * sizeof(int) : 0u;
        if (!ds41_even_admit(s, host + growth, r)) {
            snprintf(err, errlen, "logical SCR: continuation reservation refused"); goto done;
        }
        if (growth) {
            int *v = realloc(s->checkpoint.v, (size_t)physical.len * sizeof(int));
            if (!v) { snprintf(err, errlen, "logical SCR: checkpoint allocation failed"); goto done; }
            s->checkpoint.v = v; s->checkpoint.cap = physical.len;
        }
        ds41_sync_frontier frontier = {0};
        s->scr_multi_active = true;
        rc = ds41_session_prefix(s, &physical, false, &frontier, err, errlen);
        s->scr_multi_active = false;
        r->target_mutated = frontier.mutated;
        r->execution.continuation_work_unknown = frontier.unknown;
        r->execution.continuation_heads = frontier.heads;
        if (frontier.completed >= r->execution.retained_tokens)
            r->execution.continuation_completed = frontier.completed - r->execution.retained_tokens;
    } else if (r->plan.range_count) {
        r->execution.route = DS4_SCR_DENSE_REUSE;
        r->execution.reason = r->execution.trigger = DS4_SCR_DENSE_HIT;
        rc = ds41_scr_multi_owned(s, &physical, r->plan.ranges,
            r->plan.range_count, &r->execution.multi, err, errlen, &owner);
        r->target_mutated = r->execution.multi.target_mutated;
    } else {
        uint32_t end = (uint32_t)physical.len / 8u * 8u;
        uint32_t capture = cap < end ? cap : end;
        r->execution.reason = r->execution.trigger = cutoff ? DS4_SCR_DENSE_CUTOFF :
            (a ? DS4_SCR_DENSE_MISS : DS4_SCR_DENSE_MISSING);
        if (!cutoff && capture >= 128u) {
            r->execution.route = DS4_SCR_DENSE_REFRESH;
            rc = ds41_scr_refresh_owned(s, &physical, end - capture, capture,
                &r->execution.refresh, err, errlen, &owner);
            r->target_mutated = r->execution.refresh.target_mutated;
        } else {
            r->execution.route = DS4_SCR_DENSE_FRESH;
            rc = ds41_scr_multi_owned(s, &physical, NULL, 0, &r->execution.multi, err, errlen, &owner);
            r->target_mutated = r->execution.multi.target_mutated;
        }
    }
    /* Refresh may have destroyed a. Never read the old source after execution. */
    if (ds4_session_cancelled(s)) rc = DS4_SESSION_SYNC_INTERRUPTED;
    if (!rc && r->execution.route == DS4_SCR_DENSE_REUSE) {
        bool reconciled = r->execution.multi.rebase_prepared ?
            ds41_even_executed_subset(r, &pending, &budget, &executed_pads) :
            (r->execution.multi.published &&
             r->execution.multi.committed_spans == r->plan.range_count &&
             r->execution.multi.committed_tokens == r->plan.physical_reused &&
             r->plan.imported_pads <= r->execution.multi.committed_tokens);
        if (!reconciled) {
            rc = 1; snprintf(err, errlen, "logical SCR: execution/plan reconciliation failed");
        } else if (!r->execution.multi.rebase_prepared) executed_pads = r->plan.imported_pads;
    }
    if (!rc) {
        publication_attempted = true;
        /* The input copy is gone; the pre-execution check proved the projection. */
        if (!ds41_boundary_map_authorized(&pending, physical.v, NULL, &permissions, &budget) ||
            !ds41_policy_publish_coordinates(&owner, &pending)) {
            rc = 1; snprintf(err, errlen, "logical SCR: authorized headed map publication failed");
        } else if ((rc = ds41_session_rebase_commit(s, &owner, &r->execution.multi))) {
            snprintf(err, errlen, "logical SCR: source rebase commit refused");
        } else {
            r->published = r->execution.complete_n = true;
            if (continuation) {
                r->execution.partial_valid = true;
                r->execution.continuation_head_unknown =
                    r->execution.continuation_completed && !r->execution.continuation_heads;
            }
            r->imported_pads = r->execution.multi.committed_tokens ? executed_pads : 0u;
            r->useful_committed = r->execution.multi.committed_tokens - r->imported_pads;
        }
    }
done:
    if (budget.cancelled && !r->execution.multi.rebase_committed) rc = DS4_SESSION_SYNC_INTERRUPTED;
    if (rc && active && r->execution.multi.rebase_prepared && s->ds41_graph.archive_pending)
        (void)ds41_session_rebase_abort(s, &r->execution.multi);
    if (rc && active && (r->target_mutated || publication_attempted)) {
        if (ds41_session_append_writer(s))
            ds41_session_append_failure(s, err, errlen, "logical destination publication failed");
        ds4_session_invalidate(s);
    }
    if (rc) {
        r->published = r->execution.complete_n = false;
        r->execution.multi.published = false;
        r->execution.multi.committed_spans = r->execution.multi.committed_tokens = 0;
        r->execution.refresh.complete_n = false;
        r->imported_pads = r->useful_committed = 0;
    }
    r->planning_work = budget.work; r->planning_comparisons = budget.comparisons;
    r->execution.peak_outer_host_bytes = r->peak_host_bytes;
    if (rc == DS4_SESSION_SYNC_INTERRUPTED) r->execution.reason = DS4_SCR_DENSE_CANCELLED;
    ds41_coordinates_free(&pending); free(logical.v); free(physical.v);
    if (active) {
        owner.active = false; s->scr_dense_owner = NULL; s->scr_policy_active = false;
        s->progress = progress; s->display_progress = display;
        s->progress_ud = progress_ud; s->display_progress_ud = display_ud;
    }
#else
    (void)s; (void)request; (void)pad_token; (void)caller_host;
    r->execution.reason = DS4_SCR_DENSE_UNSUPPORTED;
    snprintf(err, errlen, "logical SCR: Flash41 Metal required");
#endif
    r->execution.error_code = rc; r->execution.total_seconds = now_sec() - began;
    return rc;
}
static DS4_MAYBE_UNUSED int ds41_even_policy(ds4_session *s, const ds4_tokens *request,
        uint32_t cap, uint32_t k, int pad_token, ds41_even_receipt *out,
        char *err, size_t errlen) {
    return ds41_even_policy_impl(s, request, cap, k, pad_token, out, err, errlen, 0);
}
int ds4_session_scr_logical_policy(ds4_session *s, const ds4_tokens *request,
        uint32_t cap, uint32_t k, int pad_token, ds4_scr_logical_receipt *out,
        char *err, size_t errlen) {
    ds4_scr_logical_receipt local = {0}, *r = out ? out : &local;
    ds41_even_receipt inner;
    int rc = ds41_even_policy_impl(s, request, cap, k, pad_token, &inner, err, errlen,
        sizeof(local) + (out ? sizeof(*out) : 0u));
    *r = (ds4_scr_logical_receipt){.execution = inner.execution, .logical_match = inner.logical_match,
        .logical_tokens = inner.logical_tokens, .physical_tokens = inner.physical_tokens,
        .new_pads = inner.new_pads, .carried_pads = inner.carried_pads,
        .imported_pads = inner.imported_pads, .useful_committed = inner.useful_committed,
        .placed_useful = inner.plan.logical_reused,
        .planned_ranges = inner.plan.range_count, .map_identity = inner.map_identity,
        .boundary_policy = inner.boundary_policy,
        .planning_work = inner.planning_work, .planning_comparisons = inner.planning_comparisons,
        .peak_host_bytes = inner.peak_host_bytes, .logical_known = inner.logical_tokens != 0,
        .physical_known = inner.physical_known, .planned = inner.plan.planned,
        .published = inner.published, .target_mutated = inner.target_mutated,
        .planner_status = inner.planner_status, .planner_status_known = inner.planner_status_known,
        .planner_best_useful = inner.planner_best_useful,
        .planner_admissible_min = inner.planner_admissible_min,
        .planner_covered = inner.planner_covered, .planner_padding = inner.planner_padding,
        .planner_discovered_matches = inner.planner_discovered_matches,
        .planner_longest_match = inner.planner_longest_match,
        .planner_elapsed_ns = inner.planner_elapsed_ns,
        .planner_preparation_ns = inner.planner_preparation_ns,
        .planner_discovery_ns = inner.planner_discovery_ns,
        .planner_search_ns = inner.planner_search_ns,
        .planner_discovery_known = inner.planner_discovery_known,
        .planner_search_known = inner.planner_search_known,
        .planner_timing_known = inner.planner_timing_known};
    memcpy(r->physical_plan, inner.plan.ranges, sizeof(r->physical_plan));
    return rc;
}

#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
/* Actual-token planner bridge. The Rust core sees archive tokens as content,
 * including historical artificial token IDs; native ownership and coordinates
 * remain here. This is the one normal archive-plan path, not a runtime choice. */
#include "adapter/include/cache_reuse_utils.h"

#ifndef DS41_SCR_PLANNER_BUDGET_MS
#define DS41_SCR_PLANNER_BUDGET_MS 5000u
#endif
#define DS41_SCR_PLANNER_MODEL_BYTES (32ull * 1024u * 1024u)
#ifdef DS41_SCR_PLANNER_TEST_HOOKS
extern uint32_t ds41_scr_planner_test_variant_limit;
#define DS41_SCR_PLANNER_CONFIG_VARIANT_LIMIT ds41_scr_planner_test_variant_limit
#else
#define DS41_SCR_PLANNER_CONFIG_VARIANT_LIMIT 4096u
#endif

static int ds41_scr_planner_empty_target(ds4_session *s, const ds4_tokens *request,
        ds41_even_receipt *out, ds4_tokens *physical,
        ds41_scr_coordinates *pending, char *err, size_t errlen) {
    const uint32_t target_len = (uint32_t)request->len;
    const uint64_t identity = s->scr_coordinate_serial + 1u;
    if (!identity || (size_t)target_len > SIZE_MAX / sizeof(int)) {
        snprintf(err, errlen, "actual SCR: empty target size/identity refused");
        return 1;
    }
    physical->v = malloc((size_t)target_len * sizeof(int));
    if (!physical->v) {
        snprintf(err, errlen, "actual SCR: empty target allocation failed");
        return 1;
    }
    memcpy(physical->v, request->v, (size_t)target_len * sizeof(int));
    physical->len = physical->cap = (int)target_len;
    if (!ds41_coordinates_build(pending, physical->v, target_len, DS4_N_VOCAB,
            NULL, 0, identity)) {
        snprintf(err, errlen, "actual SCR: empty target map refused");
        return 1;
    }
    memset(&out->plan, 0, sizeof(out->plan));
    out->physical_known = true;
    out->physical_tokens = target_len;
    out->new_pads = 0;
    out->carried_pads = 0;
    out->imported_pads = 0;
    out->map_identity = identity;
    out->execution.requested_tokens = target_len;
    return 0;
}

/* The one planner configuration, shared by policy planning and pool scoring
 * so that a pool candidate is scored exactly as the policy would plan it. */
static void ds41_scr_planner_config(const ds4_session *s, uint32_t target_len,
        uint32_t context, uint32_t protected_prefix, uint32_t pad_token,
        bool insertions_possible, uint32_t k, cache_reuse_config *config) {
    if (protected_prefix > target_len) protected_prefix = target_len;
    /* Native rebase clips at the eight-aligned head minus two tokens. Express
     * that stricter end boundary in the planner's original-target suffix so a
     * selected range cannot rely on a post-selection native clip.
     *
     * Without insertions the native prepared length is exactly target_len, so
     * the tight suffix is safe. If insertions are permitted, p new pads make
     * the prepared length n=target_len+p; a range ending at original E can
     * finish at E+p once every pad precedes it. Native keeps at most
     * floor(n/8)*8-2 >= n-9, hence E=target_len-9 gives E+p <= n-9. Protecting
     * nine original trailing tokens is therefore safe for every pad count,
     * while retaining the unpadded geometry for plans that cannot pad. */
    uint32_t native_fresh_suffix = 1u;
    if (target_len >= 8u) {
        uint32_t native_ceiling = (target_len & ~7u) - 2u;
        if (native_ceiling < target_len) native_fresh_suffix = target_len - native_ceiling;
    }
    if (insertions_possible) native_fresh_suffix = target_len < 9u ? target_len : 9u;
    uint32_t max_padding = context - 1u - target_len;
    if (max_padding > 4096u) max_padding = 4096u;
    *config = (cache_reuse_config){
        .max_ranges = k ? k : DS4_SCR_MULTI_MAX_RANGES,
        .minimum_match_tokens = 128u,
        .minimum_physical_reuse_tokens = 128u,
        .source_start_alignment = 2u,
        .prepared_target_start_alignment = 2u,
        .reuse_count_alignment = 2u,
        .minimum_prepared_reuse_start = 8u,
        .protected_prefix_tokens = protected_prefix,
        .fresh_suffix_tokens = native_fresh_suffix,
        .max_trim_left = 1u,
        .max_trim_right = 1u,
        .pad_token = (int32_t)pad_token,
        .loss_basis_points = 1000u,
        .max_new_padding = max_padding,
        .max_prepared_tokens = context - 1u,
        .variant_limit = DS41_SCR_PLANNER_CONFIG_VARIANT_LIMIT,
        .budget_ms = s->scr_planner_budget_ms,
        .allow_all_insertions = insertions_possible,
        .allowed_insertions = NULL,
        .allowed_insertion_count = 0u,
        .pad_penalty_tokens = 0u,   /* 0 selects the planner's tuned default (16). */
        .range_penalty_tokens = 0u, /* 0 selects the tuned per-extra-range charge (32). */
    };
}

static int ds41_scr_planner_prepare_actual(
        ds4_session *s, const ds4_tokens *request, const ds41_scr_archive *a,
        const ds41_boundary_permissions *permissions, uint32_t protected_prefix,
        uint32_t pad_token, uint32_t context, ds41_even_receipt *out,
        ds4_tokens *physical, ds41_scr_coordinates *pending,
        uint64_t *host, ds41_pad_budget *budget, char *err, size_t errlen) {
    if (!s || !request || !a || !permissions || !out || !physical || !pending || !host ||
        !request->v || request->len <= 0 || request->len >= (int)context ||
        context == 0 || context > 1048576u || pad_token >= DS4_N_VOCAB ||
        !a->tokens || !a->sealed || a->retiring ||
        !ds41_archive_shape(a, context)) {
        snprintf(err, errlen, "actual SCR: invalid planner input");
        return 1;
    }
    const uint32_t target_len = (uint32_t)request->len;
    const uint32_t source_len = ds41_archive_count(a);
    if (!source_len || source_len > 1048576u || a->start > 1048576u - source_len) {
        snprintf(err, errlen, "actual SCR: invalid actual source interval");
        return 1;
    }
    cache_reuse_config config;
    ds41_scr_planner_config(s, target_len, context, protected_prefix, pad_token,
        permissions->policy == DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1,
        out->execution.configured_k, &config);
    const uint32_t max_padding = config.max_new_padding;
    if (config.max_ranges == 0 || config.max_ranges > DS4_SCR_MULTI_MAX_RANGES) {
        snprintf(err, errlen, "actual SCR: K is outside native planner bounds");
        return 1;
    }
    /* Conservative simultaneous host model: input copies, core modeled payload,
     * target materialization and new-pad coordinate storage. The core 32MiB
     * placement model is not an RSS claim. */
    uint64_t input_bytes = ((uint64_t)target_len + source_len) * sizeof(int);
    uint64_t allowed_bytes = config.allow_all_insertions ?
        ((uint64_t)target_len + 1u) * sizeof(uint32_t) : 0u;
    uint64_t material_bytes = ((uint64_t)target_len + max_padding) * sizeof(int) +
        (uint64_t)max_padding * sizeof(uint32_t);
    uint64_t planner_bytes = input_bytes + allowed_bytes + DS41_SCR_PLANNER_MODEL_BYTES +
        material_bytes + 4096u;
    if (planner_bytes > UINT64_MAX - *host ||
        !ds41_even_admit(s, *host + planner_bytes, out)) {
        snprintf(err, errlen, "actual SCR: simultaneous planner reservation refused");
        return 1;
    }
    *host += planner_bytes;
    cache_reuse_result result = {0};
    char ffi_error[256] = {0};
    int ffi_rc = cache_reuse_plan((const int32_t *)request->v, target_len,
        (const int32_t *)a->tokens, source_len, a->source_generation, &config,
        &result, ffi_error, sizeof(ffi_error), ds4_session_cancelled_cb, s);
    if (ffi_error[0]) snprintf(err, errlen, "actual SCR: %s", ffi_error);
    /* Preserve the planner's measured stage timing even when it returns an
     * error/no-plan result. Zero is not a measured timeout, so the native
     * receipt carries the FFI's explicit known flag. */
    out->planner_status = result.status;
    out->planner_status_known = true;
    out->planner_elapsed_ns = result.elapsed_ns;
    out->planner_preparation_ns = result.preparation_ns;
    out->planner_discovery_ns = result.discovery_ns;
    out->planner_search_ns = result.search_ns;
    out->planner_discovery_known = result.discovery_entered;
    out->planner_discovered_matches = result.discovered_match_count;
    out->planner_longest_match = result.longest_match_tokens;
    out->planner_search_known = result.search_entered;
    out->planner_timing_known = result.timing_known;
    if (result.status == CACHE_REUSE_ERR_CANCELLED) return DS4_SESSION_SYNC_INTERRUPTED;
    if (ffi_rc || (result.status != CACHE_REUSE_COMPLETE &&
        result.status != CACHE_REUSE_DEADLINE_PARTIAL &&
        result.status != CACHE_REUSE_VARIANT_LIMIT &&
        result.status != CACHE_REUSE_STATE_LIMIT)) {
        /* No admissible plan is a pre-mutation refusual: materialize an empty
         * target so the existing fresh/refresh routing can own the fallback.
         * This is not a second discovery implementation. */
        return ds41_scr_planner_empty_target(s, request, out, physical, pending, err, errlen);
    }
    if (!result.range_count || result.range_count > DS4_SCR_MULTI_MAX_RANGES ||
        result.insertion_count > CACHE_REUSE_MAX_INSERTIONS) {
        return ds41_scr_planner_empty_target(s, request, out, physical, pending, err, errlen);
    }
    const uint32_t padding = result.inserted_padding;
    if (padding > max_padding ||
        target_len + padding >= context || result.prepared_tokens != target_len + padding) {
        return ds41_scr_planner_empty_target(s, request, out, physical, pending, err, errlen);
    }
    if ((size_t)target_len + padding > SIZE_MAX / sizeof(int)) {
        snprintf(err, errlen, "actual SCR: materialization size overflow");
        return 1;
    }
    physical->v = malloc(((size_t)target_len + padding) * sizeof(int));
    uint32_t *pads = padding ? malloc((size_t)padding * sizeof(uint32_t)) : NULL;
    if (!physical->v || (padding && !pads)) {
        free(physical->v); physical->v = NULL;
        free(pads);
        snprintf(err, errlen, "actual SCR: materialization allocation failed");
        return 1;
    }
    physical->cap = physical->len = (int)(target_len + padding);
    /* The planner's recipe writes the prepared target and the physical index
     * of every inserted pad; reused source tokens must equal the target. */
    if (cache_reuse_materialize((const int32_t *)request->v, target_len,
            (const int32_t *)a->tokens, source_len, &result, (int32_t *)physical->v,
            target_len + padding, pads, padding, ffi_error, sizeof(ffi_error)) !=
            CACHE_REUSE_COMPLETE) {
        free(pads);
        snprintf(err, errlen, "actual SCR: materialization refused: %s", ffi_error);
        return 1;
    }
    const uint64_t identity = s->scr_coordinate_serial + 1u;
    if (!identity || !ds41_coordinates_build(pending, physical->v, (uint32_t)physical->len,
            DS4_N_VOCAB, pads, padding, identity)) {
        free(pads);
        snprintf(err, errlen, "actual SCR: target coordinate map refused");
        return 1;
    }
    free(pads);
    if (!ds41_boundary_map_authorized(pending, physical->v, request->v, permissions, budget)) {
        snprintf(err, errlen, "actual SCR: insertion outside permitted boundary");
        return 1;
    }
    memset(&out->plan, 0, sizeof(out->plan));
    out->plan.planned = out->plan.verified = true;
    out->plan.range_count = result.range_count;
    out->plan.pad_count = out->plan.new_pads = padding;
    out->plan.physical_count = (uint32_t)physical->len;
    for (uint32_t i = 0; i < result.range_count; ++i) {
        const cache_reuse_range *src = &result.ranges[i];
        if (src->count < 128u || src->count % 2u || src->source_start % 2u ||
            src->source_start > source_len || src->count > source_len - src->source_start ||
            src->prepared_start % 2u || src->prepared_start < 8u ||
            src->prepared_start > (uint32_t)physical->len ||
            src->count > (uint32_t)physical->len - src->prepared_start) {
            snprintf(err, errlen, "actual SCR: planner range violates native geometry src=%u len=%u n=%u prep=%u plen=%u", src->source_start, source_len, src->count, src->prepared_start, physical->len);
            return 1;
        }
        out->plan.ranges[i] = (ds4_scr_range){
            a->start + src->source_start, src->prepared_start, src->count};
        out->plan.physical_reused += src->count;
        out->plan.logical_reused += src->count;
        out->logical_match.plan[i] = (ds4_scr_range){
            a->start + src->source_start, src->destination_start, src->count};
    }
    out->logical_match.selected_ranges = result.range_count;
    out->planner_status = result.status;
    out->planner_best_useful = result.best_useful;
    out->planner_admissible_min = result.admissible_min;
    out->planner_covered = result.covered_tokens;
    out->planner_padding = result.inserted_padding;
    out->planner_elapsed_ns = result.elapsed_ns;
    out->planner_preparation_ns = result.preparation_ns;
    out->planner_discovery_ns = result.discovery_ns;
    out->planner_search_ns = result.search_ns;
    out->planner_discovery_known = result.discovery_entered;
    out->planner_search_known = result.search_entered;
    out->planner_status_known = true;
    out->planner_timing_known = result.timing_known;
    out->physical_known = true;
    out->physical_tokens = (uint32_t)physical->len;
    out->new_pads = padding;
    out->carried_pads = 0;
    out->imported_pads = 0;
    out->map_identity = identity;
    out->execution.requested_tokens = (uint32_t)physical->len;
    return 0;
}
#endif
#endif

/* ==== 6. Checkpoint Pool ================================================= */
#ifdef DS4_SCR_ENGINE_SESSION
/* Sealed archives persisted under one directory so a later process, or a
 * branch that a different conversation displaced, can splice from them again.
 * The catalog is metadata only; one entry at a time is restored into the
 * session's single hot archive slot. An entry is three files:
 *
 *   <name>.scr.meta    header, tokens[count] (int32), pads[pad_count] (uint32)
 *   <name>.scr.owners  per owner (ratio 2,2,2,1): count/ratio compressed rows
 *                      of 512 floats, then count/ratio index rows of 128
 *   <name>.scr.raw     ds41_raw_bytes(count) canonical SWA records
 *
 * Owner tags of a sealed prefix are implied (row r holds start + r*ratio) and
 * terminal SWA windows are rebuilt from raw records at staging time, so
 * neither is stored. Parts are written as .tmp and renamed with the meta file
 * last: a meta file names a complete entry. The name is a content hash, so an
 * unchanged archive is never written twice. */
#include <dirent.h>
#include <stddef.h>

#define DS41_POOL_MAGIC "DS4SCR01"
#define DS41_POOL_VERSION 1u
/* Scoring deadline per catalog entry; a deadline still returns the best plan
 * found so far, and the policy replans the winner with its full budget. */
#define DS41_POOL_SCORE_BUDGET_MS 250u

typedef struct {
    char magic[8];
    uint32_t version, context, capacity, start, end, logical_start, pad_count, reserved;
    uint64_t generation, coordinate_identity, created_ns, last_used_ns;
} ds41_pool_header;

typedef struct {
    ds41_pool_header h;
    char name[17];
    int *tokens;
    uint32_t *pads;
    uint64_t bytes;
} ds41_pool_entry;

#define DS41_POOL_HOT_RECORDS 16u
struct ds4_scr_pool {
    char directory[PATH_MAX];
    uint32_t context, capacity, count;
    uint64_t budget, bytes;
    ds41_pool_entry entry[DS4_SCR_POOL_MAX_ENTRIES];
    /* Per live archive (one per session of a batched server): the archive
     * state an entry already holds, as owner, generation and end. Most recent
     * last; a full table drops its oldest record, which only costs a
     * redundant save. */
    struct ds41_pool_hot_record {
        uint64_t owner, generation;
        uint32_t end;
        char name[17];
    } hot[DS41_POOL_HOT_RECORDS];
    uint32_t hot_count;
    /* Sessions of one engine share the pool; public calls serialize here. */
    pthread_mutex_t mu;
};

static uint64_t ds41_pool_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_REALTIME, &t) || t.tv_sec < 0) return 0;
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint64_t ds41_pool_meta_bytes(uint32_t count, uint32_t pads) {
    return sizeof(ds41_pool_header) + ((uint64_t)count + pads) * 4u;
}
static uint64_t ds41_pool_entry_bytes(uint32_t count, uint32_t pads) {
    return ds41_pool_meta_bytes(count, pads) + 6400ull * count + ds41_raw_bytes(count);
}
static bool ds41_pool_path(const ds4_scr_pool *p, const char *name, const char *part,
                           bool tmp, char path[PATH_MAX]) {
    int n = snprintf(path, PATH_MAX, "%s/%s.scr.%s%s", p->directory, name, part, tmp ? ".tmp" : "");
    return n > 0 && n < PATH_MAX;
}
/* FNV-1a over the published geometry, tokens and pads. */
static void ds41_pool_name(uint32_t start, uint32_t count, const int *tokens,
                           const uint32_t *pads, uint32_t pad_count, char name[17]) {
    uint64_t h = UINT64_C(14695981039346656037);
#define DS41_POOL_MIX(v) do { uint32_t x = (uint32_t)(v); \
    for (unsigned b = 0; b < 4; ++b) { h ^= (x >> (8u * b)) & 0xffu; h *= UINT64_C(1099511628211); } } while (0)
    DS41_POOL_MIX(start); DS41_POOL_MIX(count); DS41_POOL_MIX(pad_count);
    for (uint32_t i = 0; i < count; ++i) DS41_POOL_MIX(tokens[i]);
    for (uint32_t i = 0; i < pad_count; ++i) DS41_POOL_MIX(pads[i]);
#undef DS41_POOL_MIX
    snprintf(name, 17, "%016llx", (unsigned long long)h);
}
/* A pool file name: 16 hex digits, ".scr.", a part and an optional ".tmp". */
static bool ds41_pool_file(const char *file, char name[17], bool *meta, bool *tmp) {
    for (unsigned i = 0; i < 16; ++i)
        if (!((file[i] >= '0' && file[i] <= '9') || (file[i] >= 'a' && file[i] <= 'f'))) return false;
    const char *rest = file + 16;
    static const char *const parts[] = {".scr.meta", ".scr.owners", ".scr.raw"};
    for (unsigned i = 0; i < 3; ++i) {
        size_t n = strlen(parts[i]);
        if (strncmp(rest, parts[i], n)) continue;
        if (rest[n] && strcmp(rest + n, ".tmp")) return false;
        memcpy(name, file, 16); name[16] = '\0';
        *meta = i == 0; *tmp = rest[n] != '\0';
        return true;
    }
    return false;
}
static bool ds41_pool_io(int fd, void *data, uint64_t bytes, uint64_t offset, bool write) {
    for (uint64_t done = 0; done < bytes;) {
        ssize_t n = write ?
            pwrite(fd, (const unsigned char *)data + done, (size_t)(bytes - done), (off_t)(offset + done)) :
            pread(fd, (unsigned char *)data + done, (size_t)(bytes - done), (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; return false; }
        done += (uint64_t)n;
    }
    return true;
}
static bool ds41_pool_size(const char *path, uint64_t *bytes) {
    struct stat st;
    if (lstat(path, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) return false;
    *bytes = (uint64_t)st.st_size;
    return true;
}
/* The meta file goes first: without it the remaining parts are orphans. */
static void ds41_pool_unlink(const ds4_scr_pool *p, const char *name) {
    static const char *const parts[] = {"meta", "owners", "raw"};
    char path[PATH_MAX];
    for (unsigned i = 0; i < 3; ++i)
        if (ds41_pool_path(p, name, parts[i], false, path)) (void)unlink(path);
}
static void ds41_pool_hot_drop(ds4_scr_pool *p, uint32_t i) {
    memmove(&p->hot[i], &p->hot[i + 1], (size_t)(p->hot_count - i - 1u) * sizeof(p->hot[0]));
    memset(&p->hot[--p->hot_count], 0, sizeof(p->hot[0]));
}
static void ds41_pool_remove(ds4_scr_pool *p, uint32_t index) {
    ds41_pool_entry *e = &p->entry[index];
    ds41_pool_unlink(p, e->name);
    for (uint32_t i = 0; i < p->hot_count;)
        if (!strcmp(e->name, p->hot[i].name)) ds41_pool_hot_drop(p, i); else ++i;
    p->bytes -= e->bytes;
    free(e->tokens); free(e->pads);
    memmove(e, e + 1, (size_t)(p->count - index - 1u) * sizeof(*e));
    --p->count;
    memset(&p->entry[p->count], 0, sizeof(p->entry[p->count]));
}
static int ds41_pool_find(const ds4_scr_pool *p, const char *name) {
    for (uint32_t i = 0; i < p->count; ++i)
        if (!strcmp(p->entry[i].name, name)) return (int)i;
    return -1;
}
/* Recency is persisted in place so it survives a restart. */
static void ds41_pool_touch(ds4_scr_pool *p, uint32_t index) {
    ds41_pool_entry *e = &p->entry[index];
    e->h.last_used_ns = ds41_pool_now();
    char path[PATH_MAX];
    if (!ds41_pool_path(p, e->name, "meta", false, path)) return;
    int fd = open(path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return;
    (void)ds41_pool_io(fd, &e->h.last_used_ns, sizeof(e->h.last_used_ns),
                       offsetof(ds41_pool_header, last_used_ns), true);
    close(fd);
}
/* Least recently used eviction until `incoming` more bytes and one more entry
 * fit; `keep` (an entry index or -1) is never evicted. */
static bool ds41_pool_make_room(ds4_scr_pool *p, uint64_t incoming, int *keep) {
    while (p->count >= DS4_SCR_POOL_MAX_ENTRIES || p->bytes + incoming > p->budget) {
        int victim = -1;
        for (uint32_t i = 0; i < p->count; ++i)
            if ((int)i != *keep && (victim < 0 ||
                p->entry[i].h.last_used_ns < p->entry[victim].h.last_used_ns)) victim = (int)i;
        if (victim < 0) return false;
        ds41_pool_remove(p, (uint32_t)victim);
        if (*keep > victim) --*keep;
    }
    return true;
}
static bool ds41_pool_load(ds4_scr_pool *p, const char *name, ds41_pool_entry *e) {
    char path[PATH_MAX];
    memset(e, 0, sizeof(*e));
    if (!ds41_pool_path(p, name, "meta", false, path)) return false;
    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    ds41_pool_header *h = &e->h;
    uint64_t meta = 0, owners = 0, raw = 0;
    bool ok = ds41_pool_io(fd, h, sizeof(*h), 0, false) &&
        !memcmp(h->magic, DS41_POOL_MAGIC, 8) && h->version == DS41_POOL_VERSION &&
        h->context == p->context && h->capacity == p->capacity &&
        h->start % 8u == 0 && h->end >= h->start && h->end - h->start >= 128u &&
        (h->end - h->start) % 8u == 0 && h->end - h->start <= h->capacity &&
        h->start <= p->context - h->capacity && h->end <= p->context &&
        h->pad_count <= h->end - h->start && !(h->pad_count && !h->coordinate_identity);
    const uint32_t count = ok ? h->end - h->start : 0;
    ok = ok && ds41_pool_size(path, &meta) && meta == ds41_pool_meta_bytes(count, h->pad_count) &&
        ds41_pool_path(p, name, "owners", false, path) && ds41_pool_size(path, &owners) &&
        owners == 6400ull * count &&
        ds41_pool_path(p, name, "raw", false, path) && ds41_pool_size(path, &raw) &&
        raw == ds41_raw_bytes(count);
    if (ok) {
        e->tokens = malloc((size_t)count * sizeof(int));
        e->pads = h->pad_count ? malloc((size_t)h->pad_count * sizeof(uint32_t)) : NULL;
        ok = e->tokens && (!h->pad_count || e->pads) &&
            ds41_pool_io(fd, e->tokens, (uint64_t)count * 4u, sizeof(*h), false) &&
            (!h->pad_count || ds41_pool_io(fd, e->pads, (uint64_t)h->pad_count * 4u,
                                           sizeof(*h) + (uint64_t)count * 4u, false));
        for (uint32_t i = 0; ok && i < count; ++i)
            ok = e->tokens[i] >= 0 && (uint32_t)e->tokens[i] < DS4_N_VOCAB;
        for (uint32_t i = 0; ok && i < h->pad_count; ++i)
            ok = e->pads[i] < count && (!i || e->pads[i - 1u] < e->pads[i]);
    }
    close(fd);
    if (ok) {
        char check[17];
        ds41_pool_name(h->start, count, e->tokens, e->pads, h->pad_count, check);
        ok = !strcmp(check, name);
    }
    if (!ok) { free(e->tokens); free(e->pads); memset(e, 0, sizeof(*e)); return false; }
    memcpy(e->name, name, 17);
    e->bytes = meta + owners + raw;
    return true;
}

int ds4_scr_pool_open(ds4_scr_pool **out, const char *directory, uint32_t context,
                      uint32_t capacity, uint64_t budget_bytes, char *err, size_t errlen) {
    if (out) *out = NULL;
    if (!out || !directory || !*directory || capacity < 128u || capacity % 8u ||
        capacity > DS4_SCR_ARCHIVE_MAX_CAPACITY || capacity > context) {
        if (errlen) snprintf(err, errlen, "SCR pool: invalid directory or geometry");
        return 1;
    }
    ds4_scr_pool *p = calloc(1, sizeof(*p));
    if (!p) { if (errlen) snprintf(err, errlen, "SCR pool: allocation failed"); return 1; }
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    const int mu_rc = pthread_mutex_init(&p->mu, &attr);
    pthread_mutexattr_destroy(&attr);
    if (mu_rc) { if (errlen) snprintf(err, errlen, "SCR pool: mutex init failed"); free(p); return 1; }
    int n = snprintf(p->directory, sizeof(p->directory), "%s", directory);
    DIR *dir = n > 0 && (size_t)n < sizeof(p->directory) ? opendir(directory) : NULL;
    if (!dir) {
        if (errlen) snprintf(err, errlen, "SCR pool: cannot open %s: %s", directory, strerror(errno));
        pthread_mutex_destroy(&p->mu); free(p); return 1;
    }
    p->context = context; p->capacity = capacity;
    p->budget = budget_bytes ? budget_bytes : DS4_SCR_POOL_DEFAULT_BYTES;
    /* Load every complete entry of this geometry (the oldest beyond the bound
     * are evicted below). Pool temporaries are always stale here; parts with
     * no meta file are orphans. Another geometry's entries are left alone. */
    struct { char name[17]; bool meta; } *seen = NULL;
    size_t files = 0, cap = 0;
    struct dirent *d;
    while ((d = readdir(dir))) {
        char name[17], path[PATH_MAX]; bool meta, tmp;
        if (!ds41_pool_file(d->d_name, name, &meta, &tmp)) continue;
        if (tmp) {
            if (snprintf(path, sizeof(path), "%s/%s", p->directory, d->d_name) < PATH_MAX)
                (void)unlink(path);
            continue;
        }
        if (files == cap) {
            size_t next = cap ? cap * 2u : 64u;
            void *grown = realloc(seen, next * sizeof(*seen));
            if (!grown) break;
            seen = grown; cap = next;
        }
        memcpy(seen[files].name, name, 17);
        seen[files++].meta = meta;
        ds41_pool_entry e;
        if (!meta || !ds41_pool_load(p, name, &e)) continue;
        if (p->count == DS4_SCR_POOL_MAX_ENTRIES) {
            uint32_t oldest = 0;
            for (uint32_t i = 1; i < p->count; ++i)
                if (p->entry[i].h.last_used_ns < p->entry[oldest].h.last_used_ns) oldest = i;
            if (p->entry[oldest].h.last_used_ns > e.h.last_used_ns) {
                ds41_pool_unlink(p, e.name);
                free(e.tokens); free(e.pads); continue;
            }
            ds41_pool_remove(p, oldest);
        }
        p->entry[p->count++] = e;
        p->bytes += e.bytes;
    }
    closedir(dir);
    for (size_t i = 0; i < files; ++i) {
        bool owned = false;
        for (size_t k = 0; k < files && !owned; ++k)
            owned = seen[k].meta && !strcmp(seen[k].name, seen[i].name);
        if (!owned) ds41_pool_unlink(p, seen[i].name);
    }
    free(seen);
    int none = -1;
    (void)ds41_pool_make_room(p, 0, &none);
    *out = p;
    return 0;
}
void ds4_scr_pool_close(ds4_scr_pool *p) {
    if (!p) return;
    for (uint32_t i = 0; i < p->count; ++i) { free(p->entry[i].tokens); free(p->entry[i].pads); }
    pthread_mutex_destroy(&p->mu);
    free(p);
}
uint32_t ds4_scr_pool_entries(const ds4_scr_pool *p) {
    if (!p) return 0;
    ds4_scr_pool *m = (ds4_scr_pool *)p;
    pthread_mutex_lock(&m->mu);
    const uint32_t count = p->count;
    pthread_mutex_unlock(&m->mu);
    return count;
}

#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
/* Owner planes move through the archive's host I/O buffer in bounded chunks. */
static bool ds41_pool_owners(ds41_scr_archive *a, int fd, uint32_t count, bool save) {
    const uint64_t chunk = ds41_archive_io_bytes(a->cap);
    uint64_t file = 0;
    for (uint32_t o = 0; o < 4u; ++o) {
        const uint32_t ratio = o < 3u ? 2u : 1u, rows = count / ratio;
        const uint64_t base = ds41_archive_owner(a->cap, o);
        const uint64_t plane[2] = {base, base + (uint64_t)(a->cap / ratio) * 512u * 4u};
        const uint64_t bytes[2] = {(uint64_t)rows * 512u * 4u, (uint64_t)rows * 128u * 4u};
        for (unsigned k = 0; k < 2; ++k)
            for (uint64_t done = 0; done < bytes[k];) {
                const uint64_t n = bytes[k] - done < chunk ? bytes[k] - done : chunk;
                bool ok = save ?
                    ds4_gpu_tensor_read(a->source, plane[k] + done, a->raw_io, n) &&
                    ds41_pool_io(fd, a->raw_io, n, file, true) :
                    ds41_pool_io(fd, a->raw_io, n, file, false) &&
                    ds4_gpu_tensor_write(a->source, plane[k] + done, a->raw_io, n);
                if (!ok) return false;
                done += n; file += n;
            }
    }
    return true;
}
/* Clone the committed records when the pool shares the raw file's volume;
 * otherwise copy them through the I/O buffer. */
static bool ds41_pool_save_raw(ds41_scr_archive *a, const char *path, uint32_t count) {
    const uint64_t bytes = ds41_raw_bytes(count);
#ifdef __APPLE__
    (void)unlink(path);
    if (!fclonefileat(a->raw->fd, AT_FDCWD, path, 0)) {
        if (!truncate(path, (off_t)bytes)) return true;
        (void)unlink(path);
    }
#endif
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return false;
    bool ok = true;
    for (uint32_t record = 0; ok && record < count / 8u;) {
        uint32_t records = count / 8u - record;
        if (records > DS41_RAW_READ_RECORDS) records = DS41_RAW_READ_RECORDS;
        ok = ds41_raw_read(a->raw, record, records, a->raw_io, DS41_ARCHIVE_READ_BYTES, NULL, NULL) &&
            ds41_pool_io(fd, a->raw_io, (uint64_t)records * DS41_RAW_RECORD_BYTES,
                         (uint64_t)record * DS41_RAW_RECORD_BYTES, true);
        record += records;
    }
    if (close(fd)) ok = false;
    if (!ok) (void)unlink(path);
    return ok;
}
static int ds41_pool_hot_find(const ds4_scr_pool *p, uint64_t owner) {
    for (uint32_t i = 0; i < p->hot_count; ++i)
        if (p->hot[i].owner == owner) return (int)i;
    return -1;
}
/* The entry name this live archive's current state is saved as, or NULL. */
static const char *ds41_pool_hot(const ds4_scr_pool *p, const ds41_scr_archive *a) {
    const int i = a ? ds41_pool_hot_find(p, a->observation_id) : -1;
    return i >= 0 && a->source_generation == p->hot[i].generation &&
        a->end == p->hot[i].end ? p->hot[i].name : NULL;
}
static void ds41_pool_mark_hot(ds4_scr_pool *p, const ds41_scr_archive *a, const char *name) {
    const int found = ds41_pool_hot_find(p, a->observation_id);
    if (found >= 0) ds41_pool_hot_drop(p, (uint32_t)found);
    else if (p->hot_count == DS41_POOL_HOT_RECORDS) ds41_pool_hot_drop(p, 0);
    struct ds41_pool_hot_record *h = &p->hot[p->hot_count++];
    h->owner = a->observation_id; h->generation = a->source_generation;
    h->end = a->end; memcpy(h->name, name, 17);
}
/* Persist one published archive; the entry named `keep`, if any, is neither
 * superseded nor evicted. Returns 0 when the pool holds the archive. */
static int ds41_pool_save_archive(ds4_session *s, ds4_scr_pool *p, ds41_scr_archive *a,
                                  const char *keep, ds4_scr_pool_receipt *r) {
    if (ds41_pool_hot(p, a)) return 0;
    const uint32_t count = ds41_archive_count(a);
    if (!ds41_archive_owned_sealed(a, s) || count < 128u || a->cap != p->capacity ||
        a->start > p->context - a->cap || !ds41_archive_published_valid(a)) return 1;
    const ds41_scr_coordinates *m = &a->coordinates;
    const uint32_t pads = m->identity ? m->count : 0u;
    char name[17];
    ds41_pool_name(a->start, count, a->tokens, m->pads, pads, name);
    int index = ds41_pool_find(p, name);
    if (index >= 0) { ds41_pool_touch(p, (uint32_t)index); ds41_pool_mark_hot(p, a, name); return 0; }
    const uint64_t bytes = ds41_pool_entry_bytes(count, pads);
    if (bytes > p->budget) return 1;
    /* An entry whose tokens this archive extends is redundant planner content. */
    for (uint32_t i = 0; i < p->count;) {
        const ds41_pool_entry *e = &p->entry[i];
        if ((!keep || strcmp(e->name, keep)) &&
            e->h.start == a->start && e->h.end - e->h.start < count &&
            !memcmp(e->tokens, a->tokens, (size_t)(e->h.end - e->h.start) * sizeof(int)))
            ds41_pool_remove(p, i);
        else ++i;
    }
    int kept = keep ? ds41_pool_find(p, keep) : -1;
    if (!ds41_pool_make_room(p, bytes, &kept)) return 1;
    ds41_pool_entry e = {.h = {.version = DS41_POOL_VERSION, .context = p->context,
        .capacity = a->cap, .start = a->start, .end = a->end, .logical_start = a->logical_start,
        .pad_count = pads, .generation = a->source_generation,
        .coordinate_identity = m->identity, .created_ns = ds41_pool_now()}, .bytes = bytes};
    memcpy(e.h.magic, DS41_POOL_MAGIC, 8);
    e.h.last_used_ns = e.h.created_ns;
    memcpy(e.name, name, 17);
    e.tokens = malloc((size_t)count * sizeof(int));
    e.pads = pads ? malloc((size_t)pads * sizeof(uint32_t)) : NULL;
    char owners[PATH_MAX], raw[PATH_MAX], meta[PATH_MAX], final[PATH_MAX];
    bool ok = e.tokens && (!pads || e.pads) &&
        ds41_pool_path(p, name, "owners", true, owners) &&
        ds41_pool_path(p, name, "raw", true, raw) &&
        ds41_pool_path(p, name, "meta", true, meta) &&
        !ds4_gpu_commands_active() && ds4_gpu_synchronize();
    if (ok) {
        memcpy(e.tokens, a->tokens, (size_t)count * sizeof(int));
        if (pads) memcpy(e.pads, m->pads, (size_t)pads * sizeof(uint32_t));
        int fd = open(owners, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
        ok = fd >= 0 && ds41_pool_owners(a, fd, count, true);
        if (fd >= 0 && close(fd)) ok = false;
    }
    ok = ok && ds41_pool_save_raw(a, raw, count);
    if (ok) {
        int fd = open(meta, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
        ok = fd >= 0 && ds41_pool_io(fd, &e.h, sizeof(e.h), 0, true) &&
            ds41_pool_io(fd, e.tokens, (uint64_t)count * 4u, sizeof(e.h), true) &&
            (!pads || ds41_pool_io(fd, e.pads, (uint64_t)pads * 4u,
                                   sizeof(e.h) + (uint64_t)count * 4u, true)) &&
            !fsync(fd);
        if (fd >= 0 && close(fd)) ok = false;
    }
    ok = ok && ds41_pool_path(p, name, "owners", false, final) && !rename(owners, final) &&
        ds41_pool_path(p, name, "raw", false, final) && !rename(raw, final) &&
        ds41_pool_path(p, name, "meta", false, final) && !rename(meta, final);
    if (!ok) {
        (void)unlink(owners); (void)unlink(raw); (void)unlink(meta);
        ds41_pool_unlink(p, name);
        free(e.tokens); free(e.pads);
        return 1;
    }
    p->entry[p->count++] = e;
    p->bytes += bytes;
    ds41_pool_mark_hot(p, a, name);
    if (r) { r->saved = true; r->written_bytes += bytes; }
    return 0;
}
/* Replace an empty archive slot with entry `index`, admitted like prepare
 * but without its empty-graph rule: an archive outlives graph resets. */
static int ds41_pool_restore(ds4_session *s, ds4_scr_pool *p, uint32_t index) {
    ds41_gpu_graph *g = &s->ds41_graph;
    const ds41_pool_entry *e = &p->entry[index];
    const uint32_t count = e->h.end - e->h.start, cap = p->capacity;
    const uint64_t bytes = ds41_archive_bytes(cap);
    if (g->archive || g->archive_pending || !bytes || e->h.start > g->ctx ||
        cap > g->ctx - e->h.start || bytes > UINT64_MAX - g->allocation_bytes ||
        bytes > UINT64_MAX - s->engine->ds41_session_bytes ||
        !ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + bytes, false)) return 1;
    ds41_scr_archive *a = ds41_archive_alloc(e->h.start, cap, s, &s->engine->model);
    if (!a) return 1;
    a->cancelled = ds41_archive_raw_cancel;
    g->archive = a;
    g->allocation_bytes += bytes;
    s->engine->ds41_session_bytes += bytes;
    const char *directory = getenv("TMPDIR");
    if (!directory || !*directory) directory = "/tmp";
    char path[PATH_MAX];
    bool ok = ds41_pool_path(p, e->name, "raw", false, path);
    if (ok && !ds41_raw_adopt(a->raw, directory, path, cap, count)) {
        /* No clone: copy the records into an ordinary session store. */
        ok = a->raw->fd < 0 && !a->raw->cleanup_path[0];
        if (ok) *a->raw = (ds41_raw_store)DS41_RAW_STORE_INIT;
        int fd = ok ? open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW) : -1;
        ok = fd >= 0 && ds41_raw_open(a->raw, directory, cap, ds41_raw_bytes(cap));
        for (uint32_t done = 0; ok && done < count;) {
            uint32_t n = count - done < a->raw_arena_tokens ? count - done : a->raw_arena_tokens;
            ok = ds41_pool_io(fd, a->raw_io, ds41_raw_bytes(n), ds41_raw_bytes(done), false) &&
                ds41_raw_append(a->raw, a->raw_io, n, ds41_archive_raw_cancel, s);
            done += n;
        }
        ok = ok && ds41_raw_commit(a->raw, count);
        if (fd >= 0) close(fd);
    }
    if (ok && ds41_pool_path(p, e->name, "owners", false, path)) {
        int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        ok = fd >= 0 && ds41_pool_owners(a, fd, count, false) && ds4_gpu_synchronize();
        if (fd >= 0) close(fd);
    } else ok = false;
    if (ok) {
        for (uint32_t o = 0; o < 4u; ++o) {
            const uint32_t ratio = o < 3u ? 2u : 1u;
            for (uint32_t row = 0; row < count / ratio; ++row)
                a->tags[ds41_archive_tag(cap, o) + row] = a->start + row * ratio;
        }
        memcpy(a->tokens, e->tokens, (size_t)count * sizeof(int));
        if (e->h.coordinate_identity) {
            ds41_scr_coordinates pending = {0};
            ok = ds41_coordinates_build(&pending, a->tokens, count, DS4_N_VOCAB, e->pads,
                    e->h.pad_count, e->h.coordinate_identity) &&
                ds41_coordinates_replace_charged(&a->coordinates, &pending, a->tokens, count,
                    DS4_N_VOCAB, &g->allocation_bytes, &s->engine->ds41_session_bytes,
                    &a->allocation_bytes);
            ds41_coordinates_free(&pending);
        }
    }
    if (ok) {
        a->end = a->captured_end = a->raw_origin = a->start + count;
        a->logical_start = e->h.logical_start;
        a->source_generation = a->generation;
        a->sealed = true;
        ok = ds41_archive_published_valid(a);
    }
    if (!ok) {
        a->retiring = true;
        (void)ds4_session_scr_archive_release(s);
        return 1;
    }
    ds41_pool_mark_hot(p, a, e->name);
    return 0;
}
static uint32_t ds41_pool_score(ds4_session *s, const ds4_tokens *request, const int *tokens,
                                uint32_t count, uint64_t identity, const cache_reuse_config *config) {
    cache_reuse_result result = {0};
    char error[128];
    int rc = cache_reuse_plan((const int32_t *)request->v, (uint32_t)request->len,
        (const int32_t *)tokens, count, identity ? identity : 1u, config, &result,
        error, sizeof(error), ds4_session_cancelled_cb, s);
    bool planned = !rc && result.range_count && (result.status == CACHE_REUSE_COMPLETE ||
        result.status == CACHE_REUSE_DEADLINE_PARTIAL || result.status == CACHE_REUSE_VARIANT_LIMIT ||
        result.status == CACHE_REUSE_STATE_LIMIT);
    return planned ? result.covered_tokens : 0u;
}
/* No policy, capture or encoder of this session is in flight. */
static bool ds41_pool_session_quiet(const ds4_session *s) {
    const ds41_gpu_graph *g = &s->ds41_graph;
    return s->engine && s->ds41_graph_ready && ds4_session_is_ds41(s) && !s->distributed &&
        s->engine->backend == DS4_BACKEND_METAL && !s->scr_policy_active &&
        !s->scr_multi_active && !s->scr_dense_owner && g->tp_world == 1u && !g->image_count &&
        !g->archive_pending && !g->archive_capture_pending && !g->encoder_resident;
}
static bool ds41_pool_session_idle(const ds4_session *s) {
    return ds41_pool_session_quiet(s) && !ds4_gpu_commands_active();
}

/* The planner configuration the logical policy would use for `request`, or
 * false when the policy consults no archive: it continues the live
 * transcript, carries a short tail, or cannot plan. */
static bool ds41_pool_score_config(ds4_session *s, const ds4_tokens *request,
                                   uint32_t max_ranges, int pad_token, cache_reuse_config *config) {
    ds41_gpu_graph *g = &s->ds41_graph;
    const uint32_t n = (uint32_t)request->len;
    ds41_pad_budget budget = {.limit = DS41_SCR_PLAN_WORK_MAX, .session = s};
    ds41_pad_prefix_info prefix;
    if (ds41_pad_prefix(request->v, n, s->checkpoint.v, (uint32_t)s->checkpoint.len,
            &s->scr_coordinates, &prefix, &budget) != DS41_PAD_OK) return false;
    const bool continuation = s->checkpoint.len > 0 && s->checkpoint_valid && g->valid &&
        g->pos == (uint32_t)s->checkpoint.len && prefix.common == prefix.live_logical;
    ds41_boundary_permissions permissions;
    if (continuation || n - prefix.common <= 128u ||
        ds41_even_prepare_boundaries(s, request, &permissions, &budget)) return false;
    if (permissions.policy == DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1) pad_token = permissions.pad_token;
    const uint32_t context = g->ctx < (uint32_t)s->ctx_size ? g->ctx : (uint32_t)s->ctx_size;
    if (pad_token < 0 || n >= context) return false;
    /* Scored as the policy plans: no protected live prefix (see
     * ds41_even_policy_impl). */
    ds41_scr_planner_config(s, n, context, 0u, (uint32_t)pad_token,
        permissions.policy == DS4_SCR_BOUNDARIES_FRESH_GAP_SPACE_V1, max_ranges, config);
    if (config->budget_ms > DS41_POOL_SCORE_BUDGET_MS) config->budget_ms = DS41_POOL_SCORE_BUDGET_MS;
    return true;
}

static bool ds41_pool_hot_archive(const ds4_session *s) {
    const ds41_scr_archive *a = s->ds41_graph.archive;
    return a && ds41_archive_owned_sealed(a, s) && !a->retiring && ds41_archive_count(a) >= 128u;
}
#endif

static int ds41_pool_select_locked(ds4_session *s, ds4_scr_pool *p, const ds4_tokens *request,
                                   uint32_t max_ranges, int pad_token, ds4_scr_pool_receipt *receipt) {
    ds4_scr_pool_receipt local = {0}, *r = receipt ? receipt : &local;
    memset(r, 0, sizeof(*r));
    r->entries = ds4_scr_pool_entries(p);
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (!s || !p || !p->count || !request || !request->v || request->len <= 0 ||
        request->len >= s->ctx_size || !ds41_pool_session_idle(s)) return 0;
    const double began = now_sec();
    ds41_gpu_graph *g = &s->ds41_graph;
    const uint32_t n = (uint32_t)request->len;
    /* The policy continues a live transcript or carries a short tail without
     * consulting any archive; swapping it out would only cost time. */
    cache_reuse_config config;
    if (!ds41_pool_score_config(s, request, max_ranges, pad_token, &config)) return 0;
    r->selected = true;
    ds41_scr_archive *a = g->archive;
    const bool hot = ds41_pool_hot_archive(s);
    /* Each candidate is planned here under the scoring budget; the library
     * weighs that plan against its aligned common prefix and picks one. */
    cache_reuse_candidate hot_candidate = {0}, pool[DS4_SCR_POOL_MAX_ENTRIES];
    uint32_t pool_entry[DS4_SCR_POOL_MAX_ENTRIES];
    if (hot) {
        r->hot_covered = ds41_pool_score(s, request, a->tokens, ds41_archive_count(a),
                                         a->source_generation, &config);
        hot_candidate = (cache_reuse_candidate){(const int32_t *)a->tokens,
            ds41_archive_count(a), NULL, 0, r->hot_covered};
    }
    for (uint32_t i = 0; i < p->count && i < DS4_SCR_POOL_MAX_ENTRIES; ++i) {
        const ds41_pool_entry *e = &p->entry[i];
        const char *held = hot ? ds41_pool_hot(p, a) : NULL;
        if (held && !strcmp(e->name, held)) continue;
        if (ds4_session_cancelled(s)) break;
        const uint32_t count = e->h.end - e->h.start;
        pool_entry[r->candidates] = i;
        pool[r->candidates++] = (cache_reuse_candidate){(const int32_t *)e->tokens, count,
            NULL, 0, ds41_pool_score(s, request, e->tokens, count, e->h.generation, &config)};
    }
    const cache_reuse_candidate_policy policy = {.minimum_match_tokens = 128u,
        .alignment = 2u, .restore_penalty_tokens = 0u};
    cache_reuse_candidate_selection selection = {0};
    char winner[17] = "";
    if (!ds4_session_cancelled(s) &&
        cache_reuse_select_candidate((const int32_t *)request->v, n, hot ? &hot_candidate : NULL,
            pool, r->candidates, &policy, &selection, NULL, 0) == CACHE_REUSE_COMPLETE) {
        r->best_covered = selection.reusable_tokens;
        if (selection.action == CACHE_REUSE_ACTION_RESTORE_POOL &&
            selection.pool_index < r->candidates)
            memcpy(winner, p->entry[pool_entry[selection.pool_index]].name, 17);
    }
    int rc = 0;
    if (winner[0] && !ds4_session_cancelled(s)) {
        if (hot) (void)ds41_pool_save_archive(s, p, a, winner, r);
        /* Saving may have moved entries; the winner itself is kept. */
        int index = ds41_pool_find(p, winner);
        if (index >= 0 && !ds4_session_scr_archive_release(s)) {
            rc = ds41_pool_restore(s, p, (uint32_t)index);
            if (!rc) { r->restored = true; ds41_pool_touch(p, (uint32_t)index); }
        } else rc = index >= 0;
    }
    r->entries = p->count;
    r->seconds = now_sec() - began;
    return rc;
#else
    (void)s; (void)request; (void)max_ranges; (void)pad_token;
    return 0;
#endif
}

static int ds41_pool_save_locked(ds4_session *s, ds4_scr_pool *p, bool include_appends,
                                 ds4_scr_pool_receipt *receipt) {
    ds4_scr_pool_receipt local = {0}, *r = receipt ? receipt : &local;
    r->saved = false;
    r->entries = ds4_scr_pool_entries(p);
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (!s || !p || !ds41_pool_session_idle(s) || !s->ds41_graph.archive) return 0;
    /* Appends keep the owner; refresh and rebase seal a new one. */
    if (!include_appends && ds41_pool_hot_find(p, s->ds41_graph.archive->observation_id) >= 0)
        return 0;
    const double began = now_sec();
    int rc = ds41_pool_save_archive(s, p, s->ds41_graph.archive, NULL, r);
    r->entries = p->count;
    r->seconds += now_sec() - began;
    return rc;
#else
    (void)s; (void)include_appends;
    return 0;
#endif
}

uint32_t ds4_session_scr_archive_coverage(ds4_session *s, const ds4_tokens *request,
                                          uint32_t max_ranges, int pad_token) {
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    if (!s || !request || !request->v || request->len <= 0 || request->len >= s->ctx_size ||
        !ds41_pool_session_quiet(s) || !ds41_pool_hot_archive(s)) return 0;
    cache_reuse_config config;
    if (!ds41_pool_score_config(s, request, max_ranges, pad_token, &config)) return 0;
    const ds41_scr_archive *a = s->ds41_graph.archive;
    return ds41_pool_score(s, request, a->tokens, ds41_archive_count(a), a->source_generation,
                           &config);
#else
    (void)s; (void)request; (void)max_ranges; (void)pad_token;
    return 0;
#endif
}

/* Sessions of a batched server share one pool. */
int ds4_session_scr_pool_select(ds4_session *s, ds4_scr_pool *p, const ds4_tokens *request,
                                uint32_t max_ranges, int pad_token, ds4_scr_pool_receipt *receipt) {
    if (p) pthread_mutex_lock(&p->mu);
    const int rc = ds41_pool_select_locked(s, p, request, max_ranges, pad_token, receipt);
    if (p) pthread_mutex_unlock(&p->mu);
    return rc;
}
int ds4_session_scr_pool_save(ds4_session *s, ds4_scr_pool *p, bool include_appends,
                              ds4_scr_pool_receipt *receipt) {
    if (p) pthread_mutex_lock(&p->mu);
    const int rc = ds41_pool_save_locked(s, p, include_appends, receipt);
    if (p) pthread_mutex_unlock(&p->mu);
    return rc;
}
#endif

/* ==== 7. Session Guards ================================================== */
#if defined(DS4_SCR_ENGINE_GUARDS) && !defined(DS4_SCR_ENGINE_GUARDS_DONE)
#define DS4_SCR_ENGINE_GUARDS_DONE
/* Private session entry fences; sync intentionally has the narrower contract. */
/* V4.1 DSpark is the one speculative decoder SCR coexists with: its verifier
 * rows never alias the archive, its commit appends the accepted IDs to the
 * physical map and its rewind truncates the map. Other support models have
 * no such protocol and are refused before a capture owner or mapped
 * transcript can be attached to the target graph. */
static bool ds41_scr_dspark_compatible(const ds4_session *s) {
    return s && s->engine && ds4_session_is_ds41(s) &&
        s->engine->support_kind == DS4_SUPPORT_DSPARK && s->engine->dspark_weights.v41;
}
static bool ds41_scr_engine_compatible(const ds4_session *s) {
    return s && s->engine && (!s->engine->dspark || ds41_scr_dspark_compatible(s)) &&
        !s->engine->mtp_ready && !s->engine->glm_mtp;
}

static bool ds41_coordinates_restricted(const ds4_session *s) {
    return s && (s->scr_coordinates.identity || s->scr_coordinates.pads ||
        s->scr_coordinates.count || s->scr_coordinates.capacity ||
        s->scr_coordinates.physical_tokens || s->scr_coordinate_charge ||
        s->scr_policy_active || s->scr_multi_active || s->scr_dense_owner);
}

/* Public batch admission precedes commands, prefill and peer invalidation.
 * Capture/archive ownership can exist without a logical coordinate map. */
static bool ds41_session_batch_restricted(const ds4_session *s) {
    if (ds41_coordinates_restricted(s)) return true;
#if !defined(DS4_NO_GPU) && defined(DS4_HAS_DEEPSEEK41_GPU)
    return s && (s->ds41_graph.archive ||
        s->ds41_graph.archive_pending || s->ds41_graph.archive_capture_pending);
#else
    return false;
#endif
}

/* Decode on a published transcript: no SCR prefill owns the session, and an
 * unmapped transcript carries no coordinate state at all. */
static bool ds41_coordinates_eval_supported(const ds4_session *s) {
    if (!s || s->scr_policy_active || s->scr_multi_active || s->scr_dense_owner) return false;
    if (!s->scr_coordinates.identity) return !ds41_coordinates_restricted(s);
    return s->engine && !s->distributed && !s->engine->tp.active &&
        s->engine->backend == DS4_BACKEND_METAL && ds4_session_is_ds41(s);
}

static int ds41_session_sync_entry_guard(ds4_session *s, char *err, size_t errlen) {
    if (s && (s->scr_coordinates.identity || s->scr_policy_active ||
              s->scr_multi_active || s->scr_dense_owner)) {
        snprintf(err, errlen, "sync requires an unowned, unmapped transcript; use physical sync");
        return 1;
    }
    return 0;
}

static int ds41_session_multimodal_entry_guard(ds4_session *s, char *err, size_t errlen) {
    if (s && (s->scr_coordinates.identity || s->scr_policy_active ||
              s->scr_multi_active || s->scr_dense_owner)) {
        snprintf(err, errlen, "multimodal sync does not own mapped transcripts");
        return 1;
    }
    return 0;
}
#endif
