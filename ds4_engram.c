#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"
#define DS4_SCR_OBSERVATION_CLOCK
#include "ds4_scr.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

bool ds4_engram_layout_valid(const ds4_engram_layout *l) {
    if (!l || !l->token_map || !l->vocab_size ||
        !l->compressed_vocab_size || l->compressed_vocab_size > INT32_MAX ||
        l->pad_id >= l->compressed_vocab_size) return false;
    for (uint32_t i = 0; i < l->vocab_size; i++)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
        for (int i = 0; i < DS4_ENGRAM_NGRAM; i++) {
            uint64_t m = l->multipliers[layer][i];
            if (!(m & 1) || m > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t total = 0;
        for (int i = 0; i < DS4_ENGRAM_COLS; i++) {
            if (l->primes[layer][i] < 2) return false;
            total += l->primes[layer][i];
        }
        if (total != l->rows[layer]) return false;
    }
    return true;
}

void ds4_engram_history_reset(ds4_engram_history *h) {
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) h->tail[i] = DS4_ENGRAM_DEAD;
}

bool ds4_engram_hash(const ds4_engram_layout *l, ds4_engram_history *h,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (DS4_ENGRAM_LAYERS * DS4_ENGRAM_COLS * sizeof(*rows)))
        return false;
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) {
        if (h->tail[i] < DS4_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size) return false;
    }
    for (size_t i = 0; i < count; i++) {
        int32_t current = mask && !mask[i] ? DS4_ENGRAM_DEAD :
                          (int32_t)l->token_map[tokens[i]];
        uint32_t ids[DS4_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < DS4_ENGRAM_NGRAM; j++) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == DS4_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int j = 1; j < DS4_ENGRAM_NGRAM; j++) {
                hash ^= (uint64_t)ids[j] * l->multipliers[layer][j];
                for (int head = 0; head < DS4_ENGRAM_HEADS; head++) {
                    int col = (j - 1) * DS4_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = DS4_ENGRAM_NGRAM - 2; j > 0; j--) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

bool ds4_engram_table_open(ds4_engram_table *t, const char *path,
                           uint64_t offset, uint32_t rows) {
    if (!t) return false;
    *t = (ds4_engram_table){.fd = -1};
    uint64_t bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    if (!path || !rows || offset > INT64_MAX || bytes > INT64_MAX - offset) {
        errno = EINVAL;
        return false;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto fail;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto fail;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#elif defined(POSIX_FADV_RANDOM)
    /* Hashed rows are scattered across the table. Read-ahead fetches unused
     * pages and competes with the model's limited memory headroom. */
    (void)posix_fadvise(fd, (off_t)offset, (off_t)bytes, POSIX_FADV_RANDOM);
#endif
    *t = (ds4_engram_table){.fd = fd, .offset = offset, .rows = rows};
    return true;
fail: {
        int saved = errno;
        close(fd);
        errno = saved;
        return false;
    }
}

void ds4_engram_table_close(ds4_engram_table *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static uint64_t observation_begin(ds4_engram_observation *o) {
    uint64_t start = 0;
    if (o && !o->clock_failed && !ds4_observation_clock(&start)) o->clock_failed = true;
    return start;
}
static void observation_elapsed(ds4_engram_observation *o, uint64_t start, uint64_t *sum) {
    uint64_t end;
    if (!o || o->clock_failed) return;
    if (!ds4_observation_clock(&end) || end < start) o->clock_failed = true;
    else ds4_observation_add_u64(sum, end - start, &o->time_overflow);
}
static void observation_count(ds4_engram_observation *o, uint64_t *sum, uint64_t value) {
    if (o) ds4_observation_add_u64(sum, value, &o->counters_overflow);
}

static bool read_row(int fd, uint64_t offset, uint8_t row[DS4_ENGRAM_ROW_BYTES],
                      ds4_engram_observation *o) {
    uint64_t began = observation_begin(o);
    if (o) observation_count(o, &o->row_attempts, 1);
    size_t done = 0;
    while (done < DS4_ENGRAM_ROW_BYTES) {
        if (o) observation_count(o, &o->pread_calls, 1);
        ssize_t n = pread(fd, row + done, DS4_ENGRAM_ROW_BYTES - done,
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            observation_elapsed(o, began, o ? &o->row_read_nanoseconds : NULL);
            return false;
        }
        done += (size_t)n;
        if (o) observation_count(o, &o->read_bytes, (uint64_t)n);
    }
    if (o) observation_count(o, &o->rows_completed, 1);
    observation_elapsed(o, began, o ? &o->row_read_nanoseconds : NULL);
    return true;
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

/* All code/scale pairs fit in 128 KiB; this is a format lookup, not a cache
 * of model rows. Preserve the reference's F32 scaling and BF16 rounding. */
static uint16_t engram_bf16[256][256];
static pthread_once_t engram_bf16_once = PTHREAD_ONCE_INIT;

static void init_engram_bf16(void) {
    for (unsigned scale = 0; scale < 256; scale++) {
        for (unsigned code = 0; code < 256; code++) {
            uint16_t result = UINT16_MAX;
            if ((code & 127u) != 127u && scale != 255u) {
                float value = ldexpf(e4m3((uint8_t)code), (int)scale - 127);
                uint32_t bits;
                memcpy(&bits, &value, sizeof(bits));
                bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
                if ((bits & 0x7f800000u) != 0x7f800000u) result = (uint16_t)(bits >> 16);
            }
            engram_bf16[scale][code] = result;
        }
    }
}

static bool engram_read(const ds4_engram_table *t, const uint32_t *rows,
                         size_t count, float *out, ds4_engram_observation *o) {
    if (!t || t->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (DS4_ENGRAM_DIM * sizeof(*out))) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (rows[i] >= t->rows) {
            errno = EINVAL;
            return false;
        }
    }
    pthread_once(&engram_bf16_once, init_engram_bf16);
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    for (size_t i = 0; i < count; i++) {
        if (!read_row(t->fd, t->offset + (uint64_t)rows[i] * sizeof(raw), raw, o)) return false;
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            const uint16_t value = engram_bf16[raw[DS4_ENGRAM_DIM + j / 32]][raw[j]];
            if (value == UINT16_MAX) {
                errno = EDOM;
                return false;
            }
            const uint32_t bits = (uint32_t)value << 16;
            memcpy(out + i * DS4_ENGRAM_DIM + j, &bits, sizeof(bits));
        }
    }
    return true;
}

typedef struct {
    uint32_t row, output;
} engram_request;

static int request_order(const void *a, const void *b) {
    const engram_request *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

enum { ENGRAM_READERS = 16 };
#ifdef __APPLE__
enum { ENGRAM_PARALLEL_MIN_ROWS = 8 };
#else
/* Unlike dispatch's shared pool, this path creates threads for each batch. */
enum { ENGRAM_PARALLEL_MIN_ROWS = 256 };
#endif

typedef struct {
    const ds4_engram_table *table;
    const engram_request *request;
    float *out;
    size_t count, readers;
    int error[ENGRAM_READERS];
    bool observe;
    ds4_engram_observation observations[ENGRAM_READERS];
} engram_batch;

static void read_batch_part(void *context, size_t part) {
    engram_batch *batch = context;
    ds4_engram_observation *o = batch->observe ? &batch->observations[part] : NULL;
    if (o) o->enabled = true;
    const engram_request *request = batch->request;
    const size_t begin = batch->count * part / batch->readers;
    const size_t end = batch->count * (part + 1) / batch->readers;
    const float *previous = NULL;
    for (size_t i = begin; i < end; i++) {
        float *dst = batch->out + (size_t)request[i].output * DS4_ENGRAM_DIM;
        if (i > begin && request[i].row == request[i - 1].row) {
            memcpy(dst, previous, DS4_ENGRAM_DIM * sizeof(*dst));
        } else {
            if (!engram_read(batch->table, &request[i].row, 1, dst, o)) {
                batch->error[part] = errno ? errno : EIO;
                return;
            }
            previous = dst;
        }
    }
}

#ifndef __APPLE__
typedef struct {
    engram_batch *batch;
    size_t part;
} engram_reader;

static void *read_batch_thread(void *context) {
    engram_reader *reader = context;
    read_batch_part(reader->batch, reader->part);
    return NULL;
}
#endif

static bool engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                               size_t tokens, size_t stride, float *out,
                               ds4_engram_observation *o) {
    if (!t || t->fd < 0 || (tokens && (!rows || !out || stride < DS4_ENGRAM_COLS)) ||
        tokens > SIZE_MAX / (DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(*out)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(*rows) - DS4_ENGRAM_COLS) / stride)) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < tokens; i++) {
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            if (rows[i * stride + j] >= t->rows) {
                errno = EINVAL;
                return false;
            }
        }
    }
    if (!tokens) return true;
    enum { BATCH_TOKENS = 2048 };
    const size_t cap = tokens < BATCH_TOKENS ? tokens : BATCH_TOKENS;
    engram_request *request = malloc(cap * DS4_ENGRAM_COLS * sizeof(*request));
    if (!request) return false;
    bool ok = true;
    for (size_t start = 0; ok && start < tokens; start += cap) {
        const size_t n = tokens - start < cap ? tokens - start : cap;
        const size_t count = n * DS4_ENGRAM_COLS;
        for (size_t i = 0; i < count; i++) {
            request[i] = (engram_request){
                rows[(start + i / DS4_ENGRAM_COLS) * stride + i % DS4_ENGRAM_COLS],
                (uint32_t)i
            };
        }
        qsort(request, count, sizeof(*request), request_order);
        engram_batch batch = {.table = t, .request = request, .count = count,
            .out = out + start * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM, .readers = 1, .observe = o != NULL};
        /* Fixed concurrency hides random-read latency without caching the table.
         * Each worker owns disjoint output rows; all finish before GPU use. */
        if (count >= ENGRAM_PARALLEL_MIN_ROWS) {
            batch.readers = ENGRAM_READERS;
#ifdef __APPLE__
            dispatch_apply_f(batch.readers,
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &batch, read_batch_part);
#else
            pthread_t threads[ENGRAM_READERS - 1];
            engram_reader readers[ENGRAM_READERS - 1];
            size_t started = 0;
            for (size_t part = 1; part < batch.readers; part++) {
                readers[started] = (engram_reader){&batch, part};
                if (pthread_create(&threads[started], NULL, read_batch_thread,
                                   &readers[started])) break;
                started++;
            }
            read_batch_part(&batch, 0);
            /* Thread exhaustion only reduces concurrency, not correctness. */
            for (size_t part = started + 1; part < batch.readers; part++)
                read_batch_part(&batch, part);
            for (size_t part = 0; part < started; part++)
                if (pthread_join(threads[part], NULL)) abort();
#endif
        } else
        read_batch_part(&batch, 0);
        /* All worker results, including completed bytes before failure, are
         * private until this existing join. Error precedence below is unchanged. */
        if (o) for (size_t i = 0; i < batch.readers; i++)
            ds4_engram_observation_add(o, &batch.observations[i]);
        for (size_t i = 0; i < batch.readers; i++) {
            if (batch.error[i]) {
                errno = batch.error[i];
                ok = false;
                break;
            }
        }
    }
    int saved = errno;
    free(request);
    errno = saved;
    return ok;
}

enum { TOKEN_READERS = 4 };

typedef struct {
    pthread_t thread;
    ds4_engram_reader *reader;
    unsigned index;
} engram_read_worker;

struct ds4_engram_reader {
    engram_read_worker workers[TOKEN_READERS];
    unsigned threads;
    pthread_mutex_t mutex;
    pthread_cond_t work, done;
    bool active, shutdown;
    uint64_t generation;
    unsigned completed[2];
    int errors[2];
    ds4_engram_table tables[2];
    uint32_t ids[2][DS4_ENGRAM_COLS];
    float *out;
    ds4_engram_observation *observation;
    ds4_engram_observation result, parts[2][TOKEN_READERS];
    uint64_t observation_start;
    bool observation_finalized;
};

static void reader_finalize_observation(ds4_engram_reader *r, bool complete) {
    if (!r->result.enabled || r->observation_finalized) return;
    for (unsigned t = 0; t < 2; ++t)
        for (unsigned i = 0; i < TOKEN_READERS; ++i)
            ds4_engram_observation_add(&r->result, &r->parts[t][i]);
    r->result.failed_calls = !complete || r->errors[0] || r->errors[1];
    observation_elapsed(&r->result, r->observation_start, &r->result.call_wall_nanoseconds);
    r->observation_finalized = true;
}

/* Owner only, with workers completed or joined. Never publish on wait0. */
static void reader_publish_observation(ds4_engram_reader *r) {
    if (r->observation) { *r->observation = r->result; r->observation = NULL; }
}

static void *read_token_thread(void *context) {
    const engram_read_worker *worker = context;
    ds4_engram_reader *r = worker->reader;
    const unsigned index = worker->index;
    uint64_t seen = 0;
    pthread_mutex_lock(&r->mutex);
    for (;;) {
        while (r->generation == seen && !r->shutdown) pthread_cond_wait(&r->work, &r->mutex);
        if (r->shutdown) break;
        seen = r->generation;
        pthread_mutex_unlock(&r->mutex);
        for (unsigned table = 0; table < 2; table++) {
            const unsigned first = index * DS4_ENGRAM_COLS / TOKEN_READERS;
            const unsigned end = (index + 1u) * DS4_ENGRAM_COLS / TOKEN_READERS;
            ds4_engram_observation part = {.enabled = r->result.enabled};
            const bool ok = engram_read(&r->tables[table], r->ids[table] + first,
                end - first, r->out + (table * DS4_ENGRAM_COLS + first) * DS4_ENGRAM_DIM,
                part.enabled ? &part : NULL);
            const int error = ok ? 0 : errno ? errno : EIO;
            pthread_mutex_lock(&r->mutex);
            r->parts[table][index] = part;
            if (!r->errors[table]) r->errors[table] = error;
            if (++r->completed[table] == TOKEN_READERS) {
                if (table == 1) {
                    reader_finalize_observation(r, true);
                    r->active = false;
                }
                pthread_cond_broadcast(&r->done);
            }
            pthread_mutex_unlock(&r->mutex);
        }
        pthread_mutex_lock(&r->mutex);
    }
    pthread_mutex_unlock(&r->mutex);
    return NULL;
}

ds4_engram_reader *ds4_engram_reader_create(void) {
    ds4_engram_reader *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    int error = pthread_mutex_init(&r->mutex, NULL);
    if (error) goto fail;
    error = pthread_cond_init(&r->work, NULL);
    if (error) goto mutex;
    error = pthread_cond_init(&r->done, NULL);
    if (error) goto work;
    for (unsigned i = 0; i < TOKEN_READERS; i++) {
        r->workers[i].reader = r;
        r->workers[i].index = i;
        error = pthread_create(&r->workers[i].thread, NULL, read_token_thread, &r->workers[i]);
        if (error) break;
        r->threads++;
    }
    if (!error) return r;
    pthread_mutex_lock(&r->mutex);
    r->shutdown = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    for (unsigned i = 0; i < r->threads; i++)
        if (pthread_join(r->workers[i].thread, NULL)) abort();
    pthread_cond_destroy(&r->done);
work:
    pthread_cond_destroy(&r->work);
mutex:
    pthread_mutex_destroy(&r->mutex);
fail:
    free(r);
    errno = error;
    return NULL;
}

static bool reader_start(ds4_engram_reader *r,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out,
    ds4_engram_observation *observation, const ds4_engram_observation *initial, uint64_t start) {
    if (!r || !tables || !ids || !out) { errno = EINVAL; return false; }
    for (unsigned t = 0; t < 2; t++) {
        if (tables[t].fd < 0) { errno = EINVAL; return false; }
        for (unsigned col = 0; col < DS4_ENGRAM_COLS; col++)
            if (ids[t][col] >= tables[t].rows) { errno = EINVAL; return false; }
    }
    pthread_mutex_lock(&r->mutex);
    if (r->active || r->shutdown) {
        pthread_mutex_unlock(&r->mutex);
        errno = EBUSY;
        return false;
    }
    reader_publish_observation(r);
    r->observation = observation;
    r->result = initial ? *initial : (ds4_engram_observation){0};
    if (observation) *observation = r->result;
    r->observation_start = start;
    r->observation_finalized = false;
    memset(r->parts, 0, sizeof(r->parts));
    memcpy(r->tables, tables, sizeof(r->tables));
    memcpy(r->ids, ids, sizeof(r->ids));
    memset(r->errors, 0, sizeof(r->errors));
    r->out = out;
    memset(r->completed, 0, sizeof(r->completed));
    r->generation++;
    r->active = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    return true;
}

bool ds4_engram_reader_wait(ds4_engram_reader *r, unsigned table) {
    if (!r || table >= 2) { errno = EINVAL; return false; }
    pthread_mutex_lock(&r->mutex);
    if (!r->active && !r->completed[1]) {
        pthread_mutex_unlock(&r->mutex);
        errno = EINVAL;
        return false;
    }
    while (r->completed[table] < TOKEN_READERS) pthread_cond_wait(&r->done, &r->mutex);
    const int error = r->errors[table];
    if (table == 1) reader_publish_observation(r);
    pthread_mutex_unlock(&r->mutex);
    if (error) errno = error;
    return !error;
}

void ds4_engram_reader_free(ds4_engram_reader *r) {
    if (!r) return;
    pthread_mutex_lock(&r->mutex);
    r->shutdown = true;
    pthread_cond_broadcast(&r->work);
    pthread_mutex_unlock(&r->mutex);
    for (unsigned i = 0; i < r->threads; i++)
        if (pthread_join(r->workers[i].thread, NULL)) abort();
    /* Upstream free may stop workers that have not started this generation.
     * After join, report actual partial work rather than a measured zero/success. */
    reader_finalize_observation(r, r->completed[1] == TOKEN_READERS);
    reader_publish_observation(r);
    pthread_cond_destroy(&r->done);
    pthread_cond_destroy(&r->work);
    pthread_mutex_destroy(&r->mutex);
    free(r);
}

bool ds4_engram_reader_start_observed(ds4_engram_reader *r,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out,
    ds4_engram_observation *o) {
    if (!o) return reader_start(r, tables, ids, out, NULL, NULL, 0);
    ds4_engram_observation initial = {.enabled = true, .calls = 1};
    uint64_t start = observation_begin(&initial);
    bool ok = reader_start(r, tables, ids, out, o, &initial, start);
    if (!ok) { *o = initial; o->failed_calls = 1; observation_elapsed(o, start, &o->call_wall_nanoseconds); }
    return ok;
}

bool ds4_engram_reader_start(ds4_engram_reader *r,
    const ds4_engram_table tables[2], const uint32_t ids[2][DS4_ENGRAM_COLS], float *out) {
    return ds4_engram_reader_start_observed(r, tables, ids, out, NULL);
}

bool ds4_engram_read(const ds4_engram_table *t, const uint32_t *rows,
                     size_t count, float *out) {
    return engram_read(t, rows, count, out, NULL);
}
bool ds4_engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out) {
    return engram_read_batch(t, rows, tokens, stride, out, NULL);
}

bool ds4_engram_read_observed(const ds4_engram_table *t, const uint32_t *rows,
                              size_t count, float *out, ds4_engram_observation *o) {
    if (!o) return engram_read(t, rows, count, out, NULL);
    *o = (ds4_engram_observation){.enabled = true, .calls = 1};
    uint64_t began = observation_begin(o);
    bool ok = engram_read(t, rows, count, out, o);
    observation_elapsed(o, began, &o->call_wall_nanoseconds);
    o->failed_calls = !ok;
    return ok;
}
bool ds4_engram_read_batch_observed(const ds4_engram_table *t, const uint32_t *rows,
                                    size_t tokens, size_t stride, float *out,
                                    ds4_engram_observation *o) {
    if (!o) return engram_read_batch(t, rows, tokens, stride, out, NULL);
    *o = (ds4_engram_observation){.enabled = true, .calls = 1, .batch_calls = 1};
    uint64_t began = observation_begin(o);
    bool ok = engram_read_batch(t, rows, tokens, stride, out, o);
    observation_elapsed(o, began, &o->call_wall_nanoseconds);
    o->failed_calls = !ok;
    return ok;
}

typedef struct {
    ds4_engram_table table;
    uint32_t ids[DS4_ENGRAM_COLS];
    float *out;
    bool observe, ok;
    int error;
    ds4_engram_observation observation;
} engram_async_item;

struct ds4_engram_async {
    size_t tokens;
    engram_async_item items[2][DS4_ENGRAM_ASYNC_MAX_TOKENS];
    bool pending[2];
#ifdef __APPLE__
    dispatch_group_t group[2];
#else
    pthread_t thread[2];
#endif
};

static void engram_async_item_run(void *context) {
    engram_async_item *item = context;
    errno = 0;
    item->ok = ds4_engram_read_observed(&item->table, item->ids, DS4_ENGRAM_COLS,
        item->out, item->observe ? &item->observation : NULL);
    item->error = item->ok ? 0 : errno ? errno : EIO;
}

#ifndef __APPLE__
typedef struct {
    ds4_engram_async *reads;
    unsigned table;
} engram_async_table;

static void *engram_async_table_run(void *context) {
    ds4_engram_async *reads = ((engram_async_table *)context)->reads;
    const unsigned table = ((engram_async_table *)context)->table;
    free(context);
    for (size_t i = 0; i < reads->tokens; i++) engram_async_item_run(&reads->items[table][i]);
    return NULL;
}
#endif

ds4_engram_async *ds4_engram_async_start(size_t tokens,
    const ds4_engram_table *const tables[], const uint32_t ids[][2][DS4_ENGRAM_COLS],
    float *const out[][2], const bool observe[]) {
    if (!tokens || tokens > DS4_ENGRAM_ASYNC_MAX_TOKENS || !tables || !ids || !out) {
        errno = EINVAL;
        return NULL;
    }
    ds4_engram_async *reads = calloc(1, sizeof(*reads));
    if (!reads) return NULL;
    reads->tokens = tokens;
    for (unsigned t = 0; t < 2; t++) {
        for (size_t i = 0; i < tokens; i++) {
            engram_async_item *item = &reads->items[t][i];
            item->table = tables[i][t];
            memcpy(item->ids, ids[i][t], sizeof(item->ids));
            item->out = out[i][t];
            item->observe = observe && observe[i];
        }
    }
#ifdef __APPLE__
    /* Table 0 is queued first: layer 1 needs it long before layer 14 needs 1. */
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    for (unsigned t = 0; t < 2; t++) {
        reads->group[t] = dispatch_group_create();
        if (!reads->group[t]) {
            for (size_t i = 0; i < tokens; i++) engram_async_item_run(&reads->items[t][i]);
            continue;
        }
        for (size_t i = 0; i < tokens; i++)
            dispatch_group_async_f(reads->group[t], queue, &reads->items[t][i], engram_async_item_run);
        reads->pending[t] = true;
    }
#else
    for (unsigned t = 0; t < 2; t++) {
        engram_async_table *context = malloc(sizeof(*context));
        if (context) *context = (engram_async_table){reads, t};
        if (context && !pthread_create(&reads->thread[t], NULL, engram_async_table_run, context)) {
            reads->pending[t] = true;
            continue;
        }
        /* Thread exhaustion only removes overlap, not correctness. */
        free(context);
        for (size_t i = 0; i < tokens; i++) engram_async_item_run(&reads->items[t][i]);
    }
#endif
    return reads;
}

static void engram_async_join(ds4_engram_async *reads, unsigned table) {
    if (!reads->pending[table]) return;
#ifdef __APPLE__
    dispatch_group_wait(reads->group[table], DISPATCH_TIME_FOREVER);
#else
    if (pthread_join(reads->thread[table], NULL)) abort();
#endif
    reads->pending[table] = false;
}

bool ds4_engram_async_wait(ds4_engram_async *reads, unsigned table) {
    if (!reads || table >= 2) { errno = EINVAL; return false; }
    engram_async_join(reads, table);
    for (size_t i = 0; i < reads->tokens; i++) {
        if (!reads->items[table][i].ok) {
            errno = reads->items[table][i].error;
            return false;
        }
    }
    return true;
}

bool ds4_engram_async_finish(ds4_engram_async *reads, ds4_engram_observation *const sums[]) {
    if (!reads) return false;
    bool ok = true;
    for (unsigned t = 0; t < 2; t++) engram_async_join(reads, t);
    for (size_t i = 0; i < reads->tokens; i++) {
        for (unsigned t = 0; t < 2; t++) {
            const engram_async_item *item = &reads->items[t][i];
            ok = ok && item->ok;
            if (item->observe && sums && sums[i])
                ds4_engram_observation_add(sums[i], &item->observation);
        }
    }
#ifdef __APPLE__
    for (unsigned t = 0; t < 2; t++)
        if (reads->group[t]) dispatch_release(reads->group[t]);
#endif
    free(reads);
    return ok;
}
