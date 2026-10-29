/* Real tiny Metal/file resources; simulated session/model/accounting boundaries. */
#include "../ds4_gpu.h"
#include <unistd.h>
#include <time.h>
static int fixture_close(int);
static int fixture_sync(void);
static int fixture_clock(clockid_t, struct timespec *);
#define close fixture_close
#define ds4_gpu_synchronize fixture_sync
#define clock_gettime fixture_clock
#include "../ds4.c"
#undef close
#undef ds4_gpu_synchronize
#undef clock_gettime
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s errno=%d\n", __LINE__, #x, errno); exit(3); } } while (0)
static int close_failure, sync_failure, clock_failure;
static int fixture_clock(clockid_t id, struct timespec *t) {
    if (clock_failure) { errno = ENOTSUP; return -1; }
    return clock_gettime(id, t);
}
static int fixture_close(int fd) {
    if (close_failure) { close_failure = 0; errno = EINTR; return -1; }
    return close(fd);
}
static int fixture_sync(void) {
    if (sync_failure) { sync_failure = 0; return 0; }
    return ds4_gpu_synchronize();
}
static ds4_session *session(ds4_engine *e, bool raw) {
    ds4_session *s = calloc(1, sizeof(*s));
    CHECK(s);
    s->engine = e; s->ctx_size = 4096; s->ds41_graph_ready = true;
    ds41_gpu_graph *g = &s->ds41_graph;
    g->ctx = 4096; g->tp_world = 1; g->valid = true; g->closed_io.known = true;
    g->allocation_bytes = 4096; e->ds41_session_bytes = 4096;
    g->table[0].fd = g->table[1].fd = -1;
    s->checkpoint.v = calloc(128, sizeof(int)); s->checkpoint.cap = 128;
    CHECK(s->checkpoint.v);
    if (raw) {
        CHECK(!ds4_session_scr_archive_prepare(s, 0, 128));
        ds41_scr_archive *a = g->archive;
        CHECK(a->raw && a->raw->fd >= 0 && a->raw->readable);
        s->checkpoint.len = 16; s->checkpoint.v[2] = 201; s->checkpoint_valid = true;
        uint32_t pad = 2;
        CHECK(ds41_coordinates_build(&s->scr_coordinates, s->checkpoint.v, 16,
                                      DS4_N_VOCAB, &pad, 1, 7));
        s->scr_coordinate_charge = sizeof(pad);
        g->allocation_bytes += sizeof(pad); e->ds41_session_bytes += sizeof(pad);
    }
    return s;
}
static ds41_scr_archive *pending(ds4_session *s) {
    ds41_gpu_graph *g = &s->ds41_graph;
    uint64_t bytes = ds41_archive_bytes(256u);
    CHECK(!g->archive_pending && bytes &&
        ds41_memory_admit(s->engine, s->engine->ds41_session_bytes + bytes, false));
    ds41_scr_archive *a = ds41_archive_alloc(0u, 256u, s, &s->engine->model);
    CHECK(a);
    g->archive_pending = a;
    g->allocation_bytes += bytes; s->engine->ds41_session_bytes += bytes;
    CHECK(ds41_raw_open(a->raw, "/tmp", 256u, ds41_raw_bytes(256u)));
    return a;
}
static void observe_raw_io(ds41_scr_archive *a) {
    void *record = calloc(1, DS41_RAW_RECORD_BYTES); CHECK(record);
    CHECK(ds41_raw_append(a->raw, record, 8, NULL, NULL) && ds41_raw_commit(a->raw, 8));
    CHECK(ds41_raw_read(a->raw, 0, 1, record, DS41_RAW_RECORD_BYTES, NULL, NULL));
    free(record);
}
static void closed_observations(ds4_engine *e) {
    ds4_session *s = session(e, false);
    CHECK(!ds4_session_scr_archive_prepare(s, 0, 128));
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *next = pending(s);
    observe_raw_io(g->archive); observe_raw_io(next);
    uint64_t old_id = g->archive->observation_id, extra_id = next->observation_id;
    CHECK(old_id && extra_id && old_id != extra_id);
    CHECK(!ds4_session_scr_archive_release(s));
    ds4_scr_archive_snapshot a = ds4_session_scr_archive_snapshot(s);
    ds4_scr_archive_snapshot b = ds4_session_scr_archive_snapshot(s);
    CHECK(a.observation_known && !a.present && !a.extra_owner_present && a.closed_io.known);
    CHECK(a.closed_io.owners == 2 && a.closed_io.read_bytes == 2 * DS41_RAW_RECORD_BYTES &&
        a.closed_io.written_bytes == 2 * DS41_RAW_RECORD_BYTES);
    CHECK(a.retired_io[0].owner_id == old_id && a.retired_io[1].owner_id == extra_id &&
        a.retired_io[0].retirement_complete && a.retired_io[1].retirement_complete &&
        !a.retired_io[0].descriptor_present && !a.retired_io[1].descriptor_present);
    CHECK(a.retired_io[0].allocation_known && a.retired_io[0].file_bytes == DS41_RAW_RECORD_BYTES);
    CHECK(!memcmp(&a.closed_io, &b.closed_io, sizeof(a.closed_io)));
    /* New source: the same graph retains discharged counters without readding
     * the two last-retirement diagnostic snapshots on each observation. */
    CHECK(!ds4_session_scr_archive_prepare(s, 0, 128));
    CHECK(g->archive->observation_id != old_id && g->archive->observation_id != extra_id);
    b = ds4_session_scr_archive_snapshot(s);
    CHECK(b.closed_io.owners == 2 && b.active_io.owner_id == g->archive->observation_id &&
        !b.active_io.read_bytes && !b.active_io.written_bytes && !b.active_io.allocation_known);
    CHECK(!ds4_session_dispose_checked(&s) && !s);
}
static void raw_clock_observations(ds4_engine *e) {
    ds4_session *s = session(e, false);
    CHECK(!ds4_session_scr_archive_prepare(s, 0, 128));
    ds41_scr_archive *a = s->ds41_graph.archive;
    observe_raw_io(a);
    clock_failure = 1; errno = ENOSPC;
    double began = ds41_archive_observation_time(a);
    CHECK(!began && a->raw_timing_unknown && errno == ENOSPC);
    CHECK(!ftruncate(a->raw->fd, 0));
    unsigned char record[DS41_RAW_RECORD_BYTES];
    CHECK(!ds41_raw_read(a->raw, 0, 1, record, sizeof(record), NULL, NULL) && errno == EIO);
    a->raw_read_seconds += ds41_archive_observation_time(a) - began;
    CHECK(errno == EIO && a->raw_timing_unknown);
    ds4_scr_archive_snapshot view = ds4_session_scr_archive_snapshot(s);
    CHECK(view.observation_known && !view.observation_time_known && view.active_io.raw_timing_unknown &&
        !view.active_io.allocation_time_known && view.active_io.read_bytes == DS41_RAW_RECORD_BYTES);
    clock_failure = 0;
    CHECK(!ds4_session_scr_archive_release(s));
    view = ds4_session_scr_archive_snapshot(s);
    CHECK(view.closed_io.owners == 1 && view.closed_io.timing_unknown &&
        view.retired_io[0].retirement_complete && view.retired_io[0].raw_timing_unknown);
    CHECK(!ds4_session_dispose_checked(&s) && !s);
}
static void two_owners(ds4_engine *e) {
    ds4_session *s = session(e, true), *original = s;
    ds41_gpu_graph *g = &s->ds41_graph;
    ds41_scr_archive *old = g->archive, *next = pending(s);
    const int old_fd = old->raw->fd, next_fd = next->raw->fd;
    const uint64_t charged = e->ds41_session_bytes;
    const uint64_t old_bytes = old->allocation_bytes, next_bytes = next->allocation_bytes;
    observe_raw_io(old); observe_raw_io(next);
    uint32_t *map = s->scr_coordinates.pads;
    CHECK(ds4_session_scr_archive_bytes(s) == old_bytes + next_bytes);
    CHECK(ds41_graph_capture_archive(g) == old);
    g->archive_capture_pending = true;
    CHECK(ds41_graph_capture_archive(g) == next);
    CHECK(ds41_session_archive_release_slot(s, false));
    CHECK(!old->retiring && e->ds41_session_bytes == charged);
    ds41_graph_reset(g);
    CHECK(!g->archive_capture_pending && ds41_graph_capture_archive(g) == old);
    char err[160];
    CHECK(ds41_session_append_before_eval(s, err, sizeof(err)));
    CHECK(s->checkpoint.len == 16 && s->scr_coordinates.pads == map);
    ds4_scr_archive_snapshot view = ds4_session_scr_archive_snapshot(s);
    CHECK(view.presence_known && view.present && view.extra_owner_known &&
        view.extra_owner_present && !view.extra_owner_retiring &&
        view.accounted_bytes == old_bytes && view.extra_owner_accounted_bytes == next_bytes);
    CHECK(view.observation_known && view.active_io.owner_id == old->observation_id &&
        view.extra_io.owner_id == next->observation_id && view.active_io.allocation_known &&
        view.active_io.read_bytes == DS41_RAW_RECORD_BYTES && view.extra_io.written_bytes == DS41_RAW_RECORD_BYTES);
    close_failure = 1;
    ds4_scr_retire_receipt receipt;
    CHECK(ds4_session_scr_archive_retire_once(s, &receipt) == DS4_SCR_RETIRE_FAILED_RETAINED);
    CHECK(receipt.attempted && receipt.charge_known &&
        receipt.charged_bytes_before == old_bytes + next_bytes &&
        receipt.charged_bytes_after == old_bytes + next_bytes);
    CHECK(s == original && g->archive == old && g->archive_pending == next &&
        next->retiring && !old->retiring && e->ds41_session_bytes == charged &&
        s->scr_coordinates.pads == map && fcntl(old_fd, F_GETFD) >= 0 && fcntl(next_fd, F_GETFD) >= 0);
    view = ds4_session_scr_archive_snapshot(s);
    CHECK(view.extra_owner_present && view.extra_owner_retiring &&
        view.extra_owner_accounted_bytes == next_bytes);
    CHECK(view.retired_io[1].owner_id == next->observation_id &&
        view.retired_io[1].retirement_attempted && !view.retired_io[1].retirement_complete &&
        view.retired_io[1].descriptor_present && view.retired_io[1].cleanup_state == DS41_RAW_CLEANUP_PENDING &&
        view.retired_io[1].read_bytes == DS41_RAW_RECORD_BYTES && !view.closed_io.owners);
    CHECK(ds4_session_scr_archive_retire_once(s, &receipt) == DS4_SCR_RETIRE_ALREADY_FENCED);
    CHECK(!receipt.attempted && receipt.charged_bytes_after == old_bytes + next_bytes);
    old->model = s;
    CHECK(ds4_session_dispose_checked(&s));
    CHECK(s == original && g->archive == old && !g->archive_pending &&
        e->ds41_session_bytes == charged - next_bytes && s->scr_coordinates.pads == map);
    CHECK(fcntl(next_fd, F_GETFD) < 0 && errno == EBADF);
    CHECK(g->closed_io.owners == 1 && g->closed_io.read_bytes == DS41_RAW_RECORD_BYTES &&
        g->retired_io[1].retirement_complete && !g->retired_io[1].descriptor_present);
    CHECK(fcntl(old_fd, F_GETFD) >= 0);
    old->model = &e->model;
    CHECK(!ds4_session_dispose_checked(&s) && !s && !e->ds41_session_bytes);
    CHECK(fcntl(old_fd, F_GETFD) < 0 && errno == EBADF);
    s = session(e, false); next = pending(s);
    const int only_fd = next->raw->fd;
    view = ds4_session_scr_archive_snapshot(s);
    CHECK(view.presence_known && !view.present && view.extra_owner_present &&
        ds4_session_scr_archive_bytes(s) == next->allocation_bytes);
    CHECK(!ds4_session_dispose_checked(&s) && !s && !e->ds41_session_bytes);
    CHECK(fcntl(only_fd, F_GETFD) < 0 && errno == EBADF);
}
int main(int argc, char **argv) {
    CHECK(ds4_gpu_init());
    ds4_engine e = {.backend = DS4_BACKEND_METAL};
    ds4_session *s = session(&e, true), *original = s;
    ds41_scr_archive *a = s->ds41_graph.archive;
    uint64_t charged = e.ds41_session_bytes;
    uint32_t *map = s->scr_coordinates.pads;
    int fd = a->raw->fd;
    if (argc == 2 && !strcmp(argv[1], "--fatal-pending")) {
        ds41_scr_archive *extra = pending(s);
        fprintf(stderr, "raw-dispose: isolated fatal pending fixture, no model, anonymous fd=%d\n", extra->raw->fd);
        close_failure = 1;
        ds4_session_free(s);
        return 99;
    }
    if (argc == 2 && !strcmp(argv[1], "--fatal")) {
        fprintf(stderr, "raw-dispose: isolated fatal fixture, no model, anonymous fd=%d\n", fd);
        close_failure = 1;
        ds4_session_free(s);
        return 99;
    }
    CHECK(argc == 1);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_session_dispose_checked(&s));
    CHECK(s == original && a->retiring && s->scr_coordinates.pads == map);
    CHECK(e.ds41_session_bytes == charged && fcntl(fd, F_GETFD) >= 0);
    CHECK(ds4_gpu_end_commands() && ds4_gpu_synchronize());
    sync_failure = 1;
    CHECK(ds4_session_dispose_checked(&s));
    CHECK(s == original && s->scr_coordinates.identity == 7 && e.ds41_session_bytes == charged);
    close_failure = 1;
    CHECK(ds4_session_dispose_checked(&s));
    CHECK(s == original && s->checkpoint.len == 16 && s->checkpoint.v[2] == 201);
    CHECK(s->scr_coordinates.pads == map && s->scr_coordinate_charge == sizeof(*map));
    CHECK(e.ds41_session_bytes == charged && a->raw->fd == fd);
    CHECK(a->raw->cleanup_state == DS41_RAW_CLEANUP_PENDING);
    a->model = s;
    CHECK(ds4_session_dispose_checked(&s) && s == original && e.ds41_session_bytes == charged);
    a->model = &e.model;
    CHECK(!ds4_session_dispose_checked(&s) && !s && !e.ds41_session_bytes);
    CHECK(fcntl(fd, F_GETFD) < 0 && errno == EBADF);
    CHECK(!ds4_session_dispose_checked(&s));
    CHECK(ds4_session_dispose_checked(NULL));
    s = session(&e, false);
    CHECK(!ds4_session_dispose_checked(&s) && !s && !e.ds41_session_bytes);
    g_ds4_shape = DS4_SHAPE_FLASH41;
    two_owners(&e);
    closed_observations(&e);
    raw_clock_observations(&e);
    CHECK(ds4_gpu_synchronize()); ds4_gpu_cleanup();
    puts("dispose: two owned slots, capture routing/reset, retained close failure, partial discharge and pending-only cleanup preserve maps/charges; no model");
    return 0;
}
