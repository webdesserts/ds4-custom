/* Batched SCR executor handoff while a prefill sweep yields between layers
 * (server_prefill_yield): waiting decode turns run, one each; a waiting
 * prefill of another slot does not; the yielding sweep holds the executor
 * again when the yield returns. Scheduler only: no engine call is made. */
#include "server_scr_harness.h"
#include <sched.h>

static server ys;
static server_slot yslots[3];
static pthread_mutex_t log_mu = PTHREAD_MUTEX_INITIALIZER;
static char order[16];
static int decode_turns = 1;

static void note(char c) {
    pthread_mutex_lock(&log_mu);
    size_t n = strlen(order);
    if (n + 1 < sizeof(order)) order[n] = c;
    pthread_mutex_unlock(&log_mu);
}

static void *decoder(void *arg) {
    server_slot *slot = arg;
    for (int i = 0; i < decode_turns; i++) {
        assert(server_decode_enter(&ys, slot));
        assert(ys.model_busy && !slot->decode_turn);
        note('D');
        server_prefill_leave(&ys);
    }
    return NULL;
}

static void *prefiller(void *arg) {
    server_slot *slot = arg;
    assert(server_prefill_enter(&ys, slot));
    note('Q');
    server_prefill_leave(&ys);
    return NULL;
}

/* Spin until `cond` holds under the model lock. */
#define WAIT_FOR(cond) do { \
    for (;;) { \
        pthread_mutex_lock(&ys.model_mu); \
        bool done_ = (cond); \
        pthread_mutex_unlock(&ys.model_mu); \
        if (done_) break; \
        sched_yield(); \
    } \
} while (0)

static void setup_server(void) {
    memset(&ys, 0, sizeof(ys));
    memset(yslots, 0, sizeof(yslots));
    memset(order, 0, sizeof(order));
    ys.batched_mode = true;
    ys.slots = yslots;
    ys.slot_count = 3;
    for (int i = 0; i < 3; i++) { yslots[i].id = i; yslots[i].srv = &ys; }
    pthread_mutex_init(&ys.model_mu, NULL);
    pthread_mutex_init(&ys.inference_mu, NULL);
    pthread_cond_init(&ys.model_cv, NULL);
}

static void handoff(void) {
    setup_server();
    /* Slot 0 runs a prefill; slot 1 waits to decode, slot 2 to prefill. */
    assert(server_prefill_enter(&ys, &yslots[0]));
    pthread_t d, q;
    decode_turns = 2;
    assert(!pthread_create(&d, NULL, decoder, &yslots[1]));
    assert(!pthread_create(&q, NULL, prefiller, &yslots[2]));
    WAIT_FOR(yslots[1].prefill_waiting && yslots[1].decode_turn && yslots[2].prefill_waiting);

    /* Layer boundary: the decode turn runs once; the prefill keeps waiting. */
    server_prefill_yield(&yslots[0]);
    assert(ys.model_busy && !ys.prefill_yielding && !strcmp(order, "D"));
    assert(pthread_mutex_trylock(&ys.inference_mu) == EBUSY);
    WAIT_FOR(yslots[1].prefill_waiting);
    /* Next layer: the slot's second decode turn. */
    server_prefill_yield(&yslots[0]);
    assert(ys.model_busy && !strcmp(order, "DD"));
    assert(!pthread_join(d, NULL));
    /* No decode waiting: the yield returns at once and keeps the executor. */
    server_prefill_yield(&yslots[0]);
    assert(ys.model_busy && !strcmp(order, "DD") && yslots[2].prefill_waiting);
    server_prefill_leave(&ys);
    assert(!pthread_join(q, NULL));
    assert(!strcmp(order, "DDQ") && !ys.model_busy);
    puts("prefill yield: decode turns run between layers, other prefills wait for the sweep");
}

/* Plain prefills (not decode turns) never enter during a yield. */
static void no_decode_waiting(void) {
    setup_server();
    assert(server_prefill_enter(&ys, &yslots[0]));
    pthread_t q;
    assert(!pthread_create(&q, NULL, prefiller, &yslots[2]));
    WAIT_FOR(yslots[2].prefill_waiting);
    server_prefill_yield(&yslots[0]);
    assert(ys.model_busy && !ys.prefill_yielding && order[0] == 0);
    server_prefill_leave(&ys);
    assert(!pthread_join(q, NULL) && !strcmp(order, "Q"));
}

int main(void) {
    handoff();
    no_decode_waiting();
    puts("test_server_prefill_yield: ok");
    return 0;
}
