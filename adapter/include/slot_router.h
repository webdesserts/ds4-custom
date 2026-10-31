/* Slot routing for a batched server (adapter/cache-reuse/src/slot_router.rs).
 * A pure decision over per-slot state relative to one request; see the Rust
 * module for the policy. */
#ifndef DS4_SLOT_ROUTER_H
#define DS4_SLOT_ROUTER_H

#include <stdint.h>

typedef struct {
    uint8_t available;     /* idle and unassigned */
    uint8_t bound;         /* protocol-bound live tier (tool-call ids, visible transcript) */
    uint8_t finishing;     /* busy, request finished, slot about to free */
    uint32_t live_tokens;
    uint32_t reuse_tokens; /* tokens of the request this slot can reuse */
    uint64_t idle_seconds;
} ds4_slot_route_input;

typedef struct {
    uint32_t request_tokens;
    uint32_t conversation_floor; /* prefix another conversation can share */
    uint32_t min_match;
    uint32_t wait_gain;    /* extra reuse that justifies waiting for a finishing owner */
    uint64_t stale_after_seconds;
} ds4_slot_route_request;

enum {
    DS4_SLOT_ROUTE_QUEUE = 0,
    DS4_SLOT_ROUTE_OWNER = 1,
    DS4_SLOT_ROUTE_EMPTY = 2,
    DS4_SLOT_ROUTE_STALE = 3,
    DS4_SLOT_ROUTE_CHEAPEST = 4,
    DS4_SLOT_ROUTE_WAIT = 5,
};

typedef struct {
    int32_t slot;          /* -1: queue or wait */
    uint32_t reason;
    uint32_t reuse_tokens;
    uint32_t evicted_tokens;
} ds4_slot_route_decision;

int32_t ds4_slot_route(const ds4_slot_route_request *req,
                       const ds4_slot_route_input *slots, uint32_t count,
                       ds4_slot_route_decision *out);

#endif
