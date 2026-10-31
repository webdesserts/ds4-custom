//! Slot routing for a batched server with resident sessions.
//!
//! A pure decision: given what each slot holds relative to one request, pick
//! the slot to run it on, or queue it. The caller measures the inputs (live
//! common prefix, protocol-bound reuse, idle time) and applies the result.
//!
//! Order of preference among available slots:
//! 1. **Owner**: the slot holding this conversation. A protocol binding
//!    (tool-call ids, remembered visible transcript) always qualifies;
//!    otherwise the reusable prefix must reach `min_match` and extend past
//!    `conversation_floor`, the prefix any other conversation can share (the
//!    system prompt and first user message). Most reuse wins.
//! 2. **Empty**: a slot holding nothing.
//! 3. **Stale**: idle at least `stale_after_seconds`; the longest idle first.
//! 4. **Cheapest**: the slot with the fewest live tokens (the least prefilled
//!    work lost), the longest idle on ties.
//!
//! Before those, the request **waits** when its owner is busy but finishing
//! (its response is delivered, post-request bookkeeping is running) and
//! would reuse at least `wait_gain` tokens more than the best available
//! owner: a client's immediate follow-up should not start a cold slot just
//! because the previous turn has not released its slot yet.
//!
//! With no available slot the request queues.

/// Per-slot state relative to one request.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotRouteInput {
    /// Idle and unassigned: the router may place work here.
    pub available: u8,
    /// A protocol-bound live tier matched (tool-call ids or visible transcript).
    pub bound: u8,
    /// Busy, but its request has finished and the slot is about to free.
    pub finishing: u8,
    pub live_tokens: u32,
    /// Tokens of the request this slot can reuse.
    pub reuse_tokens: u32,
    pub idle_seconds: u64,
}

/// The request and routing policy.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SlotRouteRequest {
    pub request_tokens: u32,
    pub conversation_floor: u32,
    pub min_match: u32,
    /// Extra reusable tokens that justify waiting for a finishing owner.
    pub wait_gain: u32,
    pub stale_after_seconds: u64,
}

pub const SLOT_ROUTE_QUEUE: u32 = 0;
pub const SLOT_ROUTE_OWNER: u32 = 1;
pub const SLOT_ROUTE_EMPTY: u32 = 2;
pub const SLOT_ROUTE_STALE: u32 = 3;
pub const SLOT_ROUTE_CHEAPEST: u32 = 4;
pub const SLOT_ROUTE_WAIT: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlotRouteDecision {
    /// Chosen slot index, or -1 to queue.
    pub slot: i32,
    pub reason: u32,
    pub reuse_tokens: u32,
    /// Live tokens of the chosen slot this request does not reuse.
    pub evicted_tokens: u32,
}

fn owns(req: &SlotRouteRequest, s: &SlotRouteInput) -> bool {
    s.bound != 0
        || (s.reuse_tokens >= req.min_match.max(1) && s.reuse_tokens > req.conversation_floor)
}

/// Route one request over `slots`.
pub fn route(req: &SlotRouteRequest, slots: &[SlotRouteInput]) -> SlotRouteDecision {
    let pick = |i: usize, reason: u32| {
        let s = &slots[i];
        let reuse = if reason == SLOT_ROUTE_OWNER { s.reuse_tokens } else { 0 };
        SlotRouteDecision {
            slot: i as i32,
            reason,
            reuse_tokens: reuse,
            evicted_tokens: s.live_tokens.saturating_sub(reuse),
        }
    };
    let available = || slots.iter().enumerate().filter(|(_, s)| s.available != 0);

    let best_available = available().filter(|(_, s)| owns(req, s)).map(|(_, s)| s.reuse_tokens).max().unwrap_or(0);
    if let Some(s) = slots
        .iter()
        .filter(|s| s.available == 0 && s.finishing != 0 && owns(req, s))
        .max_by_key(|s| s.reuse_tokens)
    {
        if s.reuse_tokens >= best_available.saturating_add(req.wait_gain.max(1)) {
            return SlotRouteDecision { slot: -1, reason: SLOT_ROUTE_WAIT, reuse_tokens: s.reuse_tokens, evicted_tokens: 0 };
        }
    }

    // Most reuse; then the least of the slot's own history lost; then lowest index.
    if let Some((i, _)) = available()
        .filter(|(_, s)| owns(req, s))
        .min_by_key(|(i, s)| {
            (std::cmp::Reverse(s.reuse_tokens), s.live_tokens.saturating_sub(s.reuse_tokens), *i)
        })
    {
        return pick(i, SLOT_ROUTE_OWNER);
    }
    if let Some((i, _)) = available().find(|(_, s)| s.live_tokens == 0) {
        return pick(i, SLOT_ROUTE_EMPTY);
    }
    if let Some((i, _)) = available()
        .filter(|(_, s)| s.idle_seconds >= req.stale_after_seconds)
        .min_by_key(|(i, s)| (std::cmp::Reverse(s.idle_seconds), *i))
    {
        return pick(i, SLOT_ROUTE_STALE);
    }
    if let Some((i, _)) =
        available().min_by_key(|(i, s)| (s.live_tokens, std::cmp::Reverse(s.idle_seconds), *i))
    {
        return pick(i, SLOT_ROUTE_CHEAPEST);
    }
    SlotRouteDecision { slot: -1, reason: SLOT_ROUTE_QUEUE, reuse_tokens: 0, evicted_tokens: 0 }
}

/// C entry point. Returns the chosen slot (or -1) and fills `out` when given.
///
/// # Safety
/// `req` must be valid; `slots` must point to `count` inputs (or be null when
/// `count` is zero); `out` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds4_slot_route(
    req: *const SlotRouteRequest,
    slots: *const SlotRouteInput,
    count: u32,
    out: *mut SlotRouteDecision,
) -> i32 {
    if req.is_null() || (slots.is_null() && count != 0) {
        return -1;
    }
    let req = unsafe { &*req };
    let slots: &[SlotRouteInput] =
        if count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(slots, count as usize) } };
    let d = route(req, slots);
    if !out.is_null() {
        unsafe { *out = d };
    }
    d.slot
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQ: SlotRouteRequest = SlotRouteRequest {
        request_tokens: 9000,
        conversation_floor: 6200,
        min_match: 128,
        wait_gain: 2048,
        stale_after_seconds: 3600,
    };

    fn slot(available: bool, live: u32, reuse: u32, idle: u64) -> SlotRouteInput {
        SlotRouteInput { available: available as u8, bound: 0, finishing: 0, live_tokens: live, reuse_tokens: reuse, idle_seconds: idle }
    }

    #[test]
    fn continued_conversation_returns_to_its_warm_slot() {
        let slots = [slot(true, 0, 0, 0), slot(true, 8100, 8000, 5), slot(true, 7000, 6200, 5)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason, d.reuse_tokens, d.evicted_tokens), (1, SLOT_ROUTE_OWNER, 8000, 100));
    }

    #[test]
    fn edited_prompt_returns_to_the_slot_sharing_the_conversation() {
        // Edited mid-conversation: the prefix up to the edit still passes the floor.
        let slots = [slot(true, 0, 0, 0), slot(true, 12000, 6900, 30), slot(true, 9000, 6200, 1)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason), (1, SLOT_ROUTE_OWNER));
    }

    #[test]
    fn new_caller_sharing_only_the_system_prompt_prefers_an_empty_slot() {
        let slots = [slot(true, 7000, 6200, 2), slot(true, 0, 0, 0), slot(true, 6400, 6200, 2)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason, d.evicted_tokens), (1, SLOT_ROUTE_EMPTY, 0));
    }

    #[test]
    fn all_busy_queues() {
        let slots = [slot(false, 8000, 8000, 0), slot(false, 0, 0, 0)];
        assert_eq!(route(&REQ, &slots).slot, -1);
        assert_eq!(route(&REQ, &[]).reason, SLOT_ROUTE_QUEUE);
    }

    #[test]
    fn busy_owner_is_not_chosen() {
        let slots = [slot(false, 8100, 8000, 0), slot(true, 0, 0, 0)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason), (1, SLOT_ROUTE_EMPTY));
    }

    #[test]
    fn immediate_follow_up_waits_for_its_finishing_slot() {
        let mut owner = slot(false, 11071, 10910, 0);
        owner.finishing = 1;
        let slots = [owner, slot(true, 0, 0, 0)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason, d.reuse_tokens), (-1, SLOT_ROUTE_WAIT, 10910));
        // A still-generating owner is not waited for.
        let slots = [slot(false, 11071, 10910, 0), slot(true, 0, 0, 0)];
        assert_eq!(route(&REQ, &slots).reason, SLOT_ROUTE_EMPTY);
        // Not worth waiting when an available owner reuses nearly as much.
        let slots = [owner, slot(true, 10000, 9500, 0)];
        assert_eq!(route(&REQ, &slots).reason, SLOT_ROUTE_OWNER);
        // A finishing slot of another conversation is no reason to wait.
        let mut other = slot(false, 7000, 6200, 0);
        other.finishing = 1;
        assert_eq!(route(&REQ, &[other, slot(true, 0, 0, 0)]).reason, SLOT_ROUTE_EMPTY);
    }

    #[test]
    fn eviction_takes_the_least_valuable_slot() {
        // No owner, no empty slot: a stale slot first, the longest idle of them.
        let slots = [slot(true, 9000, 6200, 4000), slot(true, 1000, 0, 60), slot(true, 50000, 0, 7200)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason, d.evicted_tokens), (2, SLOT_ROUTE_STALE, 50000));
        // Without stale slots: the fewest live tokens.
        let slots = [slot(true, 9000, 6200, 40), slot(true, 1000, 0, 60), slot(true, 50000, 0, 70)];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason, d.evicted_tokens), (1, SLOT_ROUTE_CHEAPEST, 1000));
        // Equal size: the longest idle.
        let slots = [slot(true, 1000, 0, 10), slot(true, 1000, 0, 60)];
        assert_eq!(route(&REQ, &slots).slot, 1);
    }

    #[test]
    fn protocol_binding_owns_without_a_token_prefix() {
        let mut bound = slot(true, 8000, 0, 3);
        bound.bound = 1;
        let slots = [slot(true, 0, 0, 0), bound];
        let d = route(&REQ, &slots);
        assert_eq!((d.slot, d.reason), (1, SLOT_ROUTE_OWNER));
    }

    #[test]
    fn short_match_and_floor_are_respected() {
        let req = SlotRouteRequest { conversation_floor: 0, ..REQ };
        // Below min_match: not an owner.
        let slots = [slot(true, 500, 100, 0), slot(true, 0, 0, 0)];
        assert_eq!(route(&req, &slots).reason, SLOT_ROUTE_EMPTY);
        // With no floor, any match of min_match owns.
        let slots = [slot(true, 500, 300, 0), slot(true, 0, 0, 0)];
        assert_eq!(route(&req, &slots).reason, SLOT_ROUTE_OWNER);
        // Reuse equal to the floor is shared content only.
        let slots = [slot(true, 6300, 6200, 0), slot(true, 0, 0, 0)];
        assert_eq!(route(&REQ, &slots).reason, SLOT_ROUTE_EMPTY);
    }

    #[test]
    fn ties_prefer_less_loss_then_lower_index() {
        let slots = [slot(true, 9000, 7000, 0), slot(true, 7100, 7000, 0), slot(true, 7100, 7000, 0)];
        assert_eq!(route(&REQ, &slots).slot, 1);
    }

    /// `cargo test --release -- --ignored --nocapture bench_route`
    #[test]
    #[ignore]
    fn bench_route() {
        let slots = [slot(true, 9000, 6200, 40), slot(false, 0, 0, 0), slot(true, 1000, 0, 60), slot(true, 50000, 0, 70)];
        let n = 1_000_000u32;
        let t = std::time::Instant::now();
        let mut acc = 0i64;
        for k in 0..n {
            let req = SlotRouteRequest { request_tokens: k, ..REQ };
            acc += route(std::hint::black_box(&req), std::hint::black_box(&slots)).slot as i64;
        }
        let ns = t.elapsed().as_nanos() as f64 / n as f64;
        println!("route over 4 slots: {ns:.1} ns/call ({acc})");
    }

    #[test]
    fn c_entry_point_matches_and_rejects_null() {
        let slots = [slot(true, 0, 0, 0), slot(true, 8100, 8000, 5)];
        let mut out = SlotRouteDecision::default();
        let got = unsafe { ds4_slot_route(&REQ, slots.as_ptr(), 2, &mut out) };
        assert_eq!(got, 1);
        assert_eq!(out, route(&REQ, &slots));
        assert_eq!(unsafe { ds4_slot_route(std::ptr::null(), slots.as_ptr(), 2, &mut out) }, -1);
        assert_eq!(unsafe { ds4_slot_route(&REQ, std::ptr::null(), 2, std::ptr::null_mut()) }, -1);
        assert_eq!(unsafe { ds4_slot_route(&REQ, std::ptr::null(), 0, std::ptr::null_mut()) }, -1);
    }
}
