//! Expected-gain draft length for V4.1 DSpark speculation.
//!
//! After the Markov chain proposes draft row `n - 1` with sigmoid confidence
//! `confidences[n - 1]`, drafting continues while
//!
//! ```text
//! P(rows 0..n all accepted) > value * marginal_ms[n - 1]
//! ```
//!
//! where P is the product of per-row acceptance estimates looked up by
//! confidence decile, and `marginal_ms[i]` is the verify time added by
//! verifying `i + 1` drafted rows instead of `i` (row 0: a 2-row verify
//! instead of a single-token decode). `value` is in tokens per millisecond.
//! Only the verify width depends on the rule, so greedy output does not.
//!
//! All arithmetic is `f32` in the order the server's original C rule used,
//! so decisions match it exactly (see `tests/test_dspark_policy.c`).

/// Confidence deciles in the acceptance table.
pub const BUCKETS: usize = 10;
/// Drafted rows with their own marginal cost; later rows reuse the last.
pub const ROWS: usize = 5;

/// Acceptance table, marginal verify costs and the value of a token.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainModel {
    /// P(row accepted | earlier rows accepted) by confidence decile.
    pub accept: [f32; BUCKETS],
    /// Milliseconds added by verifying draft row `i` (see the module docs).
    pub marginal_ms: [f32; ROWS],
    /// Tokens per millisecond one saved millisecond is worth.
    pub value: f32,
}

impl GainModel {
    /// Fitted with dspark-policy-sim on M-series Metal traces of six
    /// prompts at 384 tokens (9f1f59f), valued at 65 tok/s.
    pub const DEFAULT: GainModel = GainModel {
        accept: [
            0.0500, 0.0605, 0.3206, 0.3053, 0.4847, 0.5770, 0.6446, 0.7688, 0.9096, 0.9937,
        ],
        marginal_ms: [8.28, 6.84, 7.18, 8.83, 10.48],
        value: 65.0 / 1000.0,
    };
}

impl Default for GainModel {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Decile of a sigmoid confidence; out-of-range and NaN values clamp.
pub fn bucket(confidence: f32) -> usize {
    let b = (confidence * 10.0) as i32; // NaN -> 0, saturating
    b.clamp(0, BUCKETS as i32 - 1) as usize
}

/// Whether to keep drafting after the rows in `confidences` (the newest row
/// last). An empty slice continues: no row has been judged yet.
pub fn continue_drafting(model: &GainModel, confidences: &[f32]) -> bool {
    let Some(row) = confidences.len().checked_sub(1) else { return true };
    let mut survive = 1.0f32;
    for &c in confidences {
        survive *= model.accept[bucket(c)];
    }
    survive > model.value * model.marginal_ms[row.min(ROWS - 1)]
}

/// Rows a cycle verifies when the chain offers `confidences`: drafting stops
/// at the first row the rule rejects.
pub fn draft_rows(model: &GainModel, confidences: &[f32]) -> usize {
    (1..=confidences.len())
        .find(|&n| !continue_drafting(model, &confidences[..n]))
        .map_or(confidences.len(), |n| n - 1)
}

/// C ABI for [`continue_drafting`]. A null `model` uses [`GainModel::DEFAULT`];
/// `confidences` may be null only when `n` is 0.
///
/// # Safety
/// `model` is null or valid; `confidences` points at `n` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ds4_dspark_gain_continue(
    model: *const GainModel,
    confidences: *const f32,
    n: u32,
) -> bool {
    let model = unsafe { model.as_ref() }.unwrap_or(&GainModel::DEFAULT);
    let rows: &[f32] = if n == 0 || confidences.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(confidences, n as usize) }
    };
    continue_drafting(model, rows)
}

/// C ABI: the default model, valued at `value_tok_s` tokens per second, or
/// at the default value when `value_tok_s` is not positive.
#[unsafe(no_mangle)]
pub extern "C" fn ds4_dspark_gain_model_default(value_tok_s: f32) -> GainModel {
    let value = if value_tok_s > 0.0 { value_tok_s / 1000.0 } else { GainModel::DEFAULT.value };
    GainModel { value, ..GainModel::DEFAULT }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_clamp() {
        assert_eq!(bucket(-0.5), 0);
        assert_eq!(bucket(f32::NAN), 0);
        assert_eq!(bucket(0.0999), 0);
        assert_eq!(bucket(0.1), 1);
        assert_eq!(bucket(0.95), 9);
        assert_eq!(bucket(1.0), 9);
        assert_eq!(bucket(7.0), 9);
    }

    #[test]
    fn stops_when_survival_no_longer_pays_for_the_row() {
        let m = GainModel::DEFAULT;
        // 0.9937 > 0.065 * 8.28 = 0.538: a confident first row drafts.
        assert!(continue_drafting(&m, &[0.99]));
        // 0.4847 < 0.538: a 0.45 first row does not.
        assert!(!continue_drafting(&m, &[0.45]));
        // Survival compounds: 0.9096^2 * 0.7688 = 0.636 > 0.065 * 7.18 = 0.467.
        assert!(continue_drafting(&m, &[0.95, 0.95, 0.75]));
        assert!(!continue_drafting(&m, &[0.95, 0.95, 0.75, 0.55]));
        assert_eq!(draft_rows(&m, &[0.95, 0.95, 0.75, 0.55, 0.99]), 3);
        assert_eq!(draft_rows(&m, &[0.99; 5]), 5);
        assert_eq!(draft_rows(&m, &[]), 0);
    }

    #[test]
    fn value_scales_the_threshold() {
        let cheap = ds4_dspark_gain_model_default(10.0);
        assert!(continue_drafting(&cheap, &[0.45]));
        let dear = ds4_dspark_gain_model_default(200.0);
        assert!(!continue_drafting(&dear, &[0.95]));
        assert_eq!(cheap.accept, GainModel::DEFAULT.accept);
        assert_eq!(ds4_dspark_gain_model_default(0.0), GainModel::DEFAULT);
    }

    #[test]
    fn ffi_null_model_is_the_default() {
        let rows = [0.95f32, 0.45];
        let ffi = |m: *const GainModel, n| unsafe { ds4_dspark_gain_continue(m, rows.as_ptr(), n) };
        assert_eq!(ffi(std::ptr::null(), 1), continue_drafting(&GainModel::DEFAULT, &rows[..1]));
        assert_eq!(ffi(std::ptr::null(), 2), continue_drafting(&GainModel::DEFAULT, &rows));
        assert!(unsafe { ds4_dspark_gain_continue(std::ptr::null(), std::ptr::null(), 0) });
    }
}
