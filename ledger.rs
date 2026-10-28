//! Module 5 — transparent state, supply invariant and tokenomics.
//!
//! The supply is fixed at `M = 1.5 * 10^15` atoms and is partitioned into exactly
//! ten disjoint, non-negative balance categories that sum to `M` in every reachable
//! state (eq. 7.15). Every operation moves value between categories by deltas that
//! sum to zero (eq. 7.9) and commits atomically or not at all (eq. 7.10).
//!
//! At Genesis every atom is unissued: 10% in `R_I` and 90% in `R_M` (eq. 7.19a).
//! Each block `k = h - h_G >= 1` opens with the decoupled primary issuance
//! ([`Ledger::begin_block`], eqs. 7.19b–7.19g): the linear initial release
//! `R_I -> U_initial` over `N_I` blocks and the exact Q64 exponential miner release
//! `R_M -> P_pending`. The InitialProtocolPool pays out only through the rolling
//! 720-block [`InitialValve`] (eqs. 7.19h–7.19j).
//!
//! Sub-ledgers refine categories without adding a balance:
//!
//! * job escrows inside `E_esc`, locked fees inside `R_l`, and the independent
//!   MatureFeeLots with their permanently segregated rounding dust inside `R_a` (§7.8);
//! * booked rewards and the not yet booked miner release inside `P_pending`;
//! * the [`Emission`] counters `I_k` and `S_k` behind `R_I` and `R_M`, and the
//!   [`InitialValve`] ring with `InitialNonce` behind `U_initial`.
//!
//! **Testnet status:** `U`, `B_bond` and `V_L2` are aggregate balances (no per-UTXO
//! ownership). Epoch length, maturation delay `L`, emission rate `η` and fee tariffs
//! are deployment parameters; [`LedgerParams::poc`] picks illustrative values.

use core::fmt;
use std::collections::BTreeMap;

use primitives::{Atoms, ShareWeight, TOTAL_SUPPLY};

// ---------------------------------------------------------------------------
// Supply partition and issuance parameters (Table 1, eqs. 7.14, 7.19)
// ---------------------------------------------------------------------------

/// Initial protocol allocation `I = M / 10 = 150,000,000,000,000` atoms, unissued in `R_I` at Genesis.
pub const INITIAL_ALLOCATION: Atoms = Atoms(150_000_000_000_000);
/// Miner allocation `R = 9M / 10 = 1,350,000,000,000,000` atoms, unissued in `R_M` at Genesis.
pub const MINER_ALLOCATION: Atoms = Atoms(1_350_000_000_000_000);
/// `N_I`: blocks over which `R_I` releases linearly into the InitialProtocolPool.
pub const INITIAL_RELEASE_BLOCKS: u64 = 1_051_200;
/// `H_M`: half-life, in blocks, of the miner release from `R_M`.
pub const MINER_HALF_LIFE_BLOCKS: u64 = 1_051_200;
/// `Q = 2^64`: fixed-point scale of the miner reserve state `S_k`.
pub const Q64: u128 = 1 << 64;
/// `a = floor(Q * 2^(-1/H_M))`: the exact per-block miner decay multiplier (eq. 7.19e).
pub const MINER_DECAY_MULTIPLIER: u128 = 18_446_731_910_177_940_463;
/// `W`: length of the rolling InitialValve window in blocks.
pub const INITIAL_VALVE_WINDOW: u64 = 720;
/// `L_I = I / 50 = 30,000 PLAU`: gross InitialProtocolPool outflow allowed per window.
pub const INITIAL_VALVE_LIMIT: Atoms = Atoms(INITIAL_ALLOCATION.0 / 50);
/// Nominal block interval `T_B` in seconds.
pub const BLOCK_INTERVAL_SECONDS: u64 = 120;
/// Blocks per nominal day, `86,400 s / T_B`.
pub const BLOCKS_PER_DAY: u64 = 720;

const _: () = assert!(INITIAL_ALLOCATION.0 * 10 == TOTAL_SUPPLY.0);
const _: () = assert!(INITIAL_ALLOCATION.0 + MINER_ALLOCATION.0 == TOTAL_SUPPLY.0);
const _: () = assert!(INITIAL_VALVE_LIMIT.0 == 3_000_000_000_000);
const _: () = assert!(MINER_DECAY_MULTIPLIER < Q64);
const _: () = assert!(86_400 / BLOCK_INTERVAL_SECONDS == BLOCKS_PER_DAY);

// ---------------------------------------------------------------------------
// State categories and deltas (eqs. 7.9, 7.15)
// ---------------------------------------------------------------------------

/// The ten disjoint transparent balance categories of eqs. (7.9) and (7.15).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateCategory {
    /// `U` — public outputs.
    Circulating,
    /// `U_initial` — the protected InitialProtocolPool.
    InitialProtocolPool,
    /// `E_esc` — job escrows.
    Escrow,
    /// `B_bond` — executor bonds.
    Bonds,
    /// `V_L2` — L2 settlement-slot vaults.
    L2Vaults,
    /// `R_a` — available reserve: MatureFeeLots and segregated dust.
    EmissionReserve,
    /// `R_l` — locked reserve: Base Fees and other fees awaiting maturation.
    LockedBaseFees,
    /// `P_pending` — segregated reward payables: booked rewards and the not yet
    /// booked miner release.
    PendingRewards,
    /// `R_I` — the unissued initial protocol reserve.
    UnissuedInitial,
    /// `R_M` — the unissued miner reserve.
    UnissuedMiners,
}

impl StateCategory {
    /// Number of categories.
    pub const COUNT: usize = 10;

    /// All categories in canonical order.
    pub const ALL: [StateCategory; StateCategory::COUNT] = [
        StateCategory::Circulating,
        StateCategory::InitialProtocolPool,
        StateCategory::Escrow,
        StateCategory::Bonds,
        StateCategory::L2Vaults,
        StateCategory::EmissionReserve,
        StateCategory::LockedBaseFees,
        StateCategory::PendingRewards,
        StateCategory::UnissuedInitial,
        StateCategory::UnissuedMiners,
    ];

    /// Position in [`StateCategory::ALL`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The symbol used in the specification.
    pub const fn symbol(self) -> &'static str {
        match self {
            StateCategory::Circulating => "U",
            StateCategory::InitialProtocolPool => "U_initial",
            StateCategory::Escrow => "E_esc",
            StateCategory::Bonds => "B_bond",
            StateCategory::L2Vaults => "V_L2",
            StateCategory::EmissionReserve => "R_a",
            StateCategory::LockedBaseFees => "R_l",
            StateCategory::PendingRewards => "P_pending",
            StateCategory::UnissuedInitial => "R_I",
            StateCategory::UnissuedMiners => "R_M",
        }
    }
}

/// Signed per-category change produced by one operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BalanceDelta([i128; StateCategory::COUNT]);

impl BalanceDelta {
    /// The empty delta.
    pub const ZERO: BalanceDelta = BalanceDelta([0; StateCategory::COUNT]);

    /// Moves `amount` from one category to another.
    pub fn transfer(from: StateCategory, to: StateCategory, amount: Atoms) -> Self {
        Self::ZERO
            .with(from, -i128::from(amount.get()))
            .with(to, i128::from(amount.get()))
    }

    /// Adds `change` to one category's entry.
    pub fn with(mut self, category: StateCategory, change: i128) -> Self {
        self.0[category.index()] += change;
        self
    }

    /// Change for one category.
    pub fn get(&self, category: StateCategory) -> i128 {
        self.0[category.index()]
    }

    /// Sum over all categories; zero for every admissible delta (eq. 7.9).
    pub fn sum(&self) -> i128 {
        self.0.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Exact emission rate `η = u / v` with `0 < u < v` and `gcd(u, v) = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmissionRate {
    numerator: u64,
    denominator: u64,
}

impl EmissionRate {
    /// Validates and builds `η = u / v`.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, LedgerError> {
        if numerator == 0 || numerator >= denominator || gcd(numerator, denominator) != 1 {
            return Err(LedgerError::InvalidEmissionRate {
                numerator,
                denominator,
            });
        }
        Ok(EmissionRate {
            numerator,
            denominator,
        })
    }

    /// `u`.
    pub fn numerator(&self) -> u64 {
        self.numerator
    }

    /// `v`.
    pub fn denominator(&self) -> u64 {
        self.denominator
    }

    /// Exact integer ceiling budget `ceil(u A / v) = floor((u A + v - 1) / v)` (eq. 7.21).
    /// Since `u < v`, the budget never exceeds the principal `A`.
    pub fn budget(&self, principal: Atoms) -> Atoms {
        let (u, v) = (u128::from(self.numerator), u128::from(self.denominator));
        Atoms((u * u128::from(principal.get())).div_ceil(v) as u64)
    }
}

/// Deployment parameters of the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerParams {
    /// Genesis height `h_G`.
    pub genesis_height: u64,
    /// Blocks per reward epoch.
    pub blocks_per_epoch: u64,
    /// Maturation delay `L` in epochs: a fee paid in epoch `e` matures in `e + L + 1`.
    pub maturation_delay: u64,
    /// Emission rate `η` applied to every eligible fee lot each epoch.
    pub emission_rate: EmissionRate,
}

impl LedgerParams {
    /// Illustrative PoC values: one-day epochs, `L = 2`, `η = 1/1000`.
    pub fn poc() -> Self {
        LedgerParams {
            genesis_height: 0,
            blocks_per_epoch: BLOCKS_PER_DAY,
            maturation_delay: 2,
            emission_rate: EmissionRate {
                numerator: 1,
                denominator: 1_000,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Decoupled primary issuance (eqs. 7.19b–7.19g)
// ---------------------------------------------------------------------------

/// `I_k = floor(I * min(k, N_I) / N_I)`: the cumulative initial release after block
/// `k` (eq. 7.19b). Exactly `I` from block `N_I` on.
pub fn initial_release_cumulative(k: u64) -> Atoms {
    let k = u128::from(k.min(INITIAL_RELEASE_BLOCKS));
    let released = u128::from(INITIAL_ALLOCATION.get()) * k / u128::from(INITIAL_RELEASE_BLOCKS);
    Atoms(released as u64)
}

/// `S_k` from `S_{k-1}` (eqs. 7.19e–7.19f): `Ŝ_k = floor(a S_{k-1} / Q)`, evaluated
/// exactly in 128-bit arithmetic by splitting `S = uQ + v` into
/// `u a + floor(v a / Q)`. Once `Ŝ_k < Q` the final atom is released and `S_k = 0`.
pub fn miner_decay_step(state: u128) -> u128 {
    let (u, v) = (state >> 64, state & (Q64 - 1));
    // u a < 2^64 a and floor(v a / Q) < a, so the sum stays below 2^128.
    let next = u * MINER_DECAY_MULTIPLIER + ((v * MINER_DECAY_MULTIPLIER) >> 64);
    if next < Q64 { 0 } else { next }
}

/// The primary release of one block `k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimaryRelease {
    /// `k = h - h_G`.
    pub block: u64,
    /// `i_k`, released `R_I -> U_initial`.
    pub initial: Atoms,
    /// `m_k`, released `R_M -> P_pending`.
    pub miner: Atoms,
}

/// Integer release bookkeeping for the two decoupled primary releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Emission {
    block: u64,
    initial_released: Atoms,
    miner_state: u128,
}

impl Emission {
    /// `k = 0`: nothing released, `S_0 = R Q`.
    pub const fn genesis() -> Self {
        Emission {
            block: 0,
            initial_released: Atoms::ZERO,
            miner_state: (MINER_ALLOCATION.0 as u128) << 64,
        }
    }

    /// `k` of the last primary release.
    pub const fn block(&self) -> u64 {
        self.block
    }

    /// `I_k`: cumulative release into the InitialProtocolPool.
    pub const fn initial_released(&self) -> Atoms {
        self.initial_released
    }

    /// `S_k`: the Q64 fixed-point miner reserve state.
    pub const fn miner_state(&self) -> u128 {
        self.miner_state
    }

    /// `R_I(k) = I - I_k`.
    pub const fn initial_reserve(&self) -> Atoms {
        Atoms(INITIAL_ALLOCATION.0 - self.initial_released.0)
    }

    /// `R_M(k) = floor(S_k / Q)`.
    pub const fn miner_reserve(&self) -> Atoms {
        Atoms((self.miner_state >> 64) as u64)
    }

    /// `I_M(k) = R - R_M(k)`: cumulative release to miners.
    pub const fn miner_released(&self) -> Atoms {
        Atoms(MINER_ALLOCATION.0 - self.miner_reserve().0)
    }

    /// The counters after block `k + 1` and that block's release
    /// `i_k = I_k - I_{k-1}`, `m_k = R_M(k-1) - R_M(k)` (eqs. 7.19c, 7.19g).
    pub fn next(&self) -> (Emission, PrimaryRelease) {
        let block = self.block + 1;
        let initial_released = initial_release_cumulative(block);
        let miner_state = miner_decay_step(self.miner_state);
        let next = Emission {
            block,
            initial_released,
            miner_state,
        };
        // Both reserves are non-increasing: I_k is monotone and a < Q.
        let release = PrimaryRelease {
            block,
            initial: Atoms(initial_released.0 - self.initial_released.0),
            miner: Atoms(self.miner_reserve().0 - next.miner_reserve().0),
        };
        (next, release)
    }
}

// ---------------------------------------------------------------------------
// InitialValve (eqs. 7.19h–7.19j)
// ---------------------------------------------------------------------------

/// The rolling 720-block record of gross InitialProtocolPool outflow `w_k`.
///
/// Bucket `k mod W` holds `w_k`; advancing to block `k` evicts `w_{k-W}`, so
/// [`InitialValve::rolling_outflow`] is always `sum_{t=max(1,k-W+1)}^{k} w_t`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InitialValve {
    head: u64,
    rolling: Atoms,
    buckets: Vec<Atoms>,
}

impl InitialValve {
    fn genesis() -> Self {
        InitialValve {
            head: 0,
            rolling: Atoms::ZERO,
            buckets: vec![Atoms::ZERO; INITIAL_VALVE_WINDOW as usize],
        }
    }

    /// The block `k` the window currently ends at.
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Gross outflow inside the current window.
    pub fn rolling_outflow(&self) -> Atoms {
        self.rolling
    }

    /// `L_I` minus the rolling outflow.
    pub fn headroom(&self) -> Atoms {
        Atoms(INITIAL_VALVE_LIMIT.0.saturating_sub(self.rolling.0))
    }

    /// `w_t` if block `t` lies inside the current window, else zero.
    pub fn outflow_at(&self, t: u64) -> Atoms {
        if t >= 1 && t <= self.head && self.head - t < INITIAL_VALVE_WINDOW {
            self.buckets[valve_slot(t)]
        } else {
            Atoms::ZERO
        }
    }

    /// The ring indexed by `k mod W`, for state commitments.
    pub fn buckets(&self) -> &[Atoms] {
        &self.buckets
    }

    /// Whether a further gross outflow keeps the window within `L_I`.
    pub fn admits(&self, gross: Atoms) -> bool {
        self.rolling
            .checked_add(gross)
            .is_some_and(|total| total <= INITIAL_VALVE_LIMIT)
    }

    /// Slides the window to block `k`, evicting `w_{k-W}`.
    fn advance(&mut self, k: u64) {
        let evicted = core::mem::replace(&mut self.buckets[valve_slot(k)], Atoms::ZERO);
        self.rolling = Atoms(self.rolling.0 - evicted.0);
        self.head = k;
    }

    /// Adds gross outflow to `w_k` of the current block. The caller checked
    /// [`InitialValve::admits`].
    fn record(&mut self, gross: Atoms) {
        let bucket = &mut self.buckets[valve_slot(self.head)];
        *bucket = Atoms(bucket.0 + gross.0);
        self.rolling = Atoms(self.rolling.0 + gross.0);
    }

    fn is_consistent(&self) -> bool {
        let sum: u128 = self.buckets.iter().map(|a| u128::from(a.0)).sum();
        sum == u128::from(self.rolling.0) && self.rolling <= INITIAL_VALVE_LIMIT
    }
}

fn valve_slot(k: u64) -> usize {
    (k % INITIAL_VALVE_WINDOW) as usize
}

// ---------------------------------------------------------------------------
// Fee lots and reserve dynamics (eqs. 7.20–7.22)
// ---------------------------------------------------------------------------

/// Unique identifier of a fee lot, assigned in creation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LotId(pub u64);

/// An independent, immutable fee lot `c`.
///
/// Each epoch the lot budgets `b = ceil(η A)` of its unbudgeted principal `A`
/// and pays every mature share `floor(b w_s / W_e)`. The flooring remainder
/// `r = b - sum floor(...)` is dust: it leaves the lot's principal and is
/// permanently segregated in `R_a`, never funding a later budget. Lots are
/// never merged, topped up or matured twice, so for every lot
/// `source_fee = principal + booked_total + dust_total` at all times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeeLot {
    id: LotId,
    source_fee: Atoms,
    paid_epoch: u64,
    maturity_epoch: u64,
    principal: Atoms,
    booked_total: Atoms,
    dust_total: Atoms,
}

/// One lot's contribution to one epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LotEpochPayout {
    /// The lot.
    pub lot: LotId,
    /// Ceiling budget `b_{c,e}`.
    pub budget: Atoms,
    /// `Reward_{s,c,e}`, indexed like the input weights.
    pub rewards: Vec<Atoms>,
    /// Booked payouts `p_{c,e} = sum_s Reward_{s,c,e}`.
    pub booked: Atoms,
    /// Unpayable dust `r_{c,e} = b_{c,e} - p_{c,e}`.
    pub dust: Atoms,
}

impl FeeLot {
    /// A fresh lot for a matured fee `F_c` paid in `paid_epoch`.
    pub fn matured(id: LotId, fee: Atoms, paid_epoch: u64, maturity_epoch: u64) -> Self {
        FeeLot {
            id,
            source_fee: fee,
            paid_epoch,
            maturity_epoch,
            principal: fee,
            booked_total: Atoms::ZERO,
            dust_total: Atoms::ZERO,
        }
    }

    /// Lot identifier.
    pub fn id(&self) -> LotId {
        self.id
    }
    /// Original fee `F_c`.
    pub fn source_fee(&self) -> Atoms {
        self.source_fee
    }
    /// Epoch `e_c` in which the fee was paid.
    pub fn paid_epoch(&self) -> u64 {
        self.paid_epoch
    }
    /// First eligible epoch `m_c = e_c + L + 1`.
    pub fn maturity_epoch(&self) -> u64 {
        self.maturity_epoch
    }
    /// Unbudgeted principal `A_{c,e}`.
    pub fn principal(&self) -> Atoms {
        self.principal
    }
    /// Cumulative booked payouts.
    pub fn booked_total(&self) -> Atoms {
        self.booked_total
    }
    /// Cumulative segregated dust.
    pub fn dust_total(&self) -> Atoms {
        self.dust_total
    }

    /// Settles one epoch against the mature share weights `w_s` (eq. 7.21).
    ///
    /// With `W_e = 0` the budget and every payout are zero. All divisions are
    /// exact Euclidean integer divisions; `0 <= p <= b <= A` always holds.
    pub fn settle_epoch(&mut self, rate: &EmissionRate, weights: &[ShareWeight]) -> LotEpochPayout {
        let total_weight: u128 = weights.iter().map(|&w| u128::from(w)).sum();
        let budget = if total_weight == 0 {
            Atoms::ZERO
        } else {
            rate.budget(self.principal)
        };
        let rewards = pro_rata(budget, weights, total_weight);
        let booked = Atoms(rewards.iter().map(|r| r.get()).sum());
        let dust = Atoms(budget.get() - booked.get());
        self.principal = Atoms(self.principal.get() - budget.get());
        self.booked_total = Atoms(self.booked_total.get() + booked.get());
        self.dust_total = Atoms(self.dust_total.get() + dust.get());
        LotEpochPayout {
            lot: self.id,
            budget,
            rewards,
            booked,
            dust,
        }
    }
}

/// `floor(budget w_s / W)` for every weight; all zero when `W = 0`. With `W > 0` the
/// floors sum to at most `budget`.
fn pro_rata(budget: Atoms, weights: &[ShareWeight], total_weight: u128) -> Vec<Atoms> {
    weights
        .iter()
        .map(|&w| {
            let reward = (u128::from(budget.get()) * u128::from(w)).checked_div(total_weight);
            Atoms(reward.unwrap_or(0) as u64)
        })
        .collect()
}

/// A fee held in the locked reserve `R_l` until its maturity epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LockedFee {
    /// The lot this fee becomes on maturation.
    pub lot: LotId,
    /// Fee amount.
    pub amount: Atoms,
    /// Epoch of the irrevocable transfer.
    pub paid_epoch: u64,
    /// First epoch in which it may mature, `e + L + 1`.
    pub maturity_epoch: u64,
}

/// One epoch's reward booking across the fee lots and the miner release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochPayout {
    /// Epoch `e`.
    pub epoch: u64,
    /// `E_e = sum_c b_{c,e}` over the fee lots.
    pub budget: Atoms,
    /// `P_e = sum_c p_{c,e}`, moved from `R_a` to `P_pending`.
    pub booked: Atoms,
    /// `r_e = E_e - P_e`, newly segregated fee-lot dust.
    pub dust: Atoms,
    /// Miner release offered to this epoch's shares: all of it not yet booked.
    pub release_budget: Atoms,
    /// Miner release booked to shares; it stays inside `P_pending`.
    pub release_booked: Atoms,
    /// Flooring remainder of the miner release (all of it when `W_e = 0`); it stays
    /// unbooked in `P_pending` and joins the next epoch's release budget.
    pub release_carry: Atoms,
    /// `Reward_{s,e}` from fee lots and miner release, indexed like the input weights.
    pub per_share: Vec<Atoms>,
    /// Per-lot detail.
    pub lots: Vec<LotEpochPayout>,
}

impl EpochPayout {
    /// Atoms newly booked to shares, from fee lots and the miner release together.
    pub fn total_booked(&self) -> Atoms {
        Atoms(self.booked.get() + self.release_booked.get())
    }
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

/// Identifier of an escrowed job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(pub u64);

/// Rejection reasons. A rejected operation leaves the state unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// The delta does not sum to zero (eq. 7.9).
    NonZeroSumDelta(i128),
    /// A category would become negative.
    NegativeBalance(StateCategory),
    /// A category would exceed `u64`.
    BalanceOverflow(StateCategory),
    /// The categories would not sum to `M`.
    SupplyViolation {
        /// The offending total.
        total: u128,
    },
    /// `η` must satisfy `0 < u < v` and `gcd(u, v) = 1`.
    InvalidEmissionRate {
        /// `u`.
        numerator: u64,
        /// `v`.
        denominator: u64,
    },
    /// A deployment parameter is unusable.
    InvalidParams(&'static str),
    /// Blocks are applied one height at a time.
    HeightSequence {
        /// The only height the next block may have.
        expected: u64,
        /// The height requested.
        found: u64,
    },
    /// No escrow exists for the job.
    UnknownJob(JobId),
    /// The job already has an escrow.
    JobAlreadyFunded(JobId),
    /// The escrow cannot cover the requested movement.
    InsufficientEscrow {
        /// The job.
        job: JobId,
        /// Escrow balance.
        available: Atoms,
        /// Requested amount.
        required: Atoms,
    },
    /// Tariffs and withdrawals are positive integers.
    ZeroAmount,
    /// Rewards for this epoch were already booked.
    EpochAlreadyBooked(u64),
    /// `FW_REWARD` may pay only rewards already booked to shares.
    UnbookedReward {
        /// Booked, unpaid rewards.
        booked: Atoms,
        /// Requested payment.
        requested: Atoms,
    },
    /// The withdrawal does not carry the next `InitialNonce` (eq. 7.19j).
    InitialNonceMismatch {
        /// The current `InitialNonce`.
        expected: u64,
        /// The nonce the withdrawal carries.
        found: u64,
    },
    /// The InitialProtocolPool cannot cover `v + f`.
    InsufficientInitialPool {
        /// `U_initial`.
        available: Atoms,
        /// `v + f`.
        requested: Atoms,
    },
    /// The withdrawal would push the rolling 720-block outflow past `L_I` (eq. 7.19h).
    InitialValveExceeded {
        /// Gross outflow already inside the window.
        rolling: Atoms,
        /// `v + f` of the withdrawal.
        requested: Atoms,
    },
    /// A sub-ledger disagrees with its category balance.
    Inconsistent(&'static str),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerError::NonZeroSumDelta(s) => write!(f, "delta sums to {s}, not zero"),
            LedgerError::NegativeBalance(c) => write!(f, "{} would become negative", c.symbol()),
            LedgerError::BalanceOverflow(c) => write!(f, "{} would overflow", c.symbol()),
            LedgerError::SupplyViolation { total } => write!(f, "categories sum to {total}, not M"),
            LedgerError::InvalidEmissionRate {
                numerator,
                denominator,
            } => {
                write!(f, "invalid emission rate {numerator}/{denominator}")
            }
            LedgerError::InvalidParams(why) => write!(f, "invalid parameters: {why}"),
            LedgerError::HeightSequence { expected, found } => {
                write!(f, "expected block height {expected}, got {found}")
            }
            LedgerError::UnknownJob(j) => write!(f, "no escrow for job {}", j.0),
            LedgerError::JobAlreadyFunded(j) => write!(f, "job {} is already funded", j.0),
            LedgerError::InsufficientEscrow {
                job,
                available,
                required,
            } => write!(
                f,
                "job {} escrow {} atoms < required {} atoms",
                job.0,
                available.get(),
                required.get()
            ),
            LedgerError::ZeroAmount => write!(f, "amounts must be positive"),
            LedgerError::EpochAlreadyBooked(e) => write!(f, "epoch {e} already booked"),
            LedgerError::UnbookedReward { booked, requested } => write!(
                f,
                "FW_REWARD of {} atoms exceeds the {} atoms booked to shares",
                requested.get(),
                booked.get()
            ),
            LedgerError::InitialNonceMismatch { expected, found } => {
                write!(
                    f,
                    "InitialNonce is {expected}, the withdrawal carries {found}"
                )
            }
            LedgerError::InsufficientInitialPool {
                available,
                requested,
            } => write!(
                f,
                "InitialProtocolPool holds {} atoms, the withdrawal needs {}",
                available.get(),
                requested.get()
            ),
            LedgerError::InitialValveExceeded { rolling, requested } => write!(
                f,
                "InitialValve: {} atoms already in the {INITIAL_VALVE_WINDOW}-block window plus {} exceeds {}",
                rolling.get(),
                requested.get(),
                INITIAL_VALVE_LIMIT.get()
            ),
            LedgerError::Inconsistent(what) => write!(f, "inconsistent state: {what}"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// The ten-category ledger state with its sub-ledgers.
#[derive(Clone, Debug)]
pub struct Ledger {
    params: LedgerParams,
    height: u64,
    balances: [Atoms; StateCategory::COUNT],
    emission: Emission,
    valve: InitialValve,
    initial_nonce: u64,
    escrows: BTreeMap<JobId, Atoms>,
    locked: Vec<LockedFee>,
    lots: Vec<FeeLot>,
    segregated_dust: Atoms,
    booked_rewards: Atoms,
    unbooked_release: Atoms,
    next_lot: u64,
    last_booked_epoch: Option<u64>,
}

impl Ledger {
    /// Genesis (eq. 7.19a): `R_I = I` and `R_M = R`, every other category zero. There
    /// is no Genesis fee lot in `R_a`.
    pub fn genesis(params: LedgerParams) -> Result<Self, LedgerError> {
        if params.blocks_per_epoch == 0 {
            return Err(LedgerError::InvalidParams(
                "blocks_per_epoch must be positive",
            ));
        }
        let mut balances = [Atoms::ZERO; StateCategory::COUNT];
        balances[StateCategory::UnissuedInitial.index()] = INITIAL_ALLOCATION;
        balances[StateCategory::UnissuedMiners.index()] = MINER_ALLOCATION;
        let ledger = Ledger {
            params,
            height: params.genesis_height,
            balances,
            emission: Emission::genesis(),
            valve: InitialValve::genesis(),
            initial_nonce: 0,
            escrows: BTreeMap::new(),
            locked: Vec::new(),
            lots: Vec::new(),
            segregated_dust: Atoms::ZERO,
            booked_rewards: Atoms::ZERO,
            unbooked_release: Atoms::ZERO,
            next_lot: 0,
            last_booked_epoch: None,
        };
        ledger.check_invariants()?;
        Ok(ledger)
    }

    /// Deployment parameters.
    pub fn params(&self) -> &LedgerParams {
        &self.params
    }
    /// Current height.
    pub fn height(&self) -> u64 {
        self.height
    }
    /// Epoch containing the current height.
    pub fn epoch(&self) -> u64 {
        self.epoch_of(self.height)
    }
    /// Epoch `e(h)` containing `height`.
    pub fn epoch_of(&self, height: u64) -> u64 {
        height.saturating_sub(self.params.genesis_height) / self.params.blocks_per_epoch
    }
    /// Balance of one category.
    pub fn balance(&self, category: StateCategory) -> Atoms {
        self.balances[category.index()]
    }
    /// Sum over all ten categories.
    pub fn total(&self) -> u128 {
        self.balances.iter().map(|b| u128::from(b.get())).sum()
    }
    /// The primary issuance counters.
    pub fn emission(&self) -> &Emission {
        &self.emission
    }
    /// The rolling InitialProtocolPool outflow record.
    pub fn initial_valve(&self) -> &InitialValve {
        &self.valve
    }
    /// `InitialNonce`, the replay counter of InitialProtocolPool withdrawals.
    pub fn initial_nonce(&self) -> u64 {
        self.initial_nonce
    }
    /// MatureFeeLots in `R_a`.
    pub fn fee_lots(&self) -> &[FeeLot] {
        &self.lots
    }
    /// Fees still locked in `R_l`.
    pub fn locked_fees(&self) -> &[LockedFee] {
        &self.locked
    }
    /// Rounding dust permanently segregated in `R_a`.
    pub fn segregated_dust(&self) -> Atoms {
        self.segregated_dust
    }
    /// Rewards booked to shares and not yet paid (inside `P_pending`).
    pub fn booked_rewards(&self) -> Atoms {
        self.booked_rewards
    }
    /// Miner release not yet booked to shares (inside `P_pending`).
    pub fn unbooked_release(&self) -> Atoms {
        self.unbooked_release
    }
    /// Escrow balance of a job.
    pub fn escrow(&self, job: JobId) -> Atoms {
        self.escrows.get(&job).copied().unwrap_or(Atoms::ZERO)
    }

    /// Opens block `height`, which must follow the current height, with its primary
    /// issuance (eqs. 7.19b–7.19g): `ΔR_I = -i_k`, `ΔU_initial = i_k`,
    /// `ΔR_M = -m_k`, `ΔP_pending = m_k`. The InitialValve window moves to block `k`.
    pub fn begin_block(
        &mut self,
        height: u64,
    ) -> Result<(BalanceDelta, PrimaryRelease), LedgerError> {
        let expected = self
            .height
            .checked_add(1)
            .ok_or(LedgerError::HeightSequence {
                expected: self.height,
                found: height,
            })?;
        if height != expected {
            return Err(LedgerError::HeightSequence {
                expected,
                found: height,
            });
        }
        let (emission, release) = self.emission.next();
        if release.block != height - self.params.genesis_height {
            return Err(LedgerError::Inconsistent(
                "primary releases do not follow the block heights",
            ));
        }
        let delta = BalanceDelta::transfer(
            StateCategory::UnissuedInitial,
            StateCategory::InitialProtocolPool,
            release.initial,
        )
        .with(
            StateCategory::UnissuedMiners,
            -i128::from(release.miner.get()),
        )
        .with(
            StateCategory::PendingRewards,
            i128::from(release.miner.get()),
        );
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.emission = emission;
        self.unbooked_release = Atoms(self.unbooked_release.get() + release.miner.get());
        self.valve.advance(release.block);
        self.height = height;
        Ok((delta, release))
    }

    /// `INITIAL_POOL_WITHDRAWAL` (eq. 7.12): `ΔU_initial = -(v + f)`, `ΔU = v`,
    /// `ΔR_l = f`, guarded by the state conjuncts of `InitialOK` (eq. 7.19j): the
    /// expected nonce, `U_initial >= v + f`, and `w_rolling + v + f <= L_I`. The
    /// pool-funded fee becomes a locked fee like any other.
    ///
    /// The treasury dual signature, the remaining conjunct, is checked by the caller.
    pub fn initial_pool_withdrawal(
        &mut self,
        expected_nonce: u64,
        amount: Atoms,
        fee: Atoms,
    ) -> Result<BalanceDelta, LedgerError> {
        if amount == Atoms::ZERO {
            return Err(LedgerError::ZeroAmount);
        }
        if expected_nonce != self.initial_nonce {
            return Err(LedgerError::InitialNonceMismatch {
                expected: self.initial_nonce,
                found: expected_nonce,
            });
        }
        let gross = amount
            .checked_add(fee)
            .ok_or(LedgerError::BalanceOverflow(StateCategory::Circulating))?;
        let available = self.balance(StateCategory::InitialProtocolPool);
        if available < gross {
            return Err(LedgerError::InsufficientInitialPool {
                available,
                requested: gross,
            });
        }
        if !self.valve.admits(gross) {
            return Err(LedgerError::InitialValveExceeded {
                rolling: self.valve.rolling_outflow(),
                requested: gross,
            });
        }
        let delta = BalanceDelta::ZERO
            .with(StateCategory::InitialProtocolPool, -i128::from(gross.get()))
            .with(StateCategory::Circulating, i128::from(amount.get()))
            .with(StateCategory::LockedBaseFees, i128::from(fee.get()));
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.valve.record(gross);
        self.initial_nonce += 1;
        if fee > Atoms::ZERO {
            self.lock_fee(fee);
        }
        Ok(delta)
    }

    /// `ESCROW_FUND_AND_QUEUE`: `ΔU = -(D + f_queue)`, `ΔE_esc = D`, `ΔR_l = f_queue`.
    /// The queue fee is an irrevocable fee transfer and becomes its own lot.
    pub fn escrow_fund_and_queue(
        &mut self,
        job: JobId,
        deposit: Atoms,
        queue_fee: Atoms,
    ) -> Result<BalanceDelta, LedgerError> {
        if deposit == Atoms::ZERO || queue_fee == Atoms::ZERO {
            return Err(LedgerError::ZeroAmount);
        }
        if self.escrows.contains_key(&job) {
            return Err(LedgerError::JobAlreadyFunded(job));
        }
        let total = i128::from(deposit.get()) + i128::from(queue_fee.get());
        let delta = BalanceDelta::ZERO
            .with(StateCategory::Circulating, -total)
            .with(StateCategory::Escrow, i128::from(deposit.get()))
            .with(StateCategory::LockedBaseFees, i128::from(queue_fee.get()));
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.escrows.insert(job, deposit);
        self.lock_fee(queue_fee);
        Ok(delta)
    }

    /// `JOB_ADMIT_AND_LEASE`: `ΔE_esc = -F_base`, `ΔR_l = F_base`.
    ///
    /// The Base Fee moves irretrievably into `R_l[e(h_admit) + L]`; no operation
    /// ever moves value from `R_l` back to an escrow or a client (eq. 7.17).
    pub fn job_admit_and_lease(
        &mut self,
        job: JobId,
        base_fee: Atoms,
    ) -> Result<BalanceDelta, LedgerError> {
        if base_fee == Atoms::ZERO {
            return Err(LedgerError::ZeroAmount);
        }
        self.require_escrow(job, base_fee)?;
        let delta = BalanceDelta::transfer(
            StateCategory::Escrow,
            StateCategory::LockedBaseFees,
            base_fee,
        );
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.debit_escrow(job, base_fee);
        self.lock_fee(base_fee);
        Ok(delta)
    }

    /// `RESULT_SETTLE`: `ΔE_esc = -(v_exec + v_DA + v_ref)`, `ΔU = v_exec + v_DA + v_ref`.
    pub fn result_settle(
        &mut self,
        job: JobId,
        v_exec: Atoms,
        v_da: Atoms,
        v_ref: Atoms,
    ) -> Result<BalanceDelta, LedgerError> {
        let total = [v_exec, v_da, v_ref]
            .into_iter()
            .try_fold(Atoms::ZERO, Atoms::checked_add)
            .ok_or(LedgerError::BalanceOverflow(StateCategory::Escrow))?;
        self.require_escrow(job, total)?;
        let delta =
            BalanceDelta::transfer(StateCategory::Escrow, StateCategory::Circulating, total);
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.debit_escrow(job, total);
        Ok(delta)
    }

    /// `RESERVE_MATURE`: every locked fee whose maturity epoch has arrived moves
    /// `R_l -> R_a` as a new, independent MatureFeeLot (`ΔR_l = -v_mat`, `ΔR_a = v_mat`).
    pub fn reserve_mature(&mut self) -> Result<(BalanceDelta, Vec<LotId>), LedgerError> {
        let epoch = self.epoch();
        let (ready, waiting): (Vec<LockedFee>, Vec<LockedFee>) =
            self.locked.iter().partition(|f| f.maturity_epoch <= epoch);
        let v_mat = Atoms(ready.iter().map(|f| f.amount.get()).sum());
        let delta = BalanceDelta::transfer(
            StateCategory::LockedBaseFees,
            StateCategory::EmissionReserve,
            v_mat,
        );
        let staged = self.stage(&delta)?;
        self.balances = staged;
        self.locked = waiting;
        let ids = ready.iter().map(|f| f.lot).collect();
        self.lots.extend(
            ready
                .iter()
                .map(|f| FeeLot::matured(f.lot, f.amount, f.paid_epoch, f.maturity_epoch)),
        );
        Ok((delta, ids))
    }

    /// Epoch booking against the mature share weights. Every eligible fee lot settles
    /// (eq. 7.21) and its booked payouts move `R_a -> P_pending` (`ΔR_a = -P_e`,
    /// `ΔP_pending = P_e`), with its dust `r_e` segregated in `R_a`. The miner release
    /// not yet booked, already inside `P_pending`, is booked by the same exact floor
    /// rule; its remainder stays unbooked for the next epoch. Each epoch is booked at
    /// most once.
    pub fn book_epoch(
        &mut self,
        weights: &[ShareWeight],
    ) -> Result<(BalanceDelta, EpochPayout), LedgerError> {
        let epoch = self.epoch();
        if self.last_booked_epoch.is_some_and(|last| last >= epoch) {
            return Err(LedgerError::EpochAlreadyBooked(epoch));
        }
        let rate = self.params.emission_rate;
        let mut lots = self.lots.clone();
        let mut per_share = vec![Atoms::ZERO; weights.len()];
        let mut details = Vec::new();
        for lot in lots
            .iter_mut()
            .filter(|l| l.maturity_epoch <= epoch && l.principal > Atoms::ZERO)
        {
            let payout = lot.settle_epoch(&rate, weights);
            for (acc, r) in per_share.iter_mut().zip(&payout.rewards) {
                *acc = Atoms(acc.get() + r.get());
            }
            details.push(payout);
        }
        let total_weight: u128 = weights.iter().map(|&w| u128::from(w)).sum();
        let release_budget = self.unbooked_release;
        let release_rewards = pro_rata(release_budget, weights, total_weight);
        for (acc, r) in per_share.iter_mut().zip(&release_rewards) {
            *acc = Atoms(acc.get() + r.get());
        }
        let release_booked = Atoms(release_rewards.iter().map(|r| r.get()).sum());
        let sum = |f: fn(&LotEpochPayout) -> Atoms| Atoms(details.iter().map(|d| f(d).get()).sum());
        let payout = EpochPayout {
            epoch,
            budget: sum(|d| d.budget),
            booked: sum(|d| d.booked),
            dust: sum(|d| d.dust),
            release_budget,
            release_booked,
            release_carry: Atoms(release_budget.get() - release_booked.get()),
            per_share,
            lots: details,
        };
        let delta = BalanceDelta::transfer(
            StateCategory::EmissionReserve,
            StateCategory::PendingRewards,
            payout.booked,
        );
        let staged = self.stage(&delta)?;
        self.balances = staged;
        lots.retain(|lot| lot.principal > Atoms::ZERO);
        self.lots = lots;
        self.segregated_dust = Atoms(self.segregated_dust.get() + payout.dust.get());
        self.booked_rewards = Atoms(self.booked_rewards.get() + payout.total_booked().get());
        self.unbooked_release = payout.release_carry;
        self.last_booked_epoch = Some(epoch);
        Ok((delta, payout))
    }

    /// `FW_REWARD`: `ΔP_pending = -v_pay`, `ΔU = v_pay`, paying rewards already booked
    /// to shares.
    pub fn fw_reward(&mut self, amount: Atoms) -> Result<BalanceDelta, LedgerError> {
        if amount > self.booked_rewards {
            return Err(LedgerError::UnbookedReward {
                booked: self.booked_rewards,
                requested: amount,
            });
        }
        let delta = BalanceDelta::transfer(
            StateCategory::PendingRewards,
            StateCategory::Circulating,
            amount,
        );
        self.balances = self.stage(&delta)?;
        self.booked_rewards = Atoms(self.booked_rewards.get() - amount.get());
        Ok(delta)
    }

    /// Verifies the supply invariant and that every sub-ledger matches its category.
    pub fn check_invariants(&self) -> Result<(), LedgerError> {
        if self.total() != u128::from(TOTAL_SUPPLY.get()) {
            return Err(LedgerError::SupplyViolation {
                total: self.total(),
            });
        }
        if self.emission.block() != self.height - self.params.genesis_height
            || self.valve.head() != self.emission.block()
        {
            return Err(LedgerError::Inconsistent(
                "a block was entered without its primary release",
            ));
        }
        if self.balance(StateCategory::UnissuedInitial) != self.emission.initial_reserve()
            || self.balance(StateCategory::UnissuedMiners) != self.emission.miner_reserve()
        {
            return Err(LedgerError::Inconsistent(
                "R_I or R_M disagrees with the emission counters",
            ));
        }
        if !self.valve.is_consistent() {
            return Err(LedgerError::Inconsistent("InitialValve ring buffer"));
        }
        let pending = self.booked_rewards.get() + self.unbooked_release.get();
        if pending != self.balance(StateCategory::PendingRewards).get() {
            return Err(LedgerError::Inconsistent(
                "P_pending != booked rewards + unbooked miner release",
            ));
        }
        let reserve =
            self.segregated_dust.get() + self.lots.iter().map(|l| l.principal.get()).sum::<u64>();
        if reserve != self.balance(StateCategory::EmissionReserve).get() {
            return Err(LedgerError::Inconsistent(
                "R_a != dust + sum of lot principals",
            ));
        }
        let locked: u64 = self.locked.iter().map(|f| f.amount.get()).sum();
        if locked != self.balance(StateCategory::LockedBaseFees).get() {
            return Err(LedgerError::Inconsistent("R_l != sum of locked fees"));
        }
        let escrowed: u64 = self.escrows.values().map(|a| a.get()).sum();
        if escrowed != self.balance(StateCategory::Escrow).get() {
            return Err(LedgerError::Inconsistent("E_esc != sum of job escrows"));
        }
        if self.lots.iter().any(|l| {
            l.principal.get() + l.booked_total.get() + l.dust_total.get() != l.source_fee.get()
        }) {
            return Err(LedgerError::Inconsistent(
                "lot principal + booked + dust != source fee",
            ));
        }
        Ok(())
    }

    /// Validates a delta against the current state without mutating it: zero sum
    /// (eq. 7.9), non-negativity and supply `M` (eq. 7.15). Callers commit the
    /// returned balances together with their sub-ledger updates, which makes every
    /// operation all-or-nothing (eq. 7.10).
    fn stage(&self, delta: &BalanceDelta) -> Result<[Atoms; StateCategory::COUNT], LedgerError> {
        if delta.sum() != 0 {
            return Err(LedgerError::NonZeroSumDelta(delta.sum()));
        }
        let mut next = self.balances;
        for category in StateCategory::ALL {
            let value = i128::from(next[category.index()].get()) + delta.get(category);
            if value < 0 {
                return Err(LedgerError::NegativeBalance(category));
            }
            next[category.index()] =
                Atoms(u64::try_from(value).map_err(|_| LedgerError::BalanceOverflow(category))?);
        }
        let total: u128 = next.iter().map(|b| u128::from(b.get())).sum();
        if total != u128::from(TOTAL_SUPPLY.get()) {
            return Err(LedgerError::SupplyViolation { total });
        }
        Ok(next)
    }

    fn require_escrow(&self, job: JobId, required: Atoms) -> Result<(), LedgerError> {
        let available = *self.escrows.get(&job).ok_or(LedgerError::UnknownJob(job))?;
        if available < required {
            return Err(LedgerError::InsufficientEscrow {
                job,
                available,
                required,
            });
        }
        Ok(())
    }

    fn debit_escrow(&mut self, job: JobId, amount: Atoms) {
        if let Some(balance) = self.escrows.get_mut(&job) {
            *balance = Atoms(balance.get() - amount.get());
            if *balance == Atoms::ZERO {
                self.escrows.remove(&job);
            }
        }
    }

    fn lock_fee(&mut self, amount: Atoms) {
        let paid_epoch = self.epoch();
        self.locked.push(LockedFee {
            lot: LotId(self.next_lot),
            amount,
            paid_epoch,
            maturity_epoch: paid_epoch + self.params.maturation_delay + 1,
        });
        self.next_lot += 1;
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plau(n: u64) -> Atoms {
        Atoms::from_plau(n).unwrap()
    }

    fn assert_supply(ledger: &Ledger) {
        let sum: u128 = StateCategory::ALL
            .iter()
            .map(|&c| u128::from(ledger.balance(c).get()))
            .sum();
        assert_eq!(sum, 1_500_000_000_000_000);
        ledger.check_invariants().unwrap();
    }

    /// Applies blocks up to and including `height`.
    fn run_to(ledger: &mut Ledger, height: u64) {
        while ledger.height() < height {
            ledger.begin_block(ledger.height() + 1).unwrap();
        }
    }

    /// A ledger whose `U` holds paid miner rewards: blocks up to the first epoch
    /// boundary, one booking to a single share, then its payment.
    fn funded(params: LedgerParams) -> Ledger {
        let mut ledger = Ledger::genesis(params).unwrap();
        run_to(&mut ledger, params.blocks_per_epoch);
        let (_, payout) = ledger.book_epoch(&[1]).unwrap();
        ledger.fw_reward(payout.total_booked()).unwrap();
        assert!(ledger.balance(StateCategory::Circulating) > Atoms::ZERO);
        ledger
    }

    // Golden values from an independent big-integer evaluation of eqs. (7.19b)–(7.19g)
    // that computes floor(a S / Q) without splitting S.
    const GOLDEN_MINER_RESERVE: [(u64, u64); 7] = [
        (1, 1_349_999_109_828_400),
        (2, 1_349_998_219_657_388),
        (3, 1_349_997_329_486_962),
        (720, 1_349_359_228_355_031),
        (1_000, 1_349_110_121_526_365),
        (22_000, 1_330_557_580_089_425),
        (1_051_200, 674_999_999_999_975),
    ];

    #[test]
    fn genesis_is_fully_unissued_across_ten_categories() {
        let ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        assert_supply(&ledger);
        assert_eq!(StateCategory::ALL.len(), 10);
        for (i, category) in StateCategory::ALL.iter().enumerate() {
            assert_eq!(category.index(), i);
        }
        let symbols: std::collections::BTreeSet<&str> =
            StateCategory::ALL.iter().map(|c| c.symbol()).collect();
        assert_eq!(symbols.len(), 10);
        assert_eq!(
            ledger.balance(StateCategory::UnissuedInitial),
            plau(1_500_000)
        );
        assert_eq!(
            ledger.balance(StateCategory::UnissuedMiners),
            plau(13_500_000)
        );
        for category in StateCategory::ALL {
            if !matches!(
                category,
                StateCategory::UnissuedInitial | StateCategory::UnissuedMiners
            ) {
                assert_eq!(ledger.balance(category), Atoms::ZERO, "{category:?}");
            }
        }
        // No Genesis fee lot in R_a.
        assert!(ledger.fee_lots().is_empty());
        assert_eq!(ledger.emission().block(), 0);
    }

    #[test]
    fn first_block_releases_from_both_unissued_reserves() {
        let mut ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        let (delta, release) = ledger.begin_block(1).unwrap();
        assert_eq!(release.block, 1);
        assert_eq!(release.initial, Atoms(142_694_063));
        assert_eq!(release.miner, Atoms(890_171_600));
        assert_eq!(delta.sum(), 0);
        assert_eq!(delta.get(StateCategory::UnissuedInitial), -142_694_063);
        assert_eq!(delta.get(StateCategory::InitialProtocolPool), 142_694_063);
        assert_eq!(delta.get(StateCategory::UnissuedMiners), -890_171_600);
        assert_eq!(delta.get(StateCategory::PendingRewards), 890_171_600);
        assert_eq!(
            ledger.balance(StateCategory::UnissuedMiners),
            Atoms(1_349_999_109_828_400)
        );
        assert_eq!(ledger.unbooked_release(), Atoms(890_171_600));
        assert_supply(&ledger);
    }

    #[test]
    fn blocks_are_applied_one_height_at_a_time() {
        let mut ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        for bad in [0, 2, 10] {
            assert_eq!(
                ledger.begin_block(bad).unwrap_err(),
                LedgerError::HeightSequence {
                    expected: 1,
                    found: bad
                }
            );
        }
        assert_eq!(ledger.emission().block(), 0);
        ledger.begin_block(1).unwrap();
        assert_eq!(
            ledger.begin_block(1).unwrap_err(),
            LedgerError::HeightSequence {
                expected: 2,
                found: 1
            }
        );
        assert_supply(&ledger);
    }

    #[test]
    fn issuance_follows_the_closed_forms_over_many_blocks() {
        let mut params = LedgerParams::poc();
        params.genesis_height = 500;
        let mut ledger = Ledger::genesis(params).unwrap();
        for &(k, reserve) in GOLDEN_MINER_RESERVE.iter().take(6) {
            run_to(&mut ledger, 500 + k);
            assert_eq!(
                ledger.balance(StateCategory::UnissuedMiners),
                Atoms(reserve),
                "k={k}"
            );
            assert_eq!(
                ledger.balance(StateCategory::PendingRewards),
                Atoms(MINER_ALLOCATION.get() - reserve)
            );
            assert_eq!(
                ledger.balance(StateCategory::InitialProtocolPool),
                initial_release_cumulative(k)
            );
        }
        assert_supply(&ledger);
    }

    #[test]
    fn miner_decay_matches_golden_values_to_the_half_life() {
        let mut state = u128::from(MINER_ALLOCATION.get()) << 64;
        let mut golden = GOLDEN_MINER_RESERVE.iter().peekable();
        for k in 1..=MINER_HALF_LIFE_BLOCKS {
            state = miner_decay_step(state);
            if let Some(&&(at, reserve)) = golden.peek()
                && at == k
            {
                assert_eq!((state >> 64) as u64, reserve, "k={k}");
                golden.next();
            }
        }
        assert!(golden.next().is_none());
        // Half of R remains after H_M blocks, 25 atoms under from flooring a.
        assert_eq!(MINER_ALLOCATION.get() / 2 - (state >> 64) as u64, 25);
        // One atom left: Ŝ = a < Q, so the final atom is released.
        assert_eq!(miner_decay_step(Q64), 0);
        assert_eq!(miner_decay_step(Q64 - 1), 0);
        assert!(miner_decay_step(u128::MAX) < u128::MAX);
    }

    #[test]
    fn initial_release_is_linear_and_exactly_complete() {
        assert_eq!(initial_release_cumulative(0), Atoms::ZERO);
        assert_eq!(initial_release_cumulative(1), Atoms(142_694_063));
        assert_eq!(
            initial_release_cumulative(INITIAL_RELEASE_BLOCKS),
            INITIAL_ALLOCATION
        );
        assert_eq!(initial_release_cumulative(u64::MAX), INITIAL_ALLOCATION);
        let last = initial_release_cumulative(INITIAL_RELEASE_BLOCKS).get()
            - initial_release_cumulative(INITIAL_RELEASE_BLOCKS - 1).get();
        assert_eq!(last, 142_694_064);
        let done = Emission {
            block: INITIAL_RELEASE_BLOCKS,
            initial_released: INITIAL_ALLOCATION,
            miner_state: Q64,
        };
        let (after, release) = done.next();
        assert_eq!(release.initial, Atoms::ZERO);
        assert_eq!(release.miner, Atoms(1), "the final miner atom");
        assert_eq!(after.miner_reserve(), Atoms::ZERO);
    }

    #[test]
    fn zero_sum_delta_rule() {
        let mut ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        run_to(&mut ledger, 1);
        let before = ledger.balances;

        // Minting one atom is rejected and changes nothing.
        let mint = BalanceDelta::ZERO.with(StateCategory::Circulating, 1);
        assert_eq!(ledger.stage(&mint), Err(LedgerError::NonZeroSumDelta(1)));
        // A zero-sum delta that overdraws a category is rejected too.
        let overdraw =
            BalanceDelta::transfer(StateCategory::Bonds, StateCategory::Circulating, Atoms(1));
        assert_eq!(
            ledger.stage(&overdraw),
            Err(LedgerError::NegativeBalance(StateCategory::Bonds))
        );
        assert_eq!(ledger.balances, before);

        // Value spread over every category moves, the sum does not.
        let mut spread = BalanceDelta::ZERO.with(StateCategory::UnissuedMiners, -9_000);
        for category in &StateCategory::ALL[..8] {
            spread = spread.with(*category, 1_000);
        }
        spread = spread.with(StateCategory::UnissuedInitial, 1_000);
        assert_eq!(spread.sum(), 0);
        let staged = ledger.stage(&spread).unwrap();
        assert_eq!(
            staged.iter().map(|a| u128::from(a.get())).sum::<u128>(),
            1_500_000_000_000_000
        );
    }

    #[test]
    fn initial_valve_caps_gross_outflow_per_720_blocks() {
        let mut ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        // The pool holds about 31,392 PLAU after 22,000 blocks.
        run_to(&mut ledger, 22_000);
        let pool = ledger.balance(StateCategory::InitialProtocolPool);
        assert!(pool > INITIAL_VALVE_LIMIT);

        let delta = ledger
            .initial_pool_withdrawal(0, plau(29_000), plau(1))
            .unwrap();
        assert_eq!(
            delta.get(StateCategory::Circulating),
            i128::from(plau(29_000).get())
        );
        assert_eq!(
            delta.get(StateCategory::LockedBaseFees),
            i128::from(plau(1).get())
        );
        assert_eq!(ledger.initial_nonce(), 1);
        assert_eq!(ledger.initial_valve().rolling_outflow(), plau(29_001));
        assert_eq!(ledger.initial_valve().headroom(), plau(999));
        assert_eq!(
            ledger.locked_fees().len(),
            1,
            "the pool-funded fee is a locked fee"
        );
        assert_supply(&ledger);

        // Affordable from the pool, but the window already holds 29,001 PLAU.
        let snapshot = ledger.clone();
        assert_eq!(
            ledger.initial_pool_withdrawal(1, plau(1_000), Atoms::ZERO),
            Err(LedgerError::InitialValveExceeded {
                rolling: plau(29_001),
                requested: plau(1_000)
            })
        );
        assert_eq!(ledger.initial_nonce(), snapshot.initial_nonce());
        assert_eq!(ledger.balances, snapshot.balances);
        assert_eq!(
            ledger.initial_pool_withdrawal(0, plau(1), Atoms::ZERO),
            Err(LedgerError::InitialNonceMismatch {
                expected: 1,
                found: 0
            })
        );
        assert_eq!(
            ledger.initial_pool_withdrawal(1, Atoms::ZERO, Atoms(5)),
            Err(LedgerError::ZeroAmount)
        );
        assert!(
            ledger
                .initial_pool_withdrawal(1, plau(999), Atoms::ZERO)
                .is_ok()
        );

        // Block 22,000 stays in the window [k - 719, k] through k = 22,719.
        run_to(&mut ledger, 22_719);
        assert_eq!(ledger.initial_valve().rolling_outflow(), plau(30_000));
        assert!(!ledger.initial_valve().admits(Atoms(1)));
        run_to(&mut ledger, 22_720);
        assert_eq!(ledger.initial_valve().rolling_outflow(), Atoms::ZERO);
        assert_eq!(ledger.initial_valve().outflow_at(22_000), Atoms::ZERO);
        assert_eq!(ledger.initial_valve().headroom(), INITIAL_VALVE_LIMIT);
        assert_supply(&ledger);
    }

    #[test]
    fn withdrawals_cannot_overdraw_the_pool() {
        let mut ledger = Ledger::genesis(LedgerParams::poc()).unwrap();
        run_to(&mut ledger, 2);
        let available = ledger.balance(StateCategory::InitialProtocolPool);
        assert_eq!(
            ledger.initial_pool_withdrawal(0, available, Atoms(1)),
            Err(LedgerError::InsufficientInitialPool {
                available,
                requested: Atoms(available.get() + 1)
            })
        );
        ledger
            .initial_pool_withdrawal(0, available, Atoms::ZERO)
            .unwrap();
        assert_eq!(
            ledger.balance(StateCategory::InitialProtocolPool),
            Atoms::ZERO
        );
        assert_supply(&ledger);
    }

    #[test]
    fn miner_release_is_booked_by_weight_and_its_remainder_carried() {
        let mut params = LedgerParams::poc();
        params.blocks_per_epoch = 10;
        let mut ledger = Ledger::genesis(params).unwrap();
        run_to(&mut ledger, 10);
        let released = ledger.unbooked_release();
        assert_eq!(released, ledger.emission().miner_released());

        let (delta, payout) = ledger.book_epoch(&[1, 2]).unwrap();
        assert_eq!(
            delta,
            BalanceDelta::ZERO,
            "the release is already inside P_pending"
        );
        assert_eq!(payout.release_budget, released);
        assert_eq!(
            payout.per_share,
            vec![Atoms(released.get() / 3), Atoms(released.get() * 2 / 3)]
        );
        assert_eq!(payout.release_carry.get(), released.get() % 3);
        assert_eq!(ledger.unbooked_release(), payout.release_carry);
        assert_eq!(ledger.booked_rewards(), payout.total_booked());
        assert_supply(&ledger);

        // Only booked rewards can be paid.
        let booked = ledger.booked_rewards();
        assert_eq!(
            ledger.fw_reward(Atoms(booked.get() + 1)),
            Err(LedgerError::UnbookedReward {
                booked,
                requested: Atoms(booked.get() + 1)
            })
        );
        ledger.fw_reward(booked).unwrap();
        assert_eq!(ledger.balance(StateCategory::Circulating), booked);
        assert_eq!(
            ledger.balance(StateCategory::PendingRewards),
            payout.release_carry
        );

        // With no share weight the whole release is carried to the next epoch.
        run_to(&mut ledger, 20);
        let carried = ledger.unbooked_release();
        let (_, idle) = ledger.book_epoch(&[]).unwrap();
        assert_eq!(
            (idle.release_booked, idle.release_carry),
            (Atoms::ZERO, carried)
        );
        assert_supply(&ledger);
    }

    #[test]
    fn full_fee_lifecycle_conserves_supply() {
        let mut ledger = funded(LedgerParams::poc());
        assert_supply(&ledger);

        // Worked budget of eq. (7.13): D = 10 + 80 + 5 + 5 = 100 PLAU, scaled to
        // what one epoch of miner release pays out.
        let job = JobId(1);
        ledger
            .escrow_fund_and_queue(job, plau(100), plau(1))
            .unwrap();
        assert_supply(&ledger);
        ledger.job_admit_and_lease(job, plau(10)).unwrap();
        assert_supply(&ledger);
        assert_eq!(ledger.balance(StateCategory::LockedBaseFees), plau(11));
        ledger
            .result_settle(job, plau(80), plau(5), plau(5))
            .unwrap();
        assert_supply(&ledger);
        assert_eq!(ledger.escrow(job), Atoms::ZERO);

        // Nothing matures before e + L + 1.
        let (delta, ids) = ledger.reserve_mature().unwrap();
        assert!(ids.is_empty() && delta == BalanceDelta::ZERO);
        let maturity = ledger.locked_fees()[0].maturity_epoch;
        run_to(&mut ledger, maturity * BLOCKS_PER_DAY);
        let (_, ids) = ledger.reserve_mature().unwrap();
        assert_eq!(ids.len(), 2, "queue fee and Base Fee are independent lots");
        assert_eq!(ledger.balance(StateCategory::LockedBaseFees), Atoms::ZERO);
        assert_supply(&ledger);

        let (_, second) = ledger.book_epoch(&[3, 5, 7]).unwrap();
        assert_eq!(second.lots.len(), 2, "the two fee lots, and no Genesis lot");
        assert_eq!(second.budget.get(), second.booked.get() + second.dust.get());
        assert!(second.release_booked > Atoms::ZERO);
        assert_supply(&ledger);
        ledger.fw_reward(ledger.booked_rewards()).unwrap();
        assert_supply(&ledger);
        assert_eq!(
            ledger.balance(StateCategory::PendingRewards),
            ledger.unbooked_release()
        );
        assert_eq!(
            ledger.book_epoch(&[1]).unwrap_err(),
            LedgerError::EpochAlreadyBooked(maturity)
        );
    }

    #[test]
    fn fee_lot_pays_by_integer_division_and_segregates_dust() {
        let rate = EmissionRate::new(1, 1_000).unwrap();
        let mut lot = FeeLot::matured(LotId(9), Atoms(1_000_000_000), 0, 3);
        let first = lot.settle_epoch(&rate, &[1, 1, 1]);
        assert_eq!(first.budget, Atoms(1_000_000));
        assert_eq!(first.rewards, vec![Atoms(333_333); 3]);
        assert_eq!((first.booked, first.dust), (Atoms(999_999), Atoms(1)));
        assert_eq!(lot.principal(), Atoms(999_000_000));

        // Dust never returns to principal: the next budget is ceil(η * 999,000,000).
        let second = lot.settle_epoch(&rate, &[2, 5]);
        assert_eq!(second.budget, Atoms(999_000));
        assert_eq!(second.rewards, vec![Atoms(285_428), Atoms(713_571)]);
        assert_eq!(second.dust, Atoms(1));
        assert_eq!(lot.dust_total(), Atoms(2));
        assert_eq!(
            lot.principal().get() + lot.booked_total().get() + lot.dust_total().get(),
            lot.source_fee().get()
        );

        // W_e = 0 (no shares, or only zero weights): nothing is budgeted.
        let before = lot.principal();
        for weights in [&[][..], &[0, 0][..]] {
            let idle = lot.settle_epoch(&rate, weights);
            assert_eq!(
                (idle.budget, idle.booked, idle.dust),
                (Atoms::ZERO, Atoms::ZERO, Atoms::ZERO)
            );
            assert!(idle.rewards.iter().all(|&r| r == Atoms::ZERO));
        }
        assert_eq!(lot.principal(), before);

        // Ceiling budget: ceil(1001 / 1000) = 2, and a 1-atom lot budgets its last atom.
        assert_eq!(rate.budget(Atoms(1_001)), Atoms(2));
        assert_eq!(rate.budget(Atoms(1)), Atoms(1));
        assert_eq!(rate.budget(Atoms::ZERO), Atoms::ZERO);
    }

    #[test]
    fn emission_rate_is_a_reduced_proper_fraction() {
        assert!(EmissionRate::new(1, 1_000).is_ok());
        assert!(EmissionRate::new(0, 7).is_err());
        assert!(EmissionRate::new(7, 7).is_err());
        assert!(EmissionRate::new(8, 7).is_err());
        assert!(EmissionRate::new(2, 4).is_err());
    }

    #[test]
    fn base_fee_is_irretrievable() {
        let mut ledger = funded(LedgerParams::poc());
        let job = JobId(7);
        ledger
            .escrow_fund_and_queue(job, plau(100), plau(1))
            .unwrap();
        ledger.job_admit_and_lease(job, plau(10)).unwrap();
        // The client cannot settle or reclaim the full deposit any more.
        assert_eq!(
            ledger
                .result_settle(job, plau(100), Atoms::ZERO, Atoms::ZERO)
                .unwrap_err(),
            LedgerError::InsufficientEscrow {
                job,
                available: plau(90),
                required: plau(100)
            }
        );
        assert_eq!(
            ledger.job_admit_and_lease(JobId(8), plau(1)).unwrap_err(),
            LedgerError::UnknownJob(JobId(8))
        );
        assert_eq!(
            ledger
                .escrow_fund_and_queue(job, plau(1), plau(1))
                .unwrap_err(),
            LedgerError::JobAlreadyFunded(job)
        );
        assert_supply(&ledger);
    }

    #[test]
    fn exhausted_fee_lots_are_pruned_without_losing_booked_funds() {
        let mut params = LedgerParams::poc();
        params.emission_rate = EmissionRate::new(1, 2).unwrap();
        params.blocks_per_epoch = 4;
        params.maturation_delay = 0;
        let mut ledger = funded(params);
        // One 1-PLAU queue fee becomes the only fee lot.
        ledger
            .escrow_fund_and_queue(JobId(1), plau(1), plau(1))
            .unwrap();
        ledger
            .result_settle(JobId(1), plau(1), Atoms::ZERO, Atoms::ZERO)
            .unwrap();
        for epoch in 2..80 {
            run_to(&mut ledger, epoch * params.blocks_per_epoch);
            ledger.reserve_mature().unwrap();
            let (_, payout) = ledger.book_epoch(&[1]).unwrap();
            assert!(payout.lots.len() <= 1);
            ledger.check_invariants().unwrap();
            ledger.fw_reward(ledger.booked_rewards()).unwrap();
            ledger.check_invariants().unwrap();
            if epoch > 3 && ledger.fee_lots().is_empty() {
                break;
            }
        }
        assert!(ledger.fee_lots().is_empty());
        assert_eq!(ledger.total(), u128::from(TOTAL_SUPPLY.get()));
    }
}
