//! State transition function (`chain::stf`, §7).
//!
//! [`apply_body`] opens every block `k = h - h_G >= 1` with the decoupled primary
//! issuance (spec eqs. 7.19b–7.19g) and admits InitialProtocolPool withdrawals only
//! under `InitialOK` (eq. 7.19j), the 720-block InitialValve included.

use core::fmt;
use std::collections::BTreeMap;

use ledger::{BalanceDelta, INITIAL_VALVE_LIMIT, JobId, Ledger, LedgerError, StateCategory};
use primitives::{Atoms, ShareWeight};
use scheduler::{Capacity, Cohort, OwnerId, ShareID};
use wire::{
    CategoryDelta, Digest, FeeTransition, GenesisConfig, InitialPoolWithdrawal, JobTransaction,
    LeaseAssignment, MAX_WITHDRAWALS_PER_BLOCK, OwnerReward, SettlementReceipt, ShareProof,
    WithdrawalOutput,
};

use crate::core_adapter::{
    JobValidationError, LABEL_LEDGER, SignatureError, VerifiedOwner, WithdrawalError,
    compute_result_digest, compute_withdrawal_id, digest_to_hash, domain_hash,
    validate_job_stateless, validate_withdrawal_stateless, verify_share_owner,
    verify_withdrawal_signatures,
};
use crate::difficulty::share_weight;

/// Registered Fresh Work share in `NodeState` (§7.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShareRecord {
    /// ShareID.
    pub work_id: Digest,
    /// Beneficiary, whose signature was verified before registration.
    pub owner: VerifiedOwner,
    /// `w_s`, fixed by the share target at the share's height.
    pub weight: u64,
    /// Height the share was mined at.
    pub height: u64,
    /// Height of the block that registered it.
    pub included_at: u64,
    /// Cumulative D'Hondt slots `a_s`, retained until this share is pruned.
    pub allocated_slots: u64,
}

/// Admitted commercial job awaiting settlement in `NodeState` (§7.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenJob {
    /// Job identifier.
    pub job_id: u64,
    /// Transaction identifier.
    pub tx_id: Digest,
    /// Height of `JOB_ADMIT_AND_LEASE`.
    pub admitted_at: u64,
    /// Lease owner's key id (executor by attribution).
    pub owner: Digest,
    /// Leased ShareID.
    pub share_id: Digest,
    /// Executor remuneration.
    pub v_exec: u64,
    /// Data-availability remuneration.
    pub v_da: u64,
    /// Refund share.
    pub v_ref: u64,
    /// Exact dot product computed at admission.
    pub result: i128,
}

/// Immutable per-block state snapshot (§7.1).
#[derive(Clone, Debug)]
pub struct NodeState {
    /// Height of the block this state follows.
    pub height: u64,
    /// The ten-category core ledger.
    pub ledger: Ledger,
    /// Registered shares by ShareID (pruned per §6.7).
    pub shares: BTreeMap<Digest, ShareRecord>,
    /// Admitted, unsettled jobs by `job_id`.
    pub open_jobs: BTreeMap<u64, OpenJob>,
    /// Used `tx_id`s with their expiry height (pruned once expired).
    pub used_tx: BTreeMap<Digest, u64>,
    /// Used `job_id`s with their expiry height (pruned once expired).
    pub used_job_ids: BTreeMap<u64, u64>,
}

impl NodeState {
    /// Builds the genesis `NodeState` at height 0 (§3.4).
    pub fn genesis(ledger: Ledger) -> Self {
        Self {
            height: 0,
            ledger,
            shares: BTreeMap::new(),
            open_jobs: BTreeMap::new(),
            used_tx: BTreeMap::new(),
            used_job_ids: BTreeMap::new(),
        }
    }

    /// Returns the mature D'Hondt lease cohort for a block at height `h`, sorted by
    /// ShareID (§6.7).
    pub fn lease_cohort(&self, h: u64, g: &GenesisConfig) -> Vec<ShareRecord> {
        let Some(max_inc) = h.checked_sub(g.share_maturity_depth) else {
            return Vec::new();
        };
        let min_inc = h.saturating_sub(g.snapshot_window);
        self.shares_included_in(min_inc, max_inc)
    }

    /// Returns the shares weighed at the epoch boundary `h`: those included in the
    /// previous epoch, sorted by ShareID (§6.7).
    pub fn epoch_shares(&self, h: u64, g: &GenesisConfig) -> Vec<ShareRecord> {
        let Some(max_inc) = h.checked_sub(1) else {
            return Vec::new();
        };
        self.shares_included_in(h.saturating_sub(g.blocks_per_epoch), max_inc)
    }

    fn shares_included_in(&self, min_inc: u64, max_inc: u64) -> Vec<ShareRecord> {
        // `shares` is a BTreeMap keyed by work_id, so the result is in ShareID order.
        self.shares
            .values()
            .filter(|s| (min_inc..=max_inc).contains(&s.included_at))
            .cloned()
            .collect()
    }
}

/// Share pre-validated (Stage A and B) for `apply_body` (§7.2).
///
/// The fields are private and [`VerifiedShare::verify`] is the only constructor: it
/// checks the proof's own ML-DSA-87 owner signature and takes every field from that
/// same proof, so `apply_body` never registers a share its owner did not sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedShare {
    work_id: Digest,
    owner: VerifiedOwner,
    weight: u64,
    height: u64,
}

impl VerifiedShare {
    /// Verifies the share's owner signature and returns its `apply_body` view.
    pub fn verify(share: &ShareProof) -> Result<Self, SignatureError> {
        Ok(Self {
            work_id: share.work_id,
            owner: verify_share_owner(share)?,
            weight: share_weight(share.share_target),
            height: share.context.height,
        })
    }

    /// ShareID.
    pub fn work_id(&self) -> Digest {
        self.work_id
    }

    /// Beneficiary, whose signature on this share verified.
    pub fn owner(&self) -> VerifiedOwner {
        self.owner
    }

    /// `w_s`, from the share target.
    pub fn weight(&self) -> u64 {
        self.weight
    }

    /// Height the share was mined at.
    pub fn height(&self) -> u64 {
        self.height
    }
}

/// A share proof together with its [`VerifiedShare`] view, as kept in a share pool.
///
/// [`SignedShare::verify`] is the only constructor, so the two always belong
/// together and the owner signature is checked exactly once, on entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedShare {
    proof: ShareProof,
    verified: VerifiedShare,
}

impl SignedShare {
    /// Verifies the proof's owner signature (not its work) and pairs it with its view.
    pub fn verify(proof: ShareProof) -> Result<Self, SignatureError> {
        let verified = VerifiedShare::verify(&proof)?;
        Ok(Self { proof, verified })
    }

    /// The share proof.
    pub fn proof(&self) -> &ShareProof {
        &self.proof
    }

    /// Its verified `apply_body` view.
    pub fn verified(&self) -> &VerifiedShare {
        &self.verified
    }
}

/// An InitialProtocolPool withdrawal whose treasury dual signature verified under the
/// genesis treasury keys: the `Accept_treasury` conjunct of `InitialOK` (spec eqs.
/// 2.5, 7.19j).
///
/// The fields are private and [`VerifiedWithdrawal::verify`] is the only constructor,
/// so `apply_body` never debits the InitialProtocolPool for a withdrawal the treasury
/// did not sign with both keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedWithdrawal {
    tx_id: Digest,
    expected_nonce: u64,
    amount: u64,
    fee: u64,
    outputs: Vec<WithdrawalOutput>,
}

impl VerifiedWithdrawal {
    /// Checks the withdrawal against the genesis treasury policy, then verifies its
    /// ML-DSA-87 signature and SLH-DSA-SHAKE-256s co-signature for `genesis_id`.
    pub fn verify(
        withdrawal: &InitialPoolWithdrawal,
        genesis: &GenesisConfig,
        genesis_id: &Digest,
    ) -> Result<Self, WithdrawalError> {
        let (amount, fee) = validate_withdrawal_stateless(withdrawal, genesis)?;
        verify_withdrawal_signatures(withdrawal, genesis_id).map_err(WithdrawalError::Signature)?;
        let tx_id = compute_withdrawal_id(withdrawal)
            .map_err(|e| WithdrawalError::Signature(SignatureError::Encoding(e)))?;
        Ok(Self {
            tx_id,
            expected_nonce: withdrawal.expected_nonce,
            amount,
            fee,
            outputs: withdrawal.outputs.clone(),
        })
    }

    /// The withdrawal's `TxID`.
    pub fn tx_id(&self) -> Digest {
        self.tx_id
    }

    /// `ν_expected`.
    pub fn expected_nonce(&self) -> u64 {
        self.expected_nonce
    }

    /// `v`, paid to `U`.
    pub fn amount(&self) -> u64 {
        self.amount
    }

    /// `f`, paid to `R_l`.
    pub fn fee(&self) -> u64 {
        self.fee
    }

    /// The public outputs.
    pub fn outputs(&self) -> &[WithdrawalOutput] {
        &self.outputs
    }
}

/// A withdrawal together with its [`VerifiedWithdrawal`] view, as kept in a mempool.
///
/// [`SignedWithdrawal::verify`] is the only constructor, so both treasury signatures
/// are checked exactly once, on entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedWithdrawal {
    withdrawal: InitialPoolWithdrawal,
    verified: VerifiedWithdrawal,
}

impl SignedWithdrawal {
    /// Verifies the withdrawal (see [`VerifiedWithdrawal::verify`]) and pairs it with its view.
    pub fn verify(
        withdrawal: InitialPoolWithdrawal,
        genesis: &GenesisConfig,
        genesis_id: &Digest,
    ) -> Result<Self, WithdrawalError> {
        let verified = VerifiedWithdrawal::verify(&withdrawal, genesis, genesis_id)?;
        Ok(Self {
            withdrawal,
            verified,
        })
    }

    /// The withdrawal.
    pub fn withdrawal(&self) -> &InitialPoolWithdrawal {
        &self.withdrawal
    }

    /// Its verified `apply_body` view.
    pub fn verified(&self) -> &VerifiedWithdrawal {
        &self.verified
    }
}

/// Derived body commitments produced by `apply_body` (§5.3, §7.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedBody {
    /// `leases[i]` belongs to `jobs[i]`.
    pub leases: Vec<LeaseAssignment>,
    /// Receipts of the jobs settled in this block.
    pub receipts: Vec<SettlementReceipt>,
    /// Every ledger operation, in execution order.
    pub fee_transitions: Vec<FeeTransition>,
    /// Commitment to the post-state.
    pub ledger_digest: Digest,
}

/// Errors produced by `apply_body` (§7.2). Any error rejects the whole block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StfError {
    /// A core ledger operation was rejected.
    Ledger(LedgerError),
    /// More than `max_jobs_per_block` jobs.
    TooManyJobs(usize),
    /// More than `max_shares_per_block` shares.
    TooManyShares(usize),
    /// More than `MAX_WITHDRAWALS_PER_BLOCK` withdrawals.
    TooManyWithdrawals(usize),
    /// Withdrawals are not strictly increasing by `expected_nonce`.
    WithdrawalsNotSorted,
    /// `InitialOK`: the withdrawal does not carry the current `InitialNonce`.
    InitialNonce {
        /// The current `InitialNonce`.
        expected: u64,
        /// The nonce the withdrawal carries.
        found: u64,
    },
    /// `InitialOK`: the InitialProtocolPool cannot cover `v + f`.
    InsufficientInitialPool {
        /// `U_initial`.
        available: u64,
        /// `v + f`.
        requested: u64,
    },
    /// `InitialOK`: the rolling 720-block InitialValve would exceed `L_I`.
    InitialValveExceeded {
        /// Gross outflow already in the window.
        rolling: u64,
        /// `v + f`.
        requested: u64,
    },
    /// Jobs are not strictly increasing by `tx_id`.
    JobsNotSortedOrUnique,
    /// Shares are not strictly increasing by `work_id`.
    SharesNotSortedOrUnique,
    /// A job fails stateless validation.
    InvalidJob(JobValidationError),
    /// A job is included after its expiry height.
    JobExpired {
        /// The job.
        job_id: u64,
        /// Its expiry height.
        expiry: u64,
        /// The including height.
        height: u64,
    },
    /// The job would retain replay state beyond the allowed lifetime.
    JobExpiryTooFar,
    /// Jobs would spend more than one sixteenth of this block's spendable faucet.
    FaucetBudgetExceeded,
    /// The block would exceed the active plus locked fee-lot bound.
    FeeLotCapacity,
    /// The `tx_id` was already used.
    DuplicateTx(Digest),
    /// The `job_id` was already used.
    DuplicateJobId(u64),
    /// Jobs were included while the mature share cohort is empty.
    EmptyLeaseCohort,
    /// The D'Hondt scheduler rejected the cohort or did not fill every slot.
    SchedulerError(String),
    /// The share is already registered.
    DuplicateShare(Digest),
}

impl fmt::Display for StfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StfError::Ledger(e) => write!(f, "ledger error: {e}"),
            StfError::TooManyJobs(n) => write!(f, "too many jobs in block: {n}"),
            StfError::TooManyShares(n) => write!(f, "too many shares in block: {n}"),
            StfError::TooManyWithdrawals(n) => write!(f, "too many withdrawals in block: {n}"),
            StfError::WithdrawalsNotSorted => {
                write!(f, "withdrawals must be strictly sorted by expected_nonce")
            }
            StfError::InitialNonce { expected, found } => write!(
                f,
                "InitialNonce is {expected}, the withdrawal carries {found}"
            ),
            StfError::InsufficientInitialPool {
                available,
                requested,
            } => write!(
                f,
                "InitialProtocolPool holds {available} atoms, the withdrawal needs {requested}"
            ),
            StfError::InitialValveExceeded { rolling, requested } => write!(
                f,
                "InitialValve: {rolling} atoms already in the 720-block window plus {requested} exceeds {}",
                INITIAL_VALVE_LIMIT.get()
            ),
            StfError::JobsNotSortedOrUnique => {
                write!(f, "jobs must be strictly sorted by tx_id")
            }
            StfError::SharesNotSortedOrUnique => {
                write!(f, "shares must be strictly sorted by work_id")
            }
            StfError::InvalidJob(e) => write!(f, "invalid job: {e}"),
            StfError::JobExpired {
                job_id,
                expiry,
                height,
            } => write!(f, "job {job_id} expired at {expiry} (height {height})"),
            StfError::JobExpiryTooFar => write!(f, "job expiry exceeds maximum lifetime"),
            StfError::FaucetBudgetExceeded => write!(f, "per-block faucet debit limit exceeded"),
            StfError::FeeLotCapacity => write!(f, "live fee-lot capacity exceeded"),
            StfError::DuplicateTx(id) => write!(f, "duplicate tx_id {id:?}"),
            StfError::DuplicateJobId(id) => write!(f, "duplicate job_id {id}"),
            StfError::EmptyLeaseCohort => {
                write!(f, "cannot admit jobs with an empty mature share cohort")
            }
            StfError::SchedulerError(msg) => write!(f, "scheduler error: {msg}"),
            StfError::DuplicateShare(id) => write!(f, "duplicate share work_id {id:?}"),
        }
    }
}

impl std::error::Error for StfError {}

impl From<LedgerError> for StfError {
    fn from(e: LedgerError) -> Self {
        StfError::Ledger(e)
    }
}

/// Converts a `ledger::BalanceDelta` to a wire `CategoryDelta` in `StateCategory::ALL` order (§5.7).
pub fn to_category_delta(delta: &BalanceDelta) -> CategoryDelta {
    StateCategory::ALL.map(|c| delta.get(c))
}

/// Computes `ledger_digest` from the post-state (§3.3, §5.8): the height and epoch,
/// the ten category balances, the fee and reward sub-ledgers, the emission counters,
/// `InitialNonce` and the InitialValve ring, in that order.
pub fn compute_ledger_digest(state: &NodeState) -> Digest {
    let ledger = &state.ledger;
    let emission = ledger.emission();
    let valve = ledger.initial_valve();
    let balances = StateCategory::ALL.map(|c| ledger.balance(c).get().to_be_bytes());
    let head = [state.height.to_be_bytes(), ledger.epoch().to_be_bytes()];
    let tail = [
        ledger.segregated_dust().get().to_be_bytes(),
        (ledger.fee_lots().len() as u64).to_be_bytes(),
        (ledger.locked_fees().len() as u64).to_be_bytes(),
        ledger.booked_rewards().get().to_be_bytes(),
        ledger.unbooked_release().get().to_be_bytes(),
        emission.block().to_be_bytes(),
        emission.initial_released().get().to_be_bytes(),
        ledger.initial_nonce().to_be_bytes(),
        valve.head().to_be_bytes(),
        valve.rolling_outflow().get().to_be_bytes(),
    ];
    let miner_state = emission.miner_state().to_be_bytes();
    let buckets: Vec<u8> = valve
        .buckets()
        .iter()
        .flat_map(|b| b.get().to_be_bytes())
        .collect();
    let parts: Vec<&[u8]> = head
        .iter()
        .chain(&balances)
        .chain(&tail)
        .map(|p| p.as_slice())
        .chain([miner_state.as_slice(), buckets.as_slice()])
        .collect();
    domain_hash(LABEL_LEDGER, &parts)
}

/// Normative state transition function `apply_body` (§7.2).
///
/// In order: (0) primary issuance, (1) epoch maintenance, (2) settlements,
/// (3) InitialProtocolPool withdrawals, (4) job funding, (5) leases, (6) share
/// registration, (7) pruning and the invariant check.
///
/// Works on a clone of `parent`; on any error the clone is discarded, so a block
/// applies all-or-nothing (spec eq. 7.10).
pub fn apply_body(
    parent: &NodeState,
    g: &GenesisConfig,
    h: u64,
    jobs: &[JobTransaction],
    shares: &[VerifiedShare],
    withdrawals: &[VerifiedWithdrawal],
) -> Result<(NodeState, DerivedBody), StfError> {
    if jobs.len() as u64 > g.max_jobs_per_block {
        return Err(StfError::TooManyJobs(jobs.len()));
    }
    if shares.len() as u64 > g.max_shares_per_block {
        return Err(StfError::TooManyShares(shares.len()));
    }
    if withdrawals.len() as u64 > MAX_WITHDRAWALS_PER_BLOCK {
        return Err(StfError::TooManyWithdrawals(withdrawals.len()));
    }
    if withdrawals
        .windows(2)
        .any(|w| w[0].expected_nonce >= w[1].expected_nonce)
    {
        return Err(StfError::WithdrawalsNotSorted);
    }
    if shares.windows(2).any(|w| w[0].work_id >= w[1].work_id) {
        return Err(StfError::SharesNotSortedOrUnique);
    }
    if let Some(dup) = shares
        .iter()
        .find(|sh| parent.shares.contains_key(&sh.work_id))
    {
        return Err(StfError::DuplicateShare(dup.work_id));
    }

    let mut s = parent.clone();
    let mut fee_transitions = Vec::new();
    let mut receipts = Vec::new();
    let mut leases = Vec::with_capacity(jobs.len());

    // (0) Primary issuance opens block k = h - h_G (spec eqs. 7.19b–7.19g): the linear
    // initial release R_I -> U_initial and the exact Q64 miner release R_M -> P_pending.
    // `begin_block` also slides the InitialValve window to k and admits only h = parent + 1.
    let (d_primary, release) = s.ledger.begin_block(h)?;
    fee_transitions.push(FeeTransition::PrimaryRelease {
        block: release.block,
        initial: release.initial.get(),
        miner: release.miner.get(),
        delta: to_category_delta(&d_primary),
    });

    // (1) Epoch boundary, spec eqs. (7.20)-(7.22)
    if h.is_multiple_of(g.blocks_per_epoch) {
        let e = s.ledger.epoch();
        let (d_mat, lots) = s.ledger.reserve_mature()?;
        fee_transitions.push(FeeTransition::ReserveMature {
            epoch: e,
            matured_lots: lots.into_iter().map(|l| l.0).collect(),
            delta: to_category_delta(&d_mat),
        });

        // The previous boundary's booking is paid now.
        let paid = s.ledger.booked_rewards();
        let d_rew = s.ledger.fw_reward(paid)?;
        fee_transitions.push(FeeTransition::FwReward {
            epoch: e,
            amount: paid.get(),
            delta: to_category_delta(&d_rew),
        });

        let epoch_cohort = s.epoch_shares(h, g);
        let weights: Vec<ShareWeight> = epoch_cohort.iter().map(|sh| sh.weight).collect();
        let (d_book, payout) = s.ledger.book_epoch(&weights)?;

        // Rewards are keyed by the full 512-bit key id, never by the 64-bit OwnerId.
        let mut owner_totals: BTreeMap<Digest, u64> = BTreeMap::new();
        for (sh, reward) in epoch_cohort.iter().zip(&payout.per_share) {
            if reward.get() > 0 {
                *owner_totals.entry(sh.owner.key_id()).or_insert(0) += reward.get();
            }
        }
        let per_owner: Vec<OwnerReward> = owner_totals
            .into_iter()
            .map(|(owner, atoms)| OwnerReward { owner, atoms })
            .collect();

        fee_transitions.push(FeeTransition::EpochBooked {
            epoch: e,
            budget: payout.budget.get(),
            booked: payout.booked.get(),
            dust: payout.dust.get(),
            release_budget: payout.release_budget.get(),
            release_booked: payout.release_booked.get(),
            release_carry: payout.release_carry.get(),
            lots: payout.lots.len() as u32,
            shares: weights.len() as u32,
            per_owner,
            delta: to_category_delta(&d_book),
        });
    }

    // (2) Settle the jobs admitted at h - settle_delay, in job_id order
    let due_ids: Vec<u64> = s
        .open_jobs
        .values()
        .filter(|j| j.admitted_at.saturating_add(g.settle_delay) == h)
        .map(|j| j.job_id)
        .collect();
    for job_id in due_ids {
        let Some(j) = s.open_jobs.remove(&job_id) else {
            continue;
        };
        let d = s.ledger.result_settle(
            JobId(j.job_id),
            Atoms(j.v_exec),
            Atoms(j.v_da),
            Atoms(j.v_ref),
        )?;
        fee_transitions.push(FeeTransition::ResultSettle {
            job_id: j.job_id,
            v_exec: j.v_exec,
            v_da: j.v_da,
            v_ref: j.v_ref,
            delta: to_category_delta(&d),
        });
        receipts.push(SettlementReceipt {
            job_id: j.job_id,
            tx_id: j.tx_id,
            executor_owner: j.owner,
            admitted_at: j.admitted_at,
            result: j.result,
            result_digest: compute_result_digest(j.job_id, j.result),
            v_exec: j.v_exec,
            v_da: j.v_da,
            v_ref: j.v_ref,
        });
    }

    // (3) InitialProtocolPool withdrawals, each under InitialOK (spec eq. 7.19j)
    for w in withdrawals {
        fee_transitions.push(apply_initial_pool_withdrawal(&mut s.ledger, w)?);
    }

    // Maintenance, settlements and withdrawals above may free fee lots and return
    // value to U. Fix the faucet budget once; each job debits the same cumulative budget.
    let faucet_budget = s.ledger.balance(StateCategory::Circulating).get() / 16;
    let mut faucet_debit = 0u64;
    let live_lots = s
        .ledger
        .fee_lots()
        .len()
        .checked_add(s.ledger.locked_fees().len())
        .and_then(|n| {
            jobs.len()
                .checked_mul(2)
                .and_then(|extra| n.checked_add(extra))
        })
        .ok_or(StfError::FeeLotCapacity)?;
    if live_lots > wire::MAX_LIVE_FEE_LOTS {
        return Err(StfError::FeeLotCapacity);
    }

    // (4) Fund the new jobs (sorted by tx_id)
    let mut validated: Vec<(Digest, u64, i128)> = Vec::with_capacity(jobs.len());
    for tx in jobs {
        let (tx_id, job_id, deposit, dot_result) =
            validate_job_stateless(tx).map_err(StfError::InvalidJob)?;
        if validated.last().is_some_and(|&(prev, _, _)| prev >= tx_id) {
            return Err(StfError::JobsNotSortedOrUnique);
        }
        if h > tx.expiry_height {
            return Err(StfError::JobExpired {
                job_id,
                expiry: tx.expiry_height,
                height: h,
            });
        }
        if tx.expiry_height - h > wire::MAX_JOB_TTL {
            return Err(StfError::JobExpiryTooFar);
        }
        let charge = tx
            .checked_total_charge()
            .ok_or(StfError::InvalidJob(JobValidationError::AmountOverflow))?;
        faucet_debit = faucet_debit
            .checked_add(charge)
            .ok_or(StfError::FaucetBudgetExceeded)?;
        if faucet_debit > faucet_budget {
            return Err(StfError::FaucetBudgetExceeded);
        }
        if s.used_tx.contains_key(&tx_id) {
            return Err(StfError::DuplicateTx(tx_id));
        }
        if s.used_job_ids.contains_key(&job_id) || validated.iter().any(|v| v.1 == job_id) {
            return Err(StfError::DuplicateJobId(job_id));
        }

        let d =
            s.ledger
                .escrow_fund_and_queue(JobId(job_id), Atoms(deposit), Atoms(tx.queue_fee))?;
        fee_transitions.push(FeeTransition::EscrowFundAndQueue {
            job_id,
            deposit,
            queue_fee: tx.queue_fee,
            delta: to_category_delta(&d),
        });
        validated.push((tx_id, job_id, dot_result));
    }

    // (5) Allocate leases with D'Hondt over the mature cohort, spec eq. (6.4)
    if !jobs.is_empty() {
        let cohort = s.lease_cohort(h, g);
        if cohort.is_empty() {
            return Err(StfError::EmptyLeaseCohort);
        }
        let share_ids: Vec<ShareID> = cohort
            .iter()
            .map(|sh| ShareID {
                id: digest_to_hash(&sh.work_id),
                weight: sh.weight,
                owner: OwnerId(sh.owner.owner_id()),
            })
            .collect();

        // Restore each share's history from the parent state. Epoch boundaries and
        // changes to the eligible set must not reset surviving shares' counters.
        let counters: Vec<u64> = cohort.iter().map(|sh| sh.allocated_slots).collect();
        if counters
            .iter()
            .any(|&count| count.checked_add(jobs.len() as u64).is_none())
        {
            return Err(StfError::SchedulerError(
                "D'Hondt allocation counter would overflow".to_string(),
            ));
        }
        let mut scheduler = Cohort::with_counters(share_ids, counters)
            .map_err(|e| StfError::SchedulerError(e.to_string()))?;
        let alloc = scheduler.allocate_batch(jobs.len(), &Capacity::Unbounded);
        if alloc.assigned() != jobs.len() {
            return Err(StfError::SchedulerError(
                "D'Hondt did not assign all requested job slots".to_string(),
            ));
        }

        for (share, &won_slots) in cohort.iter().zip(&alloc.per_share) {
            let record = s.shares.get_mut(&share.work_id).ok_or_else(|| {
                StfError::SchedulerError("allocated share missing from state".to_string())
            })?;
            // The batch-size check above bounds every possible increment.
            record.allocated_slots += won_slots;
        }

        for ((tx, &(tx_id, job_id, dot_result)), &slot) in
            jobs.iter().zip(&validated).zip(&alloc.slots)
        {
            let share = cohort.get(slot).ok_or_else(|| {
                StfError::SchedulerError(format!("slot index {slot} outside the cohort"))
            })?;
            let d = s
                .ledger
                .job_admit_and_lease(JobId(job_id), Atoms(tx.base_fee))?;
            let owner = share.owner.key_id();
            fee_transitions.push(FeeTransition::JobAdmitAndLease {
                job_id,
                base_fee: tx.base_fee,
                owner,
                delta: to_category_delta(&d),
            });
            leases.push(LeaseAssignment {
                job_id,
                share_id: share.work_id,
                owner,
            });
            s.open_jobs.insert(
                job_id,
                OpenJob {
                    job_id,
                    tx_id,
                    admitted_at: h,
                    owner,
                    share_id: share.work_id,
                    v_exec: tx.v_exec,
                    v_da: tx.v_da,
                    v_ref: tx.v_ref,
                    result: dot_result,
                },
            );
            s.used_tx.insert(tx_id, tx.expiry_height);
            s.used_job_ids.insert(job_id, tx.expiry_height);
        }
    }

    // (6) Register the shares of the body
    for sh in shares {
        s.shares.insert(
            sh.work_id,
            ShareRecord {
                work_id: sh.work_id,
                owner: sh.owner,
                weight: sh.weight,
                height: sh.height,
                included_at: h,
                allocated_slots: 0,
            },
        );
    }

    // (7) Prune, check, commit
    s.used_tx.retain(|_, &mut expiry| expiry >= h);
    s.used_job_ids.retain(|_, &mut expiry| expiry >= h);
    let retain = g
        .snapshot_window
        .max(g.blocks_per_epoch)
        .max(g.share_inclusion_window)
        .saturating_add(1);
    if let Some(min_keep) = h.checked_sub(retain) {
        s.shares.retain(|_, sh| sh.included_at >= min_keep);
    }

    s.ledger.check_invariants()?;
    s.height = h;
    let ledger_digest = compute_ledger_digest(&s);

    Ok((
        s,
        DerivedBody {
            leases,
            receipts,
            fee_transitions,
            ledger_digest,
        },
    ))
}

/// `INITIAL_POOL_WITHDRAWAL` under `InitialOK(tx, Σ, k)` (spec eq. 7.19j):
///
/// ```text
/// Accept_treasury(tx) ∧ ν_expected = InitialNonce ∧ U_initial ≥ v + f
///                     ∧ w_rolling + v + f ≤ L_I
/// ```
///
/// `Accept_treasury` holds by construction of [`VerifiedWithdrawal`]; the other three
/// conjuncts are checked here against the state after this block's primary release,
/// whose InitialValve window already ends at block `k` (spec eq. 7.19h). On success
/// `U_initial` pays `v` to `U` and `f` to `R_l`, `InitialNonce` advances and `v + f`
/// enters the window.
fn apply_initial_pool_withdrawal(
    ledger: &mut Ledger,
    w: &VerifiedWithdrawal,
) -> Result<FeeTransition, StfError> {
    let expected = ledger.initial_nonce();
    if w.expected_nonce != expected {
        return Err(StfError::InitialNonce {
            expected,
            found: w.expected_nonce,
        });
    }
    // validate_withdrawal_stateless bounded v + f by M, so this cannot overflow.
    let gross = w.amount + w.fee;
    let available = ledger.balance(StateCategory::InitialProtocolPool).get();
    if available < gross {
        return Err(StfError::InsufficientInitialPool {
            available,
            requested: gross,
        });
    }
    let valve = ledger.initial_valve();
    if !valve.admits(Atoms(gross)) {
        return Err(StfError::InitialValveExceeded {
            rolling: valve.rolling_outflow().get(),
            requested: gross,
        });
    }
    let delta = ledger.initial_pool_withdrawal(w.expected_nonce, Atoms(w.amount), Atoms(w.fee))?;
    Ok(FeeTransition::InitialPoolWithdrawal {
        tx_id: w.tx_id,
        nonce: w.expected_nonce,
        amount: w.amount,
        fee: w.fee,
        valve_rolling: ledger.initial_valve().rolling_outflow().get(),
        delta: to_category_delta(&delta),
    })
}
