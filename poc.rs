//! Plauthium proof of concept: mock block transitions across the protocol modules.
//!
//! a) Genesis ledger: ten transparent categories, all of `M` unissued
//!    (`R_I` = 10 %, `R_M` = 90 %), no Genesis fee lot.
//! b) Blocks 1, 2 and 3: each opens with its primary issuance (the linear initial
//!    release `R_I -> U_initial` and the exact Q64 miner release `R_M -> P_pending`),
//!    then expands its Fresh Work seed with the diagnostic generator of the modulo-3
//!    schedule, computes the exact INT32 product and certifies it with six
//!    Freivalds rounds.
//! c) Scheduler: sequential D'Hondt lease allocation over discrete ShareIDs.
//! d) Ledger: the miner release is booked to shares and paid into `U`; a job's fees
//!    lock in `R_l`, mature into independent fee lots and pay rewards with
//!    segregated rounding dust.
//! e) InitialValve: InitialProtocolPool withdrawals within 30,000 PLAU per 720 blocks.
//! f) ASCII summary.
//!
//! Hashing is SHA3-512/SHAKE256 (see the `primitives` crate). The treasury's dual
//! signature (ML-DSA-87 and SLH-DSA-SHAKE-256s) is checked by `chain::stf`, which
//! this ledger-level PoC does not include.

use std::error::Error;

use ledger::{
    INITIAL_ALLOCATION, INITIAL_RELEASE_BLOCKS, INITIAL_VALVE_LIMIT, INITIAL_VALVE_WINDOW, JobId,
    Ledger, LedgerError, LedgerParams, MINER_ALLOCATION, MINER_DECAY_MULTIPLIER,
    MINER_HALF_LIFE_BLOCKS, PrimaryRelease, Q64, StateCategory, initial_release_cumulative,
    miner_decay_step,
};
use primitives::{Atoms, DomainSeparatedHash, TOTAL_SUPPLY};
use scheduler::{Allocation, OwnerId, ShareID, dhondt_allocation};
use scivm::{CONSENSUS_WEIGHT, OzakiIIProfile, RoundingMode};
use work::{
    ACCUMULATION_BOUND, CLASSICAL_MACS, DEFAULT_FREIVALDS_ROUNDS, FREIVALDS_FIELD_MULS,
    FreshWorkContext, FreshWorkIds, GENERATOR_ID_BASE, GeneratorProfile, TILE_DIM, WorkError,
    WorkUnit, verify_freivalds_rounds,
};

/// Execution slots (ready tiles) offered to the scheduler in this batch.
const LEASE_SLOTS: usize = 8;

type PocResult<T> = Result<T, Box<dyn Error>>;

fn main() -> PocResult<()> {
    println!("Plauthium PoC - mock block transitions");
    println!("(hash/XOF are SHA3-512/SHAKE256; treasury signatures are checked in chain::stf)\n");

    // ---------------------------------------------------------------- (a)
    section("a", "Genesis ledger: 100% of M unissued");
    let mut ledger = Ledger::genesis(LedgerParams::poc())?;
    print_categories(&ledger);
    ensure(
        ledger.fee_lots().is_empty(),
        "R_a must not hold a Genesis lot",
    )?;
    println!(
        "  R_I = {} (10%), R_M = {} (90%); the other 8 categories are zero, R_a has no lots\n",
        INITIAL_ALLOCATION, MINER_ALLOCATION
    );

    // ---------------------------------------------------------------- (b)
    section(
        "b",
        "Blocks 1..3: primary issuance, then Fresh Work under (h - 1) mod 3",
    );
    let mut blocks = Vec::new();
    for height in 1..=3 {
        let (_, release) = ledger.begin_block(height)?;
        print_release(&release);
        ledger.check_invariants()?;
        blocks.push(fresh_work(height)?);
    }
    println!();
    let unknown = mock_context(4, 0, GENERATOR_ID_BASE + 3).instance();
    ensure(
        matches!(unknown, Err(WorkError::UnknownGenerator(_))),
        "a fourth generator id must be rejected",
    )?;
    println!(
        "  generator id {} (a fourth profile) -> rejected: no fallback state\n",
        GENERATOR_ID_BASE + 3
    );
    let ids = blocks[0];

    // ---------------------------------------------------------------- (c)
    section("c", "Scheduler: sequential D'Hondt over discrete ShareIDs");
    let shares = mock_cohort(ids.work_id);
    let allocation = dhondt_allocation(&shares, LEASE_SLOTS)?;
    ensure(
        allocation.assigned() == LEASE_SLOTS,
        "D'Hondt left slots unassigned",
    )?;
    print_allocation(&shares, &allocation);

    // ---------------------------------------------------------------- (d)
    section(
        "d",
        "Ledger: miner release -> shares -> U; job fees -> fee lots",
    );
    let weights: Vec<u64> = shares.iter().map(|s| s.weight).collect();
    let epoch_blocks = ledger.params().blocks_per_epoch;

    // Epoch 1 ends at h = 720: the release of blocks 1..=720 is booked to the cohort
    // and, as in the STF, paid into U at the next boundary.
    advance(&mut ledger, epoch_blocks)?;
    let (_, booking) = ledger.book_epoch(&weights)?;
    step(
        &ledger,
        &format!("EPOCH_BOOKED (h={epoch_blocks})"),
        format!(
            "miner release {} booked, {} atoms carried",
            booking.release_booked,
            booking.release_carry.get()
        ),
    )?;
    advance(&mut ledger, 2 * epoch_blocks)?;
    let paid = ledger.booked_rewards();
    ledger.fw_reward(paid)?;
    step(
        &ledger,
        &format!("FW_REWARD (h={})", 2 * epoch_blocks),
        format!("{paid} paid into U"),
    )?;
    ledger.book_epoch(&weights)?;

    // The commercial job runs on the SciVM under a bound Ozaki-II profile.
    let profile = OzakiIIProfile {
        profile_id: 1,
        moduli: vec![255, 254, 253, 251, 247, 241, 239, 233],
        inner_dim: TILE_DIM as u64,
        bound_x: 1 << 23,
        bound_y: 1 << 23,
        rounding: RoundingMode::NearestTiesToEven,
    };
    profile.validate()?;
    let (xs, ys) = mock_scientific_operands();
    let dot = profile.exact_dot(&xs, &ys)?;
    let direct: i128 = xs
        .iter()
        .zip(&ys)
        .map(|(&a, &b)| a as i128 * b as i128)
        .sum();
    ensure(dot == direct, "Ozaki-II reconstruction mismatch")?;
    println!(
        "  SciVM Ozaki-II: {} moduli, Q ~ 2^{:.1} > 2*B_rec; exact dot = {} (consensus weight {CONSENSUS_WEIGHT})",
        profile.moduli.len(),
        (profile.reconstruction_modulus()? as f64).log2(),
        dot
    );

    // Worked budget of eq. (7.13): D = F_base + v_exec + v_DA + v_ref = 100 PLAU.
    let job = JobId(1);
    let (deposit, queue_fee, base_fee) = (plau(100)?, plau(1)?, plau(10)?);
    ledger.escrow_fund_and_queue(job, deposit, queue_fee)?;
    step(
        &ledger,
        "ESCROW_FUND_AND_QUEUE",
        format!("D = {deposit}, f_queue = {queue_fee}"),
    )?;
    ledger.job_admit_and_lease(job, base_fee)?;
    step(
        &ledger,
        "JOB_ADMIT_AND_LEASE",
        format!("Base Fee {base_fee} -> R_l (irretrievable)"),
    )?;
    ledger.result_settle(job, plau(80)?, plau(5)?, plau(5)?)?;
    step(
        &ledger,
        "RESULT_SETTLE",
        "v_exec 80 + v_DA 5 + v_ref 5 PLAU -> U".to_string(),
    )?;

    let maturity_epoch = ledger
        .locked_fees()
        .iter()
        .map(|f| f.maturity_epoch)
        .max()
        .ok_or("no locked fees")?;
    let maturity_height = ledger.params().genesis_height + maturity_epoch * epoch_blocks;
    advance(&mut ledger, maturity_height)?;
    let (_, matured) = ledger.reserve_mature()?;
    step(
        &ledger,
        "RESERVE_MATURE",
        format!(
            "h={maturity_height} (epoch {maturity_epoch}): {} lots R_l -> R_a",
            matured.len()
        ),
    )?;
    let paid = ledger.booked_rewards();
    ledger.fw_reward(paid)?;
    let (_, epoch) = ledger.book_epoch(&weights)?;
    step(
        &ledger,
        "EPOCH_BOOKED (fee lots)",
        format!(
            "{} from {} lots + {} miner release",
            epoch.booked,
            epoch.lots.len(),
            epoch.release_booked
        ),
    )?;
    for lot in epoch.lots.iter().filter(|l| matured.contains(&l.lot)) {
        let source = ledger
            .fee_lots()
            .iter()
            .find(|l| l.id() == lot.lot)
            .map_or(Atoms::ZERO, |l| l.source_fee());
        println!(
            "    fee lot #{} (F_c = {source}): budget {} atoms = {} booked + {} dust (segregated)",
            lot.lot.0,
            group(lot.budget.get()),
            group(lot.booked.get()),
            lot.dust.get()
        );
    }
    ledger.check_invariants()?;
    println!();
    print_categories(&ledger);

    // ---------------------------------------------------------------- (e)
    section(
        "e",
        "InitialValve: at most L_I = I/50 gross outflow per 720 blocks",
    );
    let valve = valve_demo(&mut ledger)?;
    println!();
    print_categories(&ledger);

    // ---------------------------------------------------------------- (f)
    let miner_at_half_life = issuance_projection()?;
    print_summary(
        &ledger,
        &ids,
        &allocation,
        matured.len(),
        epoch.dust,
        &valve,
        miner_at_half_life,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Fresh Work under the modulo-3 schedule
// ---------------------------------------------------------------------------

/// Expands, multiplies and certifies the Fresh Work of block `height` under its
/// scheduled generator, and prints the profile's defining properties.
fn fresh_work(height: u64) -> PocResult<FreshWorkIds> {
    let profile = GeneratorProfile::for_height(height, 0).ok_or("no Fresh Work at Genesis")?;
    let context = mock_context(height, 0, profile.generator_id());
    let instance = context.instance()?;
    ensure(
        instance.profile == profile,
        "the context names another profile",
    )?;
    println!(
        "  h={height}: profile {} = {profile} (generator id {}), seed {}..",
        profile.index(),
        profile.generator_id(),
        instance.seed.short_hex(12)
    );
    let product = instance.x.product(&instance.y);
    match profile {
        GeneratorProfile::OutlierFuzzing => {
            let rows_ok = [&instance.x, &instance.y].iter().all(|m| {
                (0..TILE_DIM).all(|i| {
                    let row = m.row(i);
                    count(row, -127) == 8 && count(row, 127) == 8
                })
            });
            ensure(rows_ok, "a row lacks its eight -127 and eight +127")?;
            let zeros = count(instance.x.as_slice(), 0) + count(instance.y.as_slice(), 0);
            println!(
                "        every row of X and Y: exactly 8 x -127 and 8 x +127 (Fisher-Yates); zeros {:.3}%",
                100.0 * zeros as f64 / (2 * TILE_DIM * TILE_DIM) as f64
            );
        }
        GeneratorProfile::ArithmeticGraph => {
            let (lo, hi) = range(&instance.x, &instance.y);
            // i8 entries never exceed 127, so the lower end is the profile's property.
            ensure(lo >= 0, "graph entries leave [0, 127]")?;
            let mean =
                (sum(&instance.x) + sum(&instance.y)) as f64 / (2 * TILE_DIM * TILE_DIM) as f64;
            println!(
                "        entries = byte & 127 in [{lo}, {hi}], mean {mean:.2}; nonnegative, so Z >= 0"
            );
        }
        GeneratorProfile::CorrelatedAccumulatorStress => {
            let magnitudes_ok = [&instance.x, &instance.y].iter().all(|m| {
                m.as_slice()
                    .iter()
                    .all(|&v| (120..=127).contains(&v.unsigned_abs()))
            });
            ensure(magnitudes_ok, "|A_ik| leaves 127 - [0, 7]")?;
            let signs_ok = (0..TILE_DIM).all(|k| {
                let sigma = instance.x.get(0, k).signum();
                (0..TILE_DIM).all(|i| instance.x.get(i, k).signum() == sigma)
                    && (0..TILE_DIM).all(|j| instance.y.get(k, j).signum() == sigma)
            });
            ensure(
                signs_ok,
                "the carrier signs differ along an inner dimension",
            )?;
            let min_z = product.as_slice().iter().copied().min().unwrap_or(0);
            ensure(
                min_z >= 120 * 120 * TILE_DIM as i32,
                "a correlated sum is too small",
            )?;
            println!(
                "        A_ik = sigma_k (127 - (byte & 7)): carrier signs agree along k; min Z = {} >= 7,372,800",
                group(min_z as u64)
            );
        }
    }
    let accepted =
        verify_freivalds_rounds(&instance.x, &instance.y, &product, DEFAULT_FREIVALDS_ROUNDS);
    let mut forged = product.clone();
    let entry = forged.get(17, 42);
    forged.set(17, 42, if entry > 0 { entry - 1 } else { entry + 1 })?;
    let forged_rejected =
        !verify_freivalds_rounds(&instance.x, &instance.y, &forged, DEFAULT_FREIVALDS_ROUNDS);
    ensure(accepted, "Freivalds rejected the honest product")?;
    ensure(forged_rejected, "Freivalds accepted a forged product")?;
    ensure(
        product.max_abs() <= i64::from(ACCUMULATION_BOUND),
        "the product exceeds the accumulation bound",
    )?;
    let ids = FreshWorkIds::derive(&context, &product);
    println!(
        "        Z = XY: max|Z| = {} <= {}; Freivalds honest ACCEPT, forged REJECT; WorkID {}..",
        group(product.max_abs() as u64),
        group(ACCUMULATION_BOUND as u64),
        ids.work_id.short_hex(12)
    );
    Ok(ids)
}

fn count(values: &[i8], target: i8) -> usize {
    values.iter().filter(|&&v| v == target).count()
}

fn range(x: &WorkUnit, y: &WorkUnit) -> (i8, i8) {
    let all = x.as_slice().iter().chain(y.as_slice());
    let lo = all.clone().copied().min().unwrap_or(0);
    let hi = all.copied().max().unwrap_or(0);
    (lo, hi)
}

fn sum(m: &WorkUnit) -> i64 {
    m.as_slice().iter().map(|&v| i64::from(v)).sum()
}

// ---------------------------------------------------------------------------
// Issuance and the InitialValve
// ---------------------------------------------------------------------------

/// Opens empty blocks until the ledger is at `height`, re-checking the invariants.
fn advance(ledger: &mut Ledger, height: u64) -> PocResult<()> {
    while ledger.height() < height {
        ledger.begin_block(ledger.height() + 1)?;
    }
    ledger.check_invariants()?;
    Ok(())
}

/// Outcome of the InitialValve demonstration.
struct ValveRun {
    first_height: u64,
    rejected_height: u64,
    reopened_height: u64,
}

/// Waits for the pool to cover 29,990 PLAU, withdraws it, shows a 10 PLAU follow-up
/// rejected by the valve, then accepted once the first outflow leaves the window.
fn valve_demo(ledger: &mut Ledger) -> PocResult<ValveRun> {
    let fee = Atoms(1_000_000);
    let large = plau(29_990)?;
    let small = plau(10)?;
    let gross = large.get() + fee.get();
    let first_height = (ledger.height()..)
        .find(|&k| initial_release_cumulative(k).get() >= gross + 1_000 * 100_000_000)
        .ok_or("the pool never fills")?;
    advance(ledger, first_height)?;
    println!(
        "  h={first_height}: U_initial = {} (sum of i_k)",
        ledger.balance(StateCategory::InitialProtocolPool)
    );
    ledger.initial_pool_withdrawal(0, large, fee)?;
    step(
        ledger,
        "INITIAL_POOL_WITHDRAWAL #0",
        format!(
            "v = {large}, f = {fee}; window {} / {}",
            ledger.initial_valve().rolling_outflow(),
            INITIAL_VALVE_LIMIT
        ),
    )?;
    let rejected_height = first_height + 1;
    advance(ledger, rejected_height)?;
    match ledger.initial_pool_withdrawal(1, small, fee) {
        Err(LedgerError::InitialValveExceeded { rolling, requested }) => println!(
            "  h={rejected_height}: #1 (v = {small}) REJECTED: {rolling} + {requested} > L_I"
        ),
        other => return Err(format!("the valve admitted #1: {other:?}").into()),
    }
    let reopened_height = first_height + INITIAL_VALVE_WINDOW;
    advance(ledger, reopened_height - 1)?;
    ensure(
        ledger.initial_pool_withdrawal(1, small, fee).is_err(),
        "the valve reopened early",
    )?;
    advance(ledger, reopened_height)?;
    ledger.initial_pool_withdrawal(1, small, fee)?;
    step(
        ledger,
        "INITIAL_POOL_WITHDRAWAL #1",
        format!(
            "h={reopened_height}: block {first_height} left the window; window {}",
            ledger.initial_valve().rolling_outflow()
        ),
    )?;
    Ok(ValveRun {
        first_height,
        rejected_height,
        reopened_height,
    })
}

/// Checks that `R_I` empties at `N_I` and returns `R_M` after `H_M` exact decay steps.
fn issuance_projection() -> PocResult<u64> {
    ensure(
        initial_release_cumulative(INITIAL_RELEASE_BLOCKS) == INITIAL_ALLOCATION,
        "R_I does not empty at N_I",
    )?;
    let mut state = u128::from(MINER_ALLOCATION.get()) * Q64;
    for _ in 0..MINER_HALF_LIFE_BLOCKS {
        state = miner_decay_step(state);
    }
    Ok((state >> 64) as u64)
}

fn print_release(release: &PrimaryRelease) {
    println!(
        "  block k={}: PRIMARY_RELEASE  i_k = {} atoms (R_I -> U_initial), m_k = {} atoms (R_M -> P_pending)",
        release.block,
        group(release.initial.get()),
        group(release.miner.get())
    );
}

// ---------------------------------------------------------------------------
// Mock inputs
// ---------------------------------------------------------------------------

fn tag(name: &str) -> DomainSeparatedHash {
    DomainSeparatedHash::new("POC/MOCK", &[name.as_bytes()])
}

fn mock_context(height: u64, attempt_nonce: u64, generator_id: u32) -> FreshWorkContext {
    FreshWorkContext {
        chain_id: 0x504c_4155, // "PLAU"
        genesis_id: tag("genesis"),
        protocol_id: 2,
        crypto_suite_id: 1,
        program_id: tag("fresh-work-program"),
        air_id: tag("int8-tile-air"),
        generator_id,
        parent_canonical_id: tag(&format!("block-{}", height - 1)),
        stable_anchor_id: tag("anchor-0"),
        height,
        miner_policy_digest: tag("miner-policy"),
        payout_policy_digest: tag("payout-policy"),
        attempt_nonce,
        body_core_root: tag(&format!("body-core-{height}")),
        parent_derived_difficulty_context: tag("difficulty-0"),
        time_bucket: height * ledger::BLOCK_INTERVAL_SECONDS,
    }
}

/// The fresh share plus five mature shares from other miners. Owner 2 holds
/// two ShareIDs: its slots are simply the sum over them.
fn mock_cohort(fresh: DomainSeparatedHash) -> Vec<ShareID> {
    let mut shares = vec![ShareID {
        id: fresh,
        weight: 1_000_003,
        owner: OwnerId(1),
    }];
    for (i, (weight, owner)) in [
        (2_400_017u64, 2u64),
        (700_001, 2),
        (1_600_009, 3),
        (300_007, 4),
        (999_983, 5),
    ]
    .into_iter()
    .enumerate()
    {
        shares.push(ShareID {
            id: tag(&format!("mature-share-{i}")),
            weight,
            owner: OwnerId(owner),
        });
    }
    shares
}

/// Deterministic scaled integer operands with |x|, |y| < 2^23.
fn mock_scientific_operands() -> (Vec<f64>, Vec<f64>) {
    let operands = |mul: u64| -> Vec<f64> {
        (0..TILE_DIM as u64)
            .map(|i| ((i.wrapping_mul(mul) >> 7) % (1 << 24)) as f64 - (1 << 23) as f64)
            .collect()
    };
    (operands(2_654_435_761), operands(40_503))
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn plau(n: u64) -> PocResult<Atoms> {
    Atoms::from_plau(n).ok_or_else(|| "PLAU amount overflows".into())
}

fn ensure(condition: bool, message: &str) -> PocResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn section(letter: &str, title: &str) {
    println!("({letter}) {title}");
}

/// Reports one operation and re-checks the supply after it.
fn step(ledger: &Ledger, op: &str, detail: String) -> PocResult<()> {
    ensure(
        ledger.total() == u128::from(TOTAL_SUPPLY.get()),
        "supply changed",
    )?;
    println!("  {op:<32} {detail}");
    Ok(())
}

fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn print_categories(ledger: &Ledger) {
    let rule = "  +---------------------+-----------+---------------------------+";
    println!("{rule}");
    println!(
        "  | {:<19} | {:<9} | {:>25} |",
        "Category", "Symbol", "Balance (atoms)"
    );
    println!("{rule}");
    for category in StateCategory::ALL {
        println!(
            "  | {:<19} | {:<9} | {:>25} |",
            format!("{category:?}"),
            category.symbol(),
            group(ledger.balance(category).get())
        );
    }
    println!("{rule}");
    let total = ledger.total() as u64;
    let verdict = if u128::from(total) == u128::from(TOTAL_SUPPLY.get()) {
        "== M"
    } else {
        "!= M"
    };
    println!(
        "  | {:<19} | {:<9} | {:>25} |",
        "SUM",
        verdict,
        group(total)
    );
    println!("{rule}\n");
}

fn print_allocation(shares: &[ShareID], allocation: &Allocation) {
    let rule = "  +----+--------------------+-------+-----------+-------+";
    println!("{rule}");
    println!("  | #  | ShareID            | Owner | Weight    | Slots |");
    println!("{rule}");
    for (i, (share, slots)) in shares.iter().zip(&allocation.per_share).enumerate() {
        println!(
            "  | {i:<2} | {}.. | {:<5} | {:>9} | {slots:>5} |",
            share.id.short_hex(8),
            share.owner.0,
            group(share.weight)
        );
    }
    println!("{rule}");
    let order: Vec<String> = allocation.slots.iter().map(|s| s.to_string()).collect();
    println!("  lease order (slot -> share #): {}\n", order.join(" "));
}

fn print_summary(
    ledger: &Ledger,
    ids: &FreshWorkIds,
    allocation: &Allocation,
    matured_lots: usize,
    epoch_dust: Atoms,
    valve: &ValveRun,
    miner_at_half_life: u64,
) -> PocResult<()> {
    const WIDTH: usize = 78;
    let border = format!("+{}+", "=".repeat(WIDTH));
    let rule = format!("+{}+", "-".repeat(WIDTH));
    let line = |key: &str, value: String| println!("| {key:<20}: {value:<w$}|", w = WIDTH - 23);
    let distinct = allocation.per_share.iter().filter(|&&a| a > 0).count();
    let emission = ledger.emission();
    let half = MINER_ALLOCATION.get() / 2;

    println!("{border}");
    println!("|{:^WIDTH$}|", "PLAUTHIUM BLOCKS EXECUTED SUCCESSFULLY");
    println!("{border}");
    line(
        "Blocks",
        format!(
            "1..3 with Fresh Work; state advanced to h={}",
            ledger.height()
        ),
    );
    line(
        "Generators",
        format!(
            "h=1,2,3 -> profiles 0,1,2 (ids {}..{}), no fallback",
            GENERATOR_ID_BASE,
            GENERATOR_ID_BASE + 2
        ),
    );
    line(
        "Block 1 BlockID",
        format!("{}..", ids.block_id.short_hex(16)),
    );
    line(
        "Work unit",
        format!(
            "{TILE_DIM}x{TILE_DIM} INT8 -> INT32, {} MACs",
            group(CLASSICAL_MACS)
        ),
    );
    line(
        "Freivalds",
        format!(
            "ACCEPT, r_F = {DEFAULT_FREIVALDS_ROUNDS}, {} mults in F_(2^61-1)",
            group(FREIVALDS_FIELD_MULS)
        ),
    );
    line(
        "D'Hondt leases",
        format!(
            "{}/{LEASE_SLOTS} slots -> {distinct} discrete ShareIDs",
            allocation.assigned()
        ),
    );
    line(
        "Fee lots matured",
        format!("{matured_lots} (queue fee + Base Fee), R_l -> R_a"),
    );
    line(
        "Dust this epoch",
        format!(
            "{} atoms (total segregated {})",
            epoch_dust.get(),
            ledger.segregated_dust().get()
        ),
    );
    line(
        "Initial release",
        format!(
            "{} atoms after k = {}",
            group(emission.initial_released().get()),
            emission.block()
        ),
    );
    line(
        "Miner release",
        format!(
            "{} atoms; a = {MINER_DECAY_MULTIPLIER}",
            group(emission.miner_released().get())
        ),
    );
    line(
        "R_I at N_I",
        format!(
            "0 (all of I released over {} blocks)",
            group(INITIAL_RELEASE_BLOCKS)
        ),
    );
    line(
        "R_M at H_M",
        format!(
            "{} = R/2 - {} atoms",
            group(miner_at_half_life),
            half - miner_at_half_life
        ),
    );
    line(
        "InitialValve",
        format!(
            "h={} 29,990 ok; h={} 10 rejected; h={} ok",
            valve.first_height, valve.rejected_height, valve.reopened_height
        ),
    );
    println!("{rule}");
    for category in StateCategory::ALL {
        line(
            category.symbol(),
            format!(
                "{:<20}{:>25}",
                format!("{category:?}"),
                group(ledger.balance(category).get())
            ),
        );
    }
    println!("{rule}");
    let total = ledger.total();
    ensure(
        total == u128::from(TOTAL_SUPPLY.get()),
        "supply invariant violated",
    )?;
    line(
        "SUM (== M)",
        format!("{:<20}{:>25}  OK", "", group(total as u64)),
    );
    println!("{border}");
    Ok(())
}
