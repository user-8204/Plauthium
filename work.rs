//! The canonical Fresh Work tile (Module 1), its three diagnostic instance
//! generators (Module 3) and its Freivalds verification.
//!
//! The native work unit is the exact integer product `Z = XY` of two
//! `512 x 512` INT8 matrices with entries in `{-127, ..., 127}` (spec §3.1).
//! Every entry of `Z` satisfies `|Z_ij| <= 512 * 127^2 = 8,258,048 < 2^23`
//! (eq. 3.2), so two admissible results that agree modulo `p = 2^61 - 1` are
//! equal over the integers (eq. 3.3). Correctness is checked by `r_F = 6`
//! Freivalds rounds over `F_p` (§3.2), whose static error is at most `p^-6`.
//!
//! The operands rotate through three [`GeneratorProfile`]s on the strict
//! schedule `j_h = (h - h_G - 1) mod 3` (spec §5.1–§5.2, eqs. 5.3a–5.3c).

use core::fmt;

use primitives::{DomainHasher, DomainSeparatedHash, Mersenne61, Shake256Xof, labels};

/// Tile dimension `n = 512`.
pub const TILE_DIM: usize = 512;
/// Entries per tile, `n^2`.
pub const TILE_ELEMS: usize = TILE_DIM * TILE_DIM;
/// Smallest admissible INT8 input. `-128` is excluded: `512 * 128^2 = 2^23`
/// would break the strict bound of eq. (3.2).
pub const INT8_MIN: i8 = -127;
/// Largest admissible INT8 input.
pub const INT8_MAX: i8 = 127;
/// Maximum absolute value of an accumulated INT32 entry, `512 * 127^2`.
pub const ACCUMULATION_BOUND: i32 = 8_258_048;

/// Normative number of Freivalds rounds, `r_F = 6`.
pub const DEFAULT_FREIVALDS_ROUNDS: usize = 6;
/// Classical multiply-accumulate count of one tile, `n^3 = 2^27` (eq. 3.4).
pub const CLASSICAL_MACS: u64 = (TILE_DIM as u64) * (TILE_DIM as u64) * (TILE_DIM as u64);
/// Field multiplications of six Freivalds rounds, `3 r_F n^2 = 4,718,592` (eq. 3.8).
pub const FREIVALDS_FIELD_MULS: u64 = 3 * DEFAULT_FREIVALDS_ROUNDS as u64 * TILE_ELEMS as u64;

const _: () = assert!(TILE_DIM as i64 * 127 * 127 == ACCUMULATION_BOUND as i64);
const _: () = assert!((ACCUMULATION_BOUND as i64) < (1 << 23));
const _: () = assert!(TILE_DIM as i64 * 128 * 128 == 1 << 23);
const _: () = assert!(CLASSICAL_MACS == 1 << 27);
const _: () = assert!(FREIVALDS_FIELD_MULS * 256 == 9 * CLASSICAL_MACS);

/// Matrix entry types admitted by the tile domain.
pub trait TileElement: Copy + Default + Into<i64> {
    /// Largest admissible absolute value.
    const MAX_ABS: i64;
}

impl TileElement for i8 {
    const MAX_ABS: i64 = INT8_MAX as i64;
}

impl TileElement for i32 {
    const MAX_ABS: i64 = ACCUMULATION_BOUND as i64;
}

/// A `512 x 512` row-major matrix whose entries are always inside the tile domain.
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    data: Box<[T]>,
}

/// The canonical Work Unit: a `512 x 512` matrix over `{-127, ..., 127}`.
pub type WorkUnit = Matrix<i8>;
/// The accumulation target: a `512 x 512` INT32 matrix with `|z| <= 8,258,048`.
pub type AccumulationTarget = Matrix<i32>;

/// Rejection reasons for tiles outside the tile domain and for unusable contexts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkError {
    /// The context names no generator profile (`generator_id` outside `3100..=3102`).
    UnknownGenerator(u32),
    /// The entry count is not `512 * 512`.
    Dimension {
        /// Required entry count.
        expected: usize,
        /// Supplied entry count.
        found: usize,
    },
    /// An entry lies outside the admissible range.
    OutOfDomain {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
        /// The offending value.
        value: i64,
        /// The admissible absolute bound.
        max_abs: i64,
    },
}

impl fmt::Display for WorkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkError::UnknownGenerator(id) => write!(f, "unknown generator id {id}"),
            WorkError::Dimension { expected, found } => {
                write!(f, "tile needs {expected} entries, got {found}")
            }
            WorkError::OutOfDomain {
                row,
                col,
                value,
                max_abs,
            } => write!(
                f,
                "entry ({row}, {col}) = {value} is outside [-{max_abs}, {max_abs}]"
            ),
        }
    }
}

impl std::error::Error for WorkError {}

impl<T: TileElement> Matrix<T> {
    /// Builds a tile from row-major entries, rejecting wrong sizes and out-of-domain values.
    pub fn from_vec(data: Vec<T>) -> Result<Self, WorkError> {
        if data.len() != TILE_ELEMS {
            return Err(WorkError::Dimension {
                expected: TILE_ELEMS,
                found: data.len(),
            });
        }
        if let Some(index) = data.iter().position(|&v| !in_domain(v)) {
            return Err(out_of_domain(index, data[index]));
        }
        Ok(Matrix {
            data: data.into_boxed_slice(),
        })
    }

    /// The all-zero tile.
    pub fn zeros() -> Self {
        Matrix {
            data: vec![T::default(); TILE_ELEMS].into_boxed_slice(),
        }
    }

    /// Entry at `(row, col)`. Panics if either index is `>= 512`.
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[index(row, col)]
    }

    /// Replaces the entry at `(row, col)` if the new value is in the domain.
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), WorkError> {
        let i = index(row, col);
        if !in_domain(value) {
            return Err(out_of_domain(i, value));
        }
        self.data[i] = value;
        Ok(())
    }

    /// Row `row` as a slice.
    pub fn row(&self, row: usize) -> &[T] {
        &self.data[row * TILE_DIM..(row + 1) * TILE_DIM]
    }

    /// All entries, row-major.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Largest absolute entry.
    pub fn max_abs(&self) -> i64 {
        self.data.iter().map(|&v| v.into().abs()).max().unwrap_or(0)
    }
}

impl Matrix<i8> {
    /// The identity tile.
    pub fn identity() -> Self {
        let mut m = Self::zeros();
        for i in 0..TILE_DIM {
            m.data[index(i, i)] = 1;
        }
        m
    }

    /// The exact INT32 product `self * rhs` (`n^3` multiply-accumulates).
    ///
    /// No partial sum can leave `[-8,258,048, 8,258,048]`, so the result is in
    /// the accumulation domain by construction.
    pub fn product(&self, rhs: &Matrix<i8>) -> Matrix<i32> {
        let mut out = vec![0i32; TILE_ELEMS];
        for (out_row, lhs_row) in out
            .chunks_exact_mut(TILE_DIM)
            .zip(self.data.chunks_exact(TILE_DIM))
        {
            for (&a, rhs_row) in lhs_row.iter().zip(rhs.data.chunks_exact(TILE_DIM)) {
                if a == 0 {
                    continue;
                }
                let a = i32::from(a);
                for (o, &b) in out_row.iter_mut().zip(rhs_row) {
                    *o += a * i32::from(b);
                }
            }
        }
        Matrix {
            data: out.into_boxed_slice(),
        }
    }
}

impl<T: TileElement + fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix<{}>[{TILE_DIM}x{TILE_DIM}; row0[..4]={:?}]",
            core::any::type_name::<T>(),
            &self.data[..4]
        )
    }
}

fn in_domain<T: TileElement>(value: T) -> bool {
    value.into().abs() <= T::MAX_ABS
}

fn out_of_domain<T: TileElement>(index: usize, value: T) -> WorkError {
    WorkError::OutOfDomain {
        row: index / TILE_DIM,
        col: index % TILE_DIM,
        value: value.into(),
        max_abs: T::MAX_ABS,
    }
}

fn index(row: usize, col: usize) -> usize {
    assert!(row < TILE_DIM && col < TILE_DIM, "tile index out of range");
    row * TILE_DIM + col
}

// ---------------------------------------------------------------------------
// Fresh Work context, seed and instance generation (spec §5.1–§5.2)
// ---------------------------------------------------------------------------

/// Number of fields bound into the Fresh Work context `x` (eq. 5.1).
pub const FRESH_WORK_CONTEXT_FIELDS: usize = 16;

/// The sixteen-field Fresh Work context `x`, bound before any computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshWorkContext {
    /// Chain domain.
    pub chain_id: u64,
    /// Genesis domain.
    pub genesis_id: DomainSeparatedHash,
    /// Protocol identifier.
    pub protocol_id: u32,
    /// Active cryptographic suite.
    pub crypto_suite_id: u32,
    /// Fresh Work program.
    pub program_id: DomainSeparatedHash,
    /// AIR of the Fresh Work proof.
    pub air_id: DomainSeparatedHash,
    /// Instance generator identifier.
    pub generator_id: u32,
    /// Parent block.
    pub parent_canonical_id: DomainSeparatedHash,
    /// Stable anchor.
    pub stable_anchor_id: DomainSeparatedHash,
    /// Height of the attempted block.
    pub height: u64,
    /// Miner policy, fixing the beneficiary before work.
    pub miner_policy_digest: DomainSeparatedHash,
    /// Payout policy.
    pub payout_policy_digest: DomainSeparatedHash,
    /// Attempt nonce.
    pub attempt_nonce: u64,
    /// Body root fixed before the attempt.
    pub body_core_root: DomainSeparatedHash,
    /// Difficulty context derived from the parent.
    pub parent_derived_difficulty_context: DomainSeparatedHash,
    /// Time bucket.
    pub time_bucket: u64,
}

impl FreshWorkContext {
    /// The injective encoding `Enc(x)` of all sixteen fields in normative order.
    pub fn encode(&self) -> Vec<u8> {
        let fields: [&[u8]; FRESH_WORK_CONTEXT_FIELDS] = [
            &self.chain_id.to_be_bytes(),
            self.genesis_id.as_bytes(),
            &self.protocol_id.to_be_bytes(),
            &self.crypto_suite_id.to_be_bytes(),
            self.program_id.as_bytes(),
            self.air_id.as_bytes(),
            &self.generator_id.to_be_bytes(),
            self.parent_canonical_id.as_bytes(),
            self.stable_anchor_id.as_bytes(),
            &self.height.to_be_bytes(),
            self.miner_policy_digest.as_bytes(),
            self.payout_policy_digest.as_bytes(),
            &self.attempt_nonce.to_be_bytes(),
            self.body_core_root.as_bytes(),
            self.parent_derived_difficulty_context.as_bytes(),
            &self.time_bucket.to_be_bytes(),
        ];
        let mut out = Vec::new();
        for field in fields {
            out.extend_from_slice(&(field.len() as u64).to_be_bytes());
            out.extend_from_slice(field);
        }
        out
    }

    /// `seed = H_512(Enc(PLAU/FW/INPUT, x))` (eq. 5.2).
    pub fn seed(&self) -> DomainSeparatedHash {
        DomainSeparatedHash::new(labels::FW_INPUT, &[&self.encode()])
    }

    /// Expands the seed into the operand tiles `X` and `Y` of the generator profile
    /// the context binds through `generator_id` (eqs. 5.3a–5.3c).
    pub fn instance(&self) -> Result<FreshWorkInstance, WorkError> {
        let profile = GeneratorProfile::from_generator_id(self.generator_id)
            .ok_or(WorkError::UnknownGenerator(self.generator_id))?;
        let seed = self.seed();
        let (x, y) = generate_operands(profile, &seed);
        Ok(FreshWorkInstance {
            profile,
            seed,
            x,
            y,
        })
    }
}

/// The operands generated from one Fresh Work seed.
#[derive(Clone, Debug)]
pub struct FreshWorkInstance {
    /// The generator profile that produced the operands.
    pub profile: GeneratorProfile,
    /// The seed both operands were expanded from.
    pub seed: DomainSeparatedHash,
    /// Left operand `X`.
    pub x: WorkUnit,
    /// Right operand `Y`.
    pub y: WorkUnit,
}

// ---------------------------------------------------------------------------
// The three diagnostic generators (spec §5.2, eqs. 5.3a–5.3c)
// ---------------------------------------------------------------------------

/// Generator id of profile `j`: `g_h = 3100 + j_h` (Table 1).
pub const GENERATOR_ID_BASE: u32 = 3100;

/// The diagnostic generator profiles. Block `k = h - h_G >= 1` uses profile
/// `j_h = (k - 1) mod 3`, which is `(h - 1) mod 3` for `h_G = 0`; there is no
/// fourth state and no fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GeneratorProfile {
    /// `j = 0`: signed outliers over small-value background arithmetic (eq. 5.3a).
    OutlierFuzzing,
    /// `j = 1`: nonnegative arithmetic graph kernels (eq. 5.3b).
    ArithmeticGraph,
    /// `j = 2`: correlated dense accumulator stress (eq. 5.3c).
    CorrelatedAccumulatorStress,
}

impl GeneratorProfile {
    /// The profiles, indexed by `j_h`.
    pub const ALL: [GeneratorProfile; 3] = [
        GeneratorProfile::OutlierFuzzing,
        GeneratorProfile::ArithmeticGraph,
        GeneratorProfile::CorrelatedAccumulatorStress,
    ];

    /// The profile scheduled for `height` on a chain whose Genesis is at
    /// `genesis_height`; `None` at or below Genesis, where no Fresh Work exists.
    pub fn for_height(height: u64, genesis_height: u64) -> Option<Self> {
        let k = height.checked_sub(genesis_height).filter(|&k| k >= 1)?;
        Some(Self::ALL[((k - 1) % 3) as usize])
    }

    /// `j_h` in `{0, 1, 2}`.
    pub const fn index(self) -> u32 {
        self as u32
    }

    /// `g_h = 3100 + j_h`, bound into the Fresh Work context.
    pub const fn generator_id(self) -> u32 {
        GENERATOR_ID_BASE + self.index()
    }

    /// The profile a context's `generator_id` names.
    pub fn from_generator_id(generator_id: u32) -> Option<Self> {
        let j = generator_id.checked_sub(GENERATOR_ID_BASE)?;
        Self::ALL.get(j as usize).copied()
    }

    /// Human-readable profile name.
    pub const fn name(self) -> &'static str {
        match self {
            GeneratorProfile::OutlierFuzzing => "Outlier Fuzzing",
            GeneratorProfile::ArithmeticGraph => "Arithmetic Graph",
            GeneratorProfile::CorrelatedAccumulatorStress => "Correlated Accumulator Stress",
        }
    }
}

impl fmt::Display for GeneratorProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Expands `seed` into the operands `(X, Y)` of `profile` in `O(n^2)`.
///
/// `X` reads `R_A = SHAKE256(Enc(PLAU/FW/A, seed))` and `Y` reads
/// `R_B = SHAKE256(Enc(PLAU/FW/B, seed))`, row by row; profile 2 also reads the
/// carrier `SHAKE256(Enc(PLAU/FW/CORRELATION/CARRIER, seed))` (eq. 5.2).
pub fn generate_operands(
    profile: GeneratorProfile,
    seed: &DomainSeparatedHash,
) -> (WorkUnit, WorkUnit) {
    let mut a = Shake256Xof::new(labels::FW_A, seed);
    let mut b = Shake256Xof::new(labels::FW_B, seed);
    match profile {
        GeneratorProfile::OutlierFuzzing => (outlier_tile(&mut a), outlier_tile(&mut b)),
        GeneratorProfile::ArithmeticGraph => (graph_tile(&mut a), graph_tile(&mut b)),
        GeneratorProfile::CorrelatedAccumulatorStress => {
            let mut carrier = Shake256Xof::new(labels::FW_CORRELATION_CARRIER, seed);
            let sigma = carrier_signs(&mut carrier);
            // A_ik = σ_k (127 - E_ik) and B_kj = σ_k (127 - F_kj): the sign follows
            // the inner index k, the column of X and the row of Y.
            let x = correlated_tile(&mut a, |_, k| sigma[k]);
            let y = correlated_tile(&mut b, |k, _| sigma[k]);
            (x, y)
        }
    }
}

/// Outlier entries of each sign per row of profile 0.
const OUTLIERS_PER_SIGN: usize = 8;

/// Background value `V` from the low nibble of one stream byte (eq. 5.3a):
/// `Pr[V = 0] = 12/16 = 3/4` and `Pr[V = -2] = Pr[V = -1] = Pr[V = 1] = Pr[V = 2] = 1/16`.
const BACKGROUND: [i8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 1, 2];

/// Profile 0: each row takes 512 background entries, then a partial Fisher–Yates
/// shuffle selects 16 distinct columns; the first eight become `-127` and the other
/// eight `+127` (a 3.125% outlier density and 72.65625% expected zeros).
fn outlier_tile(xof: &mut Shake256Xof) -> WorkUnit {
    let mut data = vec![0i8; TILE_ELEMS];
    for row in data.chunks_exact_mut(TILE_DIM) {
        for entry in row.iter_mut() {
            *entry = BACKGROUND[usize::from(xof.next_byte() & 0x0f)];
        }
        let mut columns: [u16; TILE_DIM] = core::array::from_fn(|c| c as u16);
        for t in 0..2 * OUTLIERS_PER_SIGN {
            let pick = t + uniform_below(|| xof.next_u32(), (TILE_DIM - t) as u32) as usize;
            columns.swap(t, pick);
            row[usize::from(columns[t])] = if t < OUTLIERS_PER_SIGN {
                INT8_MIN
            } else {
                INT8_MAX
            };
        }
    }
    Matrix {
        data: data.into_boxed_slice(),
    }
}

/// Draws per `UniformBelow` step of the Fisher–Yates shuffle (eq. 5.3a).
const UNIFORM_BELOW_DRAWS: u32 = 8;

/// `UniformBelow(bound)`: unbiased rejection sampling over big-endian 32-bit draws,
/// accepting a draw below the largest multiple of `bound` that fits in `2^32`.
///
/// A step takes at most eight draws. For the shuffle bounds `497..=512` a draw is
/// rejected with probability below `2^-23`, so all eight are with probability below
/// `2^-184`; that case is still defined (the eighth draw reduced modulo `bound`) so
/// that every node derives the same tile.
fn uniform_below(mut next_draw: impl FnMut() -> u32, bound: u32) -> u32 {
    let space = 1u64 << 32;
    let zone = space - space % u64::from(bound);
    let mut draw = 0;
    for _ in 0..UNIFORM_BELOW_DRAWS {
        draw = next_draw();
        if u64::from(draw) < zone {
            break;
        }
    }
    draw % bound
}

/// Profile 1: `A_ik = R_A.byte() & 127` (eq. 5.3b).
fn graph_tile(xof: &mut Shake256Xof) -> WorkUnit {
    let data: Vec<i8> = (0..TILE_ELEMS)
        .map(|_| (xof.next_byte() & 0x7f) as i8)
        .collect();
    Matrix {
        data: data.into_boxed_slice(),
    }
}

/// The carrier signs `σ_k` of profile 2: bit `k` of the carrier stream, least
/// significant bit first, with `0 ↦ +1` and `1 ↦ -1`.
fn carrier_signs(carrier: &mut Shake256Xof) -> [i8; TILE_DIM] {
    let bytes: [u8; TILE_DIM / 8] = core::array::from_fn(|_| carrier.next_byte());
    core::array::from_fn(|k| {
        if (bytes[k / 8] >> (k % 8)) & 1 == 0 {
            1
        } else {
            -1
        }
    })
}

/// Profile 2 (eq. 5.3c): entry `(row, col)` is `sign(row, col) * (127 - (byte & 7))`.
///
/// The 3-bit uniform deficit keeps every magnitude in `120..=127`, so no single
/// bit-plane pattern can be counted away. Because `X` and `Y` share `σ_k`, every
/// accumulation term `σ_k^2 (127 - E_ik)(127 - F_kj)` is positive and
/// `C_ij` lies in `[512 * 120^2, 512 * 127^2] = [7,372,800, 8,258,048]`.
fn correlated_tile(xof: &mut Shake256Xof, sign: impl Fn(usize, usize) -> i8) -> WorkUnit {
    let mut data = vec![0i8; TILE_ELEMS];
    for (row, entries) in data.chunks_exact_mut(TILE_DIM).enumerate() {
        for (col, entry) in entries.iter_mut().enumerate() {
            *entry = sign(row, col) * (INT8_MAX - (xof.next_byte() & 0x07) as i8);
        }
    }
    Matrix {
        data: data.into_boxed_slice(),
    }
}

/// Identifiers and lottery values of one evaluated attempt (eqs. 5.3–5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreshWorkIds {
    /// `outputRoot = H_512(Enc(PLAU/FW/OUTPUT, XY))`.
    pub output_root: DomainSeparatedHash,
    /// `WorkID = H_512(Enc(PLAU/FW/ID, x, outputRoot))`; also the ShareID.
    pub work_id: DomainSeparatedHash,
    /// `BlockID = H_512(Enc(PLAU/BLOCK, x, outputRoot))`.
    pub block_id: DomainSeparatedHash,
    /// Share lottery value `ϑ_S = H_512(Enc(PLAU/FW/SHARE, WorkID))`.
    pub share_ticket: DomainSeparatedHash,
    /// Seal lottery value `ϑ_B = H_512(Enc(PLAU/FW/SEAL, WorkID))`.
    pub seal_ticket: DomainSeparatedHash,
}

impl FreshWorkIds {
    /// Derives all identifiers from the context and the committed product.
    /// No witness material enters these maps (Theorem 5.1).
    pub fn derive(context: &FreshWorkContext, product: &AccumulationTarget) -> Self {
        let x = context.encode();
        let output_root = {
            let mut h = DomainHasher::new(labels::FW_OUTPUT);
            h.field_i32s(product.as_slice());
            h.finalize()
        };
        let work_id = DomainSeparatedHash::new(labels::FW_ID, &[&x, output_root.as_bytes()]);
        let block_id = DomainSeparatedHash::new(labels::BLOCK, &[&x, output_root.as_bytes()]);
        FreshWorkIds {
            output_root,
            work_id,
            block_id,
            share_ticket: DomainSeparatedHash::new(labels::FW_SHARE, &[work_id.as_bytes()]),
            seal_ticket: DomainSeparatedHash::new(labels::FW_SEAL, &[work_id.as_bytes()]),
        }
    }
}

// ---------------------------------------------------------------------------
// Freivalds verification over F_p (spec §3.2)
// ---------------------------------------------------------------------------

/// Checks `A(B v) = C v` over `F_p` for `rounds` independent challenge vectors.
///
/// The vectors are drawn only after `A`, `B` and `C` are bound into a
/// Fiat–Shamir transcript, with one domain per round and rejection sampling
/// to uniform `F_p` elements. Each round costs `3 n^2` field multiplications;
/// the normative six rounds cost `4,718,592 = (9/256) n^3`.
///
/// Returns `false` for `rounds == 0`. Consensus uses [`DEFAULT_FREIVALDS_ROUNDS`].
/// The matrix types already enforce the tile domains, which is what makes
/// agreement modulo `p` imply equality over the integers (eq. 3.3).
pub fn verify_freivalds_rounds(
    a: &Matrix<i8>,
    b: &Matrix<i8>,
    c: &Matrix<i32>,
    rounds: usize,
) -> bool {
    if rounds == 0 {
        return false;
    }
    let transcript = {
        let mut h = DomainHasher::new(labels::FW_FREIVALDS);
        h.field_i8s(a.as_slice())
            .field_i8s(b.as_slice())
            .field_i32s(c.as_slice());
        h.finalize()
    };
    (0..rounds as u64).all(|round| {
        let v = challenge_vector(&transcript, round);
        let bv = mat_vec(b, &v);
        mat_vec(a, &bv) == mat_vec(c, &v)
    })
}

/// Uniform challenge `v^(round)` in `F_p^n`: 61-bit draws, rejecting the single
/// value `p` so every element of `[0, p)` is equally likely.
fn challenge_vector(transcript: &DomainSeparatedHash, round: u64) -> Vec<Mersenne61> {
    let mut round_seed = DomainHasher::new(labels::FW_FREIVALDS);
    round_seed.field_digest(transcript).field_u64(round);
    let mut xof = Shake256Xof::new(labels::FW_FREIVALDS, &round_seed.finalize());
    let mut v = Vec::with_capacity(TILE_DIM);
    while v.len() < TILE_DIM {
        let candidate = xof.next_u64() & Mersenne61::P;
        if candidate != Mersenne61::P {
            v.push(Mersenne61::new(candidate));
        }
    }
    v
}

fn mat_vec<T: TileElement>(m: &Matrix<T>, v: &[Mersenne61]) -> Vec<Mersenne61> {
    m.data
        .chunks_exact(TILE_DIM)
        .map(|row| signed_dot(row, v))
        .collect()
}

/// `sum_k row[k] * v[k]` over `F_p`. Positive and negative terms accumulate
/// exactly in `u128`: `|row[k]| <= 2^23` and `v[k] < 2^61`, so each sum stays
/// below `512 * 2^84 = 2^93`.
fn signed_dot<T: TileElement>(row: &[T], v: &[Mersenne61]) -> Mersenne61 {
    let (mut positive, mut negative) = (0u128, 0u128);
    for (&x, y) in row.iter().zip(v) {
        let x: i64 = x.into();
        let term = u128::from(x.unsigned_abs()) * u128::from(y.value());
        if x >= 0 {
            positive += term;
        } else {
            negative += term;
        }
    }
    Mersenne61::new(Mersenne61::reduce_u128(positive))
        - Mersenne61::new(Mersenne61::reduce_u128(negative))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(nonce: u64) -> FreshWorkContext {
        context_at(1, nonce)
    }

    /// A context at `height` (Genesis at 0) that binds the scheduled generator.
    fn context_at(height: u64, nonce: u64) -> FreshWorkContext {
        let d = |tag: &[u8]| DomainSeparatedHash::new("TEST/CTX", &[tag]);
        let profile = GeneratorProfile::for_height(height, 0).expect("height after Genesis");
        FreshWorkContext {
            chain_id: 1,
            genesis_id: d(b"genesis"),
            protocol_id: 2,
            crypto_suite_id: 1,
            program_id: d(b"program"),
            air_id: d(b"air"),
            generator_id: profile.generator_id(),
            parent_canonical_id: d(b"parent"),
            stable_anchor_id: d(b"anchor"),
            height,
            miner_policy_digest: d(b"miner"),
            payout_policy_digest: d(b"payout"),
            attempt_nonce: nonce,
            body_core_root: d(b"body"),
            parent_derived_difficulty_context: d(b"difficulty"),
            time_bucket: 0,
        }
    }

    #[test]
    fn accumulation_bound_is_exact_and_strict() {
        let n: i64 = TILE_DIM as i64;
        let bound = n * 127 * 127;
        assert!(bound <= ACCUMULATION_BOUND as i64);
        assert_eq!(bound, 8_258_048);
        assert!(bound < 1 << 23);
        // Admitting -128 would reach 2^23 exactly and break strictness.
        assert_eq!(n * 128 * 128, 1 << 23);
        // Any two admissible results differ by < 2^24 < p (eq. 3.3).
        assert!(2 * bound < 1 << 24);
        assert!((1i64 << 24) < Mersenne61::P as i64);
    }

    #[test]
    fn freivalds_cost_is_nine_256ths_of_classical() {
        assert_eq!(FREIVALDS_FIELD_MULS, 4_718_592);
        assert_eq!(CLASSICAL_MACS, 134_217_728);
        assert_eq!(FREIVALDS_FIELD_MULS * 256, 9 * CLASSICAL_MACS);
    }

    #[test]
    fn work_unit_domain_excludes_minus_128() {
        let mut entries = vec![0i8; TILE_ELEMS];
        entries[7] = INT8_MIN;
        entries[8] = INT8_MAX;
        assert!(WorkUnit::from_vec(entries.clone()).is_ok());
        entries[9] = i8::MIN;
        assert_eq!(
            WorkUnit::from_vec(entries).unwrap_err(),
            WorkError::OutOfDomain {
                row: 0,
                col: 9,
                value: -128,
                max_abs: 127
            }
        );
        assert!(matches!(
            WorkUnit::from_vec(vec![0; 10]),
            Err(WorkError::Dimension { found: 10, .. })
        ));
        let mut m = WorkUnit::zeros();
        assert!(m.set(511, 511, i8::MIN).is_err());
        assert_eq!(m.get(511, 511), 0);
    }

    #[test]
    fn accumulation_target_domain_is_bounded() {
        let mut target = AccumulationTarget::zeros();
        assert!(target.set(0, 0, ACCUMULATION_BOUND).is_ok());
        assert!(target.set(0, 1, -ACCUMULATION_BOUND).is_ok());
        assert!(target.set(0, 2, ACCUMULATION_BOUND + 1).is_err());
        assert!(target.set(0, 3, -ACCUMULATION_BOUND - 1).is_err());
        assert!(AccumulationTarget::from_vec(vec![1 << 23; TILE_ELEMS]).is_err());
    }

    #[test]
    fn schedule_rotates_through_three_profiles_without_a_fourth_state() {
        use GeneratorProfile::*;
        let expected = [
            (1, OutlierFuzzing),
            (2, ArithmeticGraph),
            (3, CorrelatedAccumulatorStress),
            (4, OutlierFuzzing),
            (5, ArithmeticGraph),
            (6, CorrelatedAccumulatorStress),
            (3_001, OutlierFuzzing),
        ];
        for (height, profile) in expected {
            assert_eq!(GeneratorProfile::for_height(height, 0), Some(profile));
            assert_eq!(u64::from(profile.index()), (height - 1) % 3);
        }
        // The schedule counts blocks from Genesis, which has no Fresh Work.
        assert_eq!(GeneratorProfile::for_height(101, 100), Some(OutlierFuzzing));
        assert_eq!(
            GeneratorProfile::for_height(103, 100),
            Some(CorrelatedAccumulatorStress)
        );
        assert_eq!(GeneratorProfile::for_height(100, 100), None);
        assert_eq!(GeneratorProfile::for_height(0, 0), None);
    }

    #[test]
    fn generator_ids_are_3100_to_3102() {
        let ids = GeneratorProfile::ALL.map(GeneratorProfile::generator_id);
        assert_eq!(ids, [3100, 3101, 3102]);
        for profile in GeneratorProfile::ALL {
            assert_eq!(
                GeneratorProfile::from_generator_id(profile.generator_id()),
                Some(profile)
            );
        }
        for unknown in [0, 2, 3099, 3103, u32::MAX] {
            assert_eq!(GeneratorProfile::from_generator_id(unknown), None);
        }
        let mut ctx = context(0);
        ctx.generator_id = 3103;
        assert_eq!(
            ctx.instance().unwrap_err(),
            WorkError::UnknownGenerator(3103)
        );
    }

    #[test]
    fn instances_are_deterministic_seed_bound_and_in_domain() {
        for height in 1..=3 {
            let ctx = context_at(height, 0);
            let inst = ctx.instance().unwrap();
            assert_eq!(inst.profile.index(), (height - 1) as u32);
            assert_eq!(inst.seed, ctx.seed());
            let again = ctx.instance().unwrap();
            assert_eq!((&inst.x, &inst.y), (&again.x, &again.y));
            assert_ne!(
                inst.x, inst.y,
                "PLAU/FW/A and PLAU/FW/B are separate domains"
            );
            let other = context_at(height, 1).instance().unwrap();
            assert_ne!(inst.x, other.x, "the nonce changes the seed");
            for m in [&inst.x, &inst.y] {
                assert!(
                    m.as_slice()
                        .iter()
                        .all(|&v| (INT8_MIN..=INT8_MAX).contains(&v))
                );
            }
        }
    }

    #[test]
    fn outlier_rows_carry_exactly_eight_of_each_sign() {
        let inst = context_at(1, 0).instance().unwrap();
        let entries = (2 * TILE_ELEMS) as f64;
        let mut zeros = 0usize;
        let mut hit = [false; TILE_DIM];
        for m in [&inst.x, &inst.y] {
            for r in 0..TILE_DIM {
                let row = m.row(r);
                assert_eq!(row.iter().filter(|&&v| v == INT8_MIN).count(), 8, "row {r}");
                assert_eq!(row.iter().filter(|&&v| v == INT8_MAX).count(), 8, "row {r}");
                assert!(row.iter().all(|&v| v.abs() == 127 || (-2..=2).contains(&v)));
                zeros += row.iter().filter(|&&v| v == 0).count();
                for (c, &v) in row.iter().enumerate() {
                    hit[c] |= v.abs() == 127;
                }
            }
        }
        // (496 / 512) * 3/4 = 72.65625% expected zeros; the standard error is ~0.06%.
        let fraction = zeros as f64 / entries;
        assert!(
            (fraction - 0.726_562_5).abs() < 0.004,
            "zero fraction {fraction}"
        );
        // 1,024 rows of 16 picks reach essentially every column.
        assert!(hit.iter().filter(|&&h| h).count() > 500);
    }

    #[test]
    fn graph_entries_are_nonnegative_seven_bit_values() {
        let inst = context_at(2, 0).instance().unwrap();
        for m in [&inst.x, &inst.y] {
            assert!(m.as_slice().iter().all(|&v| (0..=127).contains(&v)));
            assert!(m.as_slice().contains(&0) && m.as_slice().contains(&127));
        }
    }

    #[test]
    fn correlated_operands_share_signs_and_stress_the_accumulator() {
        let inst = context_at(3, 0).instance().unwrap();
        let (x, y) = (&inst.x, &inst.y);
        let mut deficits = [0usize; 8];
        for k in 0..TILE_DIM {
            let sigma = x.get(0, k).signum();
            for i in 0..TILE_DIM {
                assert_eq!(x.get(i, k).signum(), sigma, "column {k} of X follows σ_k");
                assert_eq!(y.get(k, i).signum(), sigma, "row {k} of Y follows σ_k");
                assert!((120..=127).contains(&x.get(i, k).abs()));
                assert!((120..=127).contains(&y.get(k, i).abs()));
                deficits[(127 - x.get(i, k).abs()) as usize] += 1;
            }
        }
        // The 3-bit deficit is uniform: each of the eight values about 1/8 of the time.
        let expected = (TILE_ELEMS / 8) as f64;
        assert!(
            deficits
                .iter()
                .all(|&n| (n as f64 - expected).abs() < 0.02 * expected)
        );
        // Every accumulation term is positive, so C stays in the narrow positive band.
        let z = x.product(y);
        let (min, max) = z
            .as_slice()
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(min >= 512 * 120 * 120, "min {min}");
        assert!(max <= ACCUMULATION_BOUND, "max {max}");
        let mean = z.as_slice().iter().map(|&v| f64::from(v)).sum::<f64>() / TILE_ELEMS as f64;
        // E[C_ij] = 512 (127 - 3.5)^2 = 7,809,152.
        assert!((mean - 7_809_152.0).abs() < 2_000.0, "mean {mean}");
    }

    #[test]
    fn uniform_below_rejects_the_biased_tail_within_eight_draws() {
        let scripted = |draws: Vec<u32>| {
            let mut iter = draws.into_iter();
            move || iter.next().expect("no more scripted draws")
        };
        // 2^32 mod 500 = 296, so draws >= 2^32 - 296 are rejected.
        let zone = ((1u64 << 32) - (1u64 << 32) % 500) as u32;
        assert_eq!(
            uniform_below(scripted(vec![u32::MAX, zone, 1_234]), 500),
            1_234 % 500
        );
        assert_eq!(
            uniform_below(scripted(vec![zone - 1]), 500),
            (zone - 1) % 500
        );
        // A power-of-two bound never rejects.
        assert_eq!(uniform_below(scripted(vec![u32::MAX]), 512), 511);
        // Eight rejections end the step with the eighth draw.
        let mut calls = 0;
        let value = uniform_below(
            || {
                calls += 1;
                u32::MAX
            },
            500,
        );
        assert_eq!((calls, value), (8, u32::MAX % 500));
    }

    /// Reference values from an independent implementation of eqs. (5.2)–(5.3c)
    /// (Python `hashlib` with a numpy tile product) for nonce 0 at heights 1–3.
    #[test]
    fn generators_match_an_independent_implementation() {
        let vectors = [
            (
                1,
                "e8ef9e75eaf322291ce7a56875e31780",
                "184e932e6a6e47d50df161a29181af7d",
                "56f645019e8a3a0c63eb24055aa5c313",
                "7d036b2e981ae3c77260ac2393c5806e",
                [16_013, -16_286],
            ),
            (
                2,
                "ceecc61df168a084a7555a882243291b",
                "a8bcb7c132c235e4d091d22e24c5a99f",
                "09a3744896be353a8e5fda306d7391e4",
                "02abf3683c4e6fc5bddbf5d840876f93",
                [2_103_174, 1_962_711],
            ),
            (
                3,
                "0e50971c541ec455ef3df986f31a8630",
                "f8568e29a3464bfa75257d7128a03a68",
                "513caae3b58ca94a20fe9e724e20c481",
                "01d2ac2f0cca4cb2a0535221354867ba",
                [7_802_462, 7_814_933],
            ),
        ];
        for (height, seed, root, work, block, corners) in vectors {
            let ctx = context_at(height, 0);
            let inst = ctx.instance().unwrap();
            let z = inst.x.product(&inst.y);
            let ids = FreshWorkIds::derive(&ctx, &z);
            assert_eq!(inst.seed.short_hex(16), seed, "seed at h={height}");
            assert_eq!([z.get(0, 0), z.get(511, 511)], corners, "C at h={height}");
            assert_eq!(
                ids.output_root.short_hex(16),
                root,
                "outputRoot at h={height}"
            );
            assert_eq!(ids.work_id.short_hex(16), work, "WorkID at h={height}");
            assert_eq!(ids.block_id.short_hex(16), block, "BlockID at h={height}");
        }
    }

    #[test]
    fn context_binds_sixteen_fields() {
        let ctx = context(0);
        assert_eq!(
            ctx.encode().len(),
            FRESH_WORK_CONTEXT_FIELDS * 8 + 9 * 64 + 4 * 8 + 3 * 4
        );
        let mut moved = ctx.clone();
        moved.time_bucket = 1;
        assert_ne!(ctx.seed(), moved.seed());
    }

    #[test]
    fn freivalds_accepts_true_product_and_rejects_tampering() {
        let inst = context(0).instance().unwrap();
        let z = inst.x.product(&inst.y);
        assert!(z.max_abs() <= ACCUMULATION_BOUND as i64);
        assert!(verify_freivalds_rounds(
            &inst.x,
            &inst.y,
            &z,
            DEFAULT_FREIVALDS_ROUNDS
        ));

        // A single entry off by one is caught.
        let mut tampered = z.clone();
        let v = tampered.get(300, 17);
        tampered
            .set(300, 17, if v > 0 { v - 1 } else { v + 1 })
            .unwrap();
        assert!(!verify_freivalds_rounds(
            &inst.x,
            &inst.y,
            &tampered,
            DEFAULT_FREIVALDS_ROUNDS
        ));

        // Operand order matters: YX is not XY.
        assert!(!verify_freivalds_rounds(
            &inst.y,
            &inst.x,
            &z,
            DEFAULT_FREIVALDS_ROUNDS
        ));
        assert!(!verify_freivalds_rounds(&inst.x, &inst.y, &z, 0));

        let ids = FreshWorkIds::derive(&context(0), &z);
        assert_eq!(ids, FreshWorkIds::derive(&context(0), &z));
        assert_ne!(ids.work_id, ids.block_id);
        assert_ne!(ids.share_ticket, ids.seal_ticket);
        assert_ne!(
            ids.output_root,
            FreshWorkIds::derive(&context(0), &tampered).output_root
        );
    }

    #[test]
    fn freivalds_accepts_extreme_admissible_product() {
        // X = -127 everywhere, Y = 127 everywhere: every entry is exactly -8,258,048.
        let x = WorkUnit::from_vec(vec![INT8_MIN; TILE_ELEMS]).unwrap();
        let y = WorkUnit::from_vec(vec![INT8_MAX; TILE_ELEMS]).unwrap();
        let z = AccumulationTarget::from_vec(vec![-ACCUMULATION_BOUND; TILE_ELEMS]).unwrap();
        assert!(verify_freivalds_rounds(
            &x,
            &y,
            &z,
            DEFAULT_FREIVALDS_ROUNDS
        ));
        let id = WorkUnit::identity();
        assert!(verify_freivalds_rounds(
            &id,
            &y,
            &id.product(&y),
            DEFAULT_FREIVALDS_ROUNDS
        ));
    }
}
