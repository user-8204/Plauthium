# Plauthium

Plauthium is a layer-1 whose consensus mechanism is an exact integer matrix product. Each work evaluation expands two 512×512 signed-INT8 matrices from a seed that commits to the block context, multiplies them with exact INT32 accumulation, and hashes the entire result with SHA3-512; that hash enters a block lottery and a share lottery. Paid computation runs on the same INT8 tiles and adds nothing to chain weight. Signatures are ML-DSA-87 from genesis, and block, transaction and work identifiers are 512-bit SHA3 digests.

This repository implements part of the [Plauthium Technical Specification](https://www.google.com/search?q=Technical_Specification.pdf) in Rust: the arithmetic core (six crates) and a multi-node P2P testnet built on it (three crates). The Circle STARK, the data-availability layer and ML-KEM envelopes are specified but not implemented; [Spec vs. this repository](https://www.google.com/search?q=%23spec-vs-this-repository) lists every gap. This is an experimental testnet, and PLAU on it has no value.

## Architecture overview

```text
Fresh Work:  x -> seed -> A, B -> C = A·B -> outputRoot -> WorkID
             WorkID -> seal ticket  θ_B < Θ_B -> block   (chain weight)
             WorkID -> share ticket θ_S < Θ_S -> share   (rewards and execution rights, no weight)
Jobs:        escrow -> D'Hondt lease over mature ShareIDs -> SciVM result -> settlement (no weight)
Fees:        Base Fee -> locked reserve R_l -> fee lots in R_a -> paid to share weight (no weight)

```

### Consensus: Fresh Work (spec §3.1, §5)

Each attempt first binds a 16-field context `x`: chain and genesis ids, protocol and crypto-suite ids, program and AIR ids, generator id, parent and stable-anchor block, height, operator and payout policy digests, nonce, body root, parent-derived difficulty context and time bucket. Then:

```text
seed       = SHA3-512(Enc("PLAU/FW/INPUT", x))
A          = SHAKE256(Enc("PLAU/FW/A", seed)) -> 512×512 over [-127, 127], per profile
B          = SHAKE256(Enc("PLAU/FW/B", seed)) -> 512×512 over [-127, 127], per profile
C          = A·B over the integers; |C_ij| ≤ 512·127² = 8,258,048 < 2^23
outputRoot = SHA3-512(Enc("PLAU/FW/OUTPUT", C))           # all 512×512 INT32 entries
WorkID     = SHA3-512(Enc("PLAU/FW/ID", x, outputRoot))   # also the ShareID
θ_B        = SHA3-512(Enc("PLAU/FW/SEAL", WorkID))        # seal  if θ_B < Θ_B
θ_S        = SHA3-512(Enc("PLAU/FW/SHARE", WorkID))       # share if θ_S < Θ_S

```

`Enc` is an injective, length-prefixed encoding, and each label is used for exactly one purpose.

* `-128` is excluded because `512·128² = 2^23` would break the strict bound.
* Two admissible results differ by less than `2^24 < p = 2^61 − 1`, so equality mod `p` implies equality over the integers. That is what lets Freivalds over `F_p` check an integer product exactly.
* Identifiers and lottery values depend only on `(x, C)`. Signature bytes, serialization and other witness material enter none of them, so each distinct context gives exactly one draw per lottery (spec Thm 5.1). Grinding over `x` costs a full evaluation per attempt (assumption A1).
* The operand generator rotates on `(h − 1) mod 3`, and its id (3100–3102) is bound into `x`. There is no fallback profile.

| `j` | Profile | Operands |
| --- | --- | --- |
| 0 | Signed outliers | background in {−2, −1, 0, 1, 2}, zero with probability 3/4; in every row an unbiased Fisher–Yates draw sets exactly eight entries to −127 and eight to +127 |
| 1 | Arithmetic graph | `byte & 127`, all entries in [0, 127] |
| 2 | Correlated accumulator stress | a carrier stream fixes a sign `σ_k` per inner index; magnitudes are `127 − (byte & 7)`; every term is positive, so `C_ij ∈ [7,372,800, 8,258,048]`, and a 16-bit partial accumulator overflows after three terms (`3·120² > 32,767`) |

* Fork choice: heaviest cumulative seal weight; a tie keeps the current tip. Shares, jobs, fees and bonds carry zero weight, so buying computation cannot buy fork-choice influence (spec §1.1, §5.5). The node refuses reorgs deeper than 128 blocks.
* Rewards: the block author gets no premium and no fees. Work-reward issuance and recycled fees go pro rata to share weight; a sealing header also counts as a share only if its own share ticket is below `Θ_S`.
* Nobody buys the Fresh Work output itself; its electricity is a security cost for the deterministic state transition. The design bet is that the hardware it needs, integer matrix units, is the same hardware paid jobs use.
* Spec rates: one seal per 120 s and one share per second (`Θ_S/Θ_B = 120`). With a 2 s discovery-to-validation bound, an attacker holding 10% of seal capacity reverses a 34-block confirmation with probability below `2^-40` (spec §5.6, from the Guo–Ren bound).

### Verification: six Freivalds rounds (spec §3.2, §5.3)

In the spec, a seal or share ships the full `C` (1,048,576 bytes). A validator range-checks every entry, regenerates `A, B` in O(n²) and runs six Fiat–Shamir Freivalds rounds over `F_p`, `p = 2^61 − 1`:

```text
v_1..v_6 ∈ F_p^512   derived from SHA3-512(A, B, C) through SHAKE256: 61-bit draws, value p rejected
accept  ⇔  A·(B·v_l) = C·v_l   for l = 1..6

```

* Cost: `3·r_F·n² = 4,718,592` field multiplications, 9/256 (about 3.5%) of the `n³ = 2^27` multiply-accumulates of the product.
* A fixed false `C` passes with probability at most `p^-6 ≈ 2^-366`; a quantum search for an accepting false transcript costs about `p^3 ≈ 2^183`.

Encoded headers (8,203 bytes) and shares (8,059 bytes) carry `outputRoot` but not `C`, so Stage A checks every signature, then regenerates `A, B` and recomputes the full product. `work::verify_freivalds_rounds` is implemented and tested; today it runs as the node's self-check on every seal and share before broadcast, and in the `poc` binary. Moving consensus to Freivalds means putting the 1 MiB `C` on the wire with every seal and share.

### Paid compute: SciVM and Ozaki-II (spec §3.6)

Paid jobs never touch chain weight. For high-precision dense products, the spec's SciVM runs Ozaki Scheme II on the same native tiles:

1. scale the operands to integers with certified bounds `B_X`, `B_Y`;
2. split them into centered residues modulo pairwise-coprime `q_j ≤ 255`, so every residue lies in [−127, 127];
3. multiply the residue matrices as exact INT8×INT8→INT32 tiles and reduce mod `q_j`;
4. recombine by centered CRT and round under the task's bound numerical profile.

A profile is rejected unless `Q = ∏ q_j > 2·B_rec`, with `B_rec = K·B_X·B_Y`; there is no approximate fallback. In the spec, every task carries a Circle STARK proof of the whole program.

This testnet has one job type, `ExactDot`: a 512-element integer dot product with `|x|, |y| ≤ 2^23` under the moduli `{255, 254, 253, 251, 247, 241, 239, 233}` (`Q ≈ 2^63.6 > 2·512·2^23·2^23 = 2^56`). Every validator recomputes the result, and there is no STARK.

Execution slots go to discrete ShareIDs by sequential D'Hondt: quotient `w_s/(a_s + 1)`, compared by exact cross-multiplication, ties to the smallest ShareID. The unit is a share, never an address, and owner labels enter only through bonded capacity, so splitting or merging wallets gains nothing (spec Thm 6.3). The testnet has no bonds, so capacity is unbounded, and settlement credits the aggregate `U`; the lessee is named in the receipt only.

### Supply and issuance (spec §7.6–§7.9)

All amounts are integers (1 PLAU = 10^8 atoms), and ratios are compared by cross-multiplication.

|  | atoms | PLAU |
| --- | --- | --- |
| Fixed supply `M` | 1,500,000,000,000,000 | 15,000,000 |
| Unissued work-reward reserve `R_M` at genesis (90%) | 1,350,000,000,000,000 | 13,500,000 |
| Unissued initial reserve `R_I` at genesis (10%) | 150,000,000,000,000 | 1,500,000 |

* Nothing is issued at genesis and nothing is ever minted: all of `M` starts in the two unissued reserves. Every operation moves atoms between ten disjoint categories (`U`, `U_initial`, `E_esc`, `B_bond`, `V_L2`, `R_a`, `R_l`, `P_pending`, `R_I`, `R_M`) by a zero-sum delta that commits atomically or not at all. `Σ = M`, and each sub-ledger against its category, is re-checked after every block.
* Every block `k ≥ 1` opens with its primary release, before any other transition:
* Work-reward release (Q64): `S_0 = R·2^64`, `S_k = ⌊a·S_{k−1} / 2^64⌋`, `R_M(k) = ⌊S_k / 2^64⌋`, with `a = ⌊2^64·2^(−1/1,051,200)⌋ = 18,446,731,910,177,940,463`. It is exact in 128-bit integers: write `S = u·2^64 + v`, then `⌊a·S/2^64⌋ = u·a + ⌊v·a/2^64⌋`. The half-life is 1,051,200 blocks, four years at the spec's 120 s blocks. Block 1 releases 890,171,600 atoms; after one half-life `R_M` sits 25 atoms below `R/2`. Released atoms wait in `P_pending`, are booked at the next epoch boundary to that epoch's shares pro rata to weight (floor division, remainder carried forward), and are paid into `U` one boundary later.
* Initial release (linear): `I_k = ⌊I·min(k, N_I) / N_I⌋` with `N_I = 1,051,200`, into the InitialProtocolPool `U_initial`: 142,694,063 atoms in block 1, and `R_I` is exactly zero at block `N_I`.


* `U_initial` pays out only through a withdrawal signed with **both** genesis treasury keys (ML-DSA-87 and SLH-DSA-SHAKE-256s; either alone is rejected) that carries the next nonce and keeps gross outflow over any 720 consecutive blocks at or below `I/50` = 30,000 PLAU. The public testnet genesis sets both treasury key ids to zero, so nothing can leave its pool.
* Fees are recycled, never burned. A Base Fee moves irrevocably into the locked reserve `R_l` at admission, matures `L + 1` epochs after payment into an independent fee lot in `R_a`, and each lot pays a ceiling budget `⌈η·A⌉` per epoch to share weight by floor division; rounding dust is segregated permanently. Under the spec's fairness assumption, paying yourself for fictitious jobs loses money in expectation whenever the coalition's discounted payout share `ᾱ·β^(L+1)` is below 1 (spec Thm 7.2).

### Post-quantum cryptography (spec §2)

| Use | Scheme | In this repository |
| --- | --- | --- |
| Header, share and job signatures; owner key id `H(PLAU/OWNER, pk)` | ML-DSA-87, FIPS 204, NIST category 5; 2,592 B public key, 4,627 B signature | implemented with RustCrypto `ml-dsa` 0.1.1; tested against a signature produced by OpenSSL |
| Treasury co-signature, required together with ML-DSA-87 | SLH-DSA-SHAKE-256s, FIPS 205, category 5; 64 B public key, 29,792 B signature | implemented with `fips205` 0.5.0; tested against an independent implementation written from the standard |
| Identifiers, commitments, lottery values, operand streams | SHA3-512 and SHAKE256, FIPS 202 | implemented with RustCrypto `sha3` 0.12 and `shake` 0.1; known-answer tests |
| Confidential job-payload envelopes | ML-KEM-1024, FIPS 203, category 5, with HKDF-SHA-512 and AES-256-GCM | specified (§2.3); **not implemented**, no dependency |

* Every digest is 512 bits. Brassard–Høyer–Tapp collision search takes about `2^(n/3)` queries: roughly `2^85.3` for a 256-bit hash and `2^170.7` for SHA3-512 (spec §2.4).
* No ECDSA, Schnorr, BLS, pairings, Pedersen or KZG anywhere. Signatures use the FIPS 204/205 context string `PLAU/TX`, and every signed digest binds the genesis id.
* `tx_id` and `block_id` exclude signatures, and ML-DSA signing is hedged, so re-signing changes the bytes but not the id. Relay and rejection caches therefore key on a hash of the full signed encoding; a copy with a forged signature cannot shadow the genuine object.

### Node

`plauthium-node` runs a ChainState actor (the only writer of chain state), a mempool actor, the compute threads, a length-delimited TCP P2P layer (bincode 1.3.3: fixed-width little-endian, trailing bytes rejected) with sync and share gossip, an append-only `blocks.log` replayed at startup, and an axum JSON RPC that also serves the explorer. Blocks pass Stage B first (context fields, targets, generator id, policy digests; no tile work), then Stage A (signatures, the header tile, then share tiles). Stage A is limited per source (an IP address, an IPv6 /64, or a loopback address), and an invalid block or share bans its source and cancels everything that source has queued.

### Spec vs. this repository

| Component | Spec | This repository |
| --- | --- | --- |
| Fresh Work, three generator profiles, dual lottery, fork choice | §5 | implemented |
| Fresh Work verification | six Freivalds rounds on the shipped `C` | full recomputation; Freivalds is the node's self-check |
| Circle STARK, `F_{p^8}` challenges, ≤ 19.5 GiB prover peak (a modeled bound, not a measurement) | §2.5, §3.3–§3.5 | not implemented |
| SciVM / Ozaki-II | whole programs under a bound numerical profile | `ExactDot` only, recomputed by every validator |
| P-DAL: rate-1/8 Reed–Solomon, sampling, reconstruction | §4 | not implemented |
| ML-KEM-1024 envelopes | §2.3 | not implemented |
| Public UTXOs, transfers, bonds, refunds, lease timeouts, L2 slots | §6.6, §7.1–§7.5, §7.10 | not implemented: `U` is one aggregate balance, `B_bond` and `V_L2` stay zero, and every admitted job settles after `settle_delay` blocks |
| Ten-category ledger, primary issuance, InitialValve, fee lots | §7.6–§7.8 | implemented |
| D'Hondt leases | §6 | implemented, without bonds |
| Share weight `w_s = ⌊2^640/Θ_S⌋` | §6.1 | narrowed to `u64`, so every D'Hondt and payout product is exact in `u128` |

* Shared faucet: With no per-account balances, a job submitted through any node's RPC is paid from the aggregate `U`, capped at `U/16` per block and 100 PLAU per job. Work rewards are attributed to owner key ids in each block's `EpochBooked` record but land in that same `U`.
* Block timing: The public testnet targets 10 s blocks and the local dev net 5 s, against the spec's 120 s. Issuance and the InitialValve count blocks, so they run 12× and 24× faster in wall-clock time, and the confirmation depths of spec §5.6 do not apply: they assume `T_B ≥ 60·Δ_eff` with `Δ_eff = 2 s`.
* Compute node: The reference compute thread is plain CPU code, one thread by default. There is no GPU kernel.

### Workspace

| Crate | Contents |
| --- | --- |
| `primitives` | atoms and supply, `F_p` with `p = 2^61 − 1`, domain-separated SHA3-512 and SHAKE256 |
| `work` | 512×512 INT8 tile, exact product, Fresh Work context, the three generators, Freivalds |
| `scivm` | Ozaki-II profile, centered residues, centered CRT |
| `scheduler` | sequential D'Hondt over ShareIDs |
| `ledger` | ten categories, Q64 miner release, linear initial release, InitialValve, fee lots |
| `poc` | binary: genesis, three blocks under the three generators, D'Hondt, fees, InitialValve |
| `wire` | P2P messages, block/job/share/withdrawal types, bincode codec, protocol constants |
| `chain` | genesis validation, Stage A/B, state transition, templates, fork choice, keys |
| `node` | `plauthium-node`: actors, P2P, miner, `blocks.log`, JSON RPC, explorer |

`unsafe` is forbidden in every crate. `cargo test --workspace --release` runs 134 tests, including a three-node network in one process. Use `--release`: every block computes real tiles.

## Quick start: local 3-node dev net

You need:

* `bash` on a Unix-like OS. CI runs the test suite on Ubuntu only.
* Rust 1.89 or newer with `cargo`, from [https://rustup.rs](https://rustup.rs). 1.89 is the first release with `File::try_lock`, which each node uses to lock its data directory.
* Free local ports 30341–30343 (P2P) and 8081–8083 (RPC and explorer). The script stops if any of them is taken.
* Optionally `sha256sum`, `shasum` or `openssl`, which the script uses to verify the files it writes.

**1. Check the toolchain:

```sh
rustup update stable
rustc --version    # must print 1.89.0 or newer

```

2. Run the bundle: `Plauthium_Testnet.sh` is this whole workspace as one self-extracting script, generated by `scripts/make_bundle.sh`. `--local` selects the dev net; without it you get a menu, where the dev net is option 3.

```sh
bash Plauthium_Testnet.sh --local

```

The script then:

1. prints the testnet disclaimer;
2. writes the workspace to `./plauthium-workspace` and checks every file against its embedded SHA-256;
3. builds `plauthium-node` with `cargo build --release --locked -p node` (a few minutes the first time; later runs reuse `target/`);
4. writes a fresh genesis to `target/testnet/genesis.json` with `--seal-attempts 128 --target-block-secs 5 --blocks-per-epoch 12`;
5. starts three fully meshed compute nodes on `127.0.0.1:30341–30343`, each with its own ML-DSA-87 key in `target/testnet/n<i>/miner_key.bin`.

To read the code before anything is built or run:

```sh
bash Plauthium_Testnet.sh --extract-only    # write and checksum the files, nothing else
cd plauthium-workspace
scripts/testnet.sh --local                  # build and start the same dev net

```

From a clone of this repository, `scripts/testnet.sh --local` does the same.

3. Open the explorers:

* [http://127.0.0.1:8081/](http://127.0.0.1:8081/)
* [http://127.0.0.1:8082/](http://127.0.0.1:8082/)
* [http://127.0.0.1:8083/](http://127.0.0.1:8083/)

Each shows the ten category balances with a live `Σ = M` check, the next primary release, the InitialValve, a block feed with each block's generator profile, and a job form. The servers listen on `127.0.0.1` only; on a remote machine, forward the ports (`ssh -L 8081:127.0.0.1:8081 <host>`).

4. Submit a job: Jobs are fundable from height 24 (2 × `blocks_per_epoch`), when the first epoch's work-reward release is paid into `U`. A job submitted earlier waits in the mempool; it expires 120 blocks after submission. The node signs the job with its own operator key and charges 1 PLAU by default.

```sh
XS=$(seq 1 512 | paste -sd, -)
YS=$(seq 512 -1 1 | paste -sd, -)
curl -s http://127.0.0.1:8081/api/submit_job \
  -H 'content-type: application/json' \
  -d "{\"payload\":{\"type\":\"exact_dot\",\"xs\":[$XS],\"ys\":[$YS]}}"

```

The node answers `202 Accepted` with the `job_id`, the `tx_id` and `"expected_result":"22500864"` (`Σ i·(513 − i) = 512·513·514/6`). Two blocks after admission (`settle_delay = 2`), a receipt with the same result appears in the settling block's `receipts`:

```sh
curl -s http://127.0.0.1:8081/api/status             # tip, ten balances, issuance, valve, peers, verification counters
curl -s 'http://127.0.0.1:8081/api/blocks?limit=5'   # newest blocks with jobs, receipts, shares and fee transitions

```

5. Stop: Ctrl-C stops all three nodes cleanly. Every `--local` run writes a new genesis and wipes the chain data, but keeps the node keys.

Other modes of the same script:

* `--verifier` joins the public testnet without generating work; `--miner` joins and starts generating work once caught up. Both dial the seed node in `SEED_NODE_IP` on port 30333: set it on line 2 of `Plauthium_Testnet.sh`, or in the environment for `scripts/testnet.sh`. With `SEED_NODE_IP` empty, `--miner` starts a new chain on the public genesis as its first seed node. The key and chain data live in `${PLAUTHIUM_HOME:-${XDG_DATA_HOME:-~/.local/share}/plauthium}`.
* `cargo run --release -p poc` (inside `plauthium-workspace`) takes one genesis through three blocks, D'Hondt, fee maturation and the InitialValve, and prints the ten balances along the way.
* `target/release/plauthium-node --help` lists the `genesis`, `run`, `key` and `treasury` subcommands; `treasury` generates keys and signs withdrawals offline.

## Documentation

* [Plauthium Technical Specification](https://www.google.com/search?q=Technical_Specification.pdf) (24 pages). The normative document: parameter core (Table 1), adversary model A1–A7 (Table 2), post-quantum suite (§2), tile, Freivalds, Circle STARK, prover memory and SciVM/Ozaki-II (§3), data availability (§4), Fresh Work and the dual lottery (§5), D'Hondt scheduler (§6), state, issuance and fees (§7). Code comments cite its equations as `spec eq. N`.
* [Tokenisation of Computing Power in Distributed Ledger Technology](https://www.google.com/search?q=Tokenisation_of_Computing_Power_in_Distributed_Ledger_Technology.pdf) (8 pages). Why the protocol is shaped this way: sovereign debt and issuance, electricity and grid capacity, memory bandwidth and supply-chain concentration, and the post-quantum migration timeline. Table 5 maps each to a protocol rule.

Where the code and the specification disagree, that is a bug; please open an issue.

## License

MIT. See [LICENSE](https://www.google.com/search?q=LICENSE).
