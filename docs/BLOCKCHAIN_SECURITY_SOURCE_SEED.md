# Blockchain Security Source Seed v1

Status: curation seed only  
Training authority: none  
Snapshot pin required: yes

This is the first rights-compatible source map for the multi-chain security corpus. A row here is not automatically training-approved. Every source still needs an exact commit/tag snapshot, local review, `sentinel-blockchain-ingest prepare`, immutable digest pinning, privacy filtering, source-catalog admission and the final 50k-document corpus audit.

The seed deliberately separates **training candidates** from **benchmark-only material** so evaluation data cannot silently leak into continued pretraining.

## Initial training candidates

| Source | Role | Rights basis | Chain families | Primary security value |
|---|---|---|---|---|
| `ethereum/EIPs` | formal specification | CC0-1.0 | EVM | protocol/security considerations, transaction and contract standards |
| `ethereum/execution-specs` | formal specification | CC0-1.0 | EVM | executable execution-layer semantics and fork behavior |
| `OpenZeppelin/openzeppelin-contracts` | protocol/library source | MIT | EVM | secure contract patterns, access control, signatures, tokens, governance, vaults |
| `bitcoin/bitcoin` | protocol source | MIT | BITCOIN | consensus, script, wallet, transaction policy, P2P and RPC behavior |
| `anza-xyz/agave` | protocol source | Apache-2.0 | SOLANA | validator/runtime, transaction processing, accounts, consensus and RPC |
| `solana-program/libraries` | protocol/library source | Apache-2.0 | SOLANA | on-chain account/program helpers and SPL-related program primitives |
| `cosmos/cosmos-sdk` | protocol source | Apache-2.0 | COSMOS | application framework, modules, auth, staking and chain state |
| `cosmos/gaia` | protocol source | Apache-2.0 | COSMOS | production Cosmos Hub application and interchain-security integration |
| `cosmos/ibc-go` | protocol source | Apache-2.0 | COSMOS, CROSS_CHAIN | IBC clients, channels, proofs, transfers and cross-chain state machines |
| `MystenLabs/sui` | protocol source | Apache-2.0 code / CC-BY-4.0 docs | MOVE | Move execution, ownership, package upgrades, object/asset security |
| `aptos-labs/move` | language/formal source | Apache-2.0 | MOVE | Move VM, verifier, compiler, prover and language safety material |
| `tronprotocol/tips` | formal specification | CC0-1.0 | TRON | TRON core/API/contract standards and security considerations |
| `tronprotocol/tronweb` | protocol/client source | MIT | TRON | transaction construction, ABI handling, signing/client behavior |
| `tronprotocol/tron-contracts` | contract source | MIT | TRON | TRON Solidity contract patterns and security-relevant examples |

## Benchmark-only reservations

The following sources are reserved for evaluation and must **not** enter training snapshots, training corpora, retrieval corpora used during benchmark execution, or synthetic-teacher prompts derived from benchmark answers:

| Source | Rights | Reserved use |
|---|---|---|
| `openai/frontier-evals` / `project/evmbench` | MIT | EVM security held-out evaluation |
| `paradigmxyz/evmbench` and its pinned EVMbench evaluation material | Apache-2.0 | EVM detect/patch benchmark harness and held-out cases |

Any source transitively containing the same benchmark cases, audit families or answer material must be fingerprinted into the held-out set and rejected by corpus contamination gates.

## Not admitted by the current rights enum

These are technically useful primary sources, but the current `RightsBasis` contract does not represent their licenses. They remain excluded until a deliberate rights-policy review expands the enum and corresponding tests:

- `ethereum/go-ethereum` — LGPL-3.0 for the library and GPL-3.0 for `cmd`;
- `solana-foundation/solana-com` — GPL-3.0;
- `tronprotocol/java-tron` — LGPL-3.0.

Exclusion here is a project-policy choice, not a legal conclusion about whether model training would or would not be permitted by those licenses.

## Snapshot rules

For every admitted training candidate:

1. select an exact immutable commit or release tag;
2. record and independently review the license/rights basis at that snapshot;
3. clone/download into a local source directory without secrets or `.git` internals;
4. label only the chain/threat domains actually represented by the selected paths;
5. create the immutable source snapshot spec and digest;
6. ingest into canonical `sentinel.blockchain-security-document.v1` rows;
7. add the resulting manifest/corpus pair to `sentinel.blockchain-source-catalog.v1`;
8. reject the combined corpus if benchmark content/families, duplicate source content, privacy-sensitive material or source concentration gates fail.

The objective is not to maximize token count. The objective is to maximize **independent, high-signal security evidence per training token** while keeping benchmark families genuinely unseen.
