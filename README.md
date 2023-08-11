# zkProver-aggregate

zkProver-aggregate is to **fully** support the Cairo-VM, we can prove any computation process of Cairo Program and submit its proof to Ethereum, so Dapps can verify its soundness.

# Goals

Developers can develop their own zk-Dapp with [Cairo language](https://github.com/starkware-libs/cairo) for the best zk develop experience.  STARK-proofs will not be submit to the Ethereum directly, we will reduces its gas consumptions by zk-Snarks recursion proof, about 0.3M (Gas/proof verification)

# Features RoadMap
- [ ] Recrusion STARK proof verification

# Optimization RoadMap

- [ ] fully support the cairo builtin, OutPut, RangeCheck, Pedersen, Ecdsa, Keccak...
- [ ] GPU acceleration for backend STARK-prover
- [ ] Proving with Golidlocks field