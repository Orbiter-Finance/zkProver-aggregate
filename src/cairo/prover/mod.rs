use std::marker::PhantomData;

use lambdaworks_crypto::merkle_tree::proof::Proof;
use winterfell::{crypto::{ElementHasher, DefaultRandomCoin}, ProofOptions, Prover, TraceTable, StarkProof};

use crate::Felt252;

use super::{air::{CairoAIR, PublicInputs}, runner::errors::CairoProverError};


pub fn prove_cairo_trace(
    trace: TraceTable<Felt252>,
    public_inputs: PublicInputs,
    options: &ProofOptions
) -> Result<(StarkProof, PublicInputs), CairoProverError>{

    type Sha3_256_felt = winterfell::crypto::hashers::Sha3_256<Felt252>;
    let prover = CairoProver::<Sha3_256_felt>::new(public_inputs.clone(), options.clone());
    let proof = prover.prove(trace).map_err(CairoProverError::ProverError).unwrap();
    Ok((proof, public_inputs.clone()))

}

pub struct CairoProver<H: ElementHasher> {
    public_inputs: PublicInputs,
    options: ProofOptions,
    _hasher: PhantomData<H>,
}

impl <H: ElementHasher> CairoProver<H> {

    pub fn new(public_inputs: PublicInputs, options: ProofOptions) -> Self {
        Self {
            public_inputs,
            options,
            _hasher: PhantomData,
        }
    }

    pub fn build_trace() {}

}

impl <H: ElementHasher> Prover for CairoProver<H> 
where 
    H: ElementHasher<BaseField = Felt252>,
{
    type BaseField = Felt252;
    type Air = CairoAIR;
    type Trace = TraceTable<Felt252>;
    type HashFn = H;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> PublicInputs {
        self.public_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}