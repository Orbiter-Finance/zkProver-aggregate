use std::marker::PhantomData;

use lambdaworks_crypto::merkle_tree::proof::Proof;
use winterfell::{crypto::{ElementHasher, DefaultRandomCoin}, ProofOptions, Prover, TraceTable, StarkProof, math::FieldElement, DefaultTraceLde};

use crate::Felt252;

use super::{air::{CairoAIR, PublicInputs}, runner::errors::CairoProverError, cairo_trace::CairoWinterTraceTable};


// pub type CairoProverHasher = winterfell::crypto::hashers::Sha3_256<Felt252>;
// pub type CairoProverHasher = winterfell::crypto::hashers::Blake3_256<Felt252>;
pub type CairoProverHasher = winterfell::crypto::hashers::Poseidon<Felt252>;

pub fn prove_cairo_trace<H>(
    trace: CairoWinterTraceTable,
    public_inputs: PublicInputs,
    options: &ProofOptions
)-> Result<(StarkProof, PublicInputs, CairoProver<H>), CairoProverError>
where 
    H: ElementHasher<BaseField = Felt252>,
{

    let prover = CairoProver::<H>::new(public_inputs.clone(), options.clone());
    let proof = prover.prove(trace).map_err(CairoProverError::ProverError).unwrap();
    Ok((proof, public_inputs.clone(), prover))

}

pub fn verify_cairo_proof(
    proof: StarkProof,
    pub_inputs: PublicInputs,
) {
    type rand_coin = DefaultRandomCoin<CairoProverHasher>;
    // verify correct program execution
    match winterfell::verify::<CairoAIR, CairoProverHasher, rand_coin>(proof, pub_inputs) {
        Ok(_) => println!("Execution verified"),
        Err(err) => println!("Failed to verify execution: {}", err),
    }
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
    type Trace = CairoWinterTraceTable;
    type HashFn = H;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> PublicInputs {
        self.public_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}