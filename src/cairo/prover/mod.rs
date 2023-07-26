use std::marker::PhantomData;

use winterfell::{crypto::{ElementHasher, DefaultRandomCoin}, ProofOptions, Prover, TraceTable};

use crate::Felt252;

use super::air::{CairoAIR, PublicInputs};


pub struct CairoProver<H: ElementHasher> {
    pub_inputs: PublicInputs,
    options: ProofOptions,
    _hasher: PhantomData<H>,
}

impl <H: ElementHasher> CairoProver<H> {

    pub fn new() {}

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
        self.pub_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}