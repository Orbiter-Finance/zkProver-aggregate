
use winterfell::{
    crypto::{DefaultRandomCoin, ElementHasher},
    math::{FieldElement, StarkField},
    ProofOptions, Prover, StarkProof, Trace, TraceTable, VerifierError,
};
use std::time::Instant;
use log::debug;
use core::marker::PhantomData;

use crate::{FETest, cairo::felt252::felt252_fibo::prover::FibSmallProver};

use self::air::FibSmall;

mod air;
mod prover;

#[cfg(test)]
mod tests;

pub fn compute_fib_term<E: FieldElement>(n: usize) -> E {
    let mut t0 = E::ONE;
    let mut t1 = E::ONE;

    for _ in 0..(n - 1) {
        t1 = t0 + t1;
        core::mem::swap(&mut t0, &mut t1);
    }

    t1
}



#[cfg(test)]
pub fn build_proof_options(use_extension_field: bool) -> winterfell::ProofOptions {
    use winterfell::{FieldExtension, ProofOptions};

    let extension = if use_extension_field {
        FieldExtension::Quadratic
    } else {
        FieldExtension::None
    };
    ProofOptions::new(28, 8, 0, extension, 4, 7)
}



// CONSTANTS AND TYPES
// ================================================================================================

const TRACE_WIDTH: usize = 2;
type Rp64_256 = winterfell::crypto::hashers::Rp64_256;
type Sha3_256_felt = winterfell::crypto::hashers::Sha3_256<FETest>;

pub trait Example {
    fn print_result(&self) -> Result<(), VerifierError>;
    fn prove(&self) -> StarkProof;
    fn verify(&self, proof: StarkProof) -> Result<(), VerifierError>;
    fn verify_with_wrong_inputs(&self, proof: StarkProof) -> Result<(), VerifierError>;
}


// EXAMPLE IMPLEMENTATION
// ================================================================================================

pub struct FibExample<H: ElementHasher> {
    options: ProofOptions,
    sequence_length: usize,
    result: FETest,
    _hasher: PhantomData<H>,
}

impl<H: ElementHasher> FibExample<H> {
    pub fn new(sequence_length: usize, options: ProofOptions) -> Self {
        assert!(
            sequence_length.is_power_of_two(),
            "sequence length must be a power of 2"
        );

        // compute Fibonacci sequence
        let now = Instant::now();
        let result = compute_fib_term::<FETest>(sequence_length);
        debug!(
            "Computed Fibonacci sequence up to {}th term in {} ms",
            sequence_length,
            now.elapsed().as_millis()
        );

        FibExample {
            options,
            sequence_length,
            result,
            _hasher: PhantomData,
        }
    }
}

impl<H: ElementHasher> Example for FibExample<H>
where
    H: ElementHasher<BaseField = FETest>,
{
    fn prove(&self) -> StarkProof {
        debug!(
            "Generating proof for computing Fibonacci sequence (2 terms per step) up to {}th term\n\
            ---------------------",
            self.sequence_length
        );
        // create a prover
        let prover = FibSmallProver::<H>::new(self.options.clone());

        // generate execution trace
        let now = Instant::now();
        let trace = prover.build_trace(self.sequence_length);
        let trace_width = trace.width();
        let trace_length = trace.length();
        debug!(
            "Generated execution trace of {} registers and 2^{} steps in {} ms",
            trace_width,
            trace_length.ilog2(),
            now.elapsed().as_millis()
        );
        
        // generate the proof
        let result = prover.prove(trace).unwrap();
        result
    }

    fn verify(&self, proof: StarkProof) -> Result<(), VerifierError> {
        winterfell::verify::<FibSmall, H, DefaultRandomCoin<H>>(proof, self.result)
    }

    fn verify_with_wrong_inputs(&self, proof: StarkProof) -> Result<(), VerifierError> {
        winterfell::verify::<FibSmall, H, DefaultRandomCoin<H>>(
            proof,
            self.result + FETest::ONE,
        )
    }

    fn print_result(&self) -> Result<(), VerifierError> {
        println!("Field ZERO {:?}", FETest::ZERO.to_string());
        println!("Field ONE {:?}", FETest::ONE.to_string());
        println!("Field 16 {:?}", (FETest::from(16_u64)).to_string());
        println!("result {:?}", self.result.to_string());
        Ok(())
    }
}
