use winterfell::DefaultTraceLde;

use super::{
    air::FibSmall, FETest, DefaultRandomCoin, ElementHasher, FieldElement, PhantomData,
    ProofOptions, Prover, Trace, TraceTable, TRACE_WIDTH,
};

// FIBONACCI PROVER
// ================================================================================================

pub struct FibSmallProver<H: ElementHasher> {
    options: ProofOptions,
    _hasher: PhantomData<H>,
}

impl<H: ElementHasher> FibSmallProver<H> {
    pub fn new(options: ProofOptions) -> Self {
        Self {
            options,
            _hasher: PhantomData,
        }
    }

    /// Builds an execution trace for computing a Fibonacci sequence of the specified length such
    /// that each row advances the sequence by 2 terms.
    pub fn build_trace(&self, sequence_length: usize) -> TraceTable<FETest> {
        assert!(
            sequence_length.is_power_of_two(),
            "sequence length must be a power of 2"
        );

        let mut trace = TraceTable::new(TRACE_WIDTH, sequence_length / 2);
        trace.fill(
            |state| {
                state[0] = FETest::ONE;
                state[1] = FETest::ONE;
            },
            |_, state| {
                state[0] += state[1];
                state[1] += state[0];
            },
        );

        trace
    }
}

impl<H: ElementHasher> Prover for FibSmallProver<H>
where
    H: ElementHasher<BaseField = FETest>,
{
    type BaseField = FETest;
    type Air = FibSmall;
    type Trace = TraceTable<FETest>;
    type HashFn = H;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> FETest {
        let last_step = trace.length() - 1;
        let pub_inputs = trace.get(1, last_step);
        // for i in 0..trace.length() {
        //     println!("DUMP_TRACE (0,{:?}) {:?} (1,{:?}) {:?}",i, trace.get(0,i).to_string(), i, trace.get(1,i).to_string());
        // }
        // println!("GET_PUB_INPUTS {}", pub_inputs.to_raw());
        pub_inputs
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}