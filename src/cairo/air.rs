use std::{collections::HashMap, ops::Range};

use winterfell::{AirContext, Air, math::ToElements, TraceInfo, ProofOptions};

use crate::cairo::felt252::BaseElement as Felt252;

use super::{register_states::RegisterStates, cairo_mem::CairoMemory};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemorySegment {
    RangeCheck,
    Output,
}

pub type MemorySegmentMap = HashMap<MemorySegment, Range<u64>>;

#[derive(Debug, Clone)]
pub struct PublicInputs {

    pub pc_init: Felt252,
    pub ap_init: Felt252,
    pub fp_init: Felt252,
    pub pc_final: Felt252,
    pub ap_final: Felt252,
    // These are Option because they're not known until
    // the trace is obtained. They represent the minimum
    // and maximum offsets used during program execution.
    // TODO: A possible refactor is moving them to the proof.
    // minimum range check value (0 < range_check_min < range_check_max < 2^16)
    pub range_check_min: Option<u16>,
    // maximum range check value
    pub range_check_max: Option<u16>,
    // Range-check builtin address range
    pub memory_segments: MemorySegmentMap,
    pub public_memory: HashMap<Felt252, Felt252>,
    pub num_steps: usize, // number of execution steps

}

impl ToElements<Felt252> for PublicInputs {
    fn to_elements(&self) -> Vec<Felt252> {
        let mut result: Vec<Felt252> = vec![];
        result.push(self.pc_init);
        result.push(self.ap_init);
        result.push(self.fp_init);
        result.push(self.pc_final);
        result.push(self.ap_final);
        result.push(Felt252::from(self.num_steps as u64));

        // TODO: range_check_min, range_check_max, memory_segments, public_memory, num_steps 

        result
    }
}

impl PublicInputs {

    /// Creates a Public Input from register states and memory
    /// - In the future we should use the output of the Cairo Runner. This is not currently supported in Cairo RS
    /// - RangeChecks are not filled, and the prover mutates them inside the prove function. This works but also should be loaded from the Cairo RS output
    
    pub fn from_regs_and_mem(
        register_states: &RegisterStates,
        memory: &CairoMemory,
        program_size: usize,
        memory_segments: &MemorySegmentMap,
    ) -> Self {
        let output_range = memory_segments.get(&MemorySegment::Output);

        let mut public_memory = (1..=program_size as u64)
        .map(|i| (Felt252::from(i), *memory.get(&i).unwrap()))
        .collect::<HashMap<Felt252, Felt252>>();

        if let Some(output_range) = output_range {
            for addr in output_range.clone() {
                public_memory.insert(Felt252::from(addr), *memory.get(&addr).unwrap());
            }
        };

        let last_step = &register_states.rows[register_states.steps() - 1];

        PublicInputs {
            pc_init: Felt252::from(register_states.rows[0].pc),
            ap_init: Felt252::from(register_states.rows[0].ap),
            fp_init: Felt252::from(register_states.rows[0].fp),
            pc_final: Felt252::from(last_step.pc),
            ap_final: Felt252::from(last_step.ap),
            range_check_min: None,
            range_check_max: None,
            memory_segments: memory_segments.clone(),
            public_memory,
            num_steps: register_states.steps(),
        }
    }
}

#[derive(Clone)]
pub struct CairoAIR {
    pub context: AirContext<Felt252>,
    pub trace_length: usize,
    pub pub_inputs: PublicInputs,
    has_rc_builtin: bool,
}

impl Air for CairoAIR {

    type BaseField = Felt252;
    type PublicInputs = PublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        todo!()
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        todo!()
    }

    fn evaluate_transition<E: winterfell::math::FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &winterfell::EvaluationFrame<E>,
        periodic_values: &[E],
        result: &mut [E],
    ) {
        todo!()
    }

    fn get_assertions(&self) -> Vec<winterfell::Assertion<Self::BaseField>> {
        todo!()
    }
}
