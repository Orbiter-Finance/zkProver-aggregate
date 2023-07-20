use std::{collections::HashMap, ops::Range};
use crate::cairo::felt252::BaseElement as Felt252;

use super::{register_states::RegisterStates, cairo_mem::CairoMemory};



pub struct CairoAIR {

}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemorySegment {
    RangeCheck,
    Output,
}

pub type MemorySegmentMap = HashMap<MemorySegment, Range<u64>>;

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

// impl PublicInputs {

//     /// Creates a Public Input from register states and memory
//     /// - In the future we should use the output of the Cairo Runner. This is not currently supported in Cairo RS
//     /// - RangeChecks are not filled, and the prover mutates them inside the prove function. This works but also should be loaded from the Cairo RS output
    
//     pub fn from_regs_and_mem(
//         register_states: &RegisterStates,
//         memory: &CairoMemory,
//         program_size: usize,
//         memory_segments: &MemorySegmentMap,
//     ) -> Self {
//         let output_range = memory_segments.get(&MemorySegment::Output);

//         let mut public_memory = (1..=program_size as u64)
//         .map(|i| (Felt252::from(i), *memory.get(&i).unwrap()))
//         .collect::<HashMap<Felt252, Felt252>>();

//         PublicInputs {
//             pc_init: Felt252::from(register_states.rows[0].pc),
//             ap_init: Felt252::from(register_states.rows[0].ap),
//             fp_init: Felt252::from(register_states.rows[0].fp),
//             pc_final: Felt252::from(last_step.pc),
//             ap_final: Felt252::from(last_step.ap),
//             range_check_min: None,
//             range_check_max: None,
//             memory_segments: memory_segments.clone(),
//             public_memory,
//             num_steps: register_states.steps(),
//         }
//     }
// }