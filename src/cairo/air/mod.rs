use std::{collections::HashMap, ops::Range};

use winter_air::Assertion;
use winterfell::{AirContext, Air, math::{ToElements, FieldElement, fields::f128::BaseElement}, TraceInfo, ProofOptions, EvaluationFrame, TransitionConstraintDegree};

use crate::cairo::{felt252::BaseElement as Felt252, air::constraints::{MEM_A_TRACE_OFFSET, MEM_P_TRACE_OFFSET}};

use self::constraints::{evaluate_instr_constraints, evaluate_operand_constraints, evaluate_register_constraints, enforce_selector, evaluate_opcode_constraints};

use super::{register_states::RegisterStates, cairo_mem::CairoMemory};

pub mod constraints;
pub mod proof_options;


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
    // pub trace_length: usize,
    pub pub_inputs: PublicInputs,
    // has_rc_builtin: bool,
}

impl Air for CairoAIR {

    type BaseField = Felt252;
    type PublicInputs = PublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {

        debug_assert!(trace_info.length().is_power_of_two());
        let mut main_degrees = vec![];
        
        // Instruction Constraints
        // for _ in 0..=14 {
        //     main_degrees.push(TransitionConstraintDegree::new(2)); // F0-F14
        // }
        main_degrees.append(& mut vec![
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(1), // TODO: why this is always ZERO?
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
            TransitionConstraintDegree::new(2),
        ]);
        main_degrees.push(TransitionConstraintDegree::new(1)); // F15

        // // Other Constraints
        // for _ in 0..=15 {
        //     main_degrees.push(TransitionConstraintDegree::new(3));
        // }

        // // Increasing memory auxiliary constraints.
        // for _ in 0..=4 {
        //     main_degrees.push(TransitionConstraintDegree::new(2));
        // }

        // // Consistent memory auxiliary constraints.
        // for _ in 0..=4 {
        //     main_degrees.push(TransitionConstraintDegree::new(2));
        // }

        // // Permutation auxiliary constraints.
        // for _ in 0..=4 {
        //     main_degrees.push(TransitionConstraintDegree::new(2));
        // }

        // // range-check increasing constraints.
        // for _ in 0..=3 {
        //     main_degrees.push(TransitionConstraintDegree::new(2));
        // }

        // // range-check permutation argument constraints.
        // for _ in 0..=3 {
        //     main_degrees.push(TransitionConstraintDegree::new(2));
        // }

        // let mut aux_degrees =  vec![];

        // let mut num_transition_constraints = 49;
        // let mut num_transition_exemptions = 2;

        // let mut transition_exemptions = vec![];
        // transition_exemptions.extend(vec![1; main_degrees.len()]);
        // transition_exemptions.extend(vec![1; aux_degrees.len()]);
        // let context = 
        //     AirContext::new_multi_segment(trace_info, main_degrees, aux_degrees, 4, 0, options)
        //     .set_num_transition_exemptions(num_transition_exemptions);
        let context = AirContext::new(trace_info, main_degrees, 1, options);
    
        Self {
            context,
            pub_inputs,
        }

    }

    fn context(&self) -> &AirContext<Felt252> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();
        evaluate_instr_constraints(result, current);
        // evaluate_operand_constraints(result, current);
        // evaluate_register_constraints(result, current, next);
        // evaluate_opcode_constraints(result, current);
        // enforce_selector(result, current);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.pub_inputs.num_steps - 1;
        vec![
            // Assertion::single(MEM_A_TRACE_OFFSET, 0, self.pub_inputs.pc_init),
            // Assertion::single(MEM_A_TRACE_OFFSET, last_step, self.pub_inputs.pc_final),
            // Assertion::single(MEM_P_TRACE_OFFSET, 0, self.pub_inputs.ap_init),
            // Assertion::single(MEM_P_TRACE_OFFSET, last_step, self.pub_inputs.ap_final),
            Assertion::single(15, 0, Self::BaseField::from(0_u16)),
        ]
    }
}
