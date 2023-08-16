use std::{collections::HashMap, ops::Range};

use serde::{Serialize, ser::SerializeTuple};
use winter_air::Assertion;
use winter_utils::{Serializable, ByteWriter};
use winterfell::{AirContext, Air, math::{ToElements, FieldElement, fields::f128::BaseElement}, TraceInfo, ProofOptions, EvaluationFrame, TransitionConstraintDegree};

use crate::cairo::{felt252::BaseElement as Felt252, air::constraints::{MEM_A_TRACE_OFFSET, MEM_P_TRACE_OFFSET}};

use self::constraints::{evaluate_instr_constraints, evaluate_operand_constraints, evaluate_register_constraints, enforce_selector, evaluate_opcode_constraints, BUILTIN_OFFSET, evaluate_aux_memory_constraints};

use super::{register_states::RegisterStates, cairo_mem::CairoMemory};

pub mod constraints;
pub mod proof_options;


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemorySegment {
    RangeCheck,
    Output,
}

pub type MemorySegmentMap = HashMap<MemorySegment, Range<u64>>;

/// Trait for compatibility between implementations of [winterfell::Air::PublicInputs]
/// and this crate.
///
/// It simply requires that the number of public inputs be specified (through the
/// [NUM_PUB_INPUTS](WinterPublicInputs::NUM_PUB_INPUTS) constant).
pub trait WinterPublicInputs: Serialize + Clone {
    const NUM_PUB_INPUTS: usize;
}

impl WinterPublicInputs for PublicInputs {
    const NUM_PUB_INPUTS: usize = 6;
}

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

impl Serialize for PublicInputs {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
    {
        let mut state = serializer.serialize_tuple(6)?;
        state.serialize_element(&self.pc_init);
        state.serialize_element(&self.ap_init);
        state.serialize_element(&self.fp_init);
        state.serialize_element(&self.pc_final);
        state.serialize_element(&self.ap_final);
        state.serialize_element(&self.num_steps);
        state.end()
    }
}

impl Serializable for PublicInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write(self.pc_init);
        target.write(self.ap_init);
        target.write(self.fp_init);
        target.write(self.pc_init);
        target.write(self.ap_final);
        target.write(Felt252::from(self.num_steps as u64));
    }
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
    has_rc_builtin: bool,
}

impl CairoAIR {
    fn get_builtin_offset(&self) -> usize {
        if self.has_rc_builtin {
            0
        } else {
            BUILTIN_OFFSET
        }
    }
}

impl Air for CairoAIR {

    type BaseField = Felt252;
    type PublicInputs = PublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: PublicInputs, options: ProofOptions) -> Self {

        debug_assert!(trace_info.length().is_power_of_two());
        let mut main_degrees = vec![];
        let has_rc_builtin = !pub_inputs.memory_segments.is_empty();
        main_degrees.append(& mut vec![

            // evaluate_instr_constraints
            TransitionConstraintDegree::new(2), // 0  Flag0
            TransitionConstraintDegree::new(2), // 1  Flag1
            TransitionConstraintDegree::new(2), // 2  Flag2
            TransitionConstraintDegree::new(2), // 3  Flag3
            TransitionConstraintDegree::new(2), // 4  Flag4
            TransitionConstraintDegree::new(2), // 5  Flag5
            TransitionConstraintDegree::new(1), // 6  TODO: Flag6 degree num should be 2?
            TransitionConstraintDegree::new(2), // 7  Flag7
            TransitionConstraintDegree::new(2), // 8  Flag8
            TransitionConstraintDegree::new(2), // 9  Flag9
            TransitionConstraintDegree::new(2), // 10 Flag10
            TransitionConstraintDegree::new(2), // 11 Flag11
            TransitionConstraintDegree::new(2), // 12 Flag12
            TransitionConstraintDegree::new(2), // 13 Flag13
            TransitionConstraintDegree::new(2), // 14 Flag14
            TransitionConstraintDegree::new(1), // 15 Flag15
            TransitionConstraintDegree::new(2), // 16 TODO: INST degree num should be 2?
            
            // evaluate_operand_constraints
            TransitionConstraintDegree::new(3), // 17 TODO: DST_ADDR
            TransitionConstraintDegree::new(3), // 18 OP0_ADDR
            TransitionConstraintDegree::new(3), // 19 OP1_ADDR

            //evaluate_register_constraints
            TransitionConstraintDegree::new(3), // 20 NEXT_AP
            TransitionConstraintDegree::new(3), // 21 NEXT_FP
            TransitionConstraintDegree::new(3), // 22 NEXT_PC_1
            TransitionConstraintDegree::new(3), // 23 NEXT_PC_2
            TransitionConstraintDegree::new(3), // 24 T0
            TransitionConstraintDegree::new(3), // 25 T1
            
            TransitionConstraintDegree::new(3), // 26 MUL1 
            TransitionConstraintDegree::new(3), // 27 MUL2
            TransitionConstraintDegree::new(3), // 28 CALL_1
            TransitionConstraintDegree::new(3), // 29 CALL_2
            TransitionConstraintDegree::new(3), // 30 ASSERT_EQ

            TransitionConstraintDegree::new(1), // 31

        ]);

        let mut aux_degrees =  vec![
            // Memory constraints
            TransitionConstraintDegree::new(2), // A_M_PRIME 0
            TransitionConstraintDegree::new(2), //     "     1
            TransitionConstraintDegree::new(2), //     "     2
            TransitionConstraintDegree::new(2), //     "     3
            TransitionConstraintDegree::new(2), // V_M_PRIME 0
            TransitionConstraintDegree::new(2), //     "     1
            TransitionConstraintDegree::new(2), //     "     2
            TransitionConstraintDegree::new(2), //     "     3
            TransitionConstraintDegree::new(2), //    P_M    0
            TransitionConstraintDegree::new(2), //     "     1
            TransitionConstraintDegree::new(2), //     "     2
            TransitionConstraintDegree::new(2), //     "     3
            // Range check constraints
            TransitionConstraintDegree::new(1), // A_RC_PRIME 0
            TransitionConstraintDegree::new(1), //     "      1
            TransitionConstraintDegree::new(1), //     "      2
            TransitionConstraintDegree::new(1), //    P_RC    0
            TransitionConstraintDegree::new(1), //     "      1
            TransitionConstraintDegree::new(1), //     "      2
        ];

        // let context = 
        //     AirContext::new_multi_segment(trace_info, main_degrees, aux_degrees, 4, 0, options)
        //     .set_num_transition_exemptions(num_transition_exemptions);
        let context = AirContext::new(trace_info, main_degrees, 1, options);
        // let context = AirContext::new_multi_segment(
        //     trace_info, 
        //     main_degrees, 
        //     aux_degrees, 
        //     1, 
        //     1, 
        //     options
        // );
    
        Self {
            context,
            pub_inputs,
            has_rc_builtin,
        }

    }

    fn context(&self) -> &AirContext<Felt252> {
        &self.context
    }

    // fn evaluate_aux_transition<F, E>(
    //         &self,
    //         main_frame: &EvaluationFrame<F>,
    //         aux_frame: &EvaluationFrame<E>,
    //         periodic_values: &[F],
    //         aux_rand_elements: &winter_air::AuxTraceRandElements<E>,
    //         result: &mut [E],
    // ) where
    //         F: FieldElement<BaseField = Self::BaseField>,
    //         E: FieldElement<BaseField = Self::BaseField> + winterfell::math::ExtensionOf<F>, 
    // {
    //     let main_current = main_frame.current();
    //     let main_next = main_frame.next();
    //     let aux_current = aux_frame.current();
    //     let aux_next = aux_frame.next();
    //     let random_elements = aux_rand_elements.get_segment_elements(0);
    //     evaluate_aux_memory_constraints(result, main_current, main_next, aux_current, aux_next, random_elements);
    // }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();
        let builtin_offset = self.get_builtin_offset();
        evaluate_instr_constraints(result, current);
        evaluate_operand_constraints(result, current);
        evaluate_register_constraints(result, current, next);
        evaluate_opcode_constraints(result, current);
        enforce_selector(result, current);
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

    // fn get_aux_assertions<E: FieldElement<BaseField = Self::BaseField>>(
    //         &self,
    //         aux_rand_elements: &winter_air::AuxTraceRandElements<E>,
    //     ) -> Vec<Assertion<E>> {
    //     vec![
    //         Assertion::single(1, 0, E::from(0_u16)),
    //     ]
    // }
}
