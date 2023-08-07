use std::{ops::Range};

use winterfell::math::{FieldElement, StarkField, ExtensionOf};

use crate::{BaseElement};


/// Returns a [Range] initialized with the specified `start` and with `end` set to `start` + `len`.
pub const fn range(start: usize, len: usize) -> Range<usize> {
    Range {
        start,
        end: start + len,
    }
}

/// Aux constraint identifiers
pub const A_M_PRIME: Range<usize> = range(0, 4);
pub const V_M_PRIME: Range<usize> = range(4, 4);
const P_M: Range<usize> = range(8, 4);
const A_RC_PRIME: Range<usize> = range(12, 3);
const P_RC: Range<usize> = range(15, 3);


pub const MEM_A_TRACE_WIDTH: usize = 4;

// AUX TRACE LAYOUT (Range check)
// -----------------------------------------------------------------------------------------
//  D.  a_rc_prime (3) : Sorted offset values
//  E.  p_rc       (3) : Permutation product (range check)
//
//  D   E
// ├xxx|xxx┤
//

pub const A_RC_PRIME_OFFSET: usize = 12;
pub const A_RC_PRIME_WIDTH: usize = 3;

pub const P_RC_OFFSET: usize = 15;
pub const P_RC_WIDTH: usize = 3;


// AUX TRACE LAYOUT (Memory)
// -----------------------------------------------------------------------------------------
//  A.  a_m_prime  (4) : Sorted memory address
//  B.  v_m_prime  (4) : Sorted memory values
//  C.  p_m        (4) : Permutation product (memory)
//
//  A    B    C
// ├xxxx|xxxx|xxxx┤

pub const A_M_PRIME_OFFSET: usize = 0;
pub const A_M_PRIME_WIDTH: usize = 4;

pub const V_M_PRIME_OFFSET: usize = 4;
pub const V_M_PRIME_WIDTH: usize = 4;

pub const P_M_OFFSET: usize = 8;
pub const P_M_WIDTH: usize = 4;



pub const OFF_X_TRACE_OFFSET: usize = 27;
pub const OFF_X_TRACE_WIDTH: usize = 3;
pub const OFF_X_TRACE_RANGE: Range<usize> = range(OFF_X_TRACE_OFFSET, OFF_X_TRACE_WIDTH);


/// Main constraint identifiers
const INST: usize = 16;
const DST_ADDR: usize = 17;
const OP0_ADDR: usize = 18;
const OP1_ADDR: usize = 19;
const NEXT_AP: usize = 20;
const NEXT_FP: usize = 21;
const NEXT_PC_1: usize = 22;
const NEXT_PC_2: usize = 23;
const T0: usize = 24;
const T1: usize = 25;
const MUL_1: usize = 26;
const MUL_2: usize = 27;
const CALL_1: usize = 28;
const CALL_2: usize = 29;
const ASSERT_EQ: usize = 30;

// Auxiliary constraint identifiers
const MEMORY_INCREASING_0: usize = 31;
const MEMORY_INCREASING_1: usize = 32;
const MEMORY_INCREASING_2: usize = 33;
const MEMORY_INCREASING_3: usize = 34;

const MEMORY_CONSISTENCY_0: usize = 35;
const MEMORY_CONSISTENCY_1: usize = 36;
const MEMORY_CONSISTENCY_2: usize = 37;
const MEMORY_CONSISTENCY_3: usize = 38;

const PERMUTATION_ARGUMENT_0: usize = 39;
const PERMUTATION_ARGUMENT_1: usize = 40;
const PERMUTATION_ARGUMENT_2: usize = 41;
const PERMUTATION_ARGUMENT_3: usize = 42;

const RANGE_CHECK_INCREASING_0: usize = 43;
const RANGE_CHECK_INCREASING_1: usize = 44;
const RANGE_CHECK_INCREASING_2: usize = 45;

const RANGE_CHECK_0: usize = 46;
const RANGE_CHECK_1: usize = 47;
const RANGE_CHECK_2: usize = 48;

// Range-check builtin value decomposition constraint
const RANGE_CHECK_BUILTIN: usize = 49;

// Frame row identifiers
//  - Flags
const F_DST_FP: usize = 0;
const F_OP_0_FP: usize = 1;
pub(crate) const F_OP_1_VAL: usize = 2;
const F_OP_1_FP: usize = 3;
const F_OP_1_AP: usize = 4;
const F_RES_ADD: usize = 5;
const F_RES_MUL: usize = 6;
const F_PC_ABS: usize = 7;
const F_PC_REL: usize = 8;
const F_PC_JNZ: usize = 9;
const F_AP_ADD: usize = 10;
const F_AP_ONE: usize = 11;
const F_OPC_CALL: usize = 12;
const F_OPC_RET: usize = 13;
const F_OPC_AEQ: usize = 14;

//  - Others
// TODO: These should probably be in the TraceTable module.
pub const FRAME_RES: usize = 16;
pub const FRAME_AP: usize = 17;
pub const FRAME_FP: usize = 18;
pub const FRAME_PC: usize = 19;
pub const FRAME_DST_ADDR: usize = 20;
pub const FRAME_OP0_ADDR: usize = 21;
pub const FRAME_OP1_ADDR: usize = 22;
pub const FRAME_INST: usize = 23;
pub const FRAME_DST: usize = 24;
pub const FRAME_OP0: usize = 25;
pub const FRAME_OP1: usize = 26;
pub const OFF_DST: usize = 27;
pub const OFF_OP0: usize = 28;
pub const OFF_OP1: usize = 29;
pub const FRAME_T0: usize = 30;
pub const FRAME_T1: usize = 31;
pub const FRAME_MUL: usize = 32;
pub const FRAME_SELECTOR: usize = 33;

// Range-check frame identifiers
pub const RC_0: usize = 34;
pub const RC_1: usize = 35;
pub const RC_2: usize = 36;
pub const RC_3: usize = 37;
pub const RC_4: usize = 38;
pub const RC_5: usize = 39;
pub const RC_6: usize = 40;
pub const RC_7: usize = 41;
pub const RC_VALUE: usize = 42;

// Auxiliary range check columns
pub const RANGE_CHECK_COL_1: usize = 43;
pub const RANGE_CHECK_COL_2: usize = 44;
pub const RANGE_CHECK_COL_3: usize = 45;

// Auxiliary memory columns
pub const MEMORY_ADDR_SORTED_0: usize = 46;
pub const MEMORY_ADDR_SORTED_1: usize = 47;
pub const MEMORY_ADDR_SORTED_2: usize = 48;
pub const MEMORY_ADDR_SORTED_3: usize = 49;

pub const MEMORY_VALUES_SORTED_0: usize = 50;
pub const MEMORY_VALUES_SORTED_1: usize = 51;
pub const MEMORY_VALUES_SORTED_2: usize = 52;
pub const MEMORY_VALUES_SORTED_3: usize = 53;

pub const PERMUTATION_ARGUMENT_COL_0: usize = 54;
pub const PERMUTATION_ARGUMENT_COL_1: usize = 55;
pub const PERMUTATION_ARGUMENT_COL_2: usize = 56;
pub const PERMUTATION_ARGUMENT_COL_3: usize = 57;

pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_1: usize = 58;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_2: usize = 59;
pub const PERMUTATION_ARGUMENT_RANGE_CHECK_COL_3: usize = 60;

// Trace layout
pub const MEM_P_TRACE_OFFSET: usize = 17;
pub const MEM_A_TRACE_OFFSET: usize = 19;

// If Cairo AIR doesn't implement builtins, the auxiliary columns should have a smaller
// index.
pub const BUILTIN_OFFSET: usize = 9;


fn frame_inst_size<E: FieldElement + From<BaseElement>>(frame_row: &[E]) -> E {
    frame_row[F_OP_1_VAL] + E::ONE
}

/// From the Cairo whitepaper, section 9.10
pub fn evaluate_instr_constraints<E: FieldElement + From<BaseElement>>(
    constraints: &mut[E],
    current: &[E],
) {
    let ONE = E::ONE;
    let ZERO = E::ZERO;
    let TWO = ONE + ONE;
    // Bit constraints
    for (i, flag) in current[0..16].iter().enumerate() {
        constraints[i] = match i {
            0..=14 => *flag * (*flag - ONE),
            15 => *flag,
            _ => panic!("Unknown flag offset"),
        };
    }

     // Instruction unpacking
     let b16 = TWO.exp(16u32.into());
     let b32 = TWO.exp(32u32.into());
     let b48 = TWO.exp(48u32.into());

     // Named like this to match the Cairo whitepaper's notation.
     let f0_squiggle = &current[0..15]
        .iter()
        .rev()
        .fold(ZERO, |acc, flag| *flag + TWO * acc.into());

    constraints[INST] = current[OFF_DST] + (b16 * current[OFF_OP0]) + (b32 * current[OFF_OP1]) + (b48 * *f0_squiggle) - current[FRAME_INST];

}

pub fn evaluate_operand_constraints<E: FieldElement + From<BaseElement>>(
    constraints: &mut[E], 
    current: &[E],
) {
    let ap = current[FRAME_AP];
    let fp = current[FRAME_FP];
    let pc = current[FRAME_PC];

    let ONE = E::ONE;
    let TWO = ONE + ONE;
    let b15 = TWO.exp(15u32.into());

    constraints[DST_ADDR] =
        current[F_DST_FP] * fp + (ONE - current[F_DST_FP]) * ap + (current[OFF_DST] - b15)
            - current[FRAME_DST_ADDR];

    constraints[OP0_ADDR] =
        current[F_OP_0_FP] * fp + (ONE - current[F_OP_0_FP]) * ap + (current[OFF_OP0] - b15)
            - current[FRAME_OP0_ADDR];

    constraints[OP1_ADDR] = current[F_OP_1_VAL] * pc
        + current[F_OP_1_AP] * ap
        + current[F_OP_1_FP] * fp
        + (ONE - current[F_OP_1_VAL] - current[F_OP_1_AP] - current[F_OP_1_FP]) * current[FRAME_OP0]
        + (current[OFF_OP1] - b15)
        - current[FRAME_OP1_ADDR];
}

pub fn evaluate_register_constraints<E: FieldElement + From<BaseElement>>(
    constraints: &mut[E], 
    current: &[E],
    next: &[E],
) {

    let ONE = E::ONE;
    let TWO = ONE + ONE;

    // ap and fp constraints
    constraints[NEXT_AP] = current[FRAME_AP]
        + current[F_AP_ADD] * current[FRAME_RES]
        + current[F_AP_ONE]
        + current[F_OPC_CALL] * TWO
        - next[FRAME_AP];

    constraints[NEXT_FP] = current[F_OPC_RET] * current[FRAME_DST]
        + current[F_OPC_CALL] * (current[FRAME_AP] + TWO)
        + (ONE - current[F_OPC_RET] - current[F_OPC_CALL]) * current[FRAME_FP]
        - next[FRAME_FP];

    // pc constraints
    constraints[NEXT_PC_1] = (current[FRAME_T1] - current[F_PC_JNZ])
        * (next[FRAME_PC] - (current[FRAME_PC] + frame_inst_size(current)));

    constraints[NEXT_PC_2] = current[FRAME_T0]
        * (next[FRAME_PC] - (current[FRAME_PC] + current[FRAME_OP1]))
        + (ONE - current[F_PC_JNZ]) * next[FRAME_PC]
        - ((ONE - current[F_PC_ABS] - current[F_PC_REL] - current[F_PC_JNZ])
            * (current[FRAME_PC] + frame_inst_size(current))
            + current[F_PC_ABS] * current[FRAME_RES]
            + current[F_PC_REL] * (current[FRAME_PC] + current[FRAME_RES]));

    constraints[T0] = current[F_PC_JNZ] * current[FRAME_DST] - current[FRAME_T0];
    constraints[T1] = current[FRAME_T0] * current[FRAME_RES] - current[FRAME_T1];
}

pub fn evaluate_opcode_constraints<E: FieldElement + From<BaseElement>>(
    constraints: &mut[E], 
    current: &[E],
) {
    let ONE = E::ONE;

    constraints[MUL_1] = current[FRAME_MUL] - (current[FRAME_OP0] * current[FRAME_OP1]);

    constraints[MUL_2] = current[F_RES_ADD] * (current[FRAME_OP0] + current[FRAME_OP1])
        + current[F_RES_MUL] * current[FRAME_MUL]
        + (ONE - current[F_RES_ADD] - current[F_RES_MUL] - current[F_PC_JNZ]) * current[FRAME_OP1]
        - (ONE - current[F_PC_JNZ]) * current[FRAME_RES];

    constraints[CALL_1] = current[F_OPC_CALL] * (current[FRAME_DST] - current[FRAME_FP]);

    constraints[CALL_2] =
        current[F_OPC_CALL] * (current[FRAME_OP0] - (current[FRAME_PC] + frame_inst_size(current)));

    constraints[ASSERT_EQ] = current[F_OPC_AEQ] * (current[FRAME_DST] - current[FRAME_RES]);
}

pub fn enforce_selector<E: FieldElement + From<BaseElement>>(
    constraints: &mut[E], 
    current: &[E],
)
{
    for result_cell in constraints.iter_mut().take(ASSERT_EQ + 1).skip(INST) {
        *result_cell = *result_cell * current[FRAME_SELECTOR];
    }
}

pub fn evaluate_aux_memory_constraints<F, E>(
    constraints: &mut[E], 
    main_current: &[F],
    main_next: &[F],
    aux_current: &[E],
    aux_next: &[E],
    random_elements: &[E],
) 
    where F: FieldElement + From<BaseElement>,
    E: FieldElement + From<BaseElement> + ExtensionOf<F>,
{
    let z = random_elements[0];
    let alpha = random_elements[1];
    let TWO = E::ONE + E::ONE;


    // Continuity constraint
    // for (i, n) in A_M_PRIME.enumerate() {
    //     // constraints[n] = (aux.a_m_prime(i + 1) - aux.a_m_prime(i))
    //     // * (aux.a_m_prime(i + 1) - aux.a_m_prime(i) - F::ONE);
    //     constraints[n] = (aux_current[i + 1] - aux_current[i]) 
    //         * (aux_current[i + 1] - aux_current[i] - E::ONE);
    //     // println!("constarints {:?}", (aux_current[i+1] - aux_current[i]) 
    //     // * (aux_current[i + 1] - aux_current[i] - E::ONE));
    //     // println!("aux_current[{:?}] - aux_current[{:?}] {:?}",i+1, i ,aux_current[i+1] - aux_current[i]);
    //     // println!("aux_current[{:?}] - aux_current[{:?}] - E::ONE) {:?}",i+1, i, aux_current[i + 1] - aux_current[i] - E::ONE);
    // }
    constraints[0] = (aux_current[1] - aux_current[0])
        * (aux_current[1] - aux_current[0] - E::ONE);
    constraints[1] = (aux_current[2] - aux_current[1])
        * (aux_current[2] - aux_current[1] - E::ONE);
    constraints[2] = (aux_current[3] - aux_current[2])
        * (aux_current[3] - aux_current[2] - E::ONE);
    constraints[3] = (aux_next[0] - aux_current[3])
        * (aux_next[0] - aux_current[3] - E::ONE);

    // Single-valued constraint
    // for (i, n) in V_M_PRIME.enumerate() {
    //     // self[n] = (aux.v_m_prime(i + 1) - aux.v_m_prime(i))
    //     //         * (aux.a_m_prime(i + 1) - aux.a_m_prime(i) - F::ONE);
    //     constraints[n] = (aux_current[i + 1] - aux_current[i])
    //         * (aux_current[i + 1] - aux_current[i] - E::ONE);
    // }

    // constraints[4] = (aux_current[5] - aux_current[4])
    //     * (aux_current[1] - aux_current[0] - E::ONE);
    // constraints[5] = (aux_current[6] - aux_current[5])
    //     * (aux_current[2] - aux_current[1] - E::ONE);
    // constraints[6] = (aux_current[7] - aux_current[6])
    //     * (aux_current[3] - aux_current[2] - E::ONE);
    // constraints[7] = (aux_next[4] - aux_current[7])
    //     * (aux_next[0] - aux_current[3] - E::ONE);


    // Cumulative product step
    // for (i, n) in P_M.enumerate() {
    //     let a_m: F = curr.a_m(i + 1).into();
    //     let v_m: F = curr.v_m(i + 1).into();
    //     self[n] = (z - (aux.a_m_prime(i + 1) + alpha * aux.v_m_prime(i + 1))) * aux.p_m(i + 1)
    //         - (z - (a_m + alpha * v_m)) * aux.p_m(i);
    // }

}
