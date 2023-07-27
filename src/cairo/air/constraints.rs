use std::thread::current;

use winter_air::EvaluationFrame;
use winterfell::math::FieldElement;

use crate::Felt252;


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
const BUILTIN_OFFSET: usize = 9;


fn frame_inst_size(frame_row: &[Felt252]) -> Felt252 {
    frame_row[F_OP_1_VAL] + Felt252::ONE
}

/// From the Cairo whitepaper, section 9.10
pub fn evaluate_instr_constraints(
    constraints: &mut[Felt252], frame: &EvaluationFrame<Felt252>
) {
    let cur = frame.current();
    let ONE = Felt252::ONE;

    // Bit constraints
    for (i, flag) in cur[0..16].iter().enumerate() {
        constraints[i] = match i {
            0..=14 => *flag * (*flag - ONE),
            15 => *flag,
            _ => panic!("Unknown flag offset"),
        };
    }

     // Instruction unpacking
     let TWO = Felt252::from(2);
     let b16 = TWO.exp(16u32.into());
     let b32 = TWO.exp(32u32.into());
     let b48 = TWO.exp(48u32.into());

     // Named like this to match the Cairo whitepaper's notation.
     let f0_squiggle = &cur[0..15]
        .iter()
        .rev()
        .fold(Felt252::ZERO, |acc, flag| *flag + TWO * acc);

    constraints[INST] = cur[OFF_DST] + (b16 * cur[OFF_OP0]) + (b32 * cur[OFF_OP1]) + b48 * *f0_squiggle - cur[FRAME_INST];


}

pub fn evaluate_operand_constraints(constraints: &mut[Felt252], frame: &EvaluationFrame<Felt252>) {
    let cur: &[crate::BaseElement] = frame.current();

    let ap = cur[FRAME_AP];
    let fp = cur[FRAME_FP];
    let pc = cur[FRAME_PC];

    let one = Felt252::ONE;
    let b15 = Felt252::from(2).exp(15u32.into());

    constraints[DST_ADDR] =
        cur[F_DST_FP] * fp + (one - cur[F_DST_FP]) * ap + (cur[OFF_DST] - b15)
            - cur[FRAME_DST_ADDR];

    constraints[OP0_ADDR] =
        cur[F_OP_0_FP] * fp + (one - cur[F_OP_0_FP]) * ap + (cur[OFF_OP0] - b15)
            - cur[FRAME_OP0_ADDR];

    constraints[OP1_ADDR] = cur[F_OP_1_VAL] * pc
        + cur[F_OP_1_AP] * ap
        + cur[F_OP_1_FP] * fp
        + (one - cur[F_OP_1_VAL] - cur[F_OP_1_AP] - cur[F_OP_1_FP]) * cur[FRAME_OP0]
        + (cur[OFF_OP1] - b15)
        - cur[FRAME_OP1_ADDR];
}

pub fn evaluate_register_constraints(constraints: &mut[Felt252], frame: &EvaluationFrame<Felt252>) {
    let cur = frame.current();
    let next = frame.next();

    let one = Felt252::ONE;
    let two = Felt252::TWO;

    // ap and fp constraints
    constraints[NEXT_AP] = cur[FRAME_AP]
        + cur[F_AP_ADD] * cur[FRAME_RES]
        + cur[F_AP_ONE]
        + cur[F_OPC_CALL] * two
        - next[FRAME_AP];

    constraints[NEXT_FP] = cur[F_OPC_RET] * cur[FRAME_DST]
        + cur[F_OPC_CALL] * (cur[FRAME_AP] + two)
        + (one - cur[F_OPC_RET] - cur[F_OPC_CALL]) * cur[FRAME_FP]
        - next[FRAME_FP];

    // pc constraints
    constraints[NEXT_PC_1] = (cur[FRAME_T1] - cur[F_PC_JNZ])
        * (next[FRAME_PC] - (cur[FRAME_PC] + frame_inst_size(cur)));

    constraints[NEXT_PC_2] = cur[FRAME_T0]
        * (next[FRAME_PC] - (cur[FRAME_PC] + cur[FRAME_OP1]))
        + (one - cur[F_PC_JNZ]) * next[FRAME_PC]
        - ((one - cur[F_PC_ABS] - cur[F_PC_REL] - cur[F_PC_JNZ])
            * (cur[FRAME_PC] + frame_inst_size(cur))
            + cur[F_PC_ABS] * cur[FRAME_RES]
            + cur[F_PC_REL] * (cur[FRAME_PC] + cur[FRAME_RES]));

    constraints[T0] = cur[F_PC_JNZ] * cur[FRAME_DST] - cur[FRAME_T0];
    constraints[T1] = cur[FRAME_T0] * cur[FRAME_RES] - cur[FRAME_T1];
}

pub fn evaluate_opcode_constraints(constraints: &mut[Felt252], frame: &EvaluationFrame<Felt252>) {
    let cur = frame.current();
    let one = Felt252::ONE;

    constraints[MUL_1] = cur[FRAME_MUL] - (cur[FRAME_OP0] * cur[FRAME_OP1]);

    constraints[MUL_2] = cur[F_RES_ADD] * (cur[FRAME_OP0] + cur[FRAME_OP1])
        + cur[F_RES_MUL] * cur[FRAME_MUL]
        + (one - cur[F_RES_ADD] - cur[F_RES_MUL] - cur[F_PC_JNZ]) * cur[FRAME_OP1]
        - (one - cur[F_PC_JNZ]) * cur[FRAME_RES];

    constraints[CALL_1] = cur[F_OPC_CALL] * (cur[FRAME_DST] - cur[FRAME_FP]);

    constraints[CALL_2] =
        cur[F_OPC_CALL] * (cur[FRAME_OP0] - (cur[FRAME_PC] + frame_inst_size(cur)));

    constraints[ASSERT_EQ] = cur[F_OPC_AEQ] * (cur[FRAME_DST] - cur[FRAME_RES]);
}

pub fn enforce_selector(constraints: &mut[Felt252], frame: &EvaluationFrame<Felt252>) {
    let curr = frame.current();
    for result_cell in constraints.iter_mut().take(ASSERT_EQ + 1).skip(INST) {
        *result_cell = *result_cell * curr[FRAME_SELECTOR];
    }
}
