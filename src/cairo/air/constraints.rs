
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
const F_OP_1_VAL: usize = 2;
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
