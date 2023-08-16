use std::{ops::Range, iter};

use num_integer::{div_ceil};
use winterfell::{math::{FieldElement, StarkField}};

use crate::Felt252;

use super::{
    register_states::RegisterStates, 
    cairo_mem::CairoMemory, 
    air::{PublicInputs, MemorySegment, constraints::{OFF_DST, OFF_OP1, FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR, OFF_OP0, FRAME_INST, FRAME_DST, FRAME_OP0, FRAME_OP1}}, 
    decode::{
        instruction_flags::{CairoInstructionFlags, DstReg, Op0Reg, Op1Src, aux_get_last_nim_of_field_element, PcUpdate, ResLogic, CairoOpcode, ApUpdate}, instruction_offsets::InstructionOffsets
    }, 
    felt252::BigInt, cairo_trace::CairoTraceTable
};

pub const MEMORY_COLUMNS: [usize; 8] = [
    FRAME_PC,
    FRAME_DST_ADDR,
    FRAME_OP0_ADDR,
    FRAME_OP1_ADDR,
    FRAME_INST,
    FRAME_DST,
    FRAME_OP0,
    FRAME_OP1,
];

pub const ADDR_COLUMNS: [usize; 4] = [FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR];

// MAIN TRACE LAYOUT
// -----------------------------------------------------------------------------------------
//  A.  flags   (16) : Decoded instruction flags
//  B.  res     (1)  : Res value
//  C.  mem_p   (2)  : Temporary memory pointers (ap and fp)
//  D.  mem_a   (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
//  E.  mem_v   (4)  : Memory values (inst, dst, op0, op1)
//  F.  offsets (3)  : (off_dst, off_op0, off_op1)
//  G.  derived (3)  : (t0, t1, mul)
//
//  A                B C  D    E    F   G
// ├xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx┤
//
/// Builds the Cairo main trace (i.e. the trace without the auxiliary columns).
/// Builds the execution trace, fills the offset range-check holes and memory holes, adds
/// public memory dummy accesses (See section 9.8 of the Cairo whitepaper) and pads the result
/// so that it has a trace length equal to the closest power of two.
pub fn build_main_trace(
    register_states: &RegisterStates,
    memory: &CairoMemory,
    public_input: &mut PublicInputs,
) -> CairoTraceTable<Felt252> {
    // let mut trace = TraceTable::new(8, 8);
    // trace
    let mut main_trace = build_cairo_execution_trace(register_states, memory, public_input);
    println!("build_main_trace cairo-execution trace_length {:?} column_length {:?}", main_trace.n_rows(), main_trace.n_cols);
    // let mut address_cols = main_trace
    //     .get_cols(&[FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR])
    //     .table;

    // address_cols.sort_by_key(|x| x.to_string());
    // let (rc_holes, rc_min, rc_max) = get_rc_holes(&main_trace, &[OFF_DST, OFF_OP0, OFF_OP1]);
    // public_input.range_check_min = Some(rc_min);
    // public_input.range_check_max = Some(rc_max);
    // fill_rc_holes(&mut main_trace, rc_holes);

    // let mut memory_holes = get_memory_holes(&address_cols, public_input.public_memory.len());

    // if !memory_holes.is_empty() {
    //     fill_memory_holes(&mut main_trace, &mut memory_holes);
    // }

    add_pub_memory_dummy_accesses(&mut main_trace, public_input.public_memory.len());
    // println!("build_main_trace add other trace trace_length {:?} column_length {:?}", main_trace.n_rows(), main_trace.n_cols);

    let trace_len_next_power_of_two = main_trace.n_rows().next_power_of_two();
    let padding = trace_len_next_power_of_two - main_trace.n_rows();
    pad_with_last_row(&mut main_trace, padding);

    main_trace
}

fn pad_with_last_row(trace: &mut CairoTraceTable<Felt252>, number_rows: usize) {
    let last_row = trace.last_row().to_vec();
    let mut pad: Vec<_> = std::iter::repeat(&last_row)
        .take(number_rows)
        .flatten()
        .cloned()
        .collect();
    trace.table.append(&mut pad);
}

/// Artificial `(0, 0)` dummy memory accesses must be added for the public memory.
/// See section 9.8 of the Cairo whitepaper.
fn add_pub_memory_dummy_accesses(
    main_trace: &mut CairoTraceTable<Felt252>,
    pub_memory_len: usize,
) {
    pad_with_last_row_and_zeros(main_trace, (pub_memory_len >> 2) + 1, &MEMORY_COLUMNS)
}

/// Pads trace with its last row, with the exception of the columns specified in
/// `zero_pad_columns`, where the pad is done with zeros.
/// If the last row is [2, 1, 4, 1] and the zero pad columns are [0, 1], then the
/// padding will be [0, 0, 4, 1].
fn pad_with_last_row_and_zeros(
    trace: &mut CairoTraceTable<Felt252>,
    number_rows: usize,
    zero_pad_columns: &[usize],
) {
    let mut last_row = trace.last_row().to_vec();
    for exemption_column in zero_pad_columns.iter() {
        last_row[*exemption_column] = Felt252::ZERO;
    }
    let mut pad: Vec<_> = std::iter::repeat(&last_row)
        .take(number_rows)
        .flatten()
        .cloned()
        .collect();
    trace.table.append(&mut pad);
}

/// Fill memory holes in each address column of the trace with the missing address, depending on the
/// trace column. If the trace column refers to memory addresses, it will be filled with the missing
/// addresses.
fn fill_memory_holes(trace: &mut CairoTraceTable<Felt252>, memory_holes: &mut [Felt252]) {
    let last_row = trace.last_row().to_vec();

    // This number represents the amount of times we have to pad to fill the memory
    // holes into the trace.
    // There are a total of ADDR_COLUMNS.len() address columns, and we have to append
    // hole addresses in each column until there are no more holes.
    // If we have that memory_holes = [1, 2, 3, 4, 5] and there are 4 address columns,
    // we will have to pad with 2 rows, one for the first [1, 2, 3, 4] and one for the
    // 5 value.
    let padding_size = div_ceil(memory_holes.len(), ADDR_COLUMNS.len());

    let padding_row_iter = iter::repeat(last_row).take(padding_size);
    let addr_columns_iter = iter::repeat(ADDR_COLUMNS).take(padding_size);
    let mut memory_holes_iter = memory_holes.iter();

    for (addr_cols, mut padding_row) in iter::zip(addr_columns_iter, padding_row_iter) {
        // The particular placement of the holes in each column is not important,
        // the only thing that matters is that the addresses are put somewhere in the address
        // columns.
        addr_cols.iter().for_each(|a_col| {
            if let Some(hole) = memory_holes_iter.next() {
                padding_row[*a_col] = *hole;
            }
        });

        trace.table.append(&mut padding_row);
    }
}

/// Get memory holes from accessed addresses. These memory holes appear
/// as a consequence of interaction with builtins.
/// Returns a vector of addresses that were not present in the input vector (holes)
///
/// # Arguments
///
/// * `sorted_addrs` - Vector of sorted memory addresses.
/// * `codelen` - the length of the Cairo program instructions.
fn get_memory_holes(sorted_addrs: &[Felt252], codelen: usize) -> Vec<Felt252> {
    let mut memory_holes = Vec::new();
    let mut prev_addr = &sorted_addrs[0];
    let ONE = Felt252::ONE;

    for addr in sorted_addrs.iter() {
        let addr_diff = *addr - *prev_addr;

        // If the candidate memory hole has an address belonging to the program segment (public
        // memory), that is not accounted here since public memory is added in a posterior step of
        // the protocol.
        if addr_diff != Felt252::ONE
            && addr_diff != Felt252::ZERO
            && addr.as_int() > (codelen as u64).into()
        {
            let mut hole_addr = *prev_addr + ONE;

            while hole_addr.as_int() < addr.as_int() {
                if hole_addr.as_int() > (codelen as u64).into() {
                    memory_holes.push(hole_addr);
                }
                hole_addr += Felt252::ONE;
            }
        }
        prev_addr = addr;
    }

    memory_holes
}

/// Fills holes found in the range-checked columns.
fn fill_rc_holes(trace: &mut CairoTraceTable<Felt252>, holes: Vec<Felt252>) {
    let zeros_left = vec![Felt252::ZERO; OFF_DST];
    let zeros_right = vec![Felt252::ZERO; trace.n_cols - OFF_OP1 - 1];

    for i in (0..holes.len()).step_by(3) {
        trace.table.append(&mut zeros_left.clone());
        trace.table.append(&mut holes[i..(i + 3)].to_vec());
        trace.table.append(&mut zeros_right.clone());
    }
}

/// Gets holes from the range-checked columns. These holes must be filled for the
/// permutation range-checks, as can be read in section 9.9 of the Cairo whitepaper.
/// Receives the trace and the indexes of the range-checked columns.
/// Outputs the holes that must be filled to make the range continuous and the extreme
/// values rc_min and rc_max, corresponding to the minimum and maximum values of the range.
/// NOTE: These extreme values should be received as public inputs in the future and not
/// calculated here.
fn get_rc_holes(
    trace: &CairoTraceTable<Felt252>,
    columns_indices: &[usize],
) -> (Vec<Felt252>, u16, u16)
{
    let offset_columns = trace.get_cols(columns_indices).table;

    let mut sorted_offset_representatives: Vec<u16> = offset_columns
        .iter()
        .map(|x| x.as_int().to_u16())
        .collect();
    sorted_offset_representatives.sort();

    let mut all_missing_values: Vec<Felt252> = Vec::new();

    for window in sorted_offset_representatives.windows(2) {
        if window[1] != window[0] {
            let mut missing_range: Vec<_> = ((window[0] + 1)..window[1])
                .map(|x| Felt252::from(x as u64))
                .collect();
            all_missing_values.append(&mut missing_range);
        }
    }

    let multiple_of_three_padding =
        ((all_missing_values.len() + 2) / 3) * 3 - all_missing_values.len();
    let padding_element = Felt252::from(*sorted_offset_representatives.last().unwrap() as u64);
    all_missing_values.append(&mut vec![padding_element; multiple_of_three_padding]);

    (
        all_missing_values,
        sorted_offset_representatives[0],
        sorted_offset_representatives.last().cloned().unwrap(),
    )
}

/// Receives the raw Cairo trace and memory as outputted from the Cairo VM and returns
/// the trace table used to feed the Cairo STARK prover.
/// The constraints of the Cairo AIR are defined over this trace rather than the raw trace
/// obtained from the Cairo VM, this is why this function is needed.
pub fn build_cairo_execution_trace(
    raw_trace: &RegisterStates,
    memory: &CairoMemory,
    public_inputs: &PublicInputs,
// ) -> Vec<Vec<Felt252>>{
) -> CairoTraceTable<Felt252> {

    let n_steps = raw_trace.steps();

    // Instruction flags and offsets are decoded from the raw instructions and represented
    // by the CairoInstructionFlags and InstructionOffsets as an intermediate representation
    let (flags, offsets): (Vec<CairoInstructionFlags>, Vec<InstructionOffsets>) = raw_trace
        .flags_and_offsets(memory)
        .unwrap()
        .into_iter()
        .unzip();

    // dst, op0, op1 and res are computed from flags and offsets
    let (dst_addrs, mut dsts): (Vec<Felt252>, Vec<Felt252>) =
    compute_dst(&flags, &offsets, raw_trace, memory);
    let (op0_addrs, mut op0s): (Vec<Felt252>, Vec<Felt252>) =
    compute_op0(&flags, &offsets, raw_trace, memory);
    let (op1_addrs, op1s): (Vec<Felt252>, Vec<Felt252>) =
    compute_op1(&flags, &offsets, raw_trace, memory, &op0s);
    let mut res: Vec<crate::BaseElement> = compute_res(&flags, &op0s, &op1s, &dsts);

     // In some cases op0, dst or res may need to be updated from the already calculated values
     update_values(&flags, raw_trace, &mut op0s, &mut dsts, &mut res);

     // Flags and offsets are transformed to a bit representation. This is needed since
     // the flag constraints of the Cairo AIR are defined over bit representations of these
     let trace_repr_flags: Vec<[Felt252; 16]> = flags
         .iter()
         .map(CairoInstructionFlags::to_trace_representation)
         .collect();
     let trace_repr_offsets: Vec<[Felt252; 3]> = offsets
         .iter()
         .map(InstructionOffsets::to_trace_representation)
         .collect();
 
     // ap, fp, pc and instruction columns are computed
     let aps: Vec<Felt252> = raw_trace.rows.iter().map(|t| Felt252::from(t.ap)).collect();
     let fps: Vec<Felt252> = raw_trace.rows.iter().map(|t| Felt252::from(t.fp)).collect();
     let pcs: Vec<Felt252> = raw_trace.rows.iter().map(|t| Felt252::from(t.pc)).collect();
     let instructions: Vec<Felt252> = raw_trace
         .rows
         .iter()
         .map(|t| *memory.get(&t.pc).unwrap())
         .collect();
 
     // t0, t1 and mul derived values are constructed. For details refer to
     // section 9.1 of the Cairo whitepaper
     let t0: Vec<Felt252> = trace_repr_flags
         .iter()
         .zip(&dsts)
         .map(|(repr_flags, dst)| repr_flags[9] * *dst)
         .collect();
     let t1: Vec<Felt252> = t0.iter().zip(&res).map(|(t, r)| *t * *r).collect();
     let mul: Vec<Felt252> = op0s.iter().zip(&op1s).map(|(op0, op1)| *op0 * *op1).collect();
 
     // A structure change of the flags and offsets representations to fit into the arguments
     // expected by the TraceTable constructor. A vector of columns of the representations
     // is obtained from the rows representation.
     let trace_repr_flags = rows_to_cols(&trace_repr_flags);
     let trace_repr_offsets = rows_to_cols(&trace_repr_offsets);
 
     let mut selector = vec![Felt252::ONE; n_steps];
     selector[n_steps - 1] = Felt252::ZERO;
 
     // Build Cairo trace columns to instantiate TraceTable struct as defined in the trace layout
     let mut trace_cols: Vec<Vec<Felt252>> = Vec::new();
     // flags 0~15
     (0..trace_repr_flags.len()).for_each(|n| trace_cols.push(trace_repr_flags[n].clone()));
     // res   16
     trace_cols.push(res);
     // ap    17
     trace_cols.push(aps);
     // fp    18
     trace_cols.push(fps);
     // pc    19
     trace_cols.push(pcs);
     // dst_addr 20
     trace_cols.push(dst_addrs);
     // op0_addr 21
     trace_cols.push(op0_addrs);
     // op1_addr 22
     trace_cols.push(op1_addrs);
     // inst    23
     trace_cols.push(instructions);
     // dst    24
     trace_cols.push(dsts);
     // op0    25
     trace_cols.push(op0s);
     // op1   26
     trace_cols.push(op1s);
     //off_dst, off_op0, off_op1    27,28,29
     (0..trace_repr_offsets.len()).for_each(|n| trace_cols.push(trace_repr_offsets[n].clone()));
     // t0 30
     trace_cols.push(t0);
     // t1 31
     trace_cols.push(t1);
     // mul 32
     trace_cols.push(mul);
     trace_cols.push(selector);
 
     if let Some(range_check_builtin_range) = public_inputs
         .memory_segments
         .get(&MemorySegment::RangeCheck)
     {
         add_rc_builtin_columns(&mut trace_cols, range_check_builtin_range.clone(), memory);
     }

    let trace_length = trace_cols[0].len();
    // resize_to_pow2(&mut trace_cols);
    CairoTraceTable::<Felt252>::new_from_cols(&trace_cols)


}


fn resize_to_pow2<E: FieldElement>(trace_columns: &mut [Vec<E>]) {
    let trace_len_pow2 = trace_columns
    .iter()
    .map(|x| x.len().next_power_of_two())
    .max()
    .unwrap();

    for column in trace_columns.iter_mut() {
        let last_val = column.last().copied().unwrap();
        column.resize(trace_len_pow2, last_val);
    }
}

/// Returns the vector of:
/// - dst_addrs
/// - dsts
fn compute_dst(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
) -> (Vec<Felt252>, Vec<Felt252>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf

    # Compute dst
    if dst_reg == 0:
        dst = m(ap + offdst)
    else:
        dst = m(fp + offdst)
    */
    flags
        .iter()
        .zip(offsets)
        .zip(register_states.rows.iter())
        .map(|((f, o), t)| match f.dst_reg {
            DstReg::AP => {
                let addr = t.ap.checked_add_signed(o.off_dst.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            DstReg::FP => {
                let addr = t.fp.checked_add_signed(o.off_dst.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            _ => panic!("compute_dst on AP or FP"),
        })
        .unzip()
}

/// Returns the vector of:
/// - op0_addrs
/// - op0s
fn compute_op0(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
) -> (Vec<Felt252>, Vec<Felt252>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf

    # Compute op0.
    if op0_reg == 0:
        op0 = m(ap + offop0)
    else:
        op0 = m(fp + offop0)
    */
    flags
        .iter()
        .zip(offsets)
        .zip(register_states.rows.iter())
        .map(|((f, o), t)| match f.op0_reg {
            Op0Reg::AP => {
                let addr = t.ap.checked_add_signed(o.off_op0.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            Op0Reg::FP => {
                let addr = t.fp.checked_add_signed(o.off_op0.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
        })
        .unzip()
}

/// Returns the vector of:
/// - op1_addrs
/// - op1s
fn compute_op1(
    flags: &[CairoInstructionFlags],
    offsets: &[InstructionOffsets],
    register_states: &RegisterStates,
    memory: &CairoMemory,
    op0s: &[Felt252],
) -> (Vec<Felt252>, Vec<Felt252>) {
    /* Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf
    # Compute op1 and instruction_size.
    switch op1_src:
        case 0:
            instruction_size = 1
            op1 = m(op0 + offop1)
        case 1:
            instruction_size = 2
            op1 = m(pc + offop1)
            # If offop1 = 1, we have op1 = immediate_value.
        case 2:
            instruction_size = 1
            op1 = m(fp + offop1)
        case 4:
            instruction_size = 1
            op1 = m(ap + offop1)
        default:
            Undefined Behavior
    */
    flags
        .iter()
        .zip(offsets)
        .zip(op0s)
        .zip(register_states.rows.iter())
        .map(|(((flag, offset), op0), trace_state)| match flag.op1_src {
            Op1Src::Op0 => {
                let addr = aux_get_last_nim_of_field_element(op0)
                    .checked_add_signed(offset.off_op1.into())
                    .unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::Imm => {
                let pc = trace_state.pc;
                let addr = pc.checked_add_signed(offset.off_op1.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::AP => {
                let ap = trace_state.ap;
                let addr = ap.checked_add_signed(offset.off_op1.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
            Op1Src::FP => {
                let fp = trace_state.fp;
                let addr = fp.checked_add_signed(offset.off_op1.into()).unwrap();
                (Felt252::from(addr), *memory.get(&addr).unwrap())
            }
        })
        .unzip()
}

/// Returns the vector of res values.
fn compute_res(flags: &[CairoInstructionFlags], op0s: &[Felt252], op1s: &[Felt252], dsts: &[Felt252]) -> Vec<Felt252> {
    /*
    Cairo whitepaper, page 33 - https://eprint.iacr.org/2021/1063.pdf
    # Compute res.
    if pc_update == 4:
        if res_logic == 0 && opcode == 0 && ap_update != 1:
            res = Unused
        else:
            Undefined Behavior
    else if pc_update = 0, 1 or 2:
        switch res_logic:
            case 0: res = op1
            case 1: res = op0 + op1
            case 2: res = op0 * op1
            default: Undefined Behavior
    else: Undefined Behavior
    */
    flags
        .iter()
        .zip(op0s)
        .zip(op1s)
        .zip(dsts)
        .map(|(((f, op0), op1), dst)| {
            match f.pc_update {
                PcUpdate::Jnz => {
                    match (&f.res_logic, &f.opcode, &f.ap_update) {
                        (
                            ResLogic::Op1,
                            CairoOpcode::NOp,
                            ApUpdate::Regular | ApUpdate::Add1 | ApUpdate::Add2,
                        ) => {
                            // In a `jnz` instruction, res is not used, so it is used
                            // to hold the value v = dst^(-1) as an optimization.
                            // This is important for the calculation of the `t1` virtual column
                            // values later on.
                            // See section 9.5 of the Cairo whitepaper, page 53.
                            if dst == &Felt252::ZERO {
                                *dst
                            } else {
                                dst.inv()
                            }
                        }
                        _ => {
                            panic!("Undefined Behavior");
                        }
                    }
                }
                PcUpdate::Regular | PcUpdate::Jump | PcUpdate::JumpRel => match f.res_logic {
                    ResLogic::Op1 => *op1,
                    ResLogic::Add => *op0 + *op1,
                    ResLogic::Mul => *op0 * *op1,
                    ResLogic::Unconstrained => {
                        panic!("Undefined Behavior");
                    }
                },
            }
        })
        .collect()
}


/// Utility function to change from a rows representation to a columns
/// representation of a slice of arrays.   
fn rows_to_cols<const N: usize>(rows: &[[Felt252; N]]) -> Vec<Vec<Felt252>> {
    let n_cols = rows[0].len();

    (0..n_cols)
        .map(|col_idx| rows.iter().map(|elem| elem[col_idx]).collect::<Vec<Felt252>>())
        .collect::<Vec<Vec<Felt252>>>()
}

// /Build range-check builtin columns: rc_0, rc_1, ... , rc_7, rc_value
fn add_rc_builtin_columns(
    trace_cols: &mut Vec<Vec<Felt252>>,
    range_check_builtin_range: Range<u64>,
    memory: &CairoMemory,
) {
    let range_checked_values: Vec<&Felt252> = range_check_builtin_range
        .map(|addr| memory.get(&addr).unwrap())
        .collect();
    let mut rc_trace_columns = decompose_rc_values_into_trace_columns(&range_checked_values);

    // rc decomposition columns are appended with zeros and then pushed to the trace table
    rc_trace_columns.iter_mut().for_each(|column| {
        column.resize(trace_cols[0].len(), Felt252::ZERO);
        trace_cols.push(column.to_vec())
    });

    let mut rc_values_dereferenced: Vec<Felt252> = range_checked_values.iter().map(|&x| *x).collect();
    rc_values_dereferenced.resize(trace_cols[0].len(), Felt252::ZERO);

    trace_cols.push(rc_values_dereferenced);
}

fn decompose_rc_values_into_trace_columns(rc_values: &[&Felt252]) -> [Vec<Felt252>; 8] {
    let mask = BigInt::from_hex("FFFF");
    let mut rc_base_types: Vec<BigInt> =
        rc_values.iter().map(|x| x.to_raw()).collect();

    let mut decomposition_columns: Vec<Vec<Felt252>> = Vec::new();

    for _ in 0..8 {
        decomposition_columns.push(
            rc_base_types
                .iter()
                .map(|&x| Felt252::from_raw((x & mask).0))
                .collect(),
        );

        rc_base_types = rc_base_types.iter().map(|&x| x >> 16).collect();
    }

    // // This can't fail since we have 8 pushes
    decomposition_columns.try_into().unwrap()
}


/// Depending on the instruction opcodes, some values should be updated.
/// This function updates op0s, dst, res in place when the conditions hold.
fn update_values(
    flags: &[CairoInstructionFlags],
    register_states: &RegisterStates,
    op0s: &mut [Felt252],
    dst: &mut [Felt252],
    res: &mut [Felt252],
) {
    for (i, f) in flags.iter().enumerate() {
        if f.opcode == CairoOpcode::Call {
            let instruction_size = if flags[i].op1_src == Op1Src::Imm {
                2
            } else {
                1
            };
            op0s[i] = (register_states.rows[i].pc + instruction_size).into();
            dst[i] = register_states.rows[i].fp.into();
        } else if f.opcode == CairoOpcode::AssertEq {
            res[i] = dst[i];
        }
    }
}

#[cfg(test)]
mod test {
    use winter_air::ProofOptions;
    use winterfell::math::FieldElement;

    use crate::{cairo::{execution_trace::{decompose_rc_values_into_trace_columns, get_rc_holes, fill_rc_holes, get_memory_holes, fill_memory_holes}, cairo_layout::CairoLayout, runner::run::{run_program, generate_prover_args}, air::{PublicInputs, MemorySegmentMap, constraints::{OFF_DST, OFF_OP1, FRAME_SELECTOR, FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR}, proof_options::CairoProofOptions}, cairo_trace::CairoTraceTable}, Felt252};

    use super::{build_cairo_execution_trace, build_main_trace};

    #[test]
    fn test_rc_decompose() {
        let fifteen = Felt252::from_hex("000F000F000F000F000F000F000F000F");
        let sixteen = Felt252::from_hex("00100010001000100010001000100010");
        let one_two_three = Felt252::from_hex("00010002000300040005000600070008");

        let decomposition_columns =
            decompose_rc_values_into_trace_columns(&[&fifteen, &sixteen, &one_two_three]);

        for row in &decomposition_columns {
            assert_eq!(row[0], Felt252::from_hex("F"));
            assert_eq!(row[1], Felt252::from_hex("10"));
        }

        assert_eq!(decomposition_columns[0][2], Felt252::from_hex("8"));
        assert_eq!(decomposition_columns[1][2], Felt252::from_hex("7"));
        assert_eq!(decomposition_columns[2][2], Felt252::from_hex("6"));
        assert_eq!(decomposition_columns[3][2], Felt252::from_hex("5"));
        assert_eq!(decomposition_columns[4][2], Felt252::from_hex("4"));
        assert_eq!(decomposition_columns[5][2], Felt252::from_hex("3"));
        assert_eq!(decomposition_columns[6][2], Felt252::from_hex("2"));
        assert_eq!(decomposition_columns[7][2], Felt252::from_hex("1"));
    }

    #[test]
    fn test_build_main_trace_simple_program() {
        let base_dir = env!("CARGO_MANIFEST_DIR");
        let json_filename = base_dir.to_owned() + "/cairo_programs/fibonacci_cairo1.casm";
        let program_content = std::fs::read(json_filename).unwrap();

        let (register_states, memory, program_size, _rangecheck_base_end) = run_program(
            None,
            CairoLayout::AllCairo,
            &program_content,
        )
        .unwrap();
        let mut pub_inputs = PublicInputs::from_regs_and_mem(
            &register_states,
            &memory,
            program_size,
            &MemorySegmentMap::new(),
        );
        let execution_trace = build_main_trace(
            &register_states, 
            &memory, 
            &mut pub_inputs);
    }

    #[test]
    fn test_get_memory_holes_no_codelen() {
        // We construct a sorted addresses list [1, 2, 3, 6, 7, 8, 9, 13, 14, 15], and
        // set codelen = 0. With this value of codelen, any holes present between
        // the min and max addresses should be returned by the function.
        let mut addrs: Vec<Felt252> = (1..4).map(Felt252::from).collect();
        let addrs_extension: Vec<Felt252> = (6..10).map(Felt252::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let addrs_extension: Vec<Felt252> = (13..16).map(Felt252::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let codelen = 0;

        let expected_memory_holes = vec![
            Felt252::from(4),
            Felt252::from(5),
            Felt252::from(10),
            Felt252::from(11),
            Felt252::from(12),
        ];
        let calculated_memory_holes = get_memory_holes(&addrs, codelen);

        assert_eq!(expected_memory_holes, calculated_memory_holes);
    }

    #[test]
    fn test_get_memory_holes_inside_program_section() {
        // We construct a sorted addresses list [1, 2, 3, 8, 9] and we
        // set a codelen of 9. Since all the holes will be inside the
        // program segment (meaning from addresses 1 to 9), the function
        // should not return any of them.
        let mut addrs: Vec<Felt252> = (1..4).map(Felt252::from).collect();
        let addrs_extension: Vec<Felt252> = (8..10).map(Felt252::from).collect();
        addrs.extend_from_slice(&addrs_extension);
        let codelen = 9;

        let calculated_memory_holes = get_memory_holes(&addrs, codelen);
        let expected_memory_holes: Vec<Felt252> = Vec::new();

        assert_eq!(expected_memory_holes, calculated_memory_holes);
    }

    #[test]
    fn test_fill_range_check_values() {
        let columns = vec![
            vec![Felt252::from(1); 3],
            vec![Felt252::from(4); 3],
            vec![Felt252::from(7); 3],
        ];
        let expected_col = vec![
            Felt252::from(2),
            Felt252::from(3),
            Felt252::from(5),
            Felt252::from(6),
            Felt252::from(7),
            Felt252::from(7),
        ];
        let table = CairoTraceTable::<Felt252>::new_from_cols(&columns);

        let (col, rc_min, rc_max) = get_rc_holes(&table, &[0, 1, 2]);
        assert_eq!(col, expected_col);
        assert_eq!(rc_min, 1);
        assert_eq!(rc_max, 7);
    }

    #[test]
    fn test_add_missing_values_to_offsets_column() {
        let mut main_trace = CairoTraceTable::<Felt252> {
            table: (0..34 * 2).map(Felt252::from).collect(),
            n_cols: 34,
        };
        let missing_values = vec![
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
            Felt252::from(4),
            Felt252::from(5),
            Felt252::from(6),
        ];
        fill_rc_holes(&mut main_trace, missing_values);

        let mut expected: Vec<_> = (0..34 * 2).map(Felt252::from).collect();
        expected.append(&mut vec![Felt252::ZERO; OFF_DST]);
        expected.append(&mut vec![
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
        ]);
        expected.append(&mut vec![Felt252::ZERO; 34 - OFF_OP1 - 1]);
        expected.append(&mut vec![Felt252::ZERO; OFF_DST]);
        expected.append(&mut vec![
            Felt252::from(4),
            Felt252::from(5),
            Felt252::from(6),
        ]);
        expected.append(&mut vec![Felt252::ZERO; 34 - OFF_OP1 - 1]);
        assert_eq!(main_trace.table, expected);
        assert_eq!(main_trace.n_cols, 34);
        assert_eq!(main_trace.table.len(), 34 * 4);
    }

    #[test]
    fn test_fill_memory_holes() {
        const TRACE_COL_LEN: usize = 2;
        const NUM_TRACE_COLS: usize = FRAME_SELECTOR + 1;

        let mut trace_cols = vec![vec![Felt252::ONE; TRACE_COL_LEN]; NUM_TRACE_COLS];
        trace_cols[FRAME_PC][0] = Felt252::ONE;
        trace_cols[FRAME_DST_ADDR][0] = Felt252::from(2);
        trace_cols[FRAME_OP0_ADDR][0] = Felt252::from(3);
        trace_cols[FRAME_OP1_ADDR][0] = Felt252::from(5);
        trace_cols[FRAME_PC][1] = Felt252::from(6);
        trace_cols[FRAME_DST_ADDR][1] = Felt252::from(9);
        trace_cols[FRAME_OP0_ADDR][1] = Felt252::from(10);
        trace_cols[FRAME_OP1_ADDR][1] = Felt252::from(11);
        let mut trace = CairoTraceTable::new_from_cols(&trace_cols);

        let mut memory_holes = vec![Felt252::from(4), Felt252::from(7), Felt252::from(8)];
        fill_memory_holes(&mut trace, &mut memory_holes);

        let frame_pc = &trace.cols()[FRAME_PC];
        let dst_addr = &trace.cols()[FRAME_DST_ADDR];
        let op0_addr = &trace.cols()[FRAME_OP0_ADDR];
        let op1_addr = &trace.cols()[FRAME_OP1_ADDR];
        assert_eq!(frame_pc[0], Felt252::ONE);
        assert_eq!(dst_addr[0], Felt252::from(2));
        assert_eq!(op0_addr[0], Felt252::from(3));
        assert_eq!(op1_addr[0], Felt252::from(5));
        assert_eq!(frame_pc[1], Felt252::from(6));
        assert_eq!(dst_addr[1], Felt252::from(9));
        assert_eq!(op0_addr[1], Felt252::from(10));
        assert_eq!(op1_addr[1], Felt252::from(11));
    }
}