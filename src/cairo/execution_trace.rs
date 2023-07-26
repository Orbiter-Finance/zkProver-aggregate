

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

use std::ops::Range;

use winterfell::{TraceTable, math::FieldElement};

use crate::Felt252;

use super::{register_states::RegisterStates, cairo_mem::CairoMemory, air::{PublicInputs, MemorySegment}, decode::{instruction_flags::{CairoInstructionFlags, DstReg, Op0Reg, Op1Src, aux_get_last_nim_of_field_element, PcUpdate, ResLogic, CairoOpcode, ApUpdate}, instruction_offsets::InstructionOffsets}, felt252::BigInt};

/// Builds the Cairo main trace (i.e. the trace without the auxiliary columns).
/// Builds the execution trace, fills the offset range-check holes and memory holes, adds
/// public memory dummy accesses (See section 9.8 of the Cairo whitepaper) and pads the result
/// so that it has a trace length equal to the closest power of two.
pub fn build_main_trace(
    register_states: &RegisterStates,
    memory: &CairoMemory,
    public_input: &mut PublicInputs,
) -> TraceTable<Felt252> {
    let mut trace = TraceTable::new(8, 8);
    trace
}

/// Receives the raw Cairo trace and memory as outputted from the Cairo VM and returns
/// the trace table used to feed the Cairo STARK prover.
/// The constraints of the Cairo AIR are defined over this trace rather than the raw trace
/// obtained from the Cairo VM, this is why this function is needed.
pub fn build_cairo_execution_trace(
    raw_trace: &RegisterStates,
    memory: &CairoMemory,
    public_inputs: &PublicInputs,
) -> TraceTable<Felt252>{

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
    let mut res = compute_res(&flags, &op0s, &op1s, &dsts);

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
     (0..trace_repr_flags.len()).for_each(|n| trace_cols.push(trace_repr_flags[n].clone()));
     trace_cols.push(res);
     trace_cols.push(aps);
     trace_cols.push(fps);
     trace_cols.push(pcs);
     trace_cols.push(dst_addrs);
     trace_cols.push(op0_addrs);
     trace_cols.push(op1_addrs);
     trace_cols.push(instructions);
     trace_cols.push(dsts);
     trace_cols.push(op0s);
     trace_cols.push(op1s);
     (0..trace_repr_offsets.len()).for_each(|n| trace_cols.push(trace_repr_offsets[n].clone()));
     trace_cols.push(t0);
     trace_cols.push(t1);
     trace_cols.push(mul);
     trace_cols.push(selector);
 
     if let Some(range_check_builtin_range) = public_inputs
         .memory_segments
         .get(&MemorySegment::RangeCheck)
     {
         add_rc_builtin_columns(&mut trace_cols, range_check_builtin_range.clone(), memory);
     }

    let trace_length = trace_cols[0].len();
    println!("trace_length {:?}", trace_length);
    resize_to_pow2(&mut trace_cols);
    println!("trace_length after resize {:?}", trace_cols[0].len());
 
    //  TraceTable::new_from_cols(&trace_cols)
    TraceTable::init(trace_cols)

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
    use crate::{cairo::{execution_trace::decompose_rc_values_into_trace_columns, cairo_layout::CairoLayout, runner::run::run_program, air::{PublicInputs, MemorySegmentMap}}, Felt252};

    use super::build_cairo_execution_trace;

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
        let pub_inputs = PublicInputs::from_regs_and_mem(
            &register_states,
            &memory,
            program_size,
            &MemorySegmentMap::new(),
        );
        let execution_trace = build_cairo_execution_trace(&register_states, &memory, &pub_inputs);

    }
}