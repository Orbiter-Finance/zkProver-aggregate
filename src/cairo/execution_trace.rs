

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

use super::{register_states::RegisterStates, cairo_mem::CairoMemory};

/// Builds the Cairo main trace (i.e. the trace without the auxiliary columns).
/// Builds the execution trace, fills the offset range-check holes and memory holes, adds
/// public memory dummy accesses (See section 9.8 of the Cairo whitepaper) and pads the result
/// so that it has a trace length equal to the closest power of two.
pub fn build_main_trace(
    register_states: &RegisterStates,
    memory: &CairoMemory,
    // public_input: 
) {

}