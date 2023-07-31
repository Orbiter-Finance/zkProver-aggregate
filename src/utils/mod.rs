use std::ops::Range;

use winterfell::{math::StarkField, TraceTable, Trace};

/// Prints out an execution trace.
pub fn print_trace<E: StarkField>(
    trace: &TraceTable<E>,
    multiples_of: usize,
    offset: usize,
    range: Range<usize>,
) {
    let trace_width = trace.width();

    let mut state = vec![E::ZERO; trace_width];
    for i in 0..trace.length() {
        if (i.wrapping_sub(offset)) % multiples_of != 0 {
            continue;
        }
        trace.read_row_into(i, &mut state);
        println!(
            "{}\t{:?}",
            i,
            state[range.clone()]
                .iter()
                .map(|v| v.as_int())
                .collect::<Vec<E::PositiveInteger>>()
        );
    }
}