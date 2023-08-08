use std::collections::HashMap;


use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;
use itertools::Itertools;
use winter_air::{TraceLayout, TraceInfo};
use winter_utils::uninit_vector;
use winterfell::{math::{StarkField, FieldElement}, ColMatrix, Trace};

use crate::Felt252;

use super::air::PublicInputs;
use super::air::constraints::A_M_PRIME_WIDTH;
use super::air::constraints::A_RC_PRIME_WIDTH;
use super::air::constraints::OFF_X_TRACE_RANGE;
use super::air::constraints::OFF_X_TRACE_WIDTH;
use super::air::constraints::P_M_WIDTH;
use super::air::constraints::P_RC_WIDTH;
use super::air::constraints::V_M_PRIME_WIDTH;
use super::{air::{constraints::{FRAME_PC, FRAME_DST_ADDR, FRAME_OP0_ADDR, FRAME_OP1_ADDR, FRAME_INST, FRAME_OP1, MEM_A_TRACE_WIDTH}, MemorySegmentMap, MemorySegment}, cairo_mem::CairoMemory};

const AUX_WIDTHS: [usize; 1] = [12];
const AUX_RANDS: [usize; 1] = [2];

pub struct CairoWinterTraceTable {
    layout: TraceLayout,
    trace: ColMatrix<Felt252>,
    meta: Vec<u8>,
    pub public_inputs: PublicInputs,
}

impl CairoWinterTraceTable {
    // CONSTRUCTORS
    // --------------------------------------------------------------------------------------------

    /// Creates a new execution trace of the specified width and length.
    ///
    /// This allocates all the required memory for the trace, but does not initialize it. It is
    /// expected that the trace will be filled using one of the data mutator methods.
    ///
    /// # Panics
    /// Panics if:
    /// * `width` is zero or greater than 255.
    /// * `length` is smaller than 8, greater than biggest multiplicative subgroup in the field
    ///   `B`, or is not a power of two.
    pub fn new(width: usize, length: usize) -> Self {
        Self::with_meta(width, length, vec![])
    }

    /// Creates a new execution trace from a list of provided trace columns.
    ///
    /// # Panics
    /// Panics if:
    /// * The `columns` vector is empty or has over 255 columns.
    /// * Number of elements in any of the columns is smaller than 8, greater than the biggest
    ///   multiplicative subgroup in the field `B`, or is not a power of two.
    /// * Number of elements is not identical for all columns.
    pub fn init(columns: Vec<Vec<Felt252>>, public_inputs: PublicInputs) -> Self {
        assert!(
            !columns.is_empty(),
            "execution trace must consist of at least one column"
        );
        assert!(
            columns.len() <= TraceInfo::MAX_TRACE_WIDTH,
            "execution trace width cannot be greater than {}, but was {}",
            TraceInfo::MAX_TRACE_WIDTH,
            columns.len()
        );
        let trace_length = columns[0].len();
        assert!(
            trace_length >= TraceInfo::MIN_TRACE_LENGTH,
            "execution trace must be at least {} steps long, but was {}",
            TraceInfo::MIN_TRACE_LENGTH,
            trace_length
        );
        assert!(
            trace_length.is_power_of_two(),
            "execution trace length must be a power of 2"
        );
        assert!(
            trace_length.ilog2() <= Felt252::TWO_ADICITY,
            "execution trace length cannot exceed 2^{} steps, but was 2^{}",
            Felt252::TWO_ADICITY,
            trace_length.ilog2()
        );
        for column in columns.iter().skip(1) {
            assert_eq!(
                column.len(),
                trace_length,
                "all columns traces must have the same length"
            );
        }

        Self {
            layout: TraceLayout::new(
                columns.len(), 
                AUX_WIDTHS, // aux_segment widths
                AUX_RANDS,
             ),
            trace: ColMatrix::new(columns),
            meta: vec![],
            public_inputs: public_inputs,
        }
    }

    /// Creates a new execution trace of the specified width and length, and with the specified
    /// metadata.
    ///
    /// This allocates all the required memory for the trace, but does not initialize it. It is
    /// expected that the trace will be filled using one of the data mutator methods.
    ///
    /// # Panics
    /// Panics if:
    /// * `width` is zero or greater than 255.
    /// * `length` is smaller than 8, greater than the biggest multiplicative subgroup in the
    ///   field `B`, or is not a power of two.
    /// * Length of `meta` is greater than 65535;
    pub fn with_meta(width: usize, length: usize, meta: Vec<u8>) -> Self {
        assert!(
            width > 0,
            "execution trace must consist of at least one column"
        );
        assert!(
            width <= TraceInfo::MAX_TRACE_WIDTH,
            "execution trace width cannot be greater than {}, but was {}",
            TraceInfo::MAX_TRACE_WIDTH,
            width
        );
        assert!(
            length >= TraceInfo::MIN_TRACE_LENGTH,
            "execution trace must be at least {} steps long, but was {}",
            TraceInfo::MIN_TRACE_LENGTH,
            length
        );
        assert!(
            length.is_power_of_two(),
            "execution trace length must be a power of 2"
        );
        assert!(
            length.ilog2() <= Felt252::TWO_ADICITY,
            "execution trace length cannot exceed 2^{} steps, but was 2^{}",
            Felt252::TWO_ADICITY,
            length.ilog2()
        );
        assert!(
            meta.len() <= TraceInfo::MAX_META_LENGTH,
            "number of metadata bytes cannot be greater than {}, but was {}",
            TraceInfo::MAX_META_LENGTH,
            meta.len()
        );

        let columns = unsafe { (0..width).map(|_| uninit_vector(length)).collect() };
        Self {
            layout: TraceLayout::new(width, AUX_WIDTHS, AUX_RANDS),
            trace: ColMatrix::new(columns),
            meta,
            public_inputs: todo!(),
        }
    }

    // DATA MUTATORS
    // --------------------------------------------------------------------------------------------

    /// Fill all rows in the execution trace.
    ///
    /// The rows are filled by executing the provided closures as follows:
    /// - `init` closure is used to initialize the first row of the trace; it receives a mutable
    ///   reference to the first state initialized to all zeros. The contents of the state are
    ///   copied into the first row of the trace after the closure returns.
    /// - `update` closure is used to populate all subsequent rows of the trace; it receives two
    ///   parameters:
    ///   - index of the last updated row (starting with 0).
    ///   - a mutable reference to the last updated state; the contents of the state are copied
    ///     into the next row of the trace after the closure returns.
    pub fn fill<I, U>(&mut self, init: I, update: U)
    where
        I: Fn(&mut [Felt252]),
        U: Fn(usize, &mut [Felt252]),
    {
        let mut state = vec![Felt252::ZERO; self.main_trace_width()];
        init(&mut state);
        self.update_row(0, &state);

        for i in 0..self.length() - 1 {
            update(i, &mut state);
            self.update_row(i + 1, &state);
        }
    }

    /// Updates a single row in the execution trace with provided data.
    pub fn update_row(&mut self, step: usize, state: &[Felt252]) {
        self.trace.update_row(step, state);
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the number of columns in this execution trace.
    pub fn width(&self) -> usize {
        self.main_trace_width()
    }

    /// Returns value of the cell in the specified column at the specified row of this trace.
    pub fn get(&self, column: usize, step: usize) -> Felt252 {
        self.trace.get(column, step)
    }

    /// Reads a single row from this execution trace into the provided target.
    pub fn read_row_into(&self, step: usize, target: &mut [Felt252]) {
        self.trace.read_row_into(step, target);
    }

    pub fn get_public_mem(&self) -> (Vec<Felt252>, Vec<Option<Felt252>>){
        let mut addrs: Vec<Felt252> = vec![];
        let mut vals: Vec<Option<Felt252>> = vec![];

        let public_inputs = self.public_inputs.clone();

        for (key, value ) in public_inputs.public_memory.into_iter() {
            addrs.push(key);
            vals.push(Some(value))
        }
        (addrs, vals)
        
    }
}



// TRACE TRAIT IMPLEMENTATION
// ================================================================================================

impl Trace for CairoWinterTraceTable {
    type BaseField = Felt252;

    fn layout(&self) -> &TraceLayout {
        &self.layout
    }

    fn length(&self) -> usize {
        self.trace.num_rows()
    }

    fn meta(&self) -> &[u8] {
        &self.meta
    }

    fn main_segment(&self) -> &ColMatrix<Self::BaseField> {
        &self.trace
    }

    fn build_aux_segment<E: winterfell::math::FieldElement<BaseField = Self::BaseField>>(
        &mut self,
        aux_segments: &[ColMatrix<E>],
        rand_elements: &[E],
    ) -> Option<ColMatrix<E>> {
        match aux_segments.len() {
            0 => build_aux_segment_mem(self, rand_elements),
            // 1 => build_aux_segment_rc(self, rand_elements),
            _ => None,
        }
    }

    fn read_main_frame(&self, row_idx: usize, frame: &mut winter_air::EvaluationFrame<Self::BaseField>) {
        let next_row_idx = (row_idx + 1) % self.length();
        self.trace.read_row_into(row_idx, frame.current_mut());
        self.trace.read_row_into(next_row_idx, frame.next_mut());
    }
}

/// Write document
fn build_aux_segment_mem<E>(trace: &CairoWinterTraceTable, rand_elements: &[E]) -> Option<ColMatrix<E>>
where
    E: FieldElement<BaseField = Felt252>,
{
    let z = rand_elements[0];
    let alpha = rand_elements[1];

    // println!("build_aux_segment_mem z {:?} alpha {:?}", z, alpha);
    // Pack main memory access trace columns into two virtual columns
    let main = trace.main_segment();
    println!("main Trace num_cols {:?} num_rows {:?}", main.num_cols(), main.num_rows());

    let (a, v) = [(FRAME_PC..FRAME_OP1_ADDR+1),(FRAME_INST..FRAME_OP1+1)]
        .iter()
        .map(|range| {
            VirtualColumn::new(
                &range
                    .clone()
                    .map(|i| main.get_column(i).to_vec())
                    .collect::<Vec<_>>()[..],
            )
            .to_column()
        })
        .collect_tuple()
        .unwrap();

    // Construct duplicate virtual columns sorted by memory access, with dummy public
    // memory addresses/values replaced by their true values
    let mut a_prime = vec![E::ZERO; a.len()];
    let mut v_prime = vec![E::ZERO; a.len()];
    let mut a_replaced = a.clone();
    let mut v_replaced = v.clone();
    println!("a_replaced len {:?} v_replaced len {:?}", a_replaced.len(), v_replaced.len());
    let (pub_a, pub_v) = trace.get_public_mem();
    let l = a.len() - pub_a.len() - 1;
    println!("PRINT l len {:?} pub_a len {:?} pub_v len {:?} a len {:?} v len {:?}", l, pub_a.len(), pub_v.len() ,a.len(), v.len());
    for (i, (n, x)) in pub_a.iter().copied().zip(pub_v).enumerate() {
        a_replaced[l + i] = n;
        v_replaced[l + i] = x.unwrap();
    }
    let mut indices = (0..a.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| a_replaced[i].as_int());
    for (i, j) in indices.iter().copied().enumerate() {
        
        a_prime[i] = a_replaced[j].into();
        v_prime[i] = v_replaced[j].into();
        // if i != 0 {
        //     assert!(((a_prime[i] - a_prime[i-1]) * (a_prime[i] - a_prime[i-1] - E::ONE)) == E::ZERO, "a_prime[{:?}] {:?}, a_prime[{:?}] {:?}", i, a_prime[i], i-1, a_prime[i-1]);
        //     assert!(((v_prime[i] - v_prime[i-1]) * (a_prime[i] - a_prime[i-1] - E::ONE)) == E::ZERO, "v_prime[{:?}] {:?}, v_prime[{:?}] {:?} \n a_prime[{:?}] {:?}, a_prime[{:?}] {:?}", 
        //     i, v_prime[i], i-1, v_prime[i-1], i, a_prime[i], i-1, a_prime[i-1]);
        // }
    }
    // Construct virtual column of computed permutation products
    let mut p: Vec<E> = vec![E::ZERO; trace.length() * MEM_A_TRACE_WIDTH];
    println!("permutation products len {:?}", p.len());
    let a_0: E = a[0].into();
    let v_0: E = v[0].into();
    p[0] = (z - (a_0 + alpha * v_0).into()) / (z - (a_prime[0] + alpha * v_prime[0]).into());
    for i in (1..p.len()).progress() {
        let a_i: E = a[i].into();
        let v_i: E = v[i].into();
        p[i] = (z - (a_i + alpha * v_i).into()) * p[i - 1]
            / (z - (a_prime[i] + alpha * v_prime[i]).into());
    }
 
    // Split virtual columns into separate auxiliary columns
    let mut aux_columns = VirtualColumn::new(&[a_prime, v_prime, p]).to_columns(&[
        A_M_PRIME_WIDTH,
        V_M_PRIME_WIDTH,
        P_M_WIDTH,
    ]);

    resize_to_pow2(&mut aux_columns);

    Some(ColMatrix::new(aux_columns))
    
}


/// Write document
fn build_aux_segment_rc<E>(trace: &CairoWinterTraceTable, rand_elements: &[E]) -> Option<ColMatrix<E>>
where
    E: FieldElement<BaseField = Felt252>,
{
    let z = rand_elements[0];

    // Pack main offset trace columns into a single virtual column
    let main = trace.main_segment();
    let a = VirtualColumn::new(
        &OFF_X_TRACE_RANGE
            .map(|i| main.get_column(i).to_vec())
            .collect::<Vec<_>>()[..],
    )
    .to_column();

    // Construct duplicate virtual column sorted by offset value
    let mut indices = (0..a.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| a[i].as_int());
    let a_prime = indices.iter().map(|x| a[*x].into()).collect::<Vec<E>>();

    // Construct virtual column of computed permutation products
    let mut p = vec![E::ZERO; trace.length() * OFF_X_TRACE_WIDTH];
    let a_0: E = a[0].into();
    p[0] = (z - a_0) / (z - a_prime[0]);
    for i in (1..p.len()).progress() {
        let a_i: E = a[i].into();
        p[i] = (z - a_i) * p[i - 1] / (z - a_prime[i]);
    }

    // Split virtual columns into separate auxiliary columns
    let mut aux_columns =
        VirtualColumn::new(&[a_prime, p]).to_columns(&[A_RC_PRIME_WIDTH, P_RC_WIDTH]);
    resize_to_pow2(&mut aux_columns);

    Some(ColMatrix::new(aux_columns))
}

/// Resize columns to next power of two
fn resize_to_pow2<E: FieldElement>(columns: &mut [Vec<E>]) {
    let trace_len_pow2 = columns
        .iter()
        .map(|x| x.len().next_power_of_two())
        .max()
        .unwrap();
    for column in columns.iter_mut() {
        let last_value = column.last().copied().unwrap();
        column.resize(trace_len_pow2, last_value);
    }
}


/// A virtual column is composed of one or more subcolumns.
struct VirtualColumn<'a, E: FieldElement> {
    subcols: &'a [Vec<E>],
}

impl<'a, E: FieldElement> VirtualColumn<'a, E> {
    fn new(subcols: &'a [Vec<E>]) -> Self {
        Self { subcols }
    }

    /// Pack subcolumns into a single output column: cycle through each subcolumn, appending
    /// a single value to the output column for each iteration step until exhausted.
    fn to_column(&self) -> Vec<E> {
        let mut col: Vec<E> = vec![];
        for n in 0..self.subcols[0].len() {
            for subcol in self.subcols {
                col.push(subcol[n]);
            }
        }
        col
    }

    /// Split subcolumns into multiple output columns: for each subcolumn, output a single
    /// value to each output column, cycling through each output column until exhuasted.
    fn to_columns(&self, num_rows: &[usize]) -> Vec<Vec<E>> {
        let mut n = 0;
        let mut cols: Vec<Vec<E>> = vec![vec![]; num_rows.iter().sum()];
        for (subcol, width) in self.subcols.iter().zip(num_rows) {
            for (elem, idx) in subcol.iter().zip((0..*width).cycle()) {
                cols[idx + n].push(*elem);
            }
            n += width;
        }
        cols
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct CairoTraceTable<B: StarkField>{
    /// `table` is row-major trace element description
    pub table: Vec<B>,
    pub n_cols: usize,
}

impl <B: StarkField> CairoTraceTable<B> {
    pub fn empty() -> Self {
        Self {
            table: Vec::new(),
            n_cols: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.n_cols == 0
    }

    pub fn new(table: Vec<B>, n_cols: usize) -> Self {
        Self { table, n_cols }
    }

    pub fn n_rows(&self) -> usize {
        if self.n_cols == 0 {
            0
        } else {
            self.table.len() / self.n_cols
        }
    }

    pub fn get_cols(&self, columns: &[usize]) -> Self {
        let mut table = Vec::new();
        for row_index in 0..self.n_rows() {
            for column in columns {
                table.push(self.table[row_index * self.n_cols + column].clone());
            }
        }

        Self {
            table,
            n_cols: columns.len(),
        }
    }

    pub fn rows(&self) -> Vec<Vec<B>> {
        let n_rows = self.n_rows();
        (0..n_rows)
            .map(|row_idx| {
                self.table[(row_idx * self.n_cols)..(row_idx * self.n_cols + self.n_cols)].to_vec()
            })
            .collect()
    }

    pub fn get_row(&self, row_idx: usize) -> &[B] {
        let row_offset = row_idx * self.n_cols;
        &self.table[row_offset..row_offset + self.n_cols]
    }

    pub fn last_row(&self) -> &[B] {
        self.get_row(self.n_rows() - 1)
    }

    pub fn cols(&self) -> Vec<Vec<B>> {
        let n_rows = self.n_rows();
        (0..self.n_cols)
            .map(|col_idx| {
                (0..n_rows)
                    .map(|row_idx| self.table[row_idx * self.n_cols + col_idx].clone())
                    .collect()
            })
            .collect()
    }

    /// Given a step and a column index, gives stored value in that position
    pub fn get(&self, step: usize, col: usize) -> B {
        let idx = step * self.n_cols + col;
        self.table[idx].clone()
    }

    pub fn new_from_cols(cols: &[Vec<B>]) -> Self {
        let n_rows = cols[0].len();
        debug_assert!(cols.iter().all(|c| c.len() == n_rows));

        let n_cols = cols.len();

        let mut table = Vec::with_capacity(n_cols * n_rows);

        for row_idx in 0..n_rows {
            for col in cols {
                table.push(col[row_idx].clone());
            }
        }
        Self { table, n_cols }
    }

}

#[cfg(test)]
mod test {
    use winterfell::TraceTable;

    use crate::{Felt252, cairo::cairo_trace::CairoTraceTable, utils::print_trace};

    #[test]
    fn test_cols() {
        let col_1 = vec![Felt252::from(1), Felt252::from(2), Felt252::from(5), Felt252::from(13)];
        let col_2 = vec![Felt252::from(1), Felt252::from(3), Felt252::from(8), Felt252::from(21)];

        let trace_table = CairoTraceTable::<Felt252>::new_from_cols(&[col_1.clone(), col_2.clone()]);
        let res_cols = trace_table.cols();

        assert_eq!(res_cols, vec![col_1, col_2]);
    }

    #[test]
    fn test_cairo_trace_table_to_winter_trace_table() {
        let col_1 = vec![Felt252::from(1), Felt252::from(2), Felt252::from(5), Felt252::from(13), Felt252::from(1), Felt252::from(2), Felt252::from(5), Felt252::from(13)];
        let col_2 = vec![Felt252::from(1), Felt252::from(3), Felt252::from(8), Felt252::from(21), Felt252::from(1), Felt252::from(2), Felt252::from(5), Felt252::from(13)];
        let cairo_trace_table = CairoTraceTable::<Felt252>::new_from_cols(&[col_1.clone(), col_2.clone()]);
        let winter_trace_table = TraceTable::<Felt252>::init(cairo_trace_table.cols());
        // println!("winter_trace_table {:?}", winter_trace_table.get_column(0));
        // println!("cairo_trace_table {:?}", cairo_trace_table.get_cols(&[0]));
    }

    #[test]
    fn test_subtable_works() {
        let table = vec![
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
            Felt252::from(1),
            Felt252::from(2),
            Felt252::from(3),
        ];
        let trace_table = CairoTraceTable { table, n_cols: 3 };
        let subtable = trace_table.get_cols(&[0, 1]);
        assert_eq!(
            subtable.table,
            vec![
                Felt252::from(1),
                Felt252::from(2),
                Felt252::from(1),
                Felt252::from(2),
                Felt252::from(1),
                Felt252::from(2)
            ]
        );
        assert_eq!(subtable.n_cols, 2);
        let subtable = trace_table.get_cols(&[0, 2]);
        assert_eq!(
            subtable.table,
            vec![
                Felt252::from(1),
                Felt252::from(3),
                Felt252::from(1),
                Felt252::from(3),
                Felt252::from(1),
                Felt252::from(3)
            ]
        );
        assert_eq!(subtable.n_cols, 2);
        assert_eq!(trace_table.get_cols(&[]), CairoTraceTable::empty());
    }
}