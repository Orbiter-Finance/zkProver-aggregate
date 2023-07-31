use winterfell::math::StarkField;

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