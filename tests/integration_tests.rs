use zkProver_aggregate::cairo::runner::run::generate_prover_args;


// #[test_log::test]
// fn test_prover_cairo_program() {
//     let file_path = "cairo_programs/fibonacci_cairo1.casm";
//     let program_content = std::fs::read(file_path).unwrap();
//     let (main_trace, pub_inputs) =
//         generate_prover_args(&program_content, &None).unwrap();
// }