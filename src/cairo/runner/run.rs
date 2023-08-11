use super::vec_writer::VecWriter;
use crate::Felt252;
use crate::cairo::air::{PublicInputs, MemorySegmentMap, MemorySegment};
use crate::cairo::cairo_layout::CairoLayout;
use crate::cairo::cairo_mem::CairoMemory;
use crate::cairo::cairo_trace::{CairoTraceTable, CairoWinterTraceTable};
use crate::cairo::execution_trace::build_main_trace;
use crate::cairo::register_states::RegisterStates;
use crate::utils::print_trace;
use cairo_lang_starknet::casm_contract_class::CasmContractClass;
use cairo_vm::cairo_run::{self, EncodeTraceError};
use cairo_vm::hint_processor::cairo_1_hint_processor::hint_processor::Cairo1HintProcessor;
use cairo_vm::serde::deserialize_program::BuiltinName;
use cairo_vm::types::{program::Program, relocatable::MaybeRelocatable};
use cairo_vm::vm::errors::{
    cairo_run_errors::CairoRunError, trace_errors::TraceError, vm_errors::VirtualMachineError,
};
use cairo_vm::vm::runners::cairo_runner::{CairoArg, CairoRunner, RunResources};
use cairo_vm::vm::vm_core::VirtualMachine;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use winterfell::{TraceTable, Trace};
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Failed to interact with the file system")]
    IO(#[from] std::io::Error),
    #[error("The cairo program execution failed")]
    Runner(#[from] CairoRunError),
    #[error(transparent)]
    EncodeTrace(#[from] EncodeTraceError),
    #[error(transparent)]
    VirtualMachine(#[from] VirtualMachineError),
    #[error(transparent)]
    Trace(#[from] TraceError),
}

/// Runs a cairo program in JSON format and returns trace, memory and program length.
/// Uses [cairo-rs](https://github.com/lambdaclass/cairo-rs/) project to run the program.
///
///  # Params
///
/// `entrypoint_function` - the name of the entrypoint function tu run. If `None` is provided, the default value is `main`.
/// `layout` - type of layout of Cairo.
/// `program_content` - content of the input file.
/// `trace_path` - path where to store the generated trace file.
/// `memory_path` - path where to store the generated memory file.
///
/// # Returns
///
/// Ok() in case of succes, with the following values:
/// - register_states
/// - cairo_mem
/// - data_len
/// - range_check: an Option<(usize, usize)> containing the start and end of range check.
/// `Error` indicating the type of error.
#[allow(clippy::type_complexity)]
pub fn run_program(
    entrypoint_function: Option<&str>,
    layout: CairoLayout,
    program_content: &[u8],
) -> Result<(RegisterStates, CairoMemory, usize, Option<Range<u64>>), Error> {
    // default value for entrypoint is "main"
    let entrypoint = entrypoint_function.unwrap_or("main");

    let args = [];

    let (vm, runner) =  {

            let casm_contract: CasmContractClass = serde_json::from_slice(program_content).unwrap();
            let program: Program = casm_contract.clone().try_into().unwrap();
            let mut runner = CairoRunner::new(
                &(casm_contract.clone().try_into().unwrap()),
                layout.as_str(),
                false,
            )
            .unwrap();
            let mut vm: VirtualMachine = VirtualMachine::new(true);

            runner
                .initialize_function_runner_cairo_1(&mut vm, &[BuiltinName::range_check])
                .unwrap();

            // Implicit Args
            let syscall_segment = MaybeRelocatable::from(vm.add_memory_segment());

            let builtins: Vec<&'static str> = runner
                .get_program_builtins()
                .iter()
                .map(|b| b.name())
                .collect();

            let builtin_segment: Vec<MaybeRelocatable> = vm
                .get_builtin_runners()
                .iter()
                .filter(|b| builtins.contains(&b.name()))
                .flat_map(|b| b.initial_stack())
                .collect();

            let initial_gas = MaybeRelocatable::from(usize::MAX);

            let mut implicit_args = builtin_segment;
            implicit_args.extend([initial_gas]);
            implicit_args.extend([syscall_segment]);

            // Other args

            // Load builtin costs
            let builtin_costs: Vec<MaybeRelocatable> =
                vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()];
            let builtin_costs_ptr = vm.add_memory_segment();
            vm.load_data(builtin_costs_ptr, &builtin_costs).unwrap();

            // Load extra data
            let core_program_end_ptr = (runner.program_base.unwrap() + program.data_len()).unwrap();
            let program_extra_data: Vec<MaybeRelocatable> =
                vec![0x208B7FFF7FFF7FFE.into(), builtin_costs_ptr.into()];
            vm.load_data(core_program_end_ptr, &program_extra_data)
                .unwrap();

            // Load calldata
            let calldata_start = vm.add_memory_segment();
            let calldata_end = vm.load_data(calldata_start, &args.to_vec()).unwrap();

            // Create entrypoint_args

            let mut entrypoint_args: Vec<CairoArg> = implicit_args
                .iter()
                .map(|m| CairoArg::from(m.clone()))
                .collect();
            entrypoint_args.extend([
                MaybeRelocatable::from(calldata_start).into(),
                MaybeRelocatable::from(calldata_end).into(),
            ]);
            let entrypoint_args: Vec<&CairoArg> = entrypoint_args.iter().collect();

            let mut hint_processor = Cairo1HintProcessor::new(&casm_contract.hints);

            // Run contract entrypoint
            // We assume entrypoint 0 for only one function
            let mut run_resources = RunResources::default();

            runner
                .run_from_entrypoint(
                    0,
                    &entrypoint_args,
                    &mut run_resources,
                    true,
                    Some(program.data_len() + program_extra_data.len()),
                    &mut vm,
                    &mut hint_processor,
                )
                .unwrap();

            let _ = runner.relocate(&mut vm, true);

            (vm, runner)
    };

    let relocated_trace = vm.get_relocated_trace()?;

    let mut trace_vec = Vec::<u8>::new();
    let mut trace_writer = VecWriter::new(&mut trace_vec);
    trace_writer.write_encoded_trace(relocated_trace);

    let relocated_memory = &runner.relocated_memory;

    let mut memory_vec = Vec::<u8>::new();
    let mut memory_writer = VecWriter::new(&mut memory_vec);
    memory_writer.write_encoded_memory(relocated_memory);

    trace_writer.flush()?;
    memory_writer.flush()?;

    //TO DO: Better error handling
    let cairo_mem = CairoMemory::from_bytes_le(&memory_vec).unwrap();
    let register_states = RegisterStates::from_bytes_le(&trace_vec).unwrap();

    let data_len = runner.get_program().data_len();

    // get range start and end
    let range_check = vm
        .get_range_check_builtin()
        .map(|builtin| {
            let (idx, stop_offset) = builtin.get_memory_segment_addresses();
            let stop_offset = stop_offset.unwrap_or_default();
            let range_check_base =
                (0..idx).fold(1, |acc, i| acc + vm.get_segment_size(i).unwrap_or_default());
            let range_check_end = range_check_base + stop_offset;

            (range_check_base, range_check_end)
        })
        .ok();

    let range_check_builtin_range = range_check.map(|(start, end)| Range {
        start: start as u64,
        end: end as u64,
    });

    Ok((
        register_states,
        cairo_mem,
        data_len,
        range_check_builtin_range,
    ))
}



pub fn generate_prover_args(
    program_content: &[u8],
    output_range: &Option<Range<u64>>,
) -> Result<(CairoWinterTraceTable, PublicInputs), Error>{

    let (register_states, memory, program_size, range_check_builtin_range) =
        // run_program(None, CairoLayout::Plain, program_content).unwrap();
        run_program(None, CairoLayout::Plain, program_content).unwrap();
    
    let memory_segments = create_memory_segment_map(range_check_builtin_range.clone(), output_range);

     // register_states.print_trace();
     println!("programe_size {:?} range_check_builtin_range {:?} memory_segment {:?}", program_size, range_check_builtin_range, memory_segments);

    let mut pub_inputs =
        PublicInputs::from_regs_and_mem(&register_states, &memory, program_size, &memory_segments);
    
    let main_trace = build_main_trace(&register_states, &memory, &mut pub_inputs);
    // let winter_main_trace = TraceTable::init(main_trace.cols());
    let winter_main_trace = CairoWinterTraceTable::init(main_trace.cols(),pub_inputs.clone());
    // print_trace(&winter_main_trace, 8, 7 , Range { start: 3, end: 5 });
    println!("winter_main_trace {:?}", winter_main_trace.length());

    Ok((winter_main_trace, pub_inputs))

}

fn create_memory_segment_map(
    range_check_builtin_range: Option<Range<u64>>,
    output_range: &Option<Range<u64>>,
) -> MemorySegmentMap {
    let mut memory_segments = MemorySegmentMap::new();

    if let Some(range_check_builtin_range) = range_check_builtin_range {
        memory_segments.insert(MemorySegment::RangeCheck, range_check_builtin_range);
    }
    if let Some(output_range) = output_range {
        memory_segments.insert(MemorySegment::Output, output_range.clone());
    }

    memory_segments
}

mod tests {
    use winterfell::math::FieldElement;

    use crate::cairo::air::CairoAIR;
    use crate::cairo::air::proof_options::CairoProofOptions;
    use crate::cairo::prover::{prove_cairo_trace, verify_cairo_proof};
    use crate::cairo::register_states;
    use super::*;
    use super::run_program;

    #[test]
    fn test_prove_cairo_file() {
        let base_dir = env!("CARGO_MANIFEST_DIR");
        let json_filename = base_dir.to_owned() + "/cairo_programs/fibonacci_cairo1.casm";
        let program_content = std::fs::read(json_filename).unwrap();

        let proof_options = CairoProofOptions::default().into_inner();

        let (main_trace, pub_inputs) = generate_prover_args(&program_content, &None).unwrap();
        let (stark_proof, pub_inputs_proof) = prove_cairo_trace(main_trace, pub_inputs, &proof_options).unwrap();
        let proof_bytes = stark_proof.to_bytes();
        // print!("Cairo Stark Proof {:?}", stark_proof);
        println!("Proof size: {:.1} KB", proof_bytes.len() as f64 / 1024f64);
        verify_cairo_proof(stark_proof, pub_inputs_proof);
    }
}