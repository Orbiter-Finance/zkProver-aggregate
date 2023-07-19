use super::vec_writer::VecWriter;
use crate::cairo::cairo_layout::CairoLayout;
use crate::cairo::cairo_mem::CairoMemory;
use crate::cairo::register_states::RegisterStates;
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

