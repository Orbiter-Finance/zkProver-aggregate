use std::{
    collections::HashMap,
    fs::{create_dir_all, File},
    io::Write,
};
use std::time::Instant;
use colored::Colorize;
use winterfell::{
    crypto::hashers::Poseidon,
    math::{ log2, StarkField},
    Air, AirContext, Prover, TraceInfo, StarkProof,
};

use crate::{Felt252 as BaseElement, cairo::air::WinterPublicInputs, utils::json::cairo_proof_to_json};
use winterfell::crypto::{DefaultRandomCoin, ElementHasher};

use super::utils::{LoggingLevel, WinterCircomError, check_file};

/// Generate a Groth16 proof that the Winterfell proof is correct.
///
/// Only verifying the Groth16 proof attests of the validity of the Winterfell
/// proof. This makes this function the core of this crate.
///
/// This function only works if the Circom code has previously generated and
/// compiled and if the circuit-specific keys have been generated. This is
/// performed by the [circom_compile] function.
///
/// ## Steps
///
/// - Generate the Groth16 proof
/// - (Not in release mode) Verify the proof
/// - Parse the proof into a Circom-compatible JSON file
/// - Compute execution witness
/// - Generate proof
///
/// ## Soundness
///
/// The Groth16 proof generated is not self-sufficient. An additional check on
/// the out of domain trace frame and evaluations is required to ensure the
/// validity of the entire system.
///
/// This additional check, along with the Groth16 proof verification, is performed
/// by the [circom_verify] function.
///
/// See [crate documentation](crate) for more information.
pub fn circom_prove<P,H>(
    proof: StarkProof,
    prover: P,
    trace: <P as Prover>::Trace,
    circuit_name: &str,
    logging_level: LoggingLevel,
)
// ) -> Result<(), WinterCircomError>
where
    P: Prover<BaseField = BaseElement,HashFn = H>,
    H: ElementHasher<BaseField = BaseElement>,
    <<P as Prover>::Air as Air>::PublicInputs: WinterPublicInputs,
{

    // // CHECK FOR FILES
    // // ===========================================================================

    // check_file(
    //     format!("target/circom/{}/verifier.r1cs", circuit_name),
    //     Some("did you run compile?"),
    // )?;
    // check_file(
    //     format!("target/circom/{}/verifier.zkey", circuit_name),
    //     Some("did you run compile?"),
    // )?;

    // BUILD PROOF
    // ===========================================================================

    if logging_level.print_big_steps() {
        println!("{}", "Building STARK proof...".green());
    }

    // assert_eq!(hash_fn, HashFunction::Poseidon);

    let pub_inputs = prover.get_pub_inputs(&trace);
    // let proof = prover
    //     .prove(trace)
    //     .map_err(|e| WinterCircomError::ProverError(e)).unwrap();

    // VERIFY PROOF
    // ===========================================================================

    // #[cfg(debug_assertions)]
    {
        if logging_level.print_big_steps() {
            println!("{}", "Verifying STARK proof...".green());
        }

        winterfell::verify::<P::Air, H, DefaultRandomCoin<H>>(proof.clone(), pub_inputs.clone())
            .map_err(|err| WinterCircomError::InvalidProof(Some(err))).unwrap();
    }

    // BUILD JSON OUTPUTS
    // ===========================================================================

    if logging_level.print_big_steps() {
        println!("{}", "Parsing proof to JSON...".green());
    }

    // retrieve air and proof options
    let air = P::Air::new(
        proof.get_trace_info(),
        pub_inputs.clone(),
        proof.options().clone(),
    );

    // convert proof to json object
    let mut fri_tree_depths = Vec::new();
    let json = cairo_proof_to_json::<P::Air, H,DefaultRandomCoin<H>>(
        proof,
        &air,
        pub_inputs.clone(),
        &mut fri_tree_depths,
    );

    // print json to file
    let json_string = format!("{}", json);
    create_dir_all(format!("target/circom/{}", circuit_name)).map_err(|e| {
        WinterCircomError::IoError {
            io_error: e,
            comment: Some(String::from("creating Circom output directory")),
        }
    }).unwrap();
    let mut file =
        File::create(format!("target/circom/{}/input.json", circuit_name)).map_err(|e| {
            WinterCircomError::IoError {
                io_error: e,
                comment: Some(String::from("creating input.json")),
            }
        }).unwrap();
    file.write(&json_string.into_bytes())
        .map_err(|err| WinterCircomError::IoError {
            io_error: err,
            comment: Some(String::from("writing input.json")),
        }).unwrap();
    
}