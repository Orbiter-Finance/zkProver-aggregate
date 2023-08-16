use serde::Serialize;
use serde_json::{json, Value};
use winterfell::{crypto::{ElementHasher, RandomCoin}, math::{ log2, StarkField}, Air, Serializable, StarkProof, EvaluationFrame};
use winterfell::crypto::{DefaultRandomCoin, Digest};
use winterfell::math::{FieldElement, ToElements};
use std::cell::RefCell;

use crate::Felt252 as BaseElement;
// pub struct Channel< H: ElementHasher<BaseField = BaseElement>> {
//     // trace queries
//     trace_commitments: Vec<H::Digest>,
//     // constraint queries
//     constraint_commitment: H::Digest,
//     // FRI proof
//     fri_commitments: Option<Vec<H::Digest>>,
// }

/// Parse a [StarkProof] into a Circom-usable JSON object.
///
/// ## Padding
///
/// To ensure constant size arrays and therefore Circom compatibility, elements
/// of `fri_layer_proofs` and `fri_layer_queries` arrays are padded with zeroes.
///
/// The `fri_num_queries` and `fri_tree_depths` arguments are populated so that:
///
/// ```text
/// fri_layer_proofs[i].len() = fri_num_queries[i]
/// fri_layer_proofs[i][j].len() = fri_tree_depths[i]
/// fri_layer_queries[i].len() = fri_num_queries[i] * folding_factor
/// ```
///
/// ## JSON structure
///
/// ```json
/// {
///     "addicity": _,
///     "constraint_commitment": _,
///     "constraint_evaluations": [[_; trace_width]; num_queries],
///     "constraint_query_proofs": [[_; tree_depth]; num_queries],
///     "fri_commitments": [num_fri_layers + 1],
///     "fri_layer_proofs": [[[_; tree_depth]; num_queries]; num_fri_layers],
///     "fri_layer_queries": [[_; num_queries * folding_factor]; num_fri_layers],
///     "fri_remainder": [_; lde_domain_size / (folding_factor ** num_fri_layers)],
///     "ood_constraint_evaluations": [_; ce_blowup_factor],
///     "ood_frame_constraint_evaluation": [_; num_transition_constraints],
///     "ood_trace_frame": [[_; trace_width]; 2],
///     "pow_nonce": _,
///     "pub_coin_seed": [_; num_pub_coin_seed],
///     "public_inputs": [_; num_public_inputs],
///     "trace_commitment": _,
///     "trace_evaluations": [[_; trace_width]; num_queries],
///     "trace_query_proofs": [[tree_depth]; num_queries],
/// }
/// ```
///
// TODO: Return errors instead of panicking (`.map_err()` and `?` instead of `.unwrap()`)
pub fn cairo_proof_to_json<AIR, H,RandCoin>(
    proof: StarkProof,
    air: &AIR,
    pub_inputs: AIR::PublicInputs,
    fri_tree_depths: &mut Vec<usize>,
) -> Value
where
    AIR: Air<BaseField = BaseElement>,
    <AIR as Air>::PublicInputs: Serialize,
    H: ElementHasher<BaseField = AIR::BaseField>,
    RandCoin: RandomCoin<BaseField = AIR::BaseField, Hasher = H>
{
    let StarkProof {
        context,
        commitments,
        mut trace_queries,
        constraint_queries,
        ood_frame,
        fri_proof,
        pow_nonce,
    } = proof;
    // make sure AIR and proof base fields are the same
    if AIR::BaseField::get_modulus_le_bytes() != context.field_modulus_bytes() {
        // return Err(VerifierError::InconsistentBaseField);
        println!("InconsistentBaseField");
    }

    let constraint_frame_width = air.context().num_constraint_composition_columns();

    let num_trace_segments = air.trace_layout().num_segments();
    let main_trace_width = air.trace_layout().main_trace_width();
    let aux_trace_width = air.trace_layout().aux_trace_width();
    let lde_domain_size = air.lde_domain_size();
    let fri_options = air.options().to_fri_options();
    let num_queries = air.options().num_queries();
    let folding_factor = fri_options.folding_factor();

    // enforce only one trace segment to ensure compatibility with the Circom code
    assert_eq!(num_trace_segments, 1);

    // PUBLIC COIN SEED
    // ===========================================================================

    // serialize public inputs and context
    let mut pub_coin_seed = context.to_elements();
    pub_coin_seed.append(&mut pub_inputs.to_elements());

    let mut public_coin = RandCoin::new(&pub_coin_seed);

    // turn into f256 field elements

    while pub_coin_seed.len() % AIR::BaseField::ELEMENT_BYTES != 0 {
        pub_coin_seed.push(AIR::BaseField::ZERO);
    }

    // COMMITMENTS
    // ===========================================================================

    // retreive commitments
    let (trace_commitments, constraint_commitment, fri_commitments) = commitments
        .parse::<H>(
            num_trace_segments,
            fri_options.num_fri_layers(lde_domain_size),
        )
        .unwrap();


    public_coin.reseed(trace_commitments[0]);
    public_coin.reseed(constraint_commitment);

    // map commitments to BaseElements
    let trace_commitment = trace_commitments
        .iter()
        .map(|c| AIR::BaseField::from_bytes_le(&c.as_bytes()))
        .collect::<Vec<_>>()
        .remove(0);
    let constraint_commitment =
        AIR::BaseField::from_bytes_le(&constraint_commitment.as_bytes());

    // OOD FRAME
    // ===========================================================================

    // parse ood_frame, ignoring the ood_aux_trace_frame
    let (ood_trace_evaluations, ood_constraint_evaluations) = ood_frame
        .parse::<AIR::BaseField>(main_trace_width, aux_trace_width, constraint_frame_width)
        .unwrap();

    let ood_trace_frame =
        TraceOodFrame::new(ood_trace_evaluations, main_trace_width, aux_trace_width);

    public_coin.reseed(H::hash_elements(ood_trace_frame.values()));
    public_coin.reseed(H::hash_elements(&ood_constraint_evaluations));

    // OOD FRAME CONSTRAINT EVALUATIONS
    // FIXME: fix periodic values
    println!("main_trace_with {:?} aux_trace_width {:?}", air.trace_layout().aux_trace_width(), air.trace_layout().main_trace_width());
    let mut ood_frame_constraint_evaluation = AIR::BaseField::zeroed_vector(air.trace_info().width());
    air.evaluate_transition::<AIR::BaseField>(
        &ood_trace_frame.main_frame(),
        &[],
        &mut ood_frame_constraint_evaluation,
    );

    let main_frame = ood_trace_frame.main_frame();

    let ood_trace_frame = (main_frame.current(), main_frame.next());

    // FRI PROOF PART 1
    // ===========================================================================

    // only accept a fri proof with a single partition
    assert_eq!(fri_proof.num_partitions(), 1);

    for root in fri_commitments.iter() {
        public_coin.reseed(*root);
    }

    // there are fri_num_queries + 1 fri_commitments because
    // of the commitment for the remainder
    let fri_commitments = fri_commitments
        .iter()
        .map(|c| AIR::BaseField::from_bytes_le(&c.as_bytes()))
        .collect::<Vec<_>>();

    // QUERY POSITIONS
    // ===========================================================================

    public_coin.reseed_with_int(pow_nonce);

    let query_positions = public_coin
        .draw_integers(num_queries, lde_domain_size)
        .unwrap();

    // FRI PROOF PART 2
    // ===========================================================================

    // parse fri proof into Merkle proofs and queries for each layer
    let fri_remainder = fri_proof.parse_remainder::<AIR::BaseField>().unwrap();
    let (mut fri_layer_queries, fri_layer_proofs) = fri_proof
        .parse_layers::<H, AIR::BaseField>(lde_domain_size, folding_factor)
        .unwrap();

    // convert batch merkle proofs into authentication paths
    // and map digests to BaseElements
    let mut indexes = query_positions.clone();
    let mut domain_size = lde_domain_size;
    let mut fri_layer_proofs = fri_layer_proofs
        .iter()
        .map(|merkle_proof| {
            indexes = fold_positions(&indexes, domain_size, folding_factor);
            domain_size /= folding_factor;

            let rest = merkle_proof
                .into_paths(&indexes)
                .unwrap()
                .iter()
                .map(|path| {
                    path.iter()
                        .map(|digest| AIR::BaseField::from_bytes_le(&digest.as_bytes()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            rest
        })
        .collect::<Vec<_>>();

    // pad fri_query_proofs with zeroes to ensure constant size arrays
    let tree_depth = log2(lde_domain_size) as usize;
    let fri_layer_proofs = fri_layer_proofs
        .iter_mut()
        .map(|paths| {
            fri_tree_depths.push(paths[0].len());

            for path in paths.iter_mut() {
                while path.len() < tree_depth {
                    path.push(AIR::BaseField::ZERO);
                }
            }
            while paths.len() < num_queries {
                paths.push(vec![AIR::BaseField::ZERO; tree_depth]);
            }
            paths
        })
        .collect::<Vec<_>>();

    // pad fri layer queries with zeroes to ensure constant size arrays
    for queries in fri_layer_queries.iter_mut() {
        while queries.len() < num_queries * folding_factor {
            queries.push(AIR::BaseField::ZERO);
        }
    }

    // TRACE QUERIES
    // ===========================================================================

    // pick out trace queries first element (the one that corresponds to the
    // main trace segment) and parse it into a Merkle proof and trace states
    let (trace_query_proofs, trace_evaluations) = trace_queries
        .remove(0)
        .parse::<H, AIR::BaseField>(lde_domain_size, num_queries, main_trace_width)
        .unwrap();

    // convert the batch Merkle proof into authentication paths
    // and map hash digests to BaseElements
    let trace_query_proofs = trace_query_proofs
        .into_paths(&query_positions)
        .unwrap()
        .iter()
        .map(|path| {
            path.iter()
                .map(|digest| AIR::BaseField::from_bytes_le(&digest.as_bytes()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<Vec<_>>>();

    // map constraint states table into a matrix of BaseElements
    let trace_evaluations = trace_evaluations.rows().fold(vec![], |mut e, row| {
        e.push(row.to_vec());
        e
    });

    // CONSTRAINT QUERIES
    // ===========================================================================
    // let constraint_queries = ConstraintQueries::new(constraint_queries, air)?;
    // parse constraint queries back into a Merkle proof and a vector of states
    let (constraint_query_proofs, constraint_evaluations) = constraint_queries
        .parse::<H, AIR::BaseField>(lde_domain_size, num_queries, constraint_frame_width)
        .unwrap();

    // convert the batch Merkle proof into authentication paths
    // and map hash digests to BaseElements
    let constraint_query_proofs = constraint_query_proofs
        .into_paths(&query_positions)
        .unwrap()
        .iter()
        .map(|path| {
            path.iter()
                .map(|digest| AIR::BaseField::from_bytes_le(&digest.as_bytes()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // map constraint states table into a matrix of BaseElements
    let constraint_evaluations = constraint_evaluations.rows().fold(vec![], |mut e, row| {
        e.push(row.to_vec());
        e
    });

    // BUILD JSON OBJECT
    // ===========================================================================

    json!({
        "addicity_root": AIR::BaseField::TWO_ADIC_ROOT_OF_UNITY,
        "constraint_commitment": constraint_commitment,
        "constraint_evaluations": constraint_evaluations,
        "constraint_query_proofs": constraint_query_proofs,
        "fri_commitments": fri_commitments,
        "fri_layer_proofs": fri_layer_proofs,
        "fri_layer_queries": fri_layer_queries,
        "fri_remainder": fri_remainder,
        "ood_constraint_evaluations": ood_constraint_evaluations,
        "ood_frame_constraint_evaluation" : ood_frame_constraint_evaluation,
        "ood_trace_frame": ood_trace_frame,
        "pow_nonce": pow_nonce,
        "pub_coin_seed": pub_coin_seed,
        "public_inputs": pub_inputs,
        "trace_commitment": trace_commitment,
        "trace_evaluations": trace_evaluations,
        "trace_query_proofs": trace_query_proofs,
    })
}

// HELPER FUNCTIONS
// ===========================================================================

fn fold_positions(
    positions: &[usize],
    source_domain_size: usize,
    folding_factor: usize,
) -> Vec<usize> {
    let target_domain_size = source_domain_size / folding_factor;

    let mut result = Vec::new();
    for position in positions {
        let position = position % target_domain_size;
        // make sure we don't record duplicated values
        if !result.contains(&position) {
            result.push(position);
        }
    }

    result
}

// TRACE OUT-OF-DOMAIN FRAME
// ================================================================================================

pub struct TraceOodFrame<E: FieldElement> {
    values: Vec<E>,
    main_trace_width: usize,
    aux_trace_width: usize,
}

impl<E: FieldElement> TraceOodFrame<E> {
    pub fn new(values: Vec<E>, main_trace_width: usize, aux_trace_width: usize) -> Self {
        Self {
            values,
            main_trace_width,
            aux_trace_width,
        }
    }

    pub fn values(&self) -> &[E] {
        &self.values
    }

    // The out-of-domain frame is stored as one vector of interleaved values, one from the
    // current row and the other from the next row. See `OodFrame::set_trace_states`.
    // Thus we need to untangle the current and next rows stored in `Self::values` and we
    // do that for the main and auxiliary traces separately.
    // Pictorially, for the main trace portion:
    //
    // Input vector: [a1, b1, a2, b2, ..., an, bn, c1, d1, c2, d2, ..., cm, dm]
    // with n being the main trace width and m the auxiliary trace width.
    //
    // Output:
    //          +-------+-------+-------+-------+-------+
    //          |  a1   |   a2  |   a3  |  ...  |   an  |
    //          +-------+-------+-------+-------+-------+
    //          |  b1   |   b2  |   b3  |  ...  |   bn  |
    //          +-------+-------+-------+-------+-------+
    pub fn main_frame(&self) -> EvaluationFrame<E> {
        let mut current = vec![E::ZERO; self.main_trace_width];
        let mut next = vec![E::ZERO; self.main_trace_width];

        for (i, a) in self
            .values
            .chunks(2)
            .take(self.main_trace_width)
            .enumerate()
        {
            current[i] = a[0];
            next[i] = a[1];
        }

        EvaluationFrame::from_rows(current, next)
    }

    // Similar to `Self::main_frame`, the following untangles the current and next rows stored
    // in `Self::values` for the auxiliary trace portion when it exists else it returns `None`.
    // Pictorially:
    //
    // Input vector: [a1, b1, a2, b2, ..., an, bn, c1, d1, c2, d2, ..., cm, dm]
    // with n being the main trace width and m the auxiliary trace width.
    //
    // Output:
    //          +-------+-------+-------+-------+-------+
    //          |  c1   |   c2  |   c3  |  ...  |   cm  |
    //          +-------+-------+-------+-------+-------+
    //          |  d1   |   d2  |   d3  |  ...  |   dm  |
    //          +-------+-------+-------+-------+-------+
    pub fn aux_frame(&self) -> Option<EvaluationFrame<E>> {
        if self.aux_trace_width == 0 {
            None
        } else {
            let mut current_aux = vec![E::ZERO; self.aux_trace_width];
            let mut next_aux = vec![E::ZERO; self.aux_trace_width];

            for (i, a) in self
                .values
                .chunks(2)
                .skip(self.main_trace_width)
                .enumerate()
            {
                current_aux[i] = a[0];
                next_aux[i] = a[1];
            }
            Some(EvaluationFrame::from_rows(current_aux, next_aux))
        }
    }
}
