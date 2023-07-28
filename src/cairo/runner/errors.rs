use winterfell::{ProverError, VerifierError};

#[derive(Debug)]
pub enum CairoProverError {
    ProverError(ProverError)
}