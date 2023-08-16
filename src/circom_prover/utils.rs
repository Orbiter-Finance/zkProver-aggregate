

// ERRORS
// ===========================================================================

use std::{
    io, 
    fmt::{Display, Debug}, path::Path,
};

use colored::Colorize;

use winterfell::{VerifierError, ProverError};

/// Enumeration of the possible error types for this crate.
pub enum WinterCircomError {
    /// This error type is triggered when a function of this crate resulted
    /// in a [std::io::Error].
    IoError {
        io_error: io::Error,
        comment: Option<String>,
    },

    /// This error is triggered after a function of this crate failed to
    /// generate a file it further needs.
    FileNotFound {
        file: String,
        comment: Option<String>,
    },

    /// This error type is triggered when an underlying command called by a
    /// function of this crate failed (returned a non-zero exit code).
    ExitCodeError {
        executable: String,
        code: i32,
    },

    /// This error is triggered, when the generated Winterfell proof could not
    /// be verified. This only happens in debug mode.
    InvalidProof(Option<VerifierError>),

    /// This error is triggered when the Winterfell proof generation failed.
    ProverError(ProverError),
}

impl Display for WinterCircomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let error_string = match self {
            WinterCircomError::IoError { io_error, comment } => {
                if let Some(comment) = comment {
                    format!("IoError: {} ({}).", io_error, comment)
                } else {
                    format!("IoError: {}.", io_error)
                }
            }
            WinterCircomError::FileNotFound { file, comment } => {
                if let Some(comment) = comment {
                    format!("File not found: {} ({}).", file, comment)
                } else {
                    format!("File not found: {}.", file)
                }
            }
            WinterCircomError::ExitCodeError { executable, code } => {
                format!("Executable {} exited with code {}.", executable, code)
            }
            WinterCircomError::InvalidProof(verifier_error) => {
                if let Some(verifier_error) = verifier_error {
                    format!("Invalid proof: {}.", verifier_error)
                } else {
                    format!("Invalid proof.")
                }
            }
            WinterCircomError::ProverError(prover_error) => {
                format!("Prover error: {}.", prover_error)
            }
        };

        write!(f, "{}", error_string.yellow())
    }
}

impl Debug for WinterCircomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}


// LOGGING
// ===========================================================================

/// Logging level selector for functions of this crate.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LoggingLevel {
    /// Nothing is printed to stdout (errors are still printed to stderr)
    Quiet,

    /// Minimal logging (only major steps are logged to stdout)
    Default,

    /// Output of underlying executables is printed as well
    Verbose,

    /// Underlying executables are set to verbose mode, and their output is printed as well
    VeryVerbose,
}

impl LoggingLevel {
    /// Returns whether the logging level is set to [Default](LoggingLevel::Default)
    /// or above.
    ///
    /// This is used to trigger the printing of big step announcements in the functions
    /// of this crate.
    pub(crate) fn print_big_steps(&self) -> bool {
        match self {
            Self::Quiet => false,
            _ => true,
        }
    }

    /// Returns whether the logging level is set to [Verbose](LoggingLevel::Verbose)
    /// or above.
    ///
    /// This is used to trigger the printing of underlying commands stdout in the
    /// functions of this crate.
    pub(crate) fn print_command_output(&self) -> bool {
        match self {
            Self::Quiet => false,
            Self::Default => false,
            _ => true,
        }
    }

    /// Returns whether the logging level is set to
    /// [VeryVerbose](LoggingLevel::VeryVerbose).
    ///
    /// This is used to trigger verbose mode of the underlying commands of the
    /// functions in this crate.
    pub(crate) fn verbose_commands(&self) -> bool {
        match self {
            Self::VeryVerbose => true,
            _ => false,
        }
    }
}

/// Verify that a file exists, returning an error on failure.
pub(crate) fn check_file(path: String, comment: Option<&str>) -> Result<(), WinterCircomError> {
    if !Path::new(&path).exists() {
        return Err(WinterCircomError::FileNotFound {
            file: Path::new(&path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned(),
            comment: comment.map(|s| s.to_owned()),
        });
    }
    Ok(())
}