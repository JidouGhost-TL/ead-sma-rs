//! Error types for Nintendo EAD/SRD Super Mario Advance parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Nintendo EAD/SRD sample descriptors or
/// synthesis parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete header or payload could be read.
    UnexpectedEof,
    /// Descriptor or container header failed structural validation.
    InvalidHeader(&'static str),
    /// Loop range was invalid for the declared sample length.
    InvalidLoopRange {
        /// Loop start offset in samples.
        loop_start: u32,
        /// Loop end offset in samples.
        loop_end: u32,
        /// Total sample length in samples.
        sample_len: u32,
    },
    /// Sample index was out of range.
    InvalidSample(u32),
    /// Invalid synthesis parameter.
    InvalidParameter(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => {
                write!(f, "Unexpected end of data while parsing EAD/SMA stream")
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid EAD/SMA header: {msg}"),
            Self::InvalidLoopRange {
                loop_start,
                loop_end,
                sample_len,
            } => write!(
                f,
                "Invalid EAD/SMA loop range {loop_start}..{loop_end} for {sample_len}-byte sample"
            ),
            Self::InvalidSample(id) => write!(f, "Invalid EAD/SMA sample index {id}"),
            Self::InvalidParameter(msg) => write!(f, "Invalid EAD/SMA parameter: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `ead-sma`.
pub type Result<T> = core::result::Result<T, Error>;
