
#![feature(array_chunks)]

use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

pub mod cairo;

pub type PrimeField = Stark252PrimeField;
pub type FE = FieldElement<PrimeField>;