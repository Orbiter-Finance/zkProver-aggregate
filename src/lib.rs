
#![feature(array_chunks)]

pub use cairo::felt252::BaseElement;
pub mod cairo;
// pub use winterfell::math::fields::f64::BaseElement as f64BaseElement;

pub type Felt252 = BaseElement;
pub type FETest = Felt252;