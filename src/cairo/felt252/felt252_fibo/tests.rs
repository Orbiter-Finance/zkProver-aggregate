use winterfell::crypto::Hasher;

use crate::cairo::air::PublicInputs;

use super::{FibExample, Sha3_256_felt, build_proof_options, Example};


pub fn test_basic_proof_verification(e: Box<dyn Example>) {
    let proof = e.prove();
    let result = e.verify(proof);
    assert!(result.is_ok());
}

pub fn test_basic_proof_verification_fail(e: Box<dyn Example>) {
    let proof = e.prove();
    let verified = e.verify_with_wrong_inputs(proof);
    assert!(verified.is_err());
}



#[test]
fn fib_small_test_basic_proof_verification() {

    let fib = Box::new(super::FibExample::<Sha3_256_felt>::new(
        64,
        build_proof_options(false),
    ));
    test_basic_proof_verification(fib);
}

#[test]
fn fib_small_test_basic_proof_verification_fail() {
    let fib = Box::new(super::FibExample::<Sha3_256_felt>::new(
        128,
        build_proof_options(false),
    ));
    test_basic_proof_verification_fail(fib);
}


#[test]
fn Sha3_256_felt_test() {
    let b1 = [1_u8, 2, 3];
    let b2 = [1_u8, 2, 3, 0];

    // adding a zero bytes at the end of a byte string should result in a different hash
    let r1 = Sha3_256_felt::hash(&b1);
    let r2 = Sha3_256_felt::hash(&b2);
    assert_ne!(r1, r2);
}