use core::slice;

use winter_utils::{AsBytes, SliceReader, Deserializable};
use winterfell::math::{StarkField, FieldElement};
use crate::cairo::felt252::{BaseElement, rand_value, rand_vector, BigInt, ELEMENT_BYTES};


#[test]
fn add() {

    let r: BaseElement = rand_value();
    assert_eq!(r, r + BaseElement::ZERO);

     // test addition within bounds
     assert_eq!(
        BaseElement::from(5),
        BaseElement::from(2) + BaseElement::from(3)
    );

    // test overflow
    // let t = BaseElement::from(M - 1);
    // assert_eq!(BaseElement::ZERO, t + BaseElement::ONE);
    // assert_eq!(BaseElement::ONE, t + BaseElement::from(2));



    let r: BaseElement = BaseElement::from_raw([100,0,0,0]);
    assert_eq!(r, r + BaseElement::ZERO);

    // test addition within bounds
    assert_eq!(
        BaseElement::from([57,43,0,0]),
        BaseElement::from([57,0,0,0]) + BaseElement::from([0,43,0,0])
    );

    assert_eq!(
        BaseElement::from([100,0,0,0]),
        BaseElement::from([57,0,0,0]) + BaseElement::from([43,0,0,0])
    );

    assert_eq!(
        BaseElement::ONE,
        BaseElement::from([1,0,0,0]) + BaseElement::from([0,0,0,0])
    );
}

#[test]
fn sub() {

     // identity
     let r: BaseElement = rand_value();
     assert_eq!(r, r - BaseElement::ZERO);
 
     // test subtraction within bounds
     assert_eq!(
         BaseElement::from(2u8),
         BaseElement::from(5u8) - BaseElement::from(3u8)
     );
 
     // test underflow
    //  let expected = BaseElement::from(BaseElement::MODULUS - 2);
    //  assert_eq!(expected, BaseElement::from(3u8) - BaseElement::from(5u8));
    assert_eq!(
        BaseElement::from([57,0,0,0]),
        BaseElement::from([100,0,0,0]) - BaseElement::from([43,0,0,0])
    );
}

#[test]
fn mul() {

    // identity
    let r: BaseElement = rand_value();
    assert_eq!(BaseElement::ZERO, r * BaseElement::ZERO);
    assert_eq!(r, r * BaseElement::ONE);

    println!("AS BYTES {:?}", BaseElement::ZERO.as_bytes());
    // r.as_bytes();

     // test multiplication within bounds
     assert_eq!(
        BaseElement::from(15u8),
        BaseElement::from(5u8) * BaseElement::from(3u8)
    );

     // test random values
    //  let v1: Vec<BaseElement> = rand_vector(1000);
    //  let v2: Vec<BaseElement> = rand_vector(1000);
    //  for i in 0..v1.len() {
    //      let r1 = v1[i];
    //      let r2 = v2[i];
 
    //      let expected = (r1.to_big_uint() * r2.to_big_uint()) % BigUint::from(M);
    //      let expected = BaseElement::from_big_uint(expected);
 
    //      if expected != r1 * r2 {
    //          assert_eq!(expected, r1 * r2, "failed for: {r1} * {r2}");
    //      }
    //  }

    assert_eq!(
        BaseElement::from([1000,0,0,0]),
        BaseElement::from([100,0,0,0]) * BaseElement::from([10,0,0,0])
    );
}

// #[test]
// fn elements_as_bytes() {
//     let source = vec![
//         BaseElement::from(1),
//         BaseElement::from(2),
//         BaseElement::from(3),
//         BaseElement::from(4),
//     ];

//     let mut expected = vec![];
//     expected.extend_from_slice(&source[0].0.to_le_bytes());
//     expected.extend_from_slice(&source[1].0.to_le_bytes());
//     expected.extend_from_slice(&source[2].0.to_le_bytes());
//     expected.extend_from_slice(&source[3].0.to_le_bytes());

//     assert_eq!(expected, BaseElement::elements_as_bytes(&source));
// }

#[test]
fn bytes_as_elements() {
    // let elements = vec![
    //     BaseElement::from(1),
    //     BaseElement::from(2),
    //     BaseElement::from(3),
    //     BaseElement::from(4),
    // ];

    // let mut bytes = vec![];
    // bytes.extend_from_slice(&elements[0].0.to_le_bytes());
    // bytes.extend_from_slice(&elements[1].0.to_le_bytes());
    // bytes.extend_from_slice(&elements[2].0.to_le_bytes());
    // bytes.extend_from_slice(&elements[3].0.to_le_bytes());
    // bytes.extend_from_slice(&BaseElement::from(5).0.to_le_bytes());

    // let result = unsafe { BaseElement::bytes_as_elements(&bytes[..32]) };
    // assert!(result.is_ok());
    // assert_eq!(elements, result.unwrap());

    // let result = unsafe { BaseElement::bytes_as_elements(&bytes[..33]) };
    // assert!(matches!(result, Err(DeserializationError::InvalidValue(_))));

    // let result = unsafe { BaseElement::bytes_as_elements(&bytes[1..33]) };
    // assert!(matches!(result, Err(DeserializationError::InvalidValue(_))));
}

#[test]
fn exp() {
    // let a = BaseElement::ZERO;
    // assert_eq!(a.exp(0), BaseElement::ONE);
    // assert_eq!(a.exp(1), BaseElement::ZERO);

    // let a = BaseElement::ONE;
    // assert_eq!(a.exp(0), BaseElement::ONE);
    // assert_eq!(a.exp(1), BaseElement::ONE);
    // assert_eq!(a.exp(3), BaseElement::ONE);

    // let a: BaseElement = rand_value();
    // assert_eq!(a.exp(3), a * a * a);
}

#[test]
fn inv() {
    let x =  BaseElement::from([1000,0,0,0]);
    let y =  BaseElement::inv(x);
    assert_eq!(
        BaseElement::ONE,
        x * y
    );
}

#[test]
fn from() {
    assert_eq!(
        BaseElement::from(100u64),
        BaseElement::from([100,0,0,0])
    )
}

#[test]
fn serialize_and_deserialize() {
    let target = BaseElement::from(256);

    // let bytes_1: &[u8] = target.as_bytes();

    // // target.to_raw();

    println!("AS BYTES {:?}", target.as_bytes());

    // let ptr: *const BigInt = &target.to_raw();

    let mut source = SliceReader::new(target.as_bytes());

    let read_element = BaseElement::read_from(&mut source).unwrap();

    assert_eq!(target, read_element);
}

#[test]
fn element_as_int() {
    // let a = BaseElement::from_raw([3, 0, 0, 0]);
    // let b: u64 = a.as_int().try_into().unwrap();
    // assert_eq!(b, 3);

    // let v = u64::MAX;
    // let e = BaseElement::from(3);
    // assert_eq!(v % super::M, e.as_int());

    // let e1 = BaseElement::from(0);
    // let e2 = BaseElement::new(M);
    // assert_eq!(e1.as_int(), e2.as_int());
    // assert_eq!(e1.as_int(), 0);
}

#[test]
fn to_string() {
     let e = BaseElement::from(1);
     print!("TO String {}", e.to_string());
}