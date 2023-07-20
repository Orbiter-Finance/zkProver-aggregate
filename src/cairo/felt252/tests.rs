use winterfell::math::{StarkField, FieldElement};
use crate::cairo::felt252::BaseElement;

#[test]
fn add() {
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
    assert_eq!(
        BaseElement::from([57,0,0,0]),
        BaseElement::from([100,0,0,0]) - BaseElement::from([43,0,0,0])
    );
}

#[test]
fn mul() {
    assert_eq!(
        BaseElement::from([1000,0,0,0]),
        BaseElement::from([100,0,0,0]) * BaseElement::from([10,0,0,0])
    );
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
fn as_int() {
    let a = BaseElement::from_raw([3, 0, 0, 0]);
    let b: u64 = a.as_int().try_into().unwrap();
    assert_eq!(b, 3);
}