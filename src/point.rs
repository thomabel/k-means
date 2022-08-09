use ndarray::prelude::*;
type Point<'a> = ArrayView1<'a, f32>;
type PointMut = Array1<f32>;

pub fn square_length(input: &Point) -> f32 {
    let mut sum = 0.;
    for x in input {
        sum += x * x;
    }
    sum
}

fn find_smaller(lhs: &Point, rhs: &Point) -> usize {
    let left = lhs.len();
    let right = rhs.len();
    if left <= right {
        left
    }
    else {
        right
    }
}

pub fn sub(lhs: &Point, rhs: &Point) -> PointMut {
    let len = find_smaller(lhs, rhs);
    let mut point = Array1::<f32>::zeros(len);
    for i in 0..len {
        point[i] = lhs[i] - rhs[i];
    }
    point
}

pub fn _copy(into: &mut PointMut, from: &Point) {
    let len = into.len();
    for i in 0..len {
        into[i] = from[i];
    }
}
