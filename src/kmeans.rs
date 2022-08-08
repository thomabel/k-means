use ndarray::prelude::*;
pub struct KMeans {
    centroid: Vec<Point>,

}
impl KMeans {
    pub fn new(data_size: u32) -> KMeans {
        let centroid = Vec::<Point>::new();

        KMeans {
            centroid,
        }
    }

    
}

pub struct Point {
    pub x: Array1<f32>,
}
impl Point {
    pub fn square_distance(& self) -> f32 {
        let mut sum = 0.;
        for x in &self.x {
            sum += x * x;
        }
        sum
    }
    pub fn difference_from(&self, other: &Point) -> Array1<f32> {
        // Find the smaller of the two arrays.
        let len_self = self.x.len();
        let len_other = other.x.len();
        let self_smaller = len_self <= len_other;
        let len = if self_smaller {
            len_self
        }
        else {
            len_other
        };

        // Subtract each element pairwise using the index value we found earlier.
        let mut dif = Array1::<f32>::zeros(len);
        for i in 0..len {
            dif[i] = self.x[i] - other.x[i];
        }
        dif
    }
}