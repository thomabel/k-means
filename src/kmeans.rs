use ndarray::prelude::*;
use crate::point::{*, self};
type Vector<'a> = ArrayView1<'a, f32>;
type VectorMut = Array1<f32>;
type Matrix = Array2<f32>;

pub struct KMeans {
    centroid: Matrix,
    cluster: Array1<Vec<usize>>,
}
impl KMeans {
    pub fn new(k: usize, x: usize) -> KMeans {
        let centroid = Array2::<f32>::zeros((k, x));
        let cluster = Array1::<Vec<usize>>::from_elem(k, Vec::new());
        KMeans {
            centroid,
            cluster,
        }
    }

    pub fn train(&mut self, input: &Array2<f32>) {
        // For each input point, assign index to a cluster.
        let len = input.len();
        for i in 0..len {
            let point = input.row(i);
            let cluster = self.find_cluster(&point);
            self.cluster[cluster].push(i);
        }
    }

    fn find_cluster(&self, input: &Vector) -> usize {
        let mut index = 0;
        let mut distance = f32::MAX;

        let len = self.centroid.len();
        for i in 0..len {
            let dif_dist = point::square_length(&sub(input, &self.centroid.row(i)).view());
            if dif_dist < distance {
                index = i;
                distance = dif_dist;
            }
        }
        index
    }

    fn average_centroid(centroid: &mut VectorMut, input: &Matrix, index: &Vector) {
        // Zero out the centroid.
        for i in 0..centroid.len() {
            centroid[i] = 0.;
        }

        // Sum columns from input based on index vector.
        for i in 0..index.len() {
            //centroid[i] += input.row(index(i));
        }

    }

}
