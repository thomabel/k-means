use ndarray::prelude::*;
use crate::{point::{*, self}, print_data::_print_matrix};

type Vector<'a> = ArrayView1<'a, f32>;
type VectorMut<'a> = ArrayViewMut1<'a, f32>;
type Matrix = Array2<f32>;

pub struct KMeans {
    k: usize,
    x: usize,
    centroid: Matrix,               // The collection of centroid points.
    cluster: Array1<Vec<usize>>,    // Stores the points that belong to each cluster.
}
impl KMeans {
    // === PUBLIC ===
    // Constructor
    pub fn new(k: usize, x: usize) -> KMeans {
        let centroid = Array2::<f32>::zeros((k, x));
        let cluster = Array1::<Vec<usize>>::from_elem(k, Vec::new());
        KMeans {
            k, x,
            centroid,
            cluster,
        }
    }

    // Assigns each centroid based on an index array into the input array.
    pub fn assign_centroids(&mut self, input: &Matrix, index: &[usize]) {
        for k in 0..self.k {
            let mut centroid = self.centroid.row_mut(k);
            let vector = input.row(index[k]);
            for x in 0..self.x {
                centroid[x] = vector[x];
            }
        }
        //println!("INITIAL");
        //self._print_centroids();
    }

    // Prints all values of the centroids.
    pub fn _print_centroids(&self) {
        _print_matrix(&self.centroid.view(), "CENTROIDS");
    }

    // 
    pub fn get_centroid(&self) -> &Matrix {
        &self.centroid
    }

    pub fn get_cluster(&self) -> &Array1<Vec<usize>> {
        &self.cluster
    }

    // Trains the model by assigning vectors to centroids and
    // updating the centroid with the average position.
    pub fn train(&mut self, input: &Matrix) {
        let mut counter = 0;
        let mut cont = true;
        while cont {
            self.reset_cluster();
            self.assign_clusters(input);
            cont = self.update_centroids(input);
            counter += 1;
            
            //println!("GEN: {}", counter);
            //let error = self.error(input);
            //println!("ERROR: {:.6} \n", error);
            //self._print_centroids();
        }
    }

    // Gets the Sum of Squares error value for all clusters.
    pub fn error(&self, input: &Matrix) -> f32 {
        let mut sum = 0.;
        for k in 0..self.k {
            let centroid = &self.centroid.row(k);
            let index = &self.cluster[k];
            sum += KMeans::error_cluster(centroid, input, index);
        }
        sum
    }
    

    // === PRIVATE ===

    // Compares centroids to see if they're equal.
    // Returns whether the matrices are equal.
    fn centroid_equality(&self, centroid: &Matrix) -> bool {
        for k in 0..self.k {
            let c1 = self.centroid.row(k);
            let c2 = centroid.row(k);
            for x in 0..self.x {
                if c1[x] != c2[x] {
                    return false;
                }
            }
        }
        true
    }

    // Resets the cluster vectors so that they are empty.
    fn reset_cluster(&mut self) {
        for c in &mut self.cluster {
            c.clear();
        }
    }

    // Assigns each input to its closest cluster.
    fn assign_clusters(&mut self, input: &Matrix) {
        for i in 0..input.len_of(Axis(0)) {
            let point = input.row(i);
            let cluster = self.find_cluster(&point);
            self.cluster[cluster].push(i);
        }
    }

    // Takes an input, finds the closest centroid, and returns the index of that cluster.
    fn find_cluster(&self, input: &Vector) -> usize {
        let mut index = 0;
        let mut distance = f32::MAX;

        for k in 0..self.k {
            let dif_dist = point::square_length(&sub(input, &self.centroid.row(k)).view());
            if dif_dist < distance {
                index = k;
                distance = dif_dist;
            }
        }
        index
    }

    // Creates new centroids from its collection of vectors.
    // Returns whether training should continue.
    fn update_centroids(&mut self, input: &Matrix) -> bool {
        let mut new_centroid = Array2::<f32>::zeros(self.centroid.raw_dim());
        
        for k in 0..self.k {
            let mut centroid = new_centroid.row_mut(k);
            let index = &self.cluster[k];
            KMeans::average_centroid(&mut centroid, input, index);
        }

        let equal = self.centroid_equality(&new_centroid);
        if equal {
            return false;
        }
        self.centroid = new_centroid;
        true
    }

    // Finds the average point between a cluster of points.
    // Uses an index array to track cluster members.
    fn average_centroid(centroid: &mut VectorMut, input: &Matrix, index: &[usize]) {
        let cen_len = centroid.len();

        // Zero out the centroid.
        for i in 0..cen_len {
            centroid[i] = 0.;
        }

        // Sum columns from input based on index vector.
        for i in index {
            let row = input.row(*i);
            for j in 0..cen_len {
                centroid[j] += row[j];
            }
        }

        // Divide each element of the centroid by the length of index.
        let cluster_size = index.len() as f32;
        for c in centroid {
            *c /= cluster_size;
        }
    }

    // Gets the Sum of Squares error value for a single cluster.
    fn error_cluster(centroid: &Vector, input: &Matrix, index: &[usize]) -> f32 {
        let mut sum = 0.;
        for x in index {
            sum += square_length(&sub(&input.row(*x).view(), &centroid.view()).view());
        }
        sum
    }

}
