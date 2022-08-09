/*
Thomas Abel
2022-08-08
Machine Learning
*/
use constants::*;
use rand::seq::SliceRandom;
use ndarray::prelude::*;

use crate::kmeans::KMeans;

mod read;
mod print_data;
mod constants;
mod kmeans;
mod point;

type Matrix = Array2<f32>;

fn main() {
    // Read the data.
    let path = [
        "./data/545_cluster_dataset programming 3.txt"
    ];
    let path_index = 0;
    let read = read::read_csv(path[path_index], INPUTS);
    let input: Matrix;

    match read {
        Ok(o)=> {
            println!("Data read successfully.\n");
            input = o;
        },
        Err(e) => {
            println!("Error: {}", e);
            return;
        },
    }
    //_print_matrix(&input.view(), "INPUT");

    let results = model(&input, R);

    print_data::_print_vector(&Array::from_vec(results.1).view(), "ERROR");

    println!("Ending program.");
}

fn model(input: &Matrix, iterations: usize) -> (Vec<KMeans>, Vec<f32>) {
    let mut model = Vec::with_capacity(iterations);
    let mut error = Vec::with_capacity(iterations);

    for r in 0..iterations {
        // Set up the model.
        model.push(KMeans::new(K, INPUTS));
        let size = input.len_of(Axis(0));
        let index = &choose_random_index(K, size);
        model[r].assign_centroids(input, index);
    
        // Train the model.
        model[r].train(input);

        // Find the error.
        error.push(model[r].error(input));
    }

    (model, error)
}


// Chooses a random set of indicies to use as initial points.
fn choose_random_index(k: usize, size: usize) -> Vec<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());

    let mut vec2 = Vec::with_capacity(k);
    vec2.extend_from_slice(&vec[0..k]);
    vec2
}
