/*
Thomas Abel
2022-08-08
Machine Learning
*/
use print_data::*;
use constants::*;
use rand::seq::SliceRandom;
use ndarray::prelude::*;

use crate::kmeans::KMeans;

mod read;
mod print_data;
mod constants;
mod kmeans;
mod point;

fn main() {
    let path = [
        "./data/545_cluster_dataset programming 3.txt"
    ];
    let path_index = 0;
    let temp = read::read_csv(path[path_index], INPUTS);
    let input;

    match temp {
        Ok(o)=> {
            println!("Data read successfully.");
            input = o;
        },
        Err(e) => {
            println!("Error: {}", e);
            return;
        },
    }
    //_print_matrix(&input.view(), "INPUT");

    let mut kmeans = KMeans::new(K, INPUTS);
    kmeans.train(&input);

    println!("Ending program.");
}

fn choose_random_index(k: usize, size: usize) -> Vec<usize> {
    let mut vec = Vec::new();
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());

    let mut vec2 = Vec::new();
    for i in 0..k {
        vec2.push(vec[i]);
    }
    vec2
}
