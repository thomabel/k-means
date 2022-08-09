/*
Thomas Abel
2022-08-08
Machine Learning
*/
use constants::*;
use rand::{seq::SliceRandom, Rng};
use ndarray::prelude::*;
use plotters::prelude::*;

use crate::kmeans::KMeans;

mod read;
mod print_data;
mod constants;
mod kmeans;
mod point;

type Matrix = Array2<f32>;

// MAIN
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
    
    let result = model(&input, R);
    let model = result.0;
    let error = result.1;
    let index = get_min_error(&error);
    let name = format!("RESULT_K={}", K);
    let _err = visualize(&name, &input, &model[index], error[index]);

    println!("Ending program.");
}

// Uses plotters crate to visualize data.
fn visualize(
    name: &str,
    input: &Matrix, 
    model: &KMeans, 
    error: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("./data/{}.png", name);
    let root = BitMapBackend::new(&path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let dimensions = (-3f32..3f32, -3f32..3f32);
    let caption = format!("{}, with Error = {}", name, error);

    let mut chart = 
        ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(dimensions.0, dimensions.1)?;

    chart.configure_mesh().draw()?;

    // Create vector of references to points in cluster
    for vector in model.get_cluster() {
        let mut points = Vec::new();

        // Push all points into the vector.
        for index in vector {
            let point = input.row(*index);
            points.push(point);
        }

        // Choose color
        let mut rng = rand::thread_rng();
        let color = RGBColor(rng.gen(), rng.gen(), rng.gen());
        let series =
            points.into_iter().map(
            |point| 
            Circle::new((point[0], point[1]), 2.0_f64, &color)
        );

        // Draw the input points.
        chart
            .draw_series(series)?;
    }

    // Draw the centroids.
    let series = 
        model.get_centroid().rows().into_iter().map(
            |point| 
            Circle::new((point[0], point[1]), 5.0_f64, &RED)
        );
    chart
        .draw_series(series)?;

    root.present()?;

    Ok(())
}

// 
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

fn get_min_error(error: &[f32]) -> usize {
    let mut index = 0;
    let mut err = f32::MAX;
    for e in 0..error.len() {
        if error[e] < err {
            index = e;
            err = error[e];
        }
    }
    index
}