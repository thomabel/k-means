use std::error::Error;
use ndarray::prelude::*;

pub fn read_csv (path: &str, inputs: usize) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader 
        = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .delimiter(b' ')
            .from_path(path)?;
    let mut output = Vec::new();
    let mut rows: usize = 0;

    // Parse each row in the input.
    for result in reader.records() {
        let record = result?;

        // Parse each entry in that row as an f32.
        for r in record.into_iter() {
            // Filter out empty entries.
            if r.is_empty() {
                continue;
            }

            // Parse logic.
            let num = r.parse::<f32>();
            match num {
                Ok(c) => {
                    output.push(c);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        rows += 1;
    }
    let out_array 
        = Array2::from_shape_vec((rows, inputs), output)?;
    Ok(out_array)
}