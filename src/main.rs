// Intelligent perceptron
//
// An attempt to create an intelligent perceptron using my
// neuron trait, in neuron.rs
//
// Neil Crago <n.j.crago@gmail.com>
// start date:     13/02/24

mod neuron;
use crate::neuron::{neuron_chat, Neuron, SuperNeuron};

use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

/// Heaviside Step Function
trait Heaviside {
    fn heaviside(&self) -> i8;
}

/// Implement heaviside() for f64
impl Heaviside for f64 {
    fn heaviside(&self) -> i8 {
        (*self >= 0.0) as i8
    }
}

/// Dot product of input and weights
fn dot(input: (i8, i8, i8), weights: (f64, f64, f64)) -> f64 {
    input.0 as f64 * weights.0 + input.1 as f64 * weights.1 + input.2 as f64 * weights.2
}

struct TrainingDatum {
    input: (i8, i8, i8),
    expected: i8,
}

fn main() {
    println!();
    println!("Perceptron Phase (XOR)");

    let mut rng = thread_rng();

    // Provide some training data
    let training_data = [
        TrainingDatum {
            input: (0, 0, 1),
            expected: 0,
        },
        TrainingDatum {
            input: (0, 1, 1),
            expected: 1,
        },
        TrainingDatum {
            input: (1, 0, 1),
            expected: 1,
        },
        TrainingDatum {
            input: (1, 1, 1),
            expected: 1,
        },
    ];

    // Initialize weight vector with random data between 0 and 1
    //let range = rng.gen_range(0.0..1.0);
    let mut w = (
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
    );

    // Learning rate
    let eta = 0.25;

    // Number of iterations
    let n = 1_000;

    // Training
    println!("Perceptron::Starting training phase with {n} iterations");
    for _ in 0..n {
        // Choose a random training sample
        let &TrainingDatum { input: x, expected } = (training_data).choose(&mut rng).unwrap();

        // Calculate the dot product
        let result = dot(x, w);

        // Calculate the error
        let error = expected - result.heaviside();

        // Update the weights
        w.0 += eta * error as f64 * x.0 as f64;
        w.1 += eta * error as f64 * x.1 as f64;
        w.2 += eta * error as f64 * x.2 as f64;
    }

    // Show result
    for &TrainingDatum { input, .. } in &training_data {
        let result = dot(input, w);
        println!(
            "{} XOR {}: {:.*} -> {}",
            input.0,
            input.1,
            8,
            result,
            result.heaviside()
        );
    }

    println!();
    println!("SuperNeuron Phase");

    let mut sn = neuron::SuperNeuron::new(4, SuperNeuron::sigmoid);
    let mut tn = neuron::SuperNeuron::new(4, SuperNeuron::smooth_relu);
    let mut un = neuron::SuperNeuron::new(4, SuperNeuron::tanh);

    sn.set_bias(0.2);
    tn.set_bias(0.2);
    un.set_bias(0.2);

    let weightvec = vec![0.1, 0.3, 0.6, 0.7];
    let inputvec = vec![0.25, 0.37, 0.79, 0.9998];

    sn.set_weights(weightvec.clone());
    tn.set_weights(weightvec.clone());
    un.set_weights(weightvec.clone());

    println!(
        "Output from SuperNeuron with sigmoid activation = {}",
        sn.output(inputvec.clone())
    );
    println!(
        "Output from SuperNeuron with smooth ReLu activation = {}",
        tn.output(inputvec.clone())
    );
    println!(
        "Output from SuperNeuron with tanh activation = {}",
        un.output(inputvec.clone())
    );

    println!();
    println!("Neuron 2 Neuron communication (mpsc) phase");
    neuron_chat();
    println!();
}

#[cfg(test)]
mod test {
    use super::Heaviside;

    #[test]
    fn heaviside_positive() {
        assert_eq!((0.5).heaviside(), 1i8);
    }

    #[test]
    fn heaviside_zero() {
        assert_eq!((0.0).heaviside(), 1i8);
    }

    #[test]
    fn heaviside_negative() {
        assert_eq!((-0.5).heaviside(), 0i8);
    }
}
