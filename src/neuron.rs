/*
trait:          neuron
Author:         Neil Crago
start date:     13/02/24

Description:

To create a trait that models a neuron better than a standard perceptron.

-   A trait is a collection of methods that can be implemented by different types in
    Rust.

-   A neuron is a biological cell that can process and transmit information in
    the brain.

-   A perceptron is a mathematical model of a neuron that can perform binary
    classification.

Methods:

- `new(inputs: usize, activation: fn(f64) -> f64) -> Self`: This method would create
   a new neuron with a given number of inputs and an activation function.
   The activation function could be any function that maps a real number to another
   real number, such as the sigmoid, tanh, ReLU, or identity function.

- `set_weights(&mut self, weights: Vec<f64>)`: This method would set the weights of
   the neuron to a given vector of real numbers. The length of the vector should
   match the number of inputs of the neuron.

- `set_bias(&mut self, bias: f64)`: This method would set the bias of the neuron to
   a given real number. The bias is a constant value that is added to the weighted
   sum of the inputs before applying the activation function.

- `output(&self, inputs: Vec<f64>) -> f64`: This method would compute the output of
   the neuron given a vector of inputs. The output is the result of applying the
   activation function to the weighted sum of the inputs plus the bias.
*/

use rand::{thread_rng, Rng};
use std::f64::consts::E;
use std::sync::mpsc;
use std::thread;

#[derive(Clone, Debug)]
struct CodeStruct {
    a: i32,
    h: char,
    b: i32,
    g: char,
    c: i32,
    i: char,
}

pub fn neuron_chat() {
    
    let (tx, rx) = mpsc::channel();
    let mut rng = thread_rng();
    let code = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let alph: [char; 10] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];

    let mut cv: Vec<CodeStruct> = Vec::new();

    for _y in 0..21 {
        let a = rng.gen_range(0..10);
        let g = rng.gen_range(0..10);
        let b = rng.gen_range(0..10);
        let h = rng.gen_range(0..10);
        let c = rng.gen_range(0..10);
        let i = rng.gen_range(0..10);

        let cvs = CodeStruct {
            a: code[a],
            g: alph[g],
            b: code[b],
            h: alph[h],
            c: code[c],
            i: alph[i],
        };

        cv.push(cvs);
    }

    thread::spawn(move || {
        for cval in cv {
            tx.send(cval).unwrap();
        }
    });

    println!();

    for recv in rx {
        let a = recv.a;
        let b = recv.b;
        let c = recv.c;

        let g = recv.g;
        let h = recv.h;
        let i = recv.i;
        println!("Code RCV: {a:0>2}{g}{b:0>2}{h}{c:0>2}{i}");
    }
}

// define the Neuron trait
pub(crate) trait Neuron {
    
    fn new(inputs: usize, activation: fn(f64) -> f64) -> Self;
    fn set_weights(&mut self, weights: Vec<f64>);
    fn set_bias(&mut self, bias: f64);
    fn sigmoid(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
    fn smooth_relu(x: f64) -> f64;
    fn output(&self, inputs: Vec<f64>) -> f64;
}

pub struct SuperNeuron {
    
    inputs: usize,              // the number of inputs
    weights: Vec<f64>,          // the vector of weights
    bias: f64,                  // the bias
    activation: fn(f64) -> f64, // the activation function
}

// implement the Neuron trait for the SuperNeuron struct
impl Neuron for SuperNeuron {
    
    // create a new SuperNeuron with the given inputs and activation function
    fn new(inputs: usize, activation: fn(f64) -> f64) -> Self {
    
        // initialize the weights and the bias to zero
        let weights = vec![0.0; inputs];
        let bias = 0.0;
        
        // return the SuperNeuron
        SuperNeuron {
            inputs,
            weights,
            bias,
            activation,
        }
    }

    // set the weights of the SuperNeuron to the given vector
    fn set_weights(&mut self, weights: Vec<f64>) {
        
        // check if the length of the vector matches the number of inputs
        assert_eq!(weights.len(), self.inputs);
        
        // assign the weights to the SuperNeuron
        self.weights = weights;
    }

    // set the bias of the SuperNeuron to the given value
    fn set_bias(&mut self, bias: f64) {
        
        // assign the bias to the SuperNeuron
        self.bias = bias;
    }

    // sigmoid activation (Logistic Function)
    fn sigmoid(x: f64) -> f64 {
        1f64 / (1f64 + E.powf(-x))
    }

    // tanh activation
    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    fn smooth_relu(x: f64) -> f64 {
        (1f64 + E.powf(x)).ln()
    }

    // compute the output of the SuperNeuron given a vector of inputs
    fn output(&self, inputs: Vec<f64>) -> f64 {
        
        // check if the length of the vector matches the number of inputs
        assert_eq!(inputs.len(), self.inputs);
        
        // compute the weighted sum of the inputs
        let mut sum = 0.0;
        
        //for i in 0..self.inputs {
        for (i, inpiter) in inputs.iter().enumerate().take(self.inputs) {
            sum += self.weights[i] * inpiter;
        }
        
        // add the bias
        sum += self.bias;
        
        // apply the activation function
        (self.activation)(sum)
    }
}

/*
This trait could model a neuron better than a standard perceptron because it
allows more flexibility in choosing:-
    the activation function,
    the number of inputs,
    the values of the weights
    the bias.

A standard perceptron has
    a fixed activation function (the sign function),
    a fixed number of inputs (two),
    and a fixed learning rule (the perceptron algorithm) to update the weights and
    the bias.
*/
