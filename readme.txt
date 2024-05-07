trait:          neuron
Author:         Neil Crago <n.j.crago@gmail.com>
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
