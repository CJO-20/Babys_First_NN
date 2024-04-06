//
//  main.cpp
//  NN
//
//  Created by Chris on 4/22/23.
//

#include <iostream>
#include <iomanip>
#include <functional>
#include <thread>

#include "neuron.hpp"

/*
    XOR Table
 A      B      Out
 0      0       0
 0      1       1
 1      0       1
 1      1       0
 
 Current best:
 
 Input Layer to Hidden Layer Weight Matrix
 weights[0][0]: 5.07362
 weights[0][1]: -5.24759
 weights[1][0]: -5.24845
 weights[1][1]: 5.07265
 weights[2][0]: -2.52908
 weights[2][1]: -2.52857
 {5.07362, -5.24759, -5.24845, 5.07265, -2.52908, -2.52857}
 Hidden Layer to Output Layer Weight Matrix
 weights[0][0]: 2.66614
 weights[1][0]: 2.66615
 weights[2][0]: 2.5659
 {2.66614, 2.66615, 2.5659}
 For x0 = 0 and x1 = 0, the expected output is 0. Actual activation is: 0.00450543
 For x0 = 0 and x1 = 1, the expected output is 1. Actual activation is: 0.993735
 For x0 = 1 and x1 = 0, the expected output is 1. Actual activation is: 0.993735
 For x0 = 1 and x1 = 1, the expected output is 0. Actual activation is: 0.00433121
 
 */
/*
 AND Table
A      B      Out
0      0       0
0      1       0
1      0       0
1      1       1
 
 Input Layer to Hidden Layer Weight Matrix
 weights[0][0]: 3.86781
 weights[0][1]: 3.79588
 weights[1][0]: 3.79964
 weights[1][1]: 3.84321
 weights[2][0]: -5.85006
 weights[2][1]: -5.82895
 Hidden Layer to Output Layer Weight Matrix
 weights[0][0]: 3.17694
 weights[1][0]: 3.15878
 weights[2][0]: -3.17509
 For x0 = 0 and x1 = 0, the expected output is 0. Actual activation is: 0.040828
 For x0 = 0 and x1 = 1, the expected output is 0. Actual activation is: 0.080795
 For x0 = 1 and x1 = 0, the expected output is 0. Actual activation is: 0.081307
 For x0 = 1 and x1 = 1, the expected output is 1. Actual activation is: 0.906571
 */

/*
 BASIC NN DESIGN
 
 I: Input Neuron
 H: Hidden Neuron
 O: Output Neuron
 
 [I]---[H]
    \ /   \
     *    [O]
    / \   /
 [I]---[H]
 
 
 */
struct basic_neural_network
{
    const double learning_rate = 0.25;
    
    std::array<nn::input_neuron, 2> input_layer = {};
    std::array<nn::func_neuron, 2> hidden_layer = {nn::func_neuron(std::tanh, nn::tanh_prime), nn::func_neuron(std::tanh, nn::tanh_prime)};
    std::array<nn::func_neuron, 1> output_layer = {nn::func_neuron(nn::normalized_tanh, nn::normalized_tanh_prime)};
    
    //R = row; C = col
    //                            C   R
    std::array<std::array<double, 2>, 3> weights_input_to_hidden = {1, -1, -1, 1, 0, 0};
    std::array<std::array<double, 1>, 3> weights_hidden_to_output = {1, 1, 0};
    
    
    void forward_propagate (double x0, double x1)
    {
        input_layer[0].activate(x0);
        input_layer[1].activate(x1);
        
        hidden_layer[0].activate(input_layer[0].get_activation() * weights_input_to_hidden[0][0] + input_layer[1].get_activation() * weights_input_to_hidden[1][0] + weights_input_to_hidden[2][0]);
        hidden_layer[1].activate(input_layer[0].get_activation() * weights_input_to_hidden[0][1] + input_layer[1].get_activation() * weights_input_to_hidden[1][1] + weights_input_to_hidden[2][1]);
        
        output_layer[0].activate(hidden_layer[0].get_activation() * weights_hidden_to_output[0][0] + hidden_layer[1].get_activation() * weights_hidden_to_output[1][0] + weights_hidden_to_output[2][0]);
    }
    
    double back_propagate (double expected_output)
    {
        double d_error_d_out = output_layer[0].get_activation() - expected_output;
        
        //Calculate delta error with respect to delta input for each func node
        output_layer[0].calculate_d_error_d_net_input(d_error_d_out);
        hidden_layer[0].calculate_d_error_d_net_input(d_error_d_out * weights_hidden_to_output[0][0]);
        hidden_layer[1].calculate_d_error_d_net_input(d_error_d_out * weights_hidden_to_output[1][0]);
        
        //Adjust weights afterwards
        weights_hidden_to_output[0][0] -= learning_rate * output_layer[0].get_d_error_d_net_input() * hidden_layer[0].get_activation();
        weights_hidden_to_output[1][0] -= learning_rate * output_layer[0].get_d_error_d_net_input() * hidden_layer[1].get_activation();
        weights_hidden_to_output[2][0] -= learning_rate * output_layer[0].get_d_error_d_net_input();
        
        weights_input_to_hidden[0][0] -= learning_rate * hidden_layer[0].get_d_error_d_net_input() * input_layer[0].get_activation();
        weights_input_to_hidden[0][1] -= learning_rate * hidden_layer[1].get_d_error_d_net_input() * input_layer[0].get_activation();
        weights_input_to_hidden[1][0] -= learning_rate * hidden_layer[0].get_d_error_d_net_input() * input_layer[1].get_activation();
        weights_input_to_hidden[1][1] -= learning_rate * hidden_layer[1].get_d_error_d_net_input() * input_layer[1].get_activation();
        weights_input_to_hidden[2][0] -= learning_rate * hidden_layer[0].get_d_error_d_net_input();
        weights_input_to_hidden[2][1] -= learning_rate * hidden_layer[1].get_d_error_d_net_input();
        
        return 0.5 * d_error_d_out * d_error_d_out;
    }
    
    double learn (double x0, double x1, double expected_output)
    {
        forward_propagate(x0, x1);
        return back_propagate(expected_output);
    }
};


int main (int argc, const char * argv[])
{
    basic_neural_network bnn;
    
    for (size_t i = 0; i < 100; i++)
    {
        //Learning XOR Table, other logic gates with 2 input, 1 output can be trained as well
        std::cout << "Current cost: " << bnn.learn(0, 0, 0) << '\n';
        std::cout << "Current cost: " << bnn.learn(0, 1, 1) << '\n';
        std::cout << "Current cost: " << bnn.learn(1, 0, 1) << '\n';
        std::cout << "Current cost: " << bnn.learn(1, 1, 0) << '\n';
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::cout << "Input Layer to Hidden Layer Weight Matrix\n";
    std::cout << "weights[0][0]: " << bnn.weights_input_to_hidden[0][0] << '\n';
    std::cout << "weights[0][1]: " << bnn.weights_input_to_hidden[0][1] << '\n';
    std::cout << "weights[1][0]: " << bnn.weights_input_to_hidden[1][0] << '\n';
    std::cout << "weights[1][1]: " << bnn.weights_input_to_hidden[1][1] << '\n';
    std::cout << "weights[2][0]: " << bnn.weights_input_to_hidden[2][0] << '\n';
    std::cout << "weights[2][1]: " << bnn.weights_input_to_hidden[2][1] << '\n';
    std::cout << '{';
    std::cout << bnn.weights_input_to_hidden[0][0] << ", ";
    std::cout << bnn.weights_input_to_hidden[0][1] << ", ";
    std::cout << bnn.weights_input_to_hidden[1][0] << ", ";
    std::cout << bnn.weights_input_to_hidden[1][1] << ", ";
    std::cout << bnn.weights_input_to_hidden[2][0] << ", ";
    std::cout << bnn.weights_input_to_hidden[2][1] << "}\n";
    
    std::cout << "Hidden Layer to Output Layer Weight Matrix\n";
    std::cout << "weights[0][0]: " << bnn.weights_hidden_to_output[0][0] << '\n';
    std::cout << "weights[1][0]: " << bnn.weights_hidden_to_output[1][0] << '\n';
    std::cout << "weights[2][0]: " << bnn.weights_hidden_to_output[2][0] << '\n';
    std::cout << '{';
    std::cout << bnn.weights_hidden_to_output[0][0] << ", ";
    std::cout << bnn.weights_hidden_to_output[1][0] << ", ";
    std::cout << bnn.weights_hidden_to_output[2][0] << "}\n";
    
    bnn.forward_propagate(0, 0);
    std::cout << "For x0 = 0 and x1 = 0, the expected output is " << std::round(bnn.output_layer[0].get_activation()) << ". Actual activation is: " << bnn.output_layer[0].get_activation() << '\n';
    bnn.forward_propagate(0, 1);
    std::cout << "For x0 = 0 and x1 = 1, the expected output is " << std::round(bnn.output_layer[0].get_activation()) << ". Actual activation is: " << bnn.output_layer[0].get_activation() << '\n';
    bnn.forward_propagate(1, 0);
    std::cout << "For x0 = 1 and x1 = 0, the expected output is " << std::round(bnn.output_layer[0].get_activation()) << ". Actual activation is: " << bnn.output_layer[0].get_activation() << '\n';
    bnn.forward_propagate(1, 1);
    std::cout << "For x0 = 1 and x1 = 1, the expected output is " << std::round(bnn.output_layer[0].get_activation()) << ". Actual activation is: " << bnn.output_layer[0].get_activation() << '\n';
    
    return 0;
}
