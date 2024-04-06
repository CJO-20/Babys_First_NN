//
//  neuron.hpp
//  NN
//
//  Created by Chris on 4/22/23.
//

#ifndef neuron_hpp
#define neuron_hpp

#include <inttypes.h>
#include <array>
#include <functional>

namespace nn
{

struct input_neuron
{
protected:
    double activation = 0.0; //Current activation value - calculated during forward propagation
    
public:
    inline double get_activation (void)
    {
        return activation;
    }
    inline double get_net_input (void)
    {
        return activation;
    }
    inline void activate (double net_input)
    {
        activation = net_input;
    }
};

//@Struct: func_neuron
//@Description: Neuron used for hidden layers, and output
struct func_neuron
{
protected:
    double net_input = 0.0; //Net input for this neuron
    double activation = 0.0; //Net output for this neuron
    double activation_prime = 0.0; //Instantaneous derivative of the net output of this neuron with respect to the net input
    double d_error_d_net_input = 0.0; //Derivative of the total error (cost) of nn with respect to the net input of this neuron - calculated during back propagation
    //d_error_d_net_input = d_error_d_activation * activation_prime for func_neurons
    
    
    
    //Activation function, and its derivative function with respect to the net input of the neuron
    double(* act_func)(double);
    double(* act_func_prime)(double);

public:
    func_neuron (double(* f)(double), double(* df_dnet_input)(double)) : act_func(f), act_func_prime(df_dnet_input) {}
    
    inline void activate (double net_input)
    {
        this->net_input = net_input;
        activation = act_func(net_input);
        activation_prime = act_func_prime(net_input);
    }
    inline double get_activation (void)
    {
        return activation;
    }
    inline double get_net_input (void)
    {
        return net_input;
    }
    inline double get_activation_prime (void)
    {
        return activation_prime;
    }
    inline void calculate_d_error_d_net_input (double d_error_d_activation)
    {
        d_error_d_net_input = d_error_d_activation * activation_prime;
    }
    inline double get_d_error_d_net_input (void)
    {
        return d_error_d_net_input;
    }
};

/*
template <size_t N>
struct layer
{
protected:
    std::array<neuron, N> neurons;
};
*/


inline double relu (double x)
{
    return (x > 0.0) ? x : 0.0;
}

inline double relu_prime (double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

template <double a>
inline double parametric_relu (double x)
{
    return (x > 0.0) ? x : a * x;
}

template <double a>
inline double parametric_relu_prime (double x)
{
    return (x > 0.0) ? 1.0 : a;
}

inline double sigmoid (double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_prime (double x)
{
    x = sigmoid(x);
    return x * (1.0 - x);
}

inline double softplus (double x)
{
    return std::log(1.0 + std::exp(x));
}

inline double softplus_prime (double x)
{
    return sigmoid(x);
}

inline double tanh_prime (double x)
{
    x = std::tanh(x);
    return 1.0 - x * x;
}

inline double normalized_tanh (double x)
{
    return (std::tanh(x) + 1.0) / 2.0;
}

inline double normalized_tanh_prime (double x)
{
    x = std::tanh(x);
    return 0.5 * (1 - x * x);
}

}

#endif /* neuron_hpp */
