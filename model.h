// model.h
#pragma once

#include <torch/torch.h>

// Define the neural network architecture
struct SimpleNet : torch::nn::Module {
    // Define two linear layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    // Constructor to initialize layers
    SimpleNet(int input_size, int hidden_size, int output_size) {
        // Initialize the first linear layer (input to hidden)
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));

        // Initialize the second linear layer (hidden to output)
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    }

    // Define the forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x)); // Apply ReLU activation after first layer
        x = fc2->forward(x);               // Output layer
        return x;
    }
};
