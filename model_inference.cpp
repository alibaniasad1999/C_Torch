// model_inference.cpp

#include "model_inference.h"
#include <iostream>
#include <memory>

// Singleton pattern to ensure the model is loaded only once
class ModelHandler {
public:
    // Deleted methods to prevent copying
    ModelHandler(const ModelHandler&) = delete;
    ModelHandler& operator=(const ModelHandler&) = delete;

    // Static method to access the single instance
    static ModelHandler& getInstance() {
        static ModelHandler instance;
        return instance;
    }

    // Method to perform inference and return ax and ay
    std::pair<float, float> infer(const torch::Tensor& input) {
        // Prepare inputs
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input);

        // Execute the model
        torch::jit::IValue output;
        try {
            output = module.forward(inputs);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            throw std::runtime_error("Model inference failed.");
        }

        // Process the output
        if (output.isTuple()) {
            std::cout << 'here tuople';
            auto output_tuple = output.toTuple();
            auto elements = output_tuple->elements();
            if (elements.size() >= 2) {
                std::cout << 'here';
                float ax = elements[0].toTensor().item<float>();
                float ay = elements[1].toTensor().item<float>();
                return std::make_pair(ax, ay);
            }
            else {
                std::cerr << "Output tuple has fewer than 2 elements." << std::endl;
                throw std::runtime_error("Invalid model output.");
            }
        }
        else if (output.isTensor()) {
            torch::Tensor output_tensor = output.toTensor();
            if (output_tensor.dim() == 1 && output_tensor.size(0) >= 2) {
                float ax = output_tensor[0].item<float>();
                float ay = output_tensor[1].item<float>();
                return std::make_pair(ax, ay);
            }
            else {
                std::cerr << "Output tensor does not have the expected shape [2]." << std::endl;
                throw std::runtime_error("Invalid tensor shape.");
            }
        }
        else {
            std::cerr << "Unexpected output type from the model." << std::endl;
            throw std::runtime_error("Invalid model output type.");
        }
    }

private:
    torch::jit::script::Module module;

    // Private constructor to load the model
    ModelHandler() {
        try {
            module = torch::jit::load("model/pi_model_traced.pt");
            std::cout << "Model loaded successfully." << std::endl;
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            throw std::runtime_error("Model loading failed.");
        }
    }
};

// Definition of the get_ax_ay function
std::pair<float, float> get_ax_ay(const torch::Tensor& input) {
    // Access the singleton instance and perform inference
    return ModelHandler::getInstance().infer(input);
}
