#include <fstream>
#include <torch/script.h>
#include <iostream>
#include <memory>
using namespace std::chrono;

int main() {
    // Check if the model file exists
    std::ifstream model_file("model/pi_model_traced.pt");
    if (!model_file) {
        std::cerr << "Model file not found: model/pi_model_traced.pt\n";
        return -1;
    }
    auto start = high_resolution_clock::now();
    // Load the TorchScript model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model/pi_model_traced.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }

    std::cout << "Model loaded successfully.\n";

    // Prepare the input tensor (size 4)
    at::Tensor input = torch::randn({1, 4});  // Shape (1, 4)
    std::cout << "Input Tensor: " << input << "\n";

    // Perform inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Execute the model
    at::Tensor output_tensor;
    try {
        torch::jit::IValue output = module.forward(inputs);

        if (output.isTensor()) {
            // If the output is a single tensor
            output_tensor = output.toTensor();
        }
        else if (output.isTuple()) {
            // If the output is a tuple, extract the desired tensor
            auto output_tuple = output.toTuple();
            // For example, get the first element of the tuple
            // Adjust the index based on your model's actual output
            if (output_tuple->elements().size() > 0) {
                output_tensor = output_tuple->elements()[0].toTensor();
            } else {
                std::cerr << "Output tuple is empty.\n";
                return -1;
            }
        }
        else {
            std::cerr << "Unexpected output type.\n";
            return -1;
        }

        std::cout << "Inference completed successfully.\n";
        std::cout << "Output Tensor: " << output_tensor << "\n";

        // Convert the tensor to float(s)
        // Handle different tensor dimensions

        // Case 1: Scalar Tensor
        if (output_tensor.dim() == 0) {
            float output_float = output_tensor.item<float>();
            std::cout << "Output (Scalar): " << output_float << "\n";
        }
        // Case 2: 1D Tensor (Vector)
        else if (output_tensor.dim() == 1) {
            std::cout << "Output (1D Tensor): ";
            // Iterate through the tensor elements
            for (int64_t i = 0; i < output_tensor.size(0); ++i) {
                float value = output_tensor[i].item<float>();
                std::cout << value << " ";
            }
            std::cout << "\n";
        }
        // Case 3: 2D Tensor (Matrix)
        else if (output_tensor.dim() == 2) {
            std::cout << "Output (2D Tensor):\n";
            for (int64_t i = 0; i < output_tensor.size(0); ++i) {
                for (int64_t j = 0; j < output_tensor.size(1); ++j) {
                    float value = output_tensor[i][j].item<float>();
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
        }
        // Add more cases as needed for higher dimensions
        else {
            std::cout << "Output has " << output_tensor.dim() << " dimensions. Conversion not implemented.\n";
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error during inference: " << e.what() << "\n";
        return -1;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;

    return 0;
}
