// model_inference.h

#ifndef MODEL_INFERENCE_H
#define MODEL_INFERENCE_H

#include <torch/script.h> // One-stop header for loading TorchScript models
#include <utility>        // For std::pair

/**
 * @brief Performs inference using the pre-loaded TorchScript model and returns ax and ay.
 *
 * @param input A Torch tensor of shape [1, 4] representing the input to the model.
 * @return std::pair<float, float> A pair containing ax and ay.
 *
 * @throws std::runtime_error If there are issues with model loading or inference.
 */
std::pair<float, float> get_ax_ay(const torch::Tensor& input);

#endif // MODEL_INFERENCE_H
