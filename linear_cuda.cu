#include <torch/extension.h>
torch::Tensor linear_forward_cuda(
 torch::Tensor X,
 torch::Tensor W,
 torch::Tensor b
);
std::vector<torch::Tensor> linear_backward_cuda(
 torch::Tensor G,
 torch::Tensor X,
 torch::Tensor W
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("forward", &linear_forward_cuda);
 m.def("backward", &linear_backward_cuda);
}