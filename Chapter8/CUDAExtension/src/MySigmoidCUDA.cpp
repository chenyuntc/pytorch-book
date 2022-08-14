#include <torch/torch.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor sigmoid_cuda_forward(at::Tensor& x);
at::Tensor sigmoid_cuda_backward(at::Tensor& fx, at::Tensor& grad_out);

at::Tensor sigmoid_forward(at::Tensor& x){
    CHECK_INPUT(x);
    return sigmoid_cuda_forward(x);
}

at::Tensor sigmoid_backward(at::Tensor& fx, at::Tensor& grad_out){
    CHECK_INPUT(fx);
    CHECK_INPUT(grad_out);
    return sigmoid_cuda_backward(fx, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sigmoid_forward, "sigmoid forward(CUDA)");
  m.def("backward", &sigmoid_backward, "sigmoid backward(CUDA)");
}