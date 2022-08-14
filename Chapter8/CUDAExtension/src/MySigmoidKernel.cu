#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

#define THREADS 1024

template <typename scalar_t>
__global__ void sigmoid_cuda_forward_kernel(scalar_t* x, scalar_t* fx, const int state_size) {
    const uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < state_size){
        // f(x)=e^-x/1+e^-x
        fx[index] = expf(-x[index]) / (1. + expf(-x[index]));
    }
}

template <typename scalar_t>
__global__ void sigmoid_cuda_backward_kernel(scalar_t* fx, scalar_t* grad_fx, scalar_t* grad_x, const int state_size) {
    const uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < state_size){
        // f'(x)=f(x)(f(x)-1)
        grad_x[index] = fx[index] * (fx[index] - 1) * grad_fx[index];
    }
}

__host__ at::Tensor sigmoid_cuda_forward(at::Tensor& x) {
    auto fx = x.clone();
    const int state_size = x.numel();
    const int nblocks = (state_size + THREADS - 1) / THREADS;
    AT_DISPATCH_FLOATING_TYPES(x.type(), "sigmoid_forward_cuda", ([&] {
        sigmoid_cuda_forward_kernel<scalar_t><<<nblocks, THREADS>>>(
            x.data<scalar_t>(),
            fx.data<scalar_t>(),
            state_size);
  }));

  return fx;
}


__host__ at::Tensor sigmoid_cuda_backward(at::Tensor& fx, at::Tensor& grad_fx) {
    auto grad_x = grad_fx.clone();
    const int state_size = fx.numel();
    int nblocks = (state_size + THREADS - 1) / THREADS;
    AT_DISPATCH_FLOATING_TYPES(grad_fx.type(), "sigmoid_backward_cuda", ([&] {
        sigmoid_cuda_backward_kernel<scalar_t><<<nblocks, THREADS>>>(
            fx.data<scalar_t>(),
            grad_fx.data<scalar_t>(),
            grad_x.data<scalar_t>(),
            state_size);
  }));

  return grad_x;
}