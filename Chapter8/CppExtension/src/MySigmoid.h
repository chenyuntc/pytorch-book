#include<torch/torch.h>
#include<vector>
#include<iostream>

// 前向传播 过程中用到了参数x
// f(x)=e^-x/1-e^-x
at::Tensor MySigmoidForward(const at::Tensor& x);
// 反向传播 过程中用到了回传参数和f(x)的结果
at::Tensor MySigmoidBackward(const at::Tensor& fx, const at::Tensor& grad_out);