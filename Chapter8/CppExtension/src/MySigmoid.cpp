#include "MySigmoid.h" // 导入头文件
#include <math.h>

// 前向传播函数，注意Tensor的shape要一致
at::Tensor MySigmoidForward(const at::Tensor& x){
    at::Tensor fx = at::zeros(x.sizes());
    for(int i=0; i < fx.size(0); i++){
        for(int j=0; j < fx.size(1); j++){
            fx[i][j] = exp(-x[i][j]) / (1 - exp(-x[i][j]));
        }
    }
//    fx = exp(-x) / (1 - exp(-x)); // 非for循环形式，直接替换上面for循环即可
    return fx;
}

// 反向传播函数，grad_out是上一级梯度回传的值，根据链式法则需要乘上去
at::Tensor MySigmoidBackward(const at::Tensor& fx, const at::Tensor& grad_out){
    at::Tensor grad_x = at::ones(grad_out.sizes());
    for(int i=0; i < grad_x.size(0); i++){
        for(int j=0; j < grad_x.size(1); j++){
            grad_x[i][j] =  -fx[i][j] * (fx[i][j] + 1) * grad_out[i][j];
        }
    }
//    非for循环形式，直接替换上面for循环即可
//    grad_x = -fx * (fx + 1) * grad_out * at::ones(grad_out.sizes());
    return grad_x;
}

// 绑定C++函数到python上，TORCH_EXTENSION_NAME即为setup.py中setup定义的name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &MySigmoidForward,  "mysigmoid forward" );
  m.def("backward", &MySigmoidBackward, "mysigmoid backward");
}