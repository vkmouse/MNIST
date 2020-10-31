#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <vector>
#include <tensorflow/c/c_api.h>
#include "Model.hpp"

class Model;

class Tensor
{
friend class Model;
public:
    Tensor(Model& model, const char* operation);
    ~Tensor();
    void set_data(float* data, std::vector<int64_t> shape);
    void set_shape(std::vector<int64_t> shape);
    float* get_data();
private:
    TF_Output op;
    TF_Tensor* val;
    std::vector<int64_t> shape;
    bool flag;
};
#endif