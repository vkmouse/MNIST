#include "Tensor.hpp"
void NoOpDeallocatoraaa(void* data, size_t a, void* b) {}

Tensor::Tensor(Model& model, const char* operation)
{
    op = {TF_GraphOperationByName(model.graph, operation), 0};
    flag = false;
}
Tensor::~Tensor()
{
    if (flag)
    {}
}
void Tensor::set_data(float* data, std::vector<int64_t> shape)
{
    int num_bytes = sizeof(float);
    for (auto value : shape)
        num_bytes *= value;
    val = TF_NewTensor(TF_FLOAT, &shape[0], shape.size(), data, num_bytes, &NoOpDeallocatoraaa, 0);
    flag = true;
}
void Tensor::set_shape(std::vector<int64_t> shape)
{
    int num_bytes = sizeof(float);
    for (auto value : shape)
        num_bytes *= value;
    val = TF_AllocateTensor(TF_FLOAT, &shape[0], shape.size(), num_bytes);
    flag = true;
}
float* Tensor::get_data()
{
    return (float*)TF_TensorData(val);
}