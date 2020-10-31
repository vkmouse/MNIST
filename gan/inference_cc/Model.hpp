#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <vector>
#include <tensorflow/c/c_api.h>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include "Tensor.hpp"

class Tensor;

class Model
{
friend class Tensor;
public:
    Model(const char*);
    void run(Tensor input, Tensor& output);
    ~Model();
private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;
    static TF_Buffer* read_file(const char* file);
};
#endif