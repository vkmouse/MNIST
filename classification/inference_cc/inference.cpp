#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "Model.hpp"
#include "Tensor.hpp"

void inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims);
//https://gist.github.com/asimshankar/7c9f8a9b04323e93bb217109da8c7ad2
//https://github.com/serizba/cppflow

int main() 
{
    cv::Mat image = cv::imread("..\\..\\dataset\\trainingSample\\img_1.jpg", 0);
    cv::Mat image1 = cv::imread("..\\..\\dataset\\trainingSample\\img_10.jpg", 0);
    image.convertTo(image, CV_32FC1);
    image1.convertTo(image1, CV_32FC1);
    image /= 255.;
    image1 /= 255.;

    float* input_data = new float[2 * 28 * 28 * 1];
    float* output_data;
    memcpy(&input_data[0], image.data, sizeof(float) * 28 * 28);
    memcpy(&input_data[28 * 28], image1.data, sizeof(float) * 28 * 28);

    inference(input_data, output_data, std::vector<int64_t>{2, 28, 28, 1}, std::vector<int64_t>{2, 10});

    
    for (int i = 0; i < 20; i++)
        std::cout << output_data[i] << std::endl;

    cv::imshow("image", image);
    cv::imshow("image1", image1);
    cv::waitKey(0);
    return 0;
}
void inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims)
{
    Model model("classification/");
    Tensor input(model, "serving_default_input_1");
    Tensor output(model, "StatefulPartitionedCall");
    input.set_data(input_data, input_dims);
    output.set_shape(output_dims);
    
    model.run(input, output);

    output_data = output.get_data();
}
