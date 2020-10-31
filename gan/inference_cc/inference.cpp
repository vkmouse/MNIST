#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "Model.hpp"
#include "Tensor.hpp"

void discriminator_inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims);
void generator_inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims);
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

    discriminator_inference(input_data, output_data, std::vector<int64_t>{2, 28, 28, 1}, std::vector<int64_t>{2, 1});
    for (int i = 0; i < 2; i++)
        std::cout << output_data[i] << std::endl;

    float input_noise[100] = {-1.1750287 ,  0.49099255, -0.31633526,  0.95809805, -0.6055076 ,
                              1.0484773 , -0.6927337 , -0.03917084,  0.56488544,  1.4702189 ,
                              -0.9108627 ,  0.4155678 , -0.8810917 ,  1.7961727 ,  0.8604621 ,
                              -1.1106812 , -1.3945537 ,  0.08221883,  1.9019601 ,  0.46212977,
                              -0.45804057,  1.2004454 ,  1.4012384 ,  2.0886617 ,  1.3290839 ,
                              0.43453994,  0.24767497,  1.1715617 , -0.8914417 , -1.1827073 ,
                              0.12964755,  1.4309487 ,  1.2879149 ,  0.6157883 , -0.02764123,
                              0.08359215, -0.48970005,  1.2101887 , -1.4032831 ,  2.1387758 ,
                              0.5053521 , -0.11166418, -1.7307677 ,  0.01222531,  0.79674184,
                              0.74678564, -2.4236257 ,  1.1947001 , -1.5248281 , -1.5106766 ,
                              -0.20968305,  0.29775593, -0.5845091 ,  1.3973504 , -0.5588903 ,
                              1.5450251 ,  1.6855044 , -0.7556438 ,  0.69636   , -0.5613151 ,
                              1.6601524 , -0.41424787, -0.85646427, -1.0800304 , -0.07682204,
                              -0.9295773 , -0.55298096,  1.2977306 , -1.8152978 , -0.2658583 ,
                              -0.26408264,  0.6460863 ,  1.5187714 ,  0.9386747 ,  1.6675855 ,
                              -2.044065  ,  1.1745359 ,  0.3375812 , -0.9978023 ,  0.11127155,
                              0.6886519 ,  0.839662  , -0.5648673 ,  1.1908921 , -0.19822288,
                              0.8706893 , -1.5740086 ,  0.00917329, -0.40818763, -0.98421335,
                              0.00352572, -1.8546008 , -0.61477256, -0.85459316,  0.24559681,
                              -1.2835362 , -0.07178366, -0.7122301 , -0.8220526 , -0.46505198};
    generator_inference(input_noise, output_data, std::vector<int64_t>{1, 100}, std::vector<int64_t>{1, 28, 28, 1});
    cv::Mat generate_image(28, 28, CV_32FC1, output_data);
    generate_image *= 255;
    generate_image.convertTo(generate_image, CV_8UC1);

    cv::imshow("image", image);
    cv::imshow("image1", image1);
    cv::imshow("generate_image", generate_image);
    cv::waitKey(0);

    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            input_noise[j] = rand() / (double)RAND_MAX;
        }
        generator_inference(input_noise, output_data, std::vector<int64_t>{1, 100}, std::vector<int64_t>{1, 28, 28, 1});
        cv::Mat generate_image(28, 28, CV_32FC1, output_data);
        generate_image *= 255;
        generate_image.convertTo(generate_image, CV_8UC1);

        cv::imshow("generate_image", generate_image);
        cv::waitKey(0);
    }

    return 0;
}
void discriminator_inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims)
{
    Model model("discriminator/");
    Tensor input(model, "serving_default_conv2d_input");
    Tensor output(model, "StatefulPartitionedCall");
    input.set_data(input_data, input_dims);
    output.set_shape(output_dims);
    
    model.run(input, output);

    output_data = output.get_data();
}
void generator_inference(float* input_data, float* &output_data, std::vector<int64_t> input_dims, std::vector<int64_t> output_dims)
{
    Model model("generator/");
    Tensor input(model, "serving_default_dense_input");
    Tensor output(model, "StatefulPartitionedCall");
    input.set_data(input_data, input_dims);
    output.set_shape(output_dims);
    
    model.run(input, output);

    output_data = output.get_data();
}
