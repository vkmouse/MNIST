#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

int main(int argc, char *argv[])
{
    Mat image, output1, output2, blob;
    image = imread("../../dataset/trainingSample/img_0.jpg", 0);
    Size siz;

    blob = blobFromImage(image, 1 / 255., Size(28, 28), true, false);

    // autoencoder
    Net autoencoder = readNetFromTensorflow("../frozen_models/autoencoder_frozen_graph.pb"); // load model
    autoencoder.setInput(blob);
    output1 = autoencoder.forward();
    siz = Size(output1.size[2], output1.size[3]); // retrieve the calculated channels from the network output
    imshow("output1", Mat(siz, CV_32F, output1.ptr(0,0)));

    // encoder and decoder
    Net encoder = readNetFromTensorflow("../frozen_models/encoder_frozen_graph.pb");
    Net decoder = readNetFromTensorflow("../frozen_models/decoder_frozen_graph.pb");
    encoder.setInput(blob);
    output2 = encoder.forward();
    decoder.setInput(output2);
    output2 = decoder.forward();
    siz = Size(output1.size[2], output1.size[3]);
    imshow("output2", Mat(siz, CV_32F, output2.ptr(0,0)));

    waitKey(0);        
    return 0;
}