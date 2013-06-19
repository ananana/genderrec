#include "fisherfacerec.h"
#include "facerec.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

FisherFaceRec::FisherFaceRec() {
    model = createFisherFaceRecognizer();
}

FisherFaceRec::FisherFaceRec(const char* modelfile) {
    model = createFisherFaceRecognizer();
    load(modelfile);
    mean = model->getMat("mean");
}

void FisherFaceRec::load(const char* modelfile) {
    //TODO: do we need this? add it to the others as well
    model = createFisherFaceRecognizer();
    model->load(modelfile);
    mean = model->getMat("mean");
}

void FisherFaceRec::save(const char* modelfile) {
    model->save(modelfile);
}

void FisherFaceRec::train(vector<Mat>& images, vector<int>& labels) {
    model = createFisherFaceRecognizer(); // do we need this here as well?
    model->train(images, labels);
    mean = model->getMat("mean");
}

void FisherFaceRec::train() {
    train(this->images, this->labels);

    if (output) {
    // get first fisherface (only one)
        Mat W = model->getMat("eigenvectors");
        Mat ev = W.col(0).clone();
        Mat grayscale = norm_0_255(ev.reshape(1, images[0].rows));
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        imwrite(format("%s/fisherface.png", output_folder.c_str()), norm_0_255(cgrayscale));
    }
}

void FisherFaceRec::predict(Mat& testSample, int& predictedLabel, double& confidence) {
    // the test image needs to be the same size as the training images - if it's not, resize it
    if (testSample.size() != Size(WIDTH, HEIGHT)) {
      //  cerr<<"Resizing: "<<testSample.size().width<<" "<<testSample.size().height<<" -> "<<WIDTH<<" "<<HEIGHT;
        resize(testSample, testSample, Size(WIDTH, HEIGHT));
    }
    model->predict(testSample, predictedLabel, confidence);
}