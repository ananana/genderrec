#include "eigenfacerec.h"
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

EigenFaceRec::EigenFaceRec (int nrEigens = -1) {
    if (nrEigens >= 0)
        model = createEigenFaceRecognizer(nrEigens);
    else model = createEigenFaceRecognizer();
}

EigenFaceRec::EigenFaceRec (const char* modelfile, int nrEigens = -1) {
    if (nrEigens >= 0)
        model = createEigenFaceRecognizer(nrEigens);
    else
        model = createEigenFaceRecognizer();
    load(modelfile);
    mean = model->getMat("mean");
    nrEigens = -1;
}

void EigenFaceRec::setNrfaces (int nrFaces) {
    nrEigens = nrFaces;
}

void EigenFaceRec::load (const char* modelfile) {
    model->load(modelfile);
    mean = model->getMat("mean");
}

void EigenFaceRec::save (const char* modelfile) {
    model->save(modelfile);
}

void EigenFaceRec::train(vector<Mat>& images, vector<int>& labels) {
    //TODO: ! For some reason this doesn't work unless I recreate it after every train:
    if (nrEigens >= 0)
        model = createEigenFaceRecognizer(nrEigens);
    else
        model = createEigenFaceRecognizer();
    model->train(images, labels);
    mean = model->getMat("mean");
}

void EigenFaceRec::train() {
    train(this->images, this->labels);
}

//TODO: this sometimes throws an exception (image step wrong... vezi log 12)
// so maybe catch all exceptions...so it never crashes

void EigenFaceRec::predict(Mat& testSample, int& predictedLabel, double& confidence) {
    // the test image needs to be the same size as the training images - if it's not, resize it
    if (testSample.size() != Size(WIDTH, HEIGHT)) {
        //cerr<<"Resizing: "<<testSample.size().width<<" "<<testSample.size().height<<" -> "<<WIDTH<<" "<<HEIGHT;
        resize(testSample, testSample, Size(WIDTH, HEIGHT));
    }
    model->predict(testSample, predictedLabel, confidence);
}