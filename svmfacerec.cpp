#include "svmfacerec.h"
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

SVMFaceRec::SVMFaceRec() {
    // Some params
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
}

SVMFaceRec::SVMFaceRec(const char* modelfile) {
	model.load(modelfile);
}


//TODO: find a way to be able to get the number of eigenvectors that a LOADED svm used, so you can predict with it
// it seems you need to save the eigenvectors? and then do something with them. see bookmarks

void SVMFaceRec::train(vector<Mat>& trainImages, vector<int>& trainLabels) {

    int height = trainImages[0].rows;
    int width = trainImages[0].cols;

    //cout<<"nr of trainImages: "<<trainImages.size()<<"; height: "<<height<<"; width: "<<width<<endl;
    //cout<<"test image: rows: "<<testImage.rows<<"; cols: "<<testImage.cols<<endl;

    nrEigens = trainImages.size() - 1; // nr of eigenvectors(faces)

    Mat data_mat(trainImages.size(), height*width, CV_32FC1);

    assert(trainImages.size() == data_mat.rows);
    for (int i = 0; i < trainImages.size(); i++) {
        Mat X = data_mat.row(i);
        // so apparently it has to be CV_32FC1 to work
        trainImages[i].reshape(1, 1).row(0).convertTo(X, CV_32FC1);
        // cout<<i<<": rows: "<<X.rows<<"; cols: "<<X.cols<<endl;
    }

    PCA reduce(data_mat, Mat(), CV_PCA_DATA_AS_ROW, nrEigens);
    pca = reduce;

    //cout<<"Showing mean image"<<endl;

    // imwrite("avg.png", pca.mean.reshape(1, height));
    // cout<<"eigenvectors: rows: "<<pca.eigenvectors.rows<<"; cols: "<<pca.eigenvectors.cols<<endl;
    // for (int i = 0; i < pca.eigenvectors.rows; i++) {
    //     // normalization is needed to so that this picture is visible with bare eyes
        
    //     imwrite(format("pc%d.png",i), norm_0_255(pca.eigenvectors.row(i)).reshape(1, height));
    // }

    Mat projected_mat = pca.project(data_mat);

    assert(projected_mat.rows == trainImages.size()); 
    assert(projected_mat.cols == nrEigens);

    //Mat back_projected = pca.backProject(projected_mat);

    // assert(back_projected.rows == data_mat.rows);
    // assert(back_projected.rows == trainImages.size());
    // assert(back_projected.cols == data_mat.cols);
    // assert(back_projected.cols == height*width);

    // for (int i = 0; i < back_projected.rows; i++) {
    //     // clone - needed if they're not continuous
    //     Mat X = back_projected.row(i).clone();
    //     // apparently this works with CV_32FC1 as well, but not if I comment out convertTo completely
    //     X.reshape(1, height).convertTo(X, CV_8UC1);
    //     // apparently it works without this: X = normlz(X);
        
    //     //imwrite(format("backproject%d.png",i), X);
    // }


    Mat labelsmat(trainLabels,true);
    //data_mat = data_mat.reshape(1,200);

    assert(labelsmat.cols == 1);
    assert(labelsmat.rows == trainImages.size());

    //check trainLabels vec
    for (int i = 0; i < trainLabels.size(); i++)
        assert(labelsmat.at<int>(i,0) == 0 || labelsmat.at<int>(i,0) == 1);

    assert(data_mat.rows == trainImages.size() && data_mat.cols == height*width);

    cout<<"Training..."<<endl;
    

    // ! changed this to projected_mat - for speed - that was the whole point of pca
    model.train_auto(projected_mat, labelsmat, Mat(), Mat(), params);
    // trainAuto - se pare ca face singur cross-validare

    // cout<<"Testing..."<<endl;
    // Mat testmat;
    // testImage.convertTo(testmat, CV_32FC1);
    // testmat = pca.project(testmat.reshape(1,1));
    // int label = SVM.predict(testmat);
    // return label;
}

void SVMFaceRec::train() {
    train(this->images, this->labels);
}

void SVMFaceRec::predict(Mat& image, int& label, double& confidence) {
    Mat testmat;
    image.convertTo(testmat, CV_32FC1);
    testmat = pca.project(testmat.reshape(1,1));
    label = model.predict(testmat);
}
