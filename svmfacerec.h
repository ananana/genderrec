#ifndef SVMFACEREC_H
#define SVMFACEREC_H

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

class SVMFaceRec : public FaceRec {

	private:
		CvSVM model;
		int nrEigens;
		cv::PCA pca;
        CvSVMParams params;

	public:
		SVMFaceRec();
		SVMFaceRec(const char*);
		// uses given set of images to train a new classifier
		void train(std::vector<cv::Mat>&, std::vector<int>&);
        void train();
        // not yet implemented
		void predict(cv::Mat&, int&, double&);
        
		void load(const char*) {;}
		void save(const char*) { ;}
};

#endif