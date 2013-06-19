#ifndef FISHERFACEREC_H
#define FISHERFACEREC_H

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

class FisherFaceRec : public FaceRec {

	private:
		cv::Ptr<cv::FaceRecognizer> model;
        public:cv::Mat mean;

	public:
		FisherFaceRec();
		FisherFaceRec(const char*);
		// uses given set of images to train a new classifier
		void train(std::vector<cv::Mat>&, std::vector<int>&);
        void train();
		void predict(cv::Mat&, int&, double&);

		void load(const char*);
		void save(const char*);
};

#endif