#ifndef EIGENFACEREC_H
#define EIGENFACEREC_H

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

class EigenFaceRec : public FaceRec {

	private:
		cv::Ptr<cv::FaceRecognizer> model;
        //TODO: maybe lose this at somepoint
        cv::Mat mean;
        int nrEigens;


	public:
		EigenFaceRec(int);
		EigenFaceRec(const char*, int);

        void setNrfaces(int);
		// uses given set of images to train a new classifier
		void train(std::vector<cv::Mat>&, std::vector<int>&);
        void train();
		void predict(cv::Mat&, int&, double&);

		void load(const char*);
		void save(const char*);
};

#endif