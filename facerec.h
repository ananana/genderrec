#ifndef FACEREC_H
#define FACEREC_H

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>


class FaceRec {


	//private:
    public:

		//TODO: make these const? pass them to the constructor?
		int NRCLASSES;
		int SAMPLESIZE;
		bool TESTBATCH;
		bool TRAINANDSAVE;

        // dimensions of model's training images
        int HEIGHT;
        int WIDTH;

    protected:
        // are we also writing the images to output?
		bool output;

		std::string output_folder;
		const char* MODELFILE;

        std::vector<cv::Mat> images;
        std::vector<int> labels;

        std::string face_cascade_name;
        cv::CascadeClassifier face_cascade;

	public:
		FaceRec();

		static cv::Mat norm_0_255(cv::InputArray);
        // TODO: maybe understand this better? pay attention to min_size and max_size parameters
        // also take notice that this equalizes the image, so the training set should probably be too
        cv::Mat detect(cv::Mat, int&, int&);
        cv::Rect expand_Rect(cv::Rect, double, double, int, int);
		void read_csv(const std::string&, bool);

		void setOutput(std::string);

        std::vector<cv::Mat> getImages();
        std::vector<int> getLabels();

        virtual void train(std::vector<cv::Mat>&, std::vector<int>&) = 0;
        virtual void train() = 0;
        virtual void predict(cv::Mat&, int&, double&) = 0;
        cv::Mat detect_and_predict(cv::Mat&, int&, double&);
        void predict_all(std::vector<cv::Mat>&, std::vector<int>&, bool);
        void predict_all();
        void detect_and_predict_all(std::vector<cv::Mat>&, std::vector<int>&);
        void detect_and_predict_all();

        cv::Mat predict_from_webcam();
        cv::Mat predict_from_picture(cv::Mat, int&);

        virtual void load(const char*) = 0;
        virtual void save(const char*) = 0;

        int testAll_leaveOneOut();
        int testAll_leavePersonOut();
        // leaveOneOut but with a different test std::vector than train
        int testAll_leaveOneOut(std::vector<cv::Mat>&, std::vector<int>&);

    protected:

        int leaveOneOut(std::vector<cv::Mat>&, std::vector<int>&, int, int);
        int leavePersonOut(int);
};

#endif
