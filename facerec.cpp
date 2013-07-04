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

FaceRec::FaceRec() {
    // default for FEI gender 1 sample testing
	NRCLASSES = 200;
	SAMPLESIZE = 1;

	TRAINANDSAVE = false;

    // some values for initialization (will be reset later if needed)
    HEIGHT = 300;
    WIDTH = 250;

    string face_cascade_name = "haarcascade_frontalface_alt2.xml";
    if( !face_cascade.load( face_cascade_name ) ){ 
        cerr<<"(!)Error loading cascade"<<endl; 
        exit(1); 
    }

	output = false;
}

void FaceRec::setOutput(string dir) {
	output_folder = dir;
	output = true;
}

void FaceRec::read_csv(const string& filename, bool detectface = false) {
    std::ifstream file(filename.c_str(), ifstream::in);
    char separator = ';';
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    int NRLABELS=0;
    Mat image;
    string line, path, classlabel, oldlabel;
    int centerx, centery;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            // load the image as grayscale (parameter 0)
            image = imread(path, 0);

            if (detectface) {
                image = detect(image, centerx, centery);
                // if we use a detected face we must also resize the image to a standard dimension (they will vary)
                resize(image, image, Size(WIDTH, HEIGHT));
            }

            equalizeHist(image, image);
            // use this to set all to the same type. haven't tested if it changes things, but it solves an exception
            image = norm_0_255(image);
            images.push_back(image);
            labels.push_back(atoi(classlabel.c_str()));
            if (classlabel!=oldlabel)
                NRLABELS++;
            oldlabel=classlabel;
        }
    }

    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
}

// use this to make images displayable for the human eye
Mat FaceRec::norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Mat FaceRec::detect(Mat image, int& centerx, int& centery) {
    std::vector<Rect> faces;

    //equalizeHist( image, image ); // I don't need this here scince I'm equalizing everything as I read it
    //-- Detect faces

    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    // if you can't find a face return the original image
    if (faces.size() == 0)
        return image;

    Mat faceROI;
    Mat detected = image(faces[0]);

    // get the highest (just picked height, might go with width) one detected
    // TODO: ok idea?
    Rect detected_rect = faces[0];
    for( int i = 0; i < faces.size(); i++ )
    {
        Rect expanded = expand_Rect(faces[i], 0.0, 0.2, image.rows, image.cols);
        //cout<<"in detect: "<<expanded.x<<" "<<expanded.y<<" "<<expanded.height<<" "<<expanded.width<<endl;
        faceROI = image(expanded);
        if (faceROI.size().height > detected.size().height) {
            detected = faceROI;
            detected_rect = faces[i];
        }
    }
    // imwrite("detected/original.png",image);
    // imwrite("detected/detected.png",image(faces[0]));
    // imwrite("detected/expanded.png",detected);

    // equalize after you detect as well - we need it to be equalized at the face level, not the entire picture - we do it in readcsv instead
    // equalizeHist(detected, detected);

    // also set the center of the detected image
    centerx = detected_rect.x + detected_rect.width*0.5;
    centery = detected_rect.y + detected_rect.height*0.5;

    return detected;

}

// expand rectange vertically
Rect FaceRec::expand_Rect(Rect original, double up, double down, int maxh, int maxw)  {
    // up and down are fractions of entire height
    // ex: up = 0.3 => expand upwards with 0.3 of height
    // max is limit of expansion - limited by total image height
    int h_temp =original.height;                    // storing original HEIGHT
    int y = MAX(original.y - h_temp * up, 0);       // y is reduced by up*h, can't be negative
    int h = MIN(h_temp * (1 + up + down), maxh - y);   // height is increases by (up+down)%, can't be bigger than max+y
                                                      // bottom corner shouldn't exceed image height

    // set width of rectangle so that it keeps training images' aspect ratio - 
    // this way, at resize it will keep its natural aspect ratio
    //TODO: this is broken!
    double ratio = (double)WIDTH/HEIGHT;
    //cerr<<"ratio "<<ratio<<endl;
    int w = (int)(ratio * h);
    // split the extra width evenly between left and right
    int diff = w - original.width;
    int x = MAX(original.x - diff/2, 0);
    w = MIN(w, maxw - x);                       // width can't exceed image; bottom corner shouldn't exceed image limits

    // cout<<"orig: "<<original.x<<" "<<original.y<<" "<<original.height<<" "<<original.width<<endl;
    // cout<<"after: "<<x<<" "<<y<<" "<<h<<" "<<w<<endl;
    return Rect(x, y, w, h);
}


vector<Mat> FaceRec::getImages() {
    return images;
}

vector<int> FaceRec::getLabels() {
    return labels;
}

int FaceRec::leaveOneOut(vector<Mat>& testImages, vector<int>& testLabels, int testIndex, int personIndex) {

    cout<<"Testing for person "<<personIndex+1<<endl;
    cout<<"Test image index: "<<testIndex<<endl;

    // erase test image from training set
    Mat testSample = testImages[testIndex];
    int testLabel = testLabels[testIndex];

    Mat excludeSample = images[testIndex];
    int excludeLabel = labels[testIndex];
    images.erase(images.begin() + testIndex);
    labels.erase(labels.begin() + testIndex);
    

    cout<<"Training..."<<endl;


    train(images, labels);

    int predictedLabel = -1;
    double confidence = 0.0;

    cout<<"Testing..."<<endl;

    predict(testSample, predictedLabel, confidence);

    if (output)
        // also output training images
        for (int i = 0; i < images.size(); i++)
                imwrite(format("%s/%d.train%d.png", output_folder.c_str(), testIndex, i), images[i]);
   
    // insert it back where you took it from
    images.insert(images.begin() + testIndex, excludeSample);
    labels.insert(labels.begin() + testIndex, excludeLabel);

    // output results
    string result_message = format("Predicted class = %d / Actual class = %d. with confidence = %d", predictedLabel, testLabel, confidence);
    bool correct = (predictedLabel == testLabel);
    cout << result_message << endl << (correct?"CORRECT":"WRONG") << endl << endl;


    if (output) {
        imwrite(format("%s/%d.%s.png", output_folder.c_str(), testIndex, correct?"yes":"no"), testSample);
    }

    return (correct);
}

int FaceRec::leavePersonOut(int testBatch) {
    // removes a set of SAMPLESIZE images of same person and from the training set
    // tries to classify the person
    // suitable for gender recognition (or classifying people)

    int sample = SAMPLESIZE;

    cout<<"Test for person: "<<testBatch<<endl;

    // // test on random image from batch (of images of same person)
    // srand(time(0));
    // int testIndex = rand()%sample;

    int testIndexStart = testBatch * sample;

    vector<Mat> testSample(sample);
    vector<int> testLabel(sample);

    copy(images.begin() + testIndexStart, images.begin() + testIndexStart + sample, testSample.begin());
    // (label should be the same for entire batch)
    copy(labels.begin() + testIndexStart, labels.begin() + testIndexStart + sample, testLabel.begin());

    images.erase(images.begin() + testIndexStart, images.begin() + testIndexStart + sample);
    labels.erase(labels.begin() + testIndexStart, labels.begin() + testIndexStart + sample);


    cout<<"Training..."<<endl;

    train(images, labels);

    // test on all images from batch and give mean result
    // ! remember to change samplesize

    // if samplesize was set to 1 -> error
    if (SAMPLESIZE == 1)
        cerr<<"WARNING: Using leavePersonOut with SAMPLESIZE of 1!"<<endl;

    double successful = 0;

    cout<<"Testing..."<<endl;

    //TODO: test this?
    for (int testIndex = 0; testIndex < sample; testIndex++) {
        int predictedLabel = -1;
        double confidence = 0.0;

        predict(testSample[testIndex], predictedLabel, confidence);

        string result_message = format("Predicted class = %d / Actual class = %d. with confidence = %d", predictedLabel, testLabel[testIndex], confidence);
        bool correct = (predictedLabel == testLabel[testIndex]);
        cout << result_message << endl << (correct?"CORRECT":"WRONG") << endl << endl;

        successful += (correct?1:0);

        if (output) {
            imwrite(format("%s/%d.%s.png", output_folder.c_str(), testIndex + testIndexStart, correct?"yes":"no"), testSample[testIndex]);
            
            // // also output the train images
            // // this will write them twice for each batch
            // for (int i = 0; i < images.size(); i++) {
            //     imwrite(format("%s/%d.train%d.png", output_folder.c_str(), testIndex + testIndexStart, i), images[i]);
            
            //TODO: output the fisherfaces?

           // }
        }
    }
    

    // insert it back where you took it from
    images.insert(images.begin() + testIndexStart, testSample.begin(), testSample.end());
    labels.insert(labels.begin() + testIndexStart, testLabel.begin(), testLabel.end());

    string result_message = format("Accuracy for person %d: %d out of %d", testBatch, successful, sample);

    return successful;
}

int FaceRec::testAll_leaveOneOut() {

    testAll_leaveOneOut(this->images, this->labels);
}

int FaceRec::testAll_leaveOneOut(vector<Mat>& testImages, vector<int>& testLabels) {

    int succesful = 0;
    int index = 0;
    int tests = NRCLASSES; // nr of people in DB
    int samples = SAMPLESIZE; // nr of images per person
    srand(time(0));
    for (int i = 0; i < tests; i++) {
        // pick one random image in every ten image set (each subject)
        index = (rand() % samples) + samples * i; 
        succesful += leaveOneOut(testImages, testLabels, index, i);
    }

    cout<<"Succesful for "<<succesful<<" out of "<<tests<<endl;
    return succesful;
}

int FaceRec::testAll_leavePersonOut() {
    int succesful = 0;
    int index = 0;
    int tests = NRCLASSES; // nr of people in DB
    int samples = SAMPLESIZE; // nr of images per person
    
    for (int index = 0; index < tests; index++) {
        // do for every person
        succesful += leavePersonOut(index);
    }

    cout<<"Succesful for "<<succesful<<" out of "<<tests * samples<<endl;
    return succesful;
}

void FaceRec::predict_all(vector<Mat>& images, vector<int>& labels, bool detect=false) {
    int label = -1; double confidence = -1;
    int correct = 0;
    for (int i = 0; i < images.size(); i++) {
        Mat image = images[i];
        if (detect)
            image = detect_and_predict(images[i], label, confidence);
        else
            predict(images[i], label, confidence);
        if (label == labels[i]) correct++;
        cout<<"Image "<<i<<": "<<"predicted: "<<label<<"; confidence: "<<confidence<<endl
        <<(label==labels[i]?"CORRECT":"INCORRECT")<<endl<<endl;
        if (output)
            imwrite(format("%s/%d.%s.%s.png", output_folder.c_str(), i, labels[i]==0?"female":"male", label==labels[i]?"yes":"no"), image);
    }
    cout<<"Successful for "<<correct<<" / "<<images.size()<<endl;
}

void FaceRec::predict_all() {
    predict_all(this->images, this->labels);
}

Mat FaceRec::detect_and_predict(Mat& testSample, int& predictedLabel, double& confidence) {
    int centerx, centery;
    Mat detected = detect(testSample, centerx, centery);
    predict(detected, predictedLabel, confidence);
    return detected;
}

Mat FaceRec::predict_from_webcam(){
    //TODO: check if we loaded any model file
    CvCapture* capture;
    Mat frame;
    Mat frame_gray;
    capture = cvCaptureFromCAM( -1 );
    int centerx = 0, centery = 0;
    if( capture ) {
        while (true)
        {
            frame = cvQueryFrame( capture );

            //preprocess
            cvtColor( frame, frame_gray, CV_BGR2GRAY );
            equalizeHist( frame_gray, frame_gray );

            //-- 3. Apply the classifier to the frame
            int label = -1;
            double confidence = 0;
            if( !frame.empty() ) {
                frame_gray = detect(frame_gray, centerx, centery);
                Point center(centerx, centery);
                // only if something was detected
                if (centerx != 0 || centery != 0) {
                    ellipse( frame, center, Size(WIDTH/2,HEIGHT/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
                    predict(frame_gray, label, confidence);
                }
                cout<<"Predicted class: "<<(label==0?"female":"male")<<" ("<<label<<")"<<endl; 
            }
            else {
                printf(" --(!) No captured frame -- Break!"); 
                break; 
            }

            int c = waitKey(5);
            if( (char)c == 'q' ) { break; }

            std::string gender;
            if (label == 0)
                gender = "femeie";
            else
                if (label == 1)
                    gender = "barbat";
                else gender = "";

            putText(frame, gender, cvPoint(frame.cols/2-10,frame.rows-10),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);


            imshow( "Fata ta aici", frame );

        }

        cvReleaseCapture(&capture);
    }

    return frame;

}

Mat FaceRec::predict_from_picture(Mat picture, int& label) {
    Mat picture_gray, picture_resized;
    //picture_gray = picture;
    cvtColor( picture, picture_gray, CV_BGR2GRAY );
    equalizeHist( picture_gray, picture_gray );
    label = -1;
    double confidence = 0;
    int centerx = WIDTH/2, centery = HEIGHT/2;
    namedWindow("Imaginea incarcata");

    //MainWindow * win = (MainWindow *) qApp::activeWindow();

    //show it before detecting as well
    // display resized picture, but with natural aspect ratio
    Size size = Size(int(picture.cols*(1/(float)picture.rows)*HEIGHT), HEIGHT);
    cerr<<"height "<<size.height<<" width "<<size.width<<endl;
    resize(picture, picture_resized, size);
    imshow("Imaginea incarcata", picture_resized);
    //waitKey(0);

    cout<<"Detecting face..."<<endl;
    picture_gray = detect(picture_gray, centerx, centery);
    Point center(centerx, centery);
    ellipse( picture, center, Size(WIDTH/2,HEIGHT/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
    resize(picture, picture, size);
    imshow("Imaginea incarcata", picture);

    cout<<"Predicting class..."<<endl;
    predict(picture_gray, label, confidence);
    cout<<"Predicted class: "<<(label==0?"female":"male")<<" ("<<label<<")"<<endl;

    std::string gender;
    if (label == 0)
        gender = "femeie";
    else
        gender = "barbat";

    putText(picture, gender, cvPoint(picture.cols/2-10,picture.rows-10),
        FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);


    //while (1) {
        resize(picture, picture, size);
        imshow("Imaginea incarcata", picture);
        waitKey(0);
        //int c = waitKey(30);
        //if( (char)c == 'q' ) { break; }
    //}

    return picture;
}

void FaceRec::detect_and_predict_all(vector<Mat>& images, vector<int>& labels) {
    predict_all(images, labels, true);
}

void FaceRec::detect_and_predict_all() {
    detect_and_predict_all(this->images, this->labels);
}
