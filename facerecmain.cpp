#include "facerec.h"
#include "eigenfacerec.h"
#include "fisherfacerec.h"
#include "svmfacerec.h"

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

// void from_picture_menu(int argc, const char* argv[]) {
//     Mat picture = imread(argv[1]);
//     FisherFaceRec tester;
//     const char* modelfile = "model_fisher_FERET.xml";
//     tester.HEIGHT = 300;
//     tester.WIDTH = 250;
//     tester.load(modelfile);
//     tester.predict_from_picture(picture);
// }

// void main_menu(int argc, const char *argv[]) {
//     FisherFaceRec tester;
//     cout<<"Using fisherfaces"<<endl;
//     cout<<"Equalize all images (not in preprocessing, also) after face detection; expand detected face (0.0,0.2, natural a.r.)"<<endl;

//     int action = 6;
//     int model = 1;
//     bool detect = false;
//     cerr<<"DBG-1.3"<<endl;

//     const char* modelfile;
//     switch (model) {
//         case 1:
//             modelfile = "model_fisher_FERET.xml";
//             break;
//         case 2:
//             modelfile = "model_fisher_normalized.xml";
//             break;
//         case 3:
//             modelfile = "model_fisher_cropped.xml";
//             break;
//         case 4:
//             modelfile = "model_eigen_normalized.xml";
//             break;
//         case 5:
//             modelfile = "model_eigen_cropped.xml";
//             break;
//         case 6:
//             modelfile = "model_eigen_FERET.xml";
//             break;
//         case 7:
//             modelfile = "model_fisher_FERET_half.xml";
//             break;
//     }
//     cerr<<"DBG-1.2"<<endl;

//     if (action == 2 || action == 6) cout<<"Using model "<<modelfile<<endl;
//     bool TRAINANDSAVE = false;
//     bool TESTFROMLOADED;
//     switch (action) {
//         case 1:
//             TRAINANDSAVE = true;
//             TESTFROMLOADED = false;
//             break;
//         case 2:
//             TESTFROMLOADED = true;
//             TRAINANDSAVE = false;
//             break;
//         //case 3:
//         default:
//             TESTFROMLOADED = false;
//             TRAINANDSAVE = false;
//             break;
//     }
//     tester.TESTBATCH = false;

//     // Check for valid command line arguments, print usage
//     // if no arguments were given.
//     if (argc < 2 && action != 6) {
//         cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
//         exit(1);
//     }

//     cerr<<"DBG-1.1"<<endl;
//     //TODO: maybe more elaborate arguments, and then parse them
//     // like: facerec -i image.jpg -o output ...


//     // for all actions except for, behave like before: output is second argument
//     // for option 4, second argument is the second csv; and output is third argument
//     int output_arg = 0;
//     if (action <= 3)
//         output_arg = 2;
//     if (action == 4)
//         output_arg = 3;
//     string output_folder;
//     if (argc >= output_arg+1) {
//         output_folder = string(argv[output_arg]);
//         tester.setOutput(output_folder);
//     }

//     // Get the path to your CSV.
//     cerr<<"DBG-1"<<endl;
//     string fn_csv;
//     if (action != 6)
//         fn_csv = string(argv[1]);

//     try {
//         // if we're using FERET model we must set some standard photo size
//         if (model == 1) {
//             tester.HEIGHT = 300;
//             tester.WIDTH = 250;
//         }
//         if (action != 6)
//             tester.read_csv(fn_csv, detect);
//     } catch (cv::Exception& e) {
//         cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
//         // nothing more we can do
//         exit(1);
//     }
//     cerr<<"DBG0"<<endl;
//     vector<Mat> images = tester.getImages();
//     vector<int> labels = tester.getLabels();

//     // if there are 2 arguments, read the second csv as well into a different object
//     if (action == 4 && argc >= 3) {
//         FisherFaceRec imageHolder;
//             try {
//             fn_csv = string(argv[2]);
//             imageHolder.read_csv(fn_csv, detect);
//             // before setting new images set to this one; set width and height to the ones of first image set
//             tester.WIDTH = images[0].size().width;
//             tester.HEIGHT = images[0].size().height;
            
//             images = imageHolder.getImages();
//             labels = imageHolder.getLabels();
//         } catch (cv::Exception& e) {
//             cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
//             // nothing more we can do
//             exit(1);
//         }
//     }

//     // These vectors hold the images and corresponding labels.


//     // set to TESTBATCH if there are 400 images - particular to FEI
//     cerr<<"DBG0.1"<<endl;
//     if (images.size() == 400)
//         tester.TESTBATCH = true;

//     int label = -1;
//     double confidence = -1;
//     int correct = 0;

//     switch(action) {
//         // test all from loaded
//         case 2:
//         //if (TESTFROMLOADED) 
//         {
//             //TODO: store these as well somehow when you save the model then load them from there

//             switch(model) {
//                 case 1:
//                     // //full
//                     // tester.HEIGHT = 360;
//                     // tester.WIDTH = 260;
//                     //FERET - some standard size
//                     tester.HEIGHT = 300;
//                     tester.WIDTH = 250;
//                     break;
                
//                 case 3:
//                     //cropped
//                     tester.HEIGHT = 193;
//                     tester.WIDTH = 162;
//                     break;

//                 case 2:
//                     //normalized
//                     tester.HEIGHT = 300;
//                     tester.WIDTH = 250;
//                     break;
//             }

//             tester.load(modelfile);
//             cout<<"Using model "<<modelfile<<endl;
//             // if we detect from csv there's no need in detecting at predict
//             if (detect)
//                 tester.predict_all();
//             else
//                 tester.detect_and_predict_all();
//             break;
//         }

//         //else
//         // train and save
//         case 1: 
//         {


//             if (TRAINANDSAVE) {
//                 cout<<"Train images size: "<<tester.HEIGHT<<" "<<tester.WIDTH<<" "<<tester.getImages()[0].rows<<" "<<endl;
//                 tester.train();
//                 //imwrite("mean.png", tester.mean.reshape(1,images[0].rows));
//                 // TODO: merge (sa creeze un fisier mare) doar cand adaug createF.... in train()?
//                 tester.save(modelfile);
//                 break;
//             }
//             //else 
//         }

//         // train and test - testall, crossvalidation
//         // from camera
//         case 6:
//         {
//             switch(model) {
//                 case 1:
//                     // //full
//                     // tester.HEIGHT = 360;
//                     // tester.WIDTH = 260;
//                     //FERET - some standard size
//                     tester.HEIGHT = 300;
//                     tester.WIDTH = 250;
//                     break;
                
//                 case 3:
//                     //cropped
//                     tester.HEIGHT = 193;
//                     tester.WIDTH = 162;
//                     break;

//                 case 2:
//                     //normalized
//                     tester.HEIGHT = 300;
//                     tester.WIDTH = 250;
//                     break;
//             }
//             cerr<<"DBG1"<<endl;
//             tester.load(modelfile);
//             cout<<"Using model "<<modelfile<<endl;
//             tester.predict_from_webcam();
//         }
//         //case 3 && 4:
//         default:
//             {
//             if (action == 3) {
//                 // only do this if we're using one csv; for action 4 we've already set them
//                 tester.HEIGHT = images[0].size().height;
//                 tester.WIDTH = images[0].size().width;
//             }
//             if (tester.TESTBATCH) {
//                 //genderrec
//                 //tester.SAMPLESIZE = 2;
//                 tester.SAMPLESIZE = 10;
//                 tester.NRCLASSES = 38;

//                 cout<<"NRCLASSES: "<<tester.NRCLASSES<<endl;
//                 cout<<"SAMPLESIZE: "<<tester.SAMPLESIZE<<endl;

//                 cout<<"Using leavePersonOut"<<endl<<endl;
//                 int successful = tester.testAll_leavePersonOut();
//             }
//             else {
//                 //facerec
//                 tester.SAMPLESIZE = 1;
//                 tester.NRCLASSES = 200;

//                 cout<<"NRCLASSES: "<<tester.NRCLASSES<<endl;
//                 cout<<"SAMPLESIZE: "<<tester.SAMPLESIZE<<endl;

//                 // int nrEigens = 10;
//                 // tester.setNrfaces(nrEigens);
//                 // cout<<"Nr of eigenfaces: "<<nrEigens<<endl;

//                 cout<<"Using leaveOneOut"<<endl<<endl;
//                 int successful;
//                 if (action == 3) 
//                     successful = tester.testAll_leaveOneOut();
//                 if (action == 4)
//                     successful = tester.testAll_leaveOneOut(images, labels);
//                 }
//                 break;
//             }


//         }
        
    
// }
void webcam_menu();

int main(int argc, const char *argv[]) {

    //TODO: fa ceva cu cazul cand nu detecteaza niciuna
    //TODO: eventual fa pt multiple fete din poza
    //TODO: ce fac daca detectatul dureaza o vesnicie...?
    //TODO: catch friggin exceptions! just catch them!
    //TODO: fa patratul ala sa fie aplicat pe ultima imagine ca sa se vada mare si cand e o imagine mare
    try {
        webcam_menu();
    } catch (cv::Exception& e) {
        cerr<<e.msg<<endl;
    }
    
    return 0;
}

void webcam_menu() {
    FisherFaceRec tester;
    const char* modelfile = "model_fisher_FERET.xml";
    tester.HEIGHT = 300;
    tester.WIDTH = 250;
    tester.load(modelfile);
    tester.predict_from_webcam();
}