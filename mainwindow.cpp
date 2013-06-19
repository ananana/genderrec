#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QImage>
#include <QtGui/QGraphicsScene>
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsPixmapItem>
#include <QtGui/QPixmap>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "facerec.h"
#include "eigenfacerec.h"
#include "fisherfacerec.h"
#include "svmfacerec.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

 void from_picture_menu(Mat image) {
     FisherFaceRec tester;
     const char* modelfile = "model_fisher_FERET.xml";
     tester.HEIGHT = 300;
     tester.WIDTH = 250;
     tester.load(modelfile);
     tester.predict_from_picture(image);
 }

 void webcam_menu() {
     FisherFaceRec tester;
     const char* modelfile = "model_fisher_FERET.xml";
     tester.HEIGHT = 300;
     tester.WIDTH = 250;
     tester.load(modelfile);
     Mat result = tester.predict_from_webcam();
 }

 QImage Mat2QImage(const cv::Mat_<double> &src)
 {
         double scale = 255.0;
         QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
         for (int y = 0; y < src.rows; ++y) {
                 const double *srcrow = src[y];
                 QRgb *destrow = (QRgb*)dest.scanLine(y);
                 for (int x = 0; x < src.cols; ++x) {
                         unsigned int color = srcrow[x] * scale;
                         destrow[x] = qRgba(color, color, color, 255);
                 }
         }
         return dest;
 }

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    try {
        //TODO: prinde exceptia asta cumva: Image step is wrong (The matrix is not continuous, thus its number of rows can not be changed) in reshape, file
        webcam_menu();
    } catch (cv::Exception& e) {
        cerr<<e.msg<<endl;
    }
}



void MainWindow::on_pushButton_2_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                         "",
                                                         tr("Files (*.*)"));
    //TODO: ia doar fisierele foto
    cerr<<"DBG0"<<endl;
    string fileNameStr = fileName.toStdString();
    cerr<<"DBG1"<<endl;
    const char* fileNameCC = fileNameStr.c_str();
    cerr<<"DBG2"<<endl;
    //TODO: vezi ce se intampla cu -1 pt imagini alb-negru sau ceva
    Mat image = imread(fileNameCC, -1);
    cerr<<"DBG3"<<endl;

    try {
        from_picture_menu(image);
        //imshow("Some window", image);
    } catch (cv::Exception& e) {
        cerr<<e.msg<<endl;
    }
}
