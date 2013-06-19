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
#include <QString>

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

 int from_picture_menu(Mat image, int& label) {
     FisherFaceRec tester;
     const char* modelfile = "model_fisher_FERET.xml";
     tester.HEIGHT = 300;
     tester.WIDTH = 250;
     tester.load(modelfile);
     tester.predict_from_picture(image, label);
     return label;
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

//    QGraphicsScene* scene = new QGraphicsScene(this);
//    scene->addPixmap(QPixmap(qApp->applicationDirPath() +"/1.pgm"));
//    ui->graphicsView->setScene(scene);
//    ui->graphicsView->fitInView(scene->sceneRect());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    try {
        //TODO: prinde exceptia asta cumva: Image step is wrong (The matrix is not continuous, thus its number of rows can not be changed) in reshape, file
        //TODO: deseneaza elipsa doar cand ai gasit fata
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
    string fileNameStr = fileName.toStdString();
    const char* fileNameCC = fileNameStr.c_str();
    //TODO: vezi ce se intampla cu -1 pt imagini alb-negru sau ceva
    Mat image = imread(fileNameCC, 3);
    int label = -1;

    try {
        image = from_picture_menu(image, label);
        //imshow("Some window", image);
    } catch (cv::Exception& e) {
        cerr<<e.msg<<endl;
    }
//    QImage qimage = Mat2QImage(image);

//    QGraphicsScene* scene = new QGraphicsScene(this);
//    //QGraphicsPixmapItem p = scene->addPixmap(QPixmap::fromImage(qimage));
//    //QGraphicsPixmapItem p =
//    scene->addPixmap(qApp->applicationDirPath() +"/1.pgm");
//    ui->graphicsView->setScene(scene);
//    //ui->graphicsView->fitInView(p, Qt::KeepAspectRatio);


//    std::string gender;
//    if (label == 0)
//        gender = "femeie";
//    else
//        gender = "barbat";
//    QString result(("Rezultat: " + gender).c_str());
//    ui->label->setText(result);
//    cout<<gender;
}
