#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
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
    string fileNameStr = fileName.toStdString();
    const char* fileNameCC = fileNameStr.c_str();
    Mat image = imread(fileNameCC, 0);

    try {
        from_picture_menu(image);
    } catch (cv::Exception& e) {
        cerr<<e.msg<<endl;
    }
}
