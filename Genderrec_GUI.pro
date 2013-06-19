#-------------------------------------------------
#
# Project created by QtCreator 2013-06-19T08:43:44
#
#-------------------------------------------------

QT       += core gui

TARGET = Genderrec_GUI
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
        facerec.cpp \
        eigenfacerec.cpp \
        fisherfacerec.cpp \
        svmfacerec.cpp

HEADERS  += mainwindow.h \
            facerec.h \
            eigenfacerec.h \
            fisherfacerec.h \
            svmfacerec.h

INCLUDEPATH += /home/ana/sources/OpenCV-2.4.2/include

FORMS    += mainwindow.ui

LIBS += -LC/home/ana/sources/OpenCV-2.4.2/lib \
    -lopencv_calib3d \
    -lopencv_contrib \
    -lopencv_core \
    -lopencv_features2d \
    -lopencv_flann \
    -lopencv_gpu \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lopencv_legacy \
    -lopencv_ml \
    -lopencv_nonfree \
    -lopencv_objdetect \
    -lopencv_photo \
    -lopencv_stitching \
    -lopencv_ts \
    -lopencv_video \
    -lopencv_videostab
