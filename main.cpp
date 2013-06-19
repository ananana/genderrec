#include <QtGui/QApplication>
#include "mainwindow.h"
#include <QGraphicsView>
#include <QGraphicsScene>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
//    QGraphicsScene scene;
//    QGraphicsView view(&scene);
//    QGraphicsPixmapItem item(QPixmap("c:\\test.png"));
//    scene.addItem(&item);
//    view.show();
    MainWindow w;
    w.show();
    
    return a.exec();
}
