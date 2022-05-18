# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter
class MyApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.begin, self.destination = QPoint(), QPoint()
    
    def click_handler(self,i,x,y,w,h):
        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow( i,x,y,w,h)
        screenshot.save('shot.jpg', 'jpg')
      
    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.begin = event.pos()
            self.destination = self.begin
            self.update()
        print("zxczxc")

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:		
            print('Point 2')	
            self.destination = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        print('Point 3')
        if event.button() & Qt.LeftButton:
            rect = QRect(self.begin, self.destination)
            m=rect.getRect()
            print (rect.top(),rect.getRect())
            x,y,w,h=m
          
            self.click_handler(self.winId(),x,y,w,h)
           
          
# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(800, 600)
#         self.centralwidget = MyApp()
#         self.centralwidget.setObjectName("centralwidget")
#         self.label = QtWidgets.QLabel(self.centralwidget)
#         self.label.setGeometry(QtCore.QRect(80, 50, 601, 421))
#         self.label.setText("")
#         self.label.setObjectName("label")
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.menubar = QtWidgets.QMenuBar(MainWindow)
#         self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
#         self.menubar.setObjectName("menubar")
#         MainWindow.setMenuBar(self.menubar)
#         self.statusbar = QtWidgets.QStatusBar(MainWindow)
#         self.statusbar.setObjectName("statusbar")
#         MainWindow.setStatusBar(self.statusbar)
        
#         pipmap=QPixmap("./short.jpg")
#         self.label.setText("cxvcxv")

#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)

        
    
           
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())