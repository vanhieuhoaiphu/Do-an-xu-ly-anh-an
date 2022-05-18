from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
w = QtWidgets.QWidget()

grab_btn=QtWidgets.QPushButton('Grab Screen')
def click_handler():
    screen = QtWidgets.QApplication.primaryScreen()
    screenshot = screen.grabWindow( 0,50,200,740,1000 )
    print(w.winId())
    screenshot.save('shot.jpg', 'jpg')
    w.close()

grab_btn.clicked.connect(click_handler)

layout = QtWidgets.QVBoxLayout()
layout.addWidget(grab_btn)
w.setLayout(layout)
w.show()

sys.exit(app.exec_())