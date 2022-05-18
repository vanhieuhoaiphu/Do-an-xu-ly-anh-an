import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QPainter



class MyApp(QWidget):
	def __init__(self):
		super().__init__()
		self.window_width, self.window_height = 1200, 800
		self.setMinimumSize(self.window_width, self.window_height)


		self.begin, self.destination = QPoint(), QPoint()	



	def mousePressEvent(self, event):
		if event.buttons() & Qt.LeftButton:
			
			self.begin = event.pos()
			self.destination = self.begin
			self.update()

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
			print(x)
         
           
		    

if __name__ == '__main__':
	# don't auto scale when drag app to a different monitor.
	# QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
	
	app = QApplication(sys.argv)
	app.setStyleSheet('''
		QWidget {
			font-size: 30px;
		}
	''')
	
	myApp = MyApp()
	myApp.show()

	try:
		sys.exit(app.exec_())
	except SystemExit:
		print('Closing Window...')
