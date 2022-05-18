from PyQt5 import QtCore, QtGui, QtWidgets 
class Viewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QGraphicsScene(), parent)
        self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
     
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.rubberBandChanged.connect(self.onRubberBandChanged)
        self.last_rect = QtCore.QPointF()

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)


    @QtCore.pyqtSlot(QtCore.QRect, QtCore.QPointF, QtCore.QPointF)
    def onRubberBandChanged(self, rubberBandRect, fromScenePoint, toScenePoint):
        if rubberBandRect.isNull():
            pixmap = self.pixmap_item.pixmap()
            rect = self.pixmap_item.mapFromScene(self.last_rect).boundingRect().toRect()
            if not rect.intersected(pixmap.rect()).isNull():
                crop_pixmap = pixmap.copy(rect)
                label = QtWidgets.QLabel(pixmap=crop_pixmap)
                dialog = QtWidgets.QDialog(self)
                lay = QtWidgets.QVBoxLayout(dialog)
                lay.addWidget(label)
                dialog.exec_()
                
            self.last_rect = QtCore.QRectF()

        else:
            self.last_rect = QtCore.QRectF(fromScenePoint, toScenePoint)