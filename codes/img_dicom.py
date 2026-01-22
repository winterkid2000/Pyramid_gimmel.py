class ImageView(QGraphicsView):
    wheel_slice = Signal(int)
    wheel_zoom = Signal(float)

    def __init__(self, scene, edit_item):
        super().__init__(scene)
        self.edit_item = edit_item
        self.drawing = False
        self.edit_enabled = False

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

    def set_edit_mode(self, enabled):
        self.edit_enabled = enabled

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self.wheel_zoom.emit(1.1 if event.angleDelta().y() > 0 else 0.9)
        else:
            self.wheel_slice.emit(+1 if event.angleDelta().y() > 0 else -1)

    def mousePressEvent(self, event):
        if self.edit_enabled and event.buttons() & Qt.LeftButton:
            self.edit_item.start_draw(self.mapToScene(event.position().toPoint()))
            self.drawing = True
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.edit_enabled:
            self.edit_item.draw(self.mapToScene(event.position().toPoint()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.edit_item.end_draw()
            self.drawing = False
        else:
            super().mouseReleaseEvent(event)
