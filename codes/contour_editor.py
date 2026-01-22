class EditContourItem(QGraphicsPathItem):
    def __init__(self, diameter=10, colour=Qt.red, mode="erase"):
        super().__init__()
        self.diameter = diameter
        self.mode = mode
        self.current_path = QPainterPath()
        self.paths = []

        pen = QPen(colour)
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush(colour, Qt.Dense4Pattern))
        self.setZValue(100)

    def set_mode(self, mode):
        self.mode = mode
        colour = Qt.red if mode == "erase" else Qt.green
        pen = QPen(colour)
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush(colour, Qt.Dense4Pattern))

    def set_brush_size(self, diameter):
        self.diameter = diameter

    def start_draw(self, pos):
        self.current_path = QPainterPath()
        self.current_path.addEllipse(pos, self.diameter/2, self.diameter/2)
        self.setPath(self.current_path)

    def draw(self, pos):
        p = QPainterPath()
        p.addEllipse(pos, self.diameter/2, self.diameter/2)
        self.current_path = self.current_path.united(p)
        self.setPath(self.current_path)

    def end_draw(self):
        if not self.current_path.isEmpty():
            self.paths.append(self.current_path)
        self.current_path = QPainterPath()
        self.setPath(QPainterPath())

    def clear(self):
        self.paths.clear()
        self.current_path = QPainterPath()
        self.setPath(QPainterPath())

    def get_merged_path(self):
        merged = QPainterPath()
        for p in self.paths:
            merged = merged.united(p)
        return merged
