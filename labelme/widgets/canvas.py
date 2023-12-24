from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme import QT5
from labelme.shape import Shape
import labelme.utils

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import numpy as np
import cv2
import math
# import time

# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0

DEFAULT_SELECTION_TOOL_BASE_COLOR = QtGui.QColor(0, 255, 0, 128)
DEFAULT_SELECTION_TOOL_BLINK_COLOR = QtGui.QColor(0, 0, 255, 128)

class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)
    updateContrastLevelIndexTextBox = QtCore.Signal()
    updateNumFinalSelectionPolygonsLabel = QtCore.Signal()

    CREATE, EDIT, SELECT = 0, 1, 2
    SEGMENTATION_TREE, BORDER_SELECTION, MASK_SEGMENT_SELECTION, MASK_SEGMENT_DESELECTION, MASK_ADDITION, MASK_REMOVAL, MAX_CONTRAST_BOUNDARY = 0, 1, 2, 3, 4, 5, 6

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
            },
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        # Segmentation Tree Data
        self.contrast_levels = []
        self.contrast_level_index = 0
        self.segmentation_tree = None
        self.selectMode = self.BORDER_SELECTION
        self.contour_points = []
        self.contouring = False
        self.contour_editing_radius = 20
        self.segmentation_tree_selection_unary_union = []
        self.numFinalSelectionPolygons = len(self.segmentation_tree_selection_unary_union)
        self.adjustingStencilSize = False
        self.stencilSizeAdjustmentStartingPoint = 0
        self.tempStencilSize = 0
        self.image = QtGui.QImage()
        self.newPoints = []

        # self.startTime = 0
        # self.curTime = 0
        # self.saveCounter = 0

        self.blinkColors = False
        self.selectionToolColor = DEFAULT_SELECTION_TOOL_BASE_COLOR
        self.selectionToolsBaseColor = DEFAULT_SELECTION_TOOL_BASE_COLOR
        self.selectionToolsBlinkColor = DEFAULT_SELECTION_TOOL_BLINK_COLOR
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(lambda: self.updateTimerHandler())
        self.update_timer.start(1000)
        

        self.msgBox = QtWidgets.QMessageBox()
        self.msgBox.setTextFormat(QtCore.Qt.RichText)
        self.msgBox.setText("<h4>The current selection contains multiple disconencted polygons</h4>")
        self.msgBox.setInformativeText("Do you wish to save each polygon seperately?")
        self.msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        self.msgBox.setDefaultButton(QtWidgets.QMessageBox.No)
        self.msgBox.setWindowTitle(" ")

    # def polygonToMask(self):
    #     height = self.image.height()
    #     width = self.image.width()
    #     img = np.zeros((height,width,3), np.uint8)
    #     shape = self.shapes[-1]
    #     polygon = Polygon([[point.x(),point.y()] for point in shape.points])

    #     for r in range(0, height):
    #         for c in range(0, width):
    #             if polygon.contains(Point([c,r])):
    #                 img[r][c][0] = 255
    #                 img[r][c][1] = 255
    #                 img[r][c][2] = 255
    #     cv2.imwrite('mask' + str(self.saveCounter) + '.png',img)
    #     self.saveCounter += 1
        
    def swapSelectionToolColor(self):
        if self.selectionToolColor == self.selectionToolsBlinkColor:
            self.selectionToolColor = self.selectionToolsBaseColor
        else:
            self.selectionToolColor = self.selectionToolsBlinkColor

    def updateTimerHandler(self):
        if self.blinkColors:
            self.swapSelectionToolColor()
        else:
            self.selectionToolColor = self.selectionToolsBaseColor
        self.update()

    def clearSegTreeSelection(self):
        self.segmentation_tree.removeSelection()
        self.segmentation_tree_selection_unary_union = []
        self.numFinalSelectionPolygons = 0
        self.updateNumFinalSelectionPolygonsLabel.emit()

    def simplifyPolygons(self, polygons):
        result = []
        for polygon in polygons:
            result.append(polygon.simplify(0))
        return result

    def modifyFinalSelection(self, contourLineString, adding):
        modifingPolygon = Polygon(list(contourLineString.exterior.coords)).simplify(0)
        if adding:
            self.segmentation_tree_selection_unary_union.append(modifingPolygon)
            modificationResult = unary_union(self.segmentation_tree_selection_unary_union)
        else:
            if len(self.segmentation_tree_selection_unary_union) == 0:
                return
            combinedPolygons = unary_union(self.segmentation_tree_selection_unary_union)
            modificationResult = combinedPolygons.difference(modifingPolygon)

        if modificationResult.is_empty:
            modificationResult = []
        elif modificationResult.geom_type != "Polygon":
            modificationResult = list(modificationResult.geoms)
        else:
            modificationResult = [modificationResult]
        modificationResult = self.simplifyPolygons(modificationResult)
        self.segmentation_tree_selection_unary_union = modificationResult
        self.numFinalSelectionPolygons = len(self.segmentation_tree_selection_unary_union)
        self.updateNumFinalSelectionPolygonsLabel.emit()

    def updateSegTreeSelectionUnaryUnion(self):
        if self.segmentation_tree != None:
            selectedSegments = self.segmentation_tree.collectSelectedSegments()
            unaryUnionResult = unary_union(selectedSegments)
            if unaryUnionResult.is_empty:
                unaryUnionResult = []
            elif unaryUnionResult.geom_type != "Polygon":
                unaryUnionResult = list(unaryUnionResult.geoms)
            else:
                unaryUnionResult = [unaryUnionResult]
            unaryUnionResult = self.simplifyPolygons(unaryUnionResult)
            self.segmentation_tree_selection_unary_union = unaryUnionResult
            self.numFinalSelectionPolygons = len(self.segmentation_tree_selection_unary_union)
            self.updateNumFinalSelectionPolygonsLabel.emit()

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "select",
            "edit",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1 :]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def selecting(self):
        return self.mode == self.SELECT

    def setMode(self, mode="edit"):
        if mode == "edit":
            # CREATE -> EDIT
            self.mode = self.EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            if mode == "select":
                self.mode = self.SELECT
            else:
                self.mode = self.CREATE
            self.unHighlight()
            self.deSelectShape()
        self.update()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            self.repaint()
            self.current.highlightClear()
            return
        elif self.selecting():
            if self.selectMode == self.SEGMENTATION_TREE:
                self.segmentation_tree.updateHovering([pos.x(),pos.y()])
            elif self.adjustingStencilSize:
                dif = pos.x() - self.stencilSizeAdjustmentStartingPoint[0]
                self.tempStencilSize = self.contour_editing_radius + dif
                if self.tempStencilSize < 0:
                    self.tempStencilSize = 5
            elif self.contouring:
                    self.contour_points.append(pos)
            self.update()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click & drag to move point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                            # self.curTime = time.perf_counter()
                            # print(self.curTime)
                            # self.polygonToMask()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                        # self.curTime = time.perf_counter()
                        # print(self.curTime)
                        # self.polygonToMask()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                            # self.curTime = time.perf_counter()
                            # print(self.curTime)
                            # self.polygonToMask()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    # self.startTime = time.perf_counter()
                    # print(self.startTime)
                    # self.saveCounter = 0
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == "point":
                        self.finalise()
                    else:
                        if self.createMode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
            elif self.selecting():
                if self.selectMode == self.BORDER_SELECTION:
                    self.segmentation_tree.removeSelection()
                    self.contouring = True
                    self.contour_points = [pos]
                elif (
                     self.selectMode == self.MASK_SEGMENT_SELECTION or
                     self.selectMode == self.MASK_SEGMENT_DESELECTION or
                     self.selectMode == self.MASK_ADDITION or
                     self.selectMode == self.MASK_REMOVAL or
                     self.selectMode == self.MAX_CONTRAST_BOUNDARY
                    ):
                    self.contouring = True
                    self.contour_points = [pos]
        elif ev.button() == QtCore.Qt.MiddleButton:
            if self.selecting():
                if (
                    self.selectMode == self.BORDER_SELECTION or
                    self.selectMode == self.MASK_SEGMENT_SELECTION or
                    self.selectMode == self.MASK_SEGMENT_DESELECTION or
                    self.selectMode == self.MASK_ADDITION or
                    self.selectMode == self.MASK_REMOVAL or
                    self.selectMode == self.MAX_CONTRAST_BOUNDARY
                    ):
                    self.adjustingStencilSize = True
                    self.stencilSizeAdjustmentStartingPoint = (pos.x(),pos.y())
        elif ev.button() == QtCore.Qt.RightButton and self.editing():
            group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
            if not self.selectedShapes or (
                self.hShape is not None
                and self.hShape not in self.selectedShapes
            ):
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.repaint()
            self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == QtCore.Qt.RightButton:
            menu = self.menus[len(self.selectedShapesCopy) > 0]
            self.restoreCursor()
            if (
                not menu.exec_(self.mapToGlobal(ev.pos()))
                and self.selectedShapesCopy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selectedShapesCopy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )
            if self.selecting():
                if self.selectMode == self.SEGMENTATION_TREE:
                    self.segmentation_tree.editSegmentSelectionAtContrastLevel((pos.x(),pos.y()), self.contrast_levels[self.contrast_level_index])
                    self.updateSegTreeSelectionUnaryUnion()
                elif self.selectMode == self.BORDER_SELECTION:
                    self.contouring = False
                    self.contour_points.append(pos)
                    if len(self.contour_points) >= 3:
                        contour_points = [[i.x(),i.y()] for i in self.contour_points]
                        contour = LineString(contour_points).buffer(self.contour_editing_radius/2)
                        selection_polygon = Polygon(list(contour.exterior.coords)).simplify(0)
                        self.segmentation_tree.selectSegmentsWithPercentAreaAndBoundary(selection_polygon)
                    self.updateSegTreeSelectionUnaryUnion()
                    # self.curTime = time.perf_counter()
                    # print(self.curTime)
                    # self.polygonToMask()
                elif self.selectMode == self.MASK_SEGMENT_SELECTION or self.selectMode == self.MASK_SEGMENT_DESELECTION:
                    self.contouring = False
                    self.contour_points.append(pos)
                    points = [(i.x(), i.y()) for i in self.contour_points]
                    contour = LineString(points).buffer(self.contour_editing_radius/2)

                    adding = True
                    if self.selectMode == self.MASK_SEGMENT_DESELECTION:
                        adding = False

                    self.segmentation_tree.editSegmentSelectionWithVariableWidthContour(contour, adding, False)
                    self.updateSegTreeSelectionUnaryUnion()
                    # self.curTime = time.perf_counter()
                    # print(self.curTime)
                    # self.polygonToMask()
                elif self.selectMode == self.MASK_ADDITION or self.selectMode == self.MASK_REMOVAL:
                    self.contouring = False
                    self.contour_points.append(pos)
                    points = [(i.x(), i.y()) for i in self.contour_points]
                    contour = LineString(points).buffer(self.contour_editing_radius/2)

                    adding = True
                    if self.selectMode == self.MASK_REMOVAL:
                        adding = False
                    # self.curTime = time.perf_counter()
                    # print(self.curTime)
                    # self.polygonToMask()
                    self.modifyFinalSelection(contour, adding)
                elif self.selectMode == self.MAX_CONTRAST_BOUNDARY:
                    self.contouring = False
                    self.contour_points.append(pos)
                    points = [(i.x(), i.y()) for i in self.contour_points]
                    contour = LineString(points).buffer(self.contour_editing_radius/2)
                    
                    self.findSelectionFromContourContrast(contour)
                    # self.curTime = time.perf_counter()
                    # print(self.curTime)
                    # self.polygonToMask()
                self.update()
        elif ev.button() == QtCore.Qt.MiddleButton:
            if self.selecting():
                if (
                    self.selectMode == self.BORDER_SELECTION or
                    self.selectMode == self.MASK_SEGMENT_SELECTION or
                    self.selectMode == self.MASK_SEGMENT_DESELECTION or
                    self.selectMode == self.MASK_ADDITION or
                    self.selectMode == self.MASK_REMOVAL or
                    self.selectMode == self.MAX_CONTRAST_BOUNDARY
                    ):
                    self.adjustingStencilSize = False
                    dif = pos.x() - self.stencilSizeAdjustmentStartingPoint[0]
                    self.contour_editing_radius += int(dif)
                    if self.contour_editing_radius < 0:
                        self.contour_editing_radius = 5

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                self.shapesBackups[-1][index].points
                != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False
            # self.curTime = time.perf_counter()
            # print(self.curTime)
            # self.polygonToMask()

    def getSubArray(self, x, y, kernelWidth, kernelHeight):
        arr = np.ndarray((kernelHeight,kernelWidth,3))
        a = (kernelWidth-1)/2
        b = (kernelHeight-1)/2
        for i in range(0, kernelHeight):
            for j in range(0, kernelWidth):
                argb = self.image.pixel(int(x+j-a),int(y+i-b))
                arr[i,j,0] = (argb & 0x00ff0000) >> 16
                arr[i,j,1] = (argb & 0x0000ff00) >> 8
                arr[i,j,2] = (argb & 0x000000ff)
        return arr

    def calcContrastDy(self, arr):
        kernelDy = np.array([[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]],
                            [[0,0,0],[0,0,0],[0,0,0]],
                            [[1,1,1],[1,1,1],[1,1,1]]])
        arr = np.multiply(arr,kernelDy)
        return np.linalg.norm(arr.sum(0).sum(0))

    def calcContrastDx(self, arr):
        kernelDx = np.array([[[-1,-1,-1],[0,0,0],[1,1,1]],
                            [[-1,-1,-1],[0,0,0],[1,1,1]],
                            [[-1,-1,-1],[0,0,0],[1,1,1]]])
        arr = np.multiply(arr,kernelDx)
        return np.linalg.norm(arr.sum(0).sum(0))

    def findSelectionFromContourContrast(self, contour):
        contour_points = list(contour.simplify(0.1).exterior.coords)
        new_contour_points = []
        for i in range(0, len(contour_points)-1):
            new_contour_points.append([contour_points[i][0],contour_points[i][1]])
            if contour_points[i][0] < 0:
                new_contour_points[i][0] = 0
            elif contour_points[i][0] >= self.image.width():
                new_contour_points[i][0] = self.image.width()-1
            if contour_points[i][1] < 0:
                new_contour_points[i][1] = 0
            elif contour_points[i][1] >= self.image.height():
                new_contour_points[i][1] = self.image.height()-1
        contour_points = new_contour_points
                    
        newPoints = []
        for i in range(0, len(contour_points)-1):
            curX = int(contour_points[i][0])
            curY = int(contour_points[i][1])

            verticalSearch = False
            searchingUp = False
            searchingLeft = False
            if abs(contour_points[i+1][0] - contour_points[i-1][0]) >= abs(contour_points[i+1][1] - contour_points[i-1][1]):
                verticalSearch = True
                if contour.contains(Point(curX, curY-1)):
                    searchingUp = True
            else:
                if contour.contains(Point(curX-1, curY)):
                    searchingLeft = True

            maxContrastPoint = (curX, curY)
            if curY <= 0 or curY >= self.image.height()-1 or curX <= 0 or curX >= self.image.width()-1:
                maxContrastVal = 255
            else:
                arr = self.getSubArray(curX,curY,3, 3)
                maxContrastVal = self.calcContrastDy(arr) + self.calcContrastDx(arr)
            
            # if verticalSearch and searchingUp:
            #     LineString([(curX, curY), (curX, curY-j)])
            # elif verticalSearch and not searchingUp:
                
            # elif not verticalSearch and searchingLeft:
                
            # elif not verticalSearch and not searchingLeft:
            for j in range(0, self.contour_editing_radius):
                if verticalSearch and searchingUp:
                    if self.selectedSegmentContainsPoint(Point(curX, curY-j)):
                        maxContrastPoint = (curX, curY-j)
                        break
                    elif curY-j <= 0 or curY-j >= self.image.height()-1 or curX <= 0 or curX >= self.image.width()-1:
                        newContrastVal = 255
                    else:
                        arr = self.getSubArray(curX,curY-j,3, 3)
                        newContrastVal = self.calcContrastDy(arr) + self.calcContrastDx(arr)
                elif verticalSearch and not searchingUp:
                    if self.selectedSegmentContainsPoint(Point(curX, curY+j)):
                        maxContrastPoint = (curX, curY+j)
                        break
                    elif curY+j <= 0 or curY+j >= self.image.height()-1 or curX <= 0 or curX >= self.image.width()-1:
                        newContrastVal = 255
                    else:
                        arr = self.getSubArray(curX,curY+j,3, 3)
                        newContrastVal = self.calcContrastDy(arr) + self.calcContrastDx(arr)
                elif not verticalSearch and searchingLeft:
                    if self.selectedSegmentContainsPoint(Point(curX-j, curY)):
                        maxContrastPoint = (curX-j, curY)
                        break
                    elif curY <= 0 or curY >= self.image.height()-1 or curX-j <= 0 or curX-j >= self.image.width()-1:
                        newContrastVal = 255
                    else:
                        arr = self.getSubArray(curX-j,curY,3, 3)
                        newContrastVal = self.calcContrastDy(arr) + self.calcContrastDx(arr)
                elif not verticalSearch and not searchingLeft:
                    if self.selectedSegmentContainsPoint(Point(curX+j, curY)):
                        maxContrastPoint = (curX+j, curY)
                        break
                    elif curY <= 0 or curY >= self.image.height()-1 or curX+j <= 0 or curX+j >= self.image.width()-1:
                        newContrastVal = 255
                    else:
                        arr = self.getSubArray(curX+j,curY,3, 3)
                        newContrastVal = self.calcContrastDy(arr) + self.calcContrastDx(arr)
                if newContrastVal > maxContrastVal:
                    maxContrastVal = newContrastVal
                    if verticalSearch and searchingUp:
                        maxContrastPoint = (curX, curY-j)
                    elif verticalSearch and not searchingUp:
                        maxContrastPoint = (curX, curY+j)
                    elif not verticalSearch and searchingLeft:
                        maxContrastPoint = (curX-j, curY)
                    elif not verticalSearch and not searchingLeft:
                        maxContrastPoint = (curX+j, curY)
                        
            newPoints.append(maxContrastPoint)
        unmodified = False
        while not unmodified:
            unmodified = True
            smothedPoints = []
            for i in range(0, len(newPoints)-1):
                point0 = newPoints[i-1]
                point1 = newPoints[i]
                point2 = newPoints[i+1]
                vector1 = (point0[0]-point1[0],point0[1]-point1[1])
                vector2 = (point2[0]-point1[0],point2[1]-point1[1])

                vectorL2Norm = (math.sqrt(pow(vector1[0],2)+pow(vector1[1],2))*math.sqrt(pow(vector2[0],2)+pow(vector2[1],2)))
                angle = 0
                if vectorL2Norm != 0:
                    val = (vector1[0]*vector2[0]+vector1[1]*vector2[1])/vectorL2Norm
                    if val >= -1 and val <= 1:
                        angle = math.acos(val)
                
                if angle < math.pi/2:
                    unmodified = False
                else:
                    smothedPoints.append(point1)
            newPoints = smothedPoints
        convexHullSelectionPolygon = Polygon(newPoints).convex_hull
        selectionPolygon = unary_union(Polygon(newPoints).buffer(0))
        while hasattr(selectionPolygon, 'geoms') and len(selectionPolygon.geoms) != 1:
            selectionPolygon = unary_union(selectionPolygon.buffer(0.5))
        self.newPoints = newPoints#list(selectionPolygon.exterior.coords)
        selectionPolygon = selectionPolygon.intersection(convexHullSelectionPolygon)
        resultingGeom = unary_union(self.segmentation_tree_selection_unary_union + [selectionPolygon])
        self.segmentation_tree_selection_unary_union = []
        if resultingGeom.is_empty:
            pass
        elif resultingGeom.geom_type != "Polygon":
            self.segmentation_tree_selection_unary_union += list(resultingGeom.geoms)
        else:
            self.segmentation_tree_selection_unary_union += [resultingGeom]
        self.numFinalSelectionPolygons = len(self.segmentation_tree_selection_unary_union)
        self.updateNumFinalSelectionPolygonsLabel.emit()

    def selectedSegmentContainsPoint(self, point):
        for segment in self.segmentation_tree_selection_unary_union:
            if segment.contains(point):
                return True

        return False
    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if (
            self.double_click == "close"
            and self.canCloseShape()
            and len(self.current) > 3
        ):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                min(0, self.pixmap.width() - o2.x()),
                min(0, self.pixmap.height() - o2.y()),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintSegTreeSelectionUnaryUnion(self, painter):
        for polygon in self.segmentation_tree_selection_unary_union:
            if polygon != None:
                pen = QtGui.QPen(self.selectionToolColor)
                # Try using integer sizes for smoother drawing(?)
                pen.setWidth(max(1, int(round(2.0 / self.scale))))
                painter.setPen(pen)

                line_path = QtGui.QPainterPath()
                line_path.moveTo(QtCore.QPointF(list(polygon.exterior.coords)[0][0],list(polygon.exterior.coords)[0][1]))

                for i, p in enumerate(list(polygon.exterior.coords)):
                    line_path.lineTo(QtCore.QPointF(p[0],p[1]))
                line_path.lineTo(QtCore.QPointF(list(polygon.exterior.coords)[0][0],list(polygon.exterior.coords)[0][1]))

                painter.drawPath(line_path)
                painter.fillPath(line_path, self.selectionToolColor)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)

        if not self.selecting() and not self.editing():
        # draw crosshair
            if (
                self._crosshair[self._createMode]
                and self.drawing()
                and self.prevMovePoint
                and not self.outOfPixmap(self.prevMovePoint)
            ):
                p.setPen(QtGui.QColor(0, 0, 0))
                p.drawLine(
                    0,
                    int(self.prevMovePoint.y()),
                    self.image.width() - 1,
                    int(self.prevMovePoint.y()),
                )
                p.drawLine(
                    int(self.prevMovePoint.x()),
                    0,
                    int(self.prevMovePoint.x()),
                    self.image.height() - 1,
                )

        Shape.scale = self.scale

        if self.selecting():
            if self.selectMode == self.SEGMENTATION_TREE:
                self.segmentation_tree.paintContrastLevel(p, self.contrast_levels[self.contrast_level_index], self.selectionToolColor)
            elif (
                 self.selectMode == self.BORDER_SELECTION or
                 self.selectMode == self.MASK_SEGMENT_SELECTION or
                 self.selectMode == self.MASK_SEGMENT_DESELECTION or
                 self.selectMode == self.MASK_ADDITION or
                 self.selectMode == self.MASK_REMOVAL or
                 self.selectMode == self.MAX_CONTRAST_BOUNDARY
                ):
                pen = p.pen()
                save_pen = pen
                pen.setColor(QtGui.QColor(0, 255, 0, 255))
                p.setPen(pen)
                if self.adjustingStencilSize:
                    p.drawEllipse(self.stencilSizeAdjustmentStartingPoint[0] - self.tempStencilSize/2, self.stencilSizeAdjustmentStartingPoint[1] - self.tempStencilSize/2, self.tempStencilSize, self.tempStencilSize)
                else:
                    p.drawEllipse(int(self.prevMovePoint.x() - self.contour_editing_radius/2), int(self.prevMovePoint.y() - self.contour_editing_radius/2), self.contour_editing_radius, self.contour_editing_radius)
                if self.contouring:
                    pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 255), self.contour_editing_radius, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                    p.setPen(pen)
                    for i in range(0, len(self.contour_points)-1):
                        p.drawLine(self.contour_points[i], self.contour_points[i+1])

                self.paintSegTreeSelectionUnaryUnion(p)
            # if self.selectMode == self.MAX_CONTRAST_BOUNDARY:
            #     for i in range(0, len(self.newPoints)-1):
            #             p.drawLine(QtCore.QPointF(*self.newPoints[i]), QtCore.QPointF(*self.newPoints[i+1]))
            #     pen = p.pen()
            #     pen.setColor(QtGui.QColor(255, 0, 0, 255))
            #     p.setPen(pen)
            #     for point in self.newPoints:
            #         p.drawPoint(QtCore.QPointF(*point))
                # if not self.contouring:
                #     points = [(i.x(), i.y()) for i in self.contour_points]
                #     contour = LineString(points).buffer(self.contour_editing_radius/2)
                #     contour_points = list(contour.exterior.coords)
                #     for i in range(0,len(contour_points)-1):
                #         if abs(contour_points[i+1][0] - contour_points[i-1][0]) >= abs(contour_points[i+1][1] - contour_points[i-1][1]):
                #             if contour.contains(Point(contour_points[i][0], contour_points[i][1]-1)):
                #                 pen = p.pen()
                #                 pen.setColor(QtGui.QColor(255, 0, 0, 255))
                #                 p.setPen(pen)
                #                 p.drawPoint(QtCore.QPointF(contour_points[i][0], contour_points[i][1]))
                #             else:
                #                 pen = p.pen()
                #                 pen.setColor(QtGui.QColor(0, 255, 0, 255))
                #                 p.setPen(pen)
                #                 p.drawPoint(QtCore.QPointF(contour_points[i][0], contour_points[i][1]))
                #         else:
                #             if contour.contains(Point(contour_points[i][0]-1, contour_points[i][1])):
                #                 pen = p.pen()
                #                 pen.setColor(QtGui.QColor(0, 0, 255, 255))
                #                 p.setPen(pen)
                #                 p.drawPoint(QtCore.QPointF(contour_points[i][0], contour_points[i][1]))
                #             else:
                #                 pen = p.pen()
                #                 pen.setColor(QtGui.QColor(255, 255, 0, 255))
                #                 p.setPen(pen)
                #                 p.drawPoint(QtCore.QPointF(contour_points[i][0], contour_points[i][1]))

        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(
                shape
            ):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        if (
            self.fillDrawing()
            and self.createMode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.paint(p)

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        assert self.current
        self.current.close()
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def adjustContourRadius(self, scroll_delta):
        self.contour_editing_radius += int(scroll_delta * 0.1)
        if self.contour_editing_radius < 0:
            self.contour_editing_radius = 5
        self.update()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            elif QtCore.Qt.NoModifier == int(mods):
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
            elif QtCore.Qt.ShiftModifier == int(mods) and (self.selectMode == self.BORDER_SELECTION or self.selectMode == self.MASK_SEGMENT_SELECTION or self.selectMode == self.MASK_SEGMENT_DESELECTION or self.selectMode == self.MASK_ADDITION or self.selectMode == self.MASK_REMOVAL or self.selectMode == self.MAX_CONTRAST_BOUNDARY):
                self.adjustContourRadius(delta.y())
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                elif QtCore.Qt.NoModifier == int(mods):
                    self.scrollRequest.emit(ev.delta(), QtCore.Qt.Vertical)
                elif QtCore.Qt.AltModifier == int(mods):
                    self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)

            if QtCore.Qt.ShiftModifier == int(mods) and (self.selectMode == self.BORDER_SELECTION or self.selectMode == self.MASK_SEGMENT_SELECTION or self.selectMode == self.MASK_SEGMENT_DESELECTION or self.selectMode == self.MASK_ADDITION or self.selectMode == self.MASK_REMOVAL or self.selectMode == self.MAX_CONTRAST_BOUNDARY):
                self.adjustContourRadius(ev.delta())
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))
        if self.selecting():
            if key == QtCore.Qt.Key_Escape:
                self.clearSegTreeSelection()
                self.update()
            elif key == QtCore.Qt.Key_Return:
                if self.numFinalSelectionPolygons == 0:
                    return
                if self.numFinalSelectionPolygons > 1:
                    ret = self.msgBox.exec()
                    if ret == QtWidgets.QMessageBox.No:
                        return
                for polygon in self.segmentation_tree_selection_unary_union:
                    new_shape = Shape(shape_type="polygon")
                    points = list(polygon.simplify(0.4).exterior.coords)
                    for point in points:
                        new_shape.addPoint(QtCore.QPoint(point[0], point[1]))
                    new_shape.close()
                    self.shapes.append(new_shape)
                    self.storeShapes()
                    self.setHiding(False)
                    self.newShape.emit()
                self.clearSegTreeSelection()
                self.update()
            elif key == QtCore.Qt.Key_Q:
                if self.selectMode == self.SEGMENTATION_TREE and self.contrast_level_index > 0:
                    self.contrast_level_index -= 1
                    self.updateContrastLevelIndexTextBox.emit()
                    self.update()
            elif key == QtCore.Qt.Key_E:
                if self.selectMode == self.SEGMENTATION_TREE and self.contrast_level_index < len(self.contrast_levels)-1:
                    self.contrast_level_index += 1
                    self.updateContrastLevelIndexTextBox.emit()
                    self.update()
            elif key == QtCore.Qt.Key_W:
                pass

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                index = self.shapes.index(self.selectedShapes[0])
                if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode == "point":
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        self.update()

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()
