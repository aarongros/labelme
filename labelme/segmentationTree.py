import cv2
import numpy as np
from scipy.io import loadmat
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid, explain_validity
import json
from qtpy import QtCore
from qtpy import QtGui

import subprocess
import os 

def ReadRegionList(generalDataDir, boundaryDataDir):
    data = np.fromfile(generalDataDir, dtype=np.int32)
    num_regions = data[0]
    regions = []
    num_boundary_points = []
    boundary_points_start_index = []
    index_sum = 0
    for i in range(0,num_regions):
        index_sum = index_sum + data[i*3+1]
        boundary_points_start_index.append(index_sum)
        num_boundary_points.append(data[i*3+1])
        regions.append([])
        regions[i].append(int(data[i*3+2]))
        regions[i].append(int(data[i*3+3]))

    array = np.fromfile(boundaryDataDir, dtype=np.int32).reshape(index_sum,2)
    boundary_points = np.split(array, boundary_points_start_index)
    boundary_points = [[[int(boundary_point[1]), int(boundary_point[0])] for boundary_point in boundary_point_list] for boundary_point_list in boundary_points]

    for i in range(0,num_regions):
        regions[i].append(boundary_points[i])

    return regions, num_regions

def createSegTree(image_filename):
# def convertMatToTree(filename):
    # matlab_file = loadmat(filename)
    # key = list(matlab_file.keys())[3]
    # segmentation_tree_array = loadmat(filename)[key][0]
    # num_segments = len(segmentation_tree_array)
    # trees = []
    # contrast_levels = {}
    # for i in range(0,num_segments):
    #     segment_array = segmentation_tree_array[i][0][0]
    #     boundary = segment_array[2]
    #     contrast = int(segment_array[3][0][0])

    #     contrast_levels[contrast] = contrast
    #     if len(boundary) > 3:
    #         boundary = [(coord[0]-1, coord[1]-1) for coord in boundary]
    #         contour = np.flip(boundary, 1)
    #         
    #         if len(contour) < 3:
    #             continue
    #         new_node = SegmentationTree(contour, contrast)

    #         if new_node.polygon.is_valid == False:
    #             new_node.polygon = new_node.polygon.buffer(0.5).simplify(0)
    #             if new_node.polygon.geom_type != "MultiPolygon" and len(new_node.getCoords()) < 3:
    #                 pass
    #         if new_node.polygon.geom_type == "MultiPolygon":
    #             for polygon in list(new_node.polygon.geoms):
    #                 if len(list(polygon.exterior.coords)) < 3:
    #                     continue
    #                 trees.append(SegmentationTree(list(polygon.exterior.coords), contrast))
    #         else:
    #             trees.append(new_node)
    # print("in")
    baseDir = os.getcwd() + "/"
    # print(baseDir)
    image_name = os.path.splitext(os.path.split(image_filename)[1])[0]
    # print(baseDir + "segmentation_tree.exe", image_filename, baseDir + "temp\\", image_name)
    result = subprocess.run([baseDir + "segmentation_tree.out", image_filename, baseDir + "temp/", image_name])
    # print(result.returncode)
    # print("1")
    generalDataDir = baseDir + "temp/"+ image_name + "_GeneralData.bin"
    boundaryDataDir = baseDir + "temp/"+ image_name + "_BoundaryData.bin"
    # print(generalDataDir, boundaryDataDir)
    regions, num_regions = ReadRegionList(generalDataDir, boundaryDataDir)

    trees = []
    contrast_levels = {}
    for i in range(0,num_regions):
        boundary = regions[i][2]
        contrast = regions[i][1]
        contrast_levels[contrast] = contrast
        if len(boundary) < 3:
            continue

        new_node = SegmentationTree(boundary, contrast)
        if new_node.polygon.is_valid == False:
            new_node.polygon = new_node.polygon.buffer(0.5).simplify(0)
            if new_node.polygon.geom_type != "MultiPolygon" and len(new_node.getCoords()) < 3:
                pass
        if new_node.polygon.geom_type == "MultiPolygon":
            for polygon in list(new_node.polygon.geoms):
                if len(list(polygon.exterior.coords)) < 3:
                    continue
                trees.append(SegmentationTree(list(polygon.exterior.coords), contrast))
        else:
            trees.append(new_node)
    # print("2")
    # sort shapes based on area
    is_sorted = True
    prev_area = trees[0].polygon.area
    for i in range(1, len(trees)):
        if trees[i].polygon.area > prev_area:
            # print("ERROR: Unsorted list of trees")
            is_sorted = False
    # print("3")
    if not is_sorted:
        trees_copy = []
        for i in range(0, len(trees)):
            max = -1
            k = 0
            for j in range(0, len(trees)):
                if trees[j] == None:
                    continue

                if max == -1:
                    max = trees[j].polygon.area
                    k = j
                    continue

                if max < trees[j].polygon.area:
                    max = trees[j].polygon.area
                    k = j
            trees_copy.append(trees[k])
            trees[k] = None
        trees = trees_copy

        is_sorted = True
        prev_area = trees[0].polygon.area
        for i in range(1, len(trees)):
            if trees[i].polygon.area > prev_area:
                print("ERROR: Unsorted list of trees")
                is_sorted = False
    # print("4")
    if is_sorted:
        roots = []
        # check containment of shapes to build tree
        for i in range(len(trees)-1,-1,-1):
            parent_found = False
            for j in range(i-1,-1,-1):
                if (trees[j].polygon.contains(trees[i].polygon)):
                    # make child
                    trees[j].children.append(trees[i])
                    parent_found = True
                    break
            if not parent_found:
                roots.append(trees[i])
    rootNode = trees[0]
    if len(roots) > 1:
        polygons = []
        for root in roots:
            polygons.append(root.polygon)
        polygonsCombination = MultiPolygon(polygons)
        rootNode = SegmentationTree(list(polygonsCombination.convex_hull.exterior.coords), -1)
        for root in roots:
            rootNode.children.append(root)
        
    sortedContrastLevels = list(contrast_levels.keys())
    sortedContrastLevels.sort(reverse=True)
    return rootNode, sortedContrastLevels

class SegmentationTree(object):

    scale = 1.0

    def __init__(self, pts=None, contrast_level=None):
        self.polygon = Polygon(pts).simplify(0)
        self.children = []
        self.selected = False
        self.contrast_level = contrast_level
        self.hovered = False
    
    def getCoords(self):
        return list(self.polygon.exterior.coords)

    def selectSegmentsWithPercentAreaAndBoundary(self, selection_polygon):
        try:
            intersection = self.polygon.intersection(selection_polygon)
            if self.polygon.area > 0 and (intersection.area / self.polygon.area) > 0.95:
                segment_points = self.getCoords()
                num_boundary_points = len(segment_points)
                num_boundary_points_enclosed = 0
                for point in segment_points:
                    if selection_polygon.contains(Point(point)):
                        num_boundary_points_enclosed += 1
                
                if (num_boundary_points_enclosed / num_boundary_points) > 0.95:
                    self.selected = True
                    return
        except:
            print('invalid object')
        
        for child in self.children:
                child.selectSegmentsWithPercentAreaAndBoundary(selection_polygon)

    def groupChildSelection(self):
        allChildrenSelected = True
        for child in self.children:
            allChildrenSelected = allChildrenSelected and child.selected
                
        if allChildrenSelected:
            self.removeSelection()
            self.selected = True
            return True
        return False

    def editSegmentSelectionWithVariableWidthContour(self, contour, adding, parent_selected):
        # TODO optimize to not check all segments when parents don't overlap
        if adding and not self.selected:
            intersection = self.polygon.intersection(contour)
            if self.polygon.area > 0 and (intersection.area / self.polygon.area) > 0.9:
                self.removeSelection()
                self.selected = True
                return True
        elif not adding and (self.selected or parent_selected):
            intersection = self.polygon.intersection(contour)
            if self.polygon.area > 0 and (intersection.area / self.polygon.area) > 0.9:
                self.removeSelection()
                return True
        
        modified = False
        modified_children = []
        for child in self.children:
            child_modified = child.editSegmentSelectionWithVariableWidthContour(contour, adding, parent_selected or self.selected)
            modified = modified or child_modified
            if child_modified:
                modified_children.append(child)

        if modified:
            if adding:
                # TODO fix grouping because a segment's children don't always span the entire segment so it could be included even when it shouldn't
                # return self.groupChildSelection()
                return False
            elif not adding and (self.selected or parent_selected):
                for child in self.children:
                    child_unmodified = True
                    # TODO can lead to removing more than intended when a segments children don't span the entire parent's segment since there won't be children to cover the non removed areas
                    for modified_child in modified_children:
                        if modified_child is child:
                            child_unmodified = False
                            break
                    
                    if child_unmodified:
                        child.selected = True

                if self.selected:
                    self.selected = False
                    return False
                return True

        return False
    
    def editSegmentSelectionAtContrastLevel(self, pos, contrastLevel):
        if self.polygon == None or not self.polygon.contains(Point(pos)):
            return

        if self.contrast_level == contrastLevel:
            self.selected = not self.selected
            return

        for child in self.children:
            child.editSegmentSelectionAtContrastLevel(pos, contrastLevel)

    def paintContrastLevel(self, painter, curr_contrast_level, color):
        if ((self.contrast_level == curr_contrast_level) or self.selected) and self.polygon != None:
            pen = QtGui.QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QtGui.QPainterPath()
            line_path.moveTo(QtCore.QPointF(self.getCoords()[0][0],self.getCoords()[0][1]))

            for i, p in enumerate(self.getCoords()):
                line_path.lineTo(QtCore.QPointF(p[0],p[1]))
            line_path.lineTo(QtCore.QPointF(self.getCoords()[0][0],self.getCoords()[0][1]))

            painter.drawPath(line_path)
            if self.hovered or self.selected:
                painter.fillPath(line_path, color)
        
        for child in self.children:
            child.paintContrastLevel(painter, curr_contrast_level, color)

    def removeSelection(self):
        self.selected = False
        for child in self.children:
            child.removeSelection()

    def updateHovering(self, pos):
        if self.polygon.contains(Point(pos)):
            self.hovered = True
        else:
            self.hovered = False
        
        for child in self.children:
            child.updateHovering(pos)

    def collectSelectedSegments(self):
        selectedSegments = []
        if self.selected:
            selectedSegments.append(self.polygon.buffer(0.5))
        for child in self.children:
            selectedSegments += child.collectSelectedSegments()
        return selectedSegments
    
    def convertSegTreeToDictArray(self):
        nodes = [{
            "polygon": self.getCoords(),
            "children": len(self.children),
            "contrast_level": self.contrast_level
        }]
        for child in self.children:
            nodes += child.convertSegTreeToDictArray()
        return nodes

    def convertDictArrayToSegTree(self, dict):
        contrastLevelList = dict[0]
        self.convertDictArrayToSegTreeHelper(dict, 1)
        return contrastLevelList

    def convertDictArrayToSegTreeHelper(self, dict, dictArrayIndex):
        self.polygon = Polygon(dict[dictArrayIndex]["polygon"]).simplify(0)
        self.contrast_level = dict[dictArrayIndex]["contrast_level"]
        num_children = dict[dictArrayIndex]["children"]
        self.children = [None] * num_children
        for childIndex in range(0,num_children):
            new_child = SegmentationTree()
            dictArrayIndex = new_child.convertDictArrayToSegTreeHelper(dict, dictArrayIndex+1)
            self.children[childIndex] = new_child
        return dictArrayIndex
    
    def getSegTreeAsDictArray(self, contrastLevelList):
        return [contrastLevelList] + self.convertSegTreeToDictArray()

    def loadSegTreeFromDictArray(self, dictArraySegTree):
        return self.convertDictArrayToSegTree(dictArraySegTree)

    def createMissingChildren(self):
        newPolygons = []
        childPolygons = [i.polygon for i in self.children]
        childrenUnion = unary_union(childPolygons)
        if childrenUnion.area > 0:
            difference = self.polygon.difference(childrenUnion)
            if difference.geom_type == "Polygon":
                if difference.area > 5:
                    newPolygons.append(difference)
            else:
                for polygon in list(difference.geoms):
                    if polygon.area > 5:
                        newPolygons.append(polygon)
            for polygon in newPolygons:
                self.children.append(SegmentationTree(list(polygon.exterior.coords), self.children[0].contrast_level))

        for child in self.children:
            child.createMissingChildren()

    def selectedSegmentContainsPoint(self, point):
        if self.polygon.contains(point):
            if self.selected:
                return True
        
            for child in self.children:
                if child.selectedSegmentContainsPoint(point):
                    return True
        return False