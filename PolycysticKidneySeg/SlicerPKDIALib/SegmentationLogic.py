from pathlib import Path

import slicer

from .pkdia.utils.modality import ModalityEnum


class SegmentationLogic:
    def __init__(self):
        fileDir = Path(__file__).parent
        self.weightsDir = fileDir.parent / "weights"
        self.weightsPaths = {
            ModalityEnum.T2: self.weightsDir / "PKDIAv1-weights.pth",
            ModalityEnum.CT: self.weightsDir / "PKDIAv2-weights.pth",
        }

        self.segmentColors = [(0.7, 0.4, 0.3), (0.8, 0.3, 0.3)]

    def areWeightsFound(self):
        for weightPath in self.weightsPaths.values():
            if not weightPath.exists():
                return False
        return True

    def applySegmentation(self, inputFilePath, outputFolder, modality):
        from .pkdia.PKDIA import applyPKDIA

        weightsPath = self.weightsPaths[modality]
        predLKPath, predRKPath, _, _ = applyPKDIA(inputFilePath, outputFolder, modality, weightsPath)
        return self.generateSegmentationNodes(predLKPath, predRKPath)

    def generateSegmentationNodes(self, predLKPath, predRKPath):
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "PKDIASegmentation")
        segmentationNode.CreateDefaultDisplayNodes()

        self._loadSegment(predLKPath, segmentationNode, self.segmentColors[0], "Left Kidney", "Segment_1")
        self._loadSegment(predRKPath, segmentationNode, self.segmentColors[1], "Right Kidney", "Segment_2")

        return segmentationNode

    def _loadSegment(self, dataPath, segmentationNode, segmentColor, segmentName, segmentID):
        sourceSegmentationNode = slicer.util.loadSegmentation(dataPath)
        sourceSegmentation = sourceSegmentationNode.GetSegmentation()
        sourceSegmentID = sourceSegmentation.GetSegmentIDs()[0]
        segment = sourceSegmentation.GetSegment(sourceSegmentID)
        segment.SetColor(segmentColor)
        segment.SetName(segmentName)
        segmentationNode.GetSegmentation().AddSegment(segment, segmentID)
        slicer.mrmlScene.RemoveNode(sourceSegmentationNode)
