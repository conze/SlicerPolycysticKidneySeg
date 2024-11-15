import unittest

import pytest
import SampleData
import slicer
from SlicerPKDIALib import InstallLogic, SegmentationLogic, Widget


@pytest.mark.slow
class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self._clearScene()

    @staticmethod
    def _clearScene():
        slicer.app.processEvents()
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def test_run_segmentation_logic(self):
        installLogic = InstallLogic()
        self.assertTrue(installLogic.areRequirementsInstalled())
        segmentationLogic = SegmentationLogic()
        widget = Widget(segmentationLogic, installLogic, doShowInfoWindows=False)

        SampleData.SampleDataLogic().downloadMRHead()
        widget.ui.modalityComboBox.setCurrentIndex(0)  # T2

        widget.ui.applyButton.click()

        segmentations = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))
        self.assertEqual(len(segmentations), 1)
        segmentation = segmentations[0]
        self.assertEqual(len(segmentation.GetSegmentation().GetSegmentIDs()), 2)
