import tempfile
import traceback
from pathlib import Path

import qt
import slicer
from slicer.i18n import tr as _  # noqa

from .pkdia.utils.modality import ModalityEnum


class Widget(qt.QWidget):
    def __init__(self, segmentationLogic, installLogic, doShowInfoWindows=True, parent=None):
        super().__init__(parent)

        self.logic = segmentationLogic
        self.installLogic = installLogic
        self._doShowErrorWindows = doShowInfoWindows

        self.installLogic.progressInfo.connect(self.onProgressInfo)

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        uiWidget = slicer.util.loadUI(self.resourcePath().joinpath("UI/PolycysticKidneySeg.ui").as_posix())
        uiWidget.setMRMLScene(slicer.mrmlScene)
        layout.addWidget(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        for modality in ModalityEnum:
            self.ui.modalityComboBox.addItem(modality.value)

        self.ui.installButton.pressed.connect(self.onInstall)
        self.ui.applyButton.pressed.connect(self.onApply)
        self.ui.inputVolumeComboBox.setMRMLScene(slicer.mrmlScene)

    @staticmethod
    def resourcePath() -> Path:
        return Path(__file__).parent.joinpath("..", "Resources")

    def _setButtonsEnabled(self, isEnabled):
        self.ui.installButton.setEnabled(isEnabled)
        self.ui.applyButton.setEnabled(isEnabled)
        self.ui.inputVolumeComboBox.setEnabled(isEnabled)
        self.ui.modalityComboBox.setEnabled(isEnabled)

    def onInstall(self, *, doReportFinished=True):
        self._setButtonsEnabled(False)

        if doReportFinished:
            self.ui.logTextEdit.clear()

        success = self.installLogic.setupPythonRequirements()
        if doReportFinished:
            if success:
                self._reportFinished("Install finished correctly.")
            else:
                self._reportError("Install failed.")
        self._setButtonsEnabled(True)
        return success

    def _reportError(self, msg, doTraceback=True):
        translatedMsg = _(msg)
        self.onProgressInfo(translatedMsg)
        if self._doShowErrorWindows:
            all_msgs = (translatedMsg,) if not doTraceback else (translatedMsg, traceback.format_exc())
            slicer.util.errorDisplay(*all_msgs)

    def _reportFinished(self, msg):
        translatedMsg = _(msg)
        self.onProgressInfo("*" * 80)
        self.onProgressInfo(translatedMsg)
        if self._doShowErrorWindows:
            slicer.util.infoDisplay(translatedMsg)

    def onProgressInfo(self, infoMsg):
        translatedMsg = _(infoMsg)
        self.ui.logTextEdit.insertPlainText(self._formatMsg(translatedMsg) + "\n")
        self.moveTextEditToEnd(self.ui.logTextEdit)
        slicer.app.processEvents()

    @staticmethod
    def _formatMsg(infoMsg):
        return "\n".join([msg for msg in infoMsg.strip().splitlines()])

    @staticmethod
    def moveTextEditToEnd(textEdit):
        textEdit.verticalScrollBar().setValue(textEdit.verticalScrollBar().maximum)

    def onApply(self):
        self._setButtonsEnabled(False)
        errorMessage = None

        modality = self.getModality()
        inputVolume = self.getInputVolume()

        if inputVolume is None:
            errorMessage = "Invalid input volume"
        if modality not in ModalityEnum:
            errorMessage = "Invalid modality"
        if not self.installLogic.areRequirementsInstalled():
            errorMessage = "Missing dependencies. Please install necesary dependencies."
        if errorMessage is not None:
            self._reportError(errorMessage)
        else:
            try:
                with tempfile.TemporaryDirectory() as tempDirPath:
                    self.ui.logTextEdit.clear()
                    self.onProgressInfo("Start")
                    self.onProgressInfo("*" * 80)

                    inputFileName = "volume.nii.gz"
                    inputFilePath = Path(tempDirPath) / inputFileName
                    slicer.util.saveNode(inputVolume, str(inputFilePath))
                    self.onProgressInfo("Loading inference results...")
                    segmentationNode = self.logic.applySegmentation(str(inputFilePath), tempDirPath, modality)
                    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
                    self._reportFinished("Inference ended successfully.")
            except RuntimeError as e:
                self._reportError(f"Inference ended in error:\n{e}")
        self._setButtonsEnabled(True)

    def getInputVolume(self):
        return self.ui.inputVolumeComboBox.currentNode()

    def getModality(self):
        return self.ui.modalityComboBox.currentText
