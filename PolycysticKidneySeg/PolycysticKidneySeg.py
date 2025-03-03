import os

import slicer
from slicer.i18n import tr as _  # noqa
from slicer.i18n import translate  # noqa
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from SlicerPKDIALib.InstallLogic import InstallLogic
from SlicerPKDIALib.SegmentationLogic import SegmentationLogic
from SlicerPKDIALib.Widget import Widget


class PolycysticKidneySeg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PolycysticKidneySeg")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Pierre-Henri Conze (IMT Atlantique)", "Jonathan Bouyer (Kitware SAS), Thibault Pelletier (Kitware SAS), Julien Finet (Kitware SAS)"]

        self.parent.helpText = _(
            """
            This extension allows the use of PKDIA neural networks in 3DSlicer.<br><br>
            These neural networks perform polycystic kidney segmentation in T2 MR and CT imaging modalities.<br>
            The segmentation is performed on a volume loaded into 3DSlicer, and provides the results as a 3DSlicer segmentation.<br><br>
            Test files can be downloaded on the <a href="https://github.com/conze/SlicerPolycysticKidneySeg/releases/">releases page</a>.
            """
        )
        self.parent.acknowledgementText = _(
            "This work was funded by the "
            '<a href="https://www.sfndt.org/">Société Francophone de Néphrologie, Dialyse et Transplantation (SFNDT)</a>.<br>'
            "Models were developed and trained by Pierre-Henri Conze (IMT Atlantique) using imaging data from the Genkyst cohort."
            "Please refer to the following paper: P.-H. Conze et al., Dual-task kidney MR segmentation with Transformers in autosomal-dominant polycystic kidney disease. Computerized Medical Imaging and Graphics, 2024."
        )

        slicer.app.connect("startupCompleted()", self.registerSampleData)

    def registerSampleData(self):
        import SampleData

        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="PolycysticKidneySeg Sample Files",
            sampleName="T2PolycysticKidney",
            thumbnailFileName=os.path.join(iconsPath, "T2SampleData.png"),
            uris="https://github.com/conze/SlicerPolycysticKidneySeg/blob/main/PolycysticKidneySeg/Resources/SampleFiles/T2SampleFile.nii.gz?raw=true",
            fileNames="T2SampleFile.nii.gz",
            checksums="SHA256:293f172a02a19e41c27bf992819447bd2e5c90e61c0f2906d92edc0fd05ffb95",
            nodeNames="T2PolycysticKidney",
        )

        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            # Category and sample name displayed in Sample Data module
            category="PolycysticKidneySeg Sample Files",
            sampleName="CTPolycysticKidney",
            thumbnailFileName=os.path.join(iconsPath, "CTSampleData.png"),
            uris="https://github.com/conze/SlicerPolycysticKidneySeg/blob/main/PolycysticKidneySeg/Resources/SampleFiles/CTSampleFile.nii.gz?raw=true",
            fileNames="CTSampleFile.nii.gz",
            checksums="SHA256:45eb4ef81672c4686442b4982f91635900bff2c6b7e6144b5f847a8031f873ff",
            nodeNames="CTPolycysticKidney",
        )


class PolycysticKidneySegWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)

        self.widget = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        self.widget = Widget(SegmentationLogic(), InstallLogic())
        self.layout.addWidget(self.widget)


class PolycysticKidneySegTest(ScriptedLoadableModuleTest):
    def runTest(self):
        from pathlib import Path

        try:
            from SlicerPythonTestRunnerLib import (
                RunnerLogic,
                RunSettings,
                isRunningInTestMode,
            )
        except ImportError:
            slicer.util.warningDisplay("Please install SlicerPythonTestRunner extension to run the self tests.")
            return
        InstallLogic().setupPythonRequirements()

        currentDirTest = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            currentDirTest,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"]),
            doRunInSubProcess=not isRunningInTestMode(),
        )

        if results.failuresNumber:
            raise AssertionError(f"Test failed: \n{results.getFailingCasesString()}")

        slicer.util.delayDisplay(f"Tests OK. {results.getSummaryString()}")
