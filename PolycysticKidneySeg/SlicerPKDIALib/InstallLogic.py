import logging
from subprocess import CalledProcessError

import qt
import slicer

from .Signal import Signal


class InstallLogic:
    r"""
    Class responsible for installing dependencies.
    Makes sure that SimpleITK and requests packages are not overwritten during install.
    Makes sure that torch is installed separately by PyTorch module.

    Copied and adapted from:
    https://github.com/KitwareMedical/SlicerNNUnet/blob/main/SlicerNNUnet/SlicerNNUNetLib/InstallLogic.py
    """

    def __init__(self, doAskConfirmation=True):
        self.progressInfo = Signal("str")
        self.doAskConfirmation = doAskConfirmation
        self.needsRestart = False

    def _log(self, text):
        logging.info(text)
        self.progressInfo(text)

    def areRequirementsInstalled(self) -> bool:
        installed = True
        try:
            import numpy  # noqa
        except ImportError:
            installed = False
        try:
            import skimage  # noqa
        except ImportError:
            installed = False
        try:
            import nibabel  # noqa
        except ImportError:
            installed = False
        try:
            import torch  # noqa
        except ImportError:
            installed = False
        try:
            import timm  # noqa
        except ImportError:
            installed = False
        return installed

    def setupPythonRequirements(self) -> bool:
        """
        Setups 3D Slicer's Python environment with the needed dependencies.
        Install will proceed with the best PyTorch version for environment.

        Setup may require 3D Slicer to be restarted to fully proceed.
        """
        try:
            import numpy  # noqa
        except ImportError:
            self.pip_install("numpy")
        try:
            import skimage  # noqa
        except ImportError:
            self.pip_install("scikit-image")
        try:
            import nibabel  # noqa
        except ImportError:
            self.pip_install("nibabel")
        try:
            import torch  # noqa
        except ImportError:
            try:
                self.installPyTorchExtensionAndRestartIfNeeded()
                if self.needsRestart:
                    self._log("Slicer needs to be restarted before continuing install.")
                    return True

                if self.doAskConfirmation:
                    self._requestPermissionToInstallOrRaise()

                self._installPyTorch()
                self._log("Installation completed successfully.")
            except Exception as e:
                self._log(f"Error occurred during install : {e}")
                return False
        try:
            import timm  # noqa
        except ImportError:
            self.pip_install("timm")
        return True

    def _installPyTorch(self) -> None:
        torchLogic = self._getTorchLogic()
        self._log("PyTorch Python package is required. Installing... (it may take several minutes)")
        if torchLogic.installTorch(askConfirmation=False) is None:
            raise RuntimeError(
                "Failed to correctly install PyTorch. PyTorch extension needs to be installed to use this module."
            )

    @classmethod
    def _requestPermissionToInstallOrRaise(cls) -> None:
        """
        Request user permission to install PyTorch.
        """
        ret = qt.QMessageBox.question(
            None,
            "PyTorch about to be installed",
            "PyTorch will be installed to 3D Slicer. "
            "This install can take a few minutes. "
            "Would you like to proceed?",
        )

        if ret == qt.QMessageBox.No:
            raise RuntimeError("Install process was manually canceled by user.")

    def installPyTorchExtensionAndRestartIfNeeded(self):
        """
        Install PytorchUtils if not installed and raises RuntimeError if canceled by user or install was unsuccessful.
        """
        try:
            self._getTorchLogic()
        except RuntimeError:
            if not self.doAskConfirmation:
                raise

            ret = qt.QMessageBox.question(
                None,
                "Pytorch extension not found.",
                "This module requires PyTorch extension. Would you like to install it?\n\n"
                "Slicer will need to be restarted before continuing the install.",
            )
            if ret == qt.QMessageBox.No:
                raise

            self.needsRestart = True
            self.installTorchUtils()

    @staticmethod
    def installTorchUtils() -> None:
        """
        Installs PytorchUtils from server and raises RuntimeError if install was unsuccessful.
        """
        extensionManager = slicer.app.extensionsManagerModel()
        extName = "PyTorch"
        if extensionManager.isExtensionInstalled(extName):
            return

        if not extensionManager.installExtensionFromServer(extName):
            raise RuntimeError("Failed to install PyTorch extension from the servers. " "Manually install to continue.")

    @classmethod
    def _getTorchLogic(cls) -> "PyTorchUtilsLogic":  # noqa
        """
        Returns torch utils logic if available. Otherwise, raise RuntimeError.
        """
        try:
            import PyTorchUtils  # noqa

            return PyTorchUtils.PyTorchUtilsLogic()
        except ModuleNotFoundError:
            raise RuntimeError(
                "This module requires PyTorch extension. "
                "Install it from the Extensions Manager and restart Slicer to continue."
            )

    def pip_install(self, package) -> None:
        """
        Install and log install of input package.
        """
        self._log(f"- Installing {package}...")
        try:
            slicer.util.pip_install(package)
        except CalledProcessError as e:
            self._log(f"Install returned non-zero exit status : {e}. Attempting to continue...")

    def pip_uninstall(self, package) -> None:
        """
        Uninstall and log uninstall of input package.
        """
        self._log(f"- Uninstall {package}...")
        try:
            slicer.util.pip_uninstall(package)
        except CalledProcessError as e:
            self._log(f"Uninstall returned non-zero exit status : {e}. Attempting to continue...")
