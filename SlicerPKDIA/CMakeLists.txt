#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerPKDIA)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SlicerPKDIALib/__init__.py
  SlicerPKDIALib/InstallLogic.py
  SlicerPKDIALib/SegmentationLogic.py
  SlicerPKDIALib/Signal.py
  SlicerPKDIALib/Widget.py
  SlicerPKDIALib/pkdia/__init__.py
  SlicerPKDIALib/pkdia/PKDIA.py
  SlicerPKDIALib/pkdia/datasets/__init__.py
  SlicerPKDIALib/pkdia/datasets/dataset_genkyst.py
  SlicerPKDIALib/pkdia/exams/__init__.py
  SlicerPKDIALib/pkdia/exams/exam_genkyst_prod.py
  SlicerPKDIALib/pkdia/manage/__init__.py
  SlicerPKDIALib/pkdia/manage/manage_genkyst.py
  SlicerPKDIALib/pkdia/nets/__init__.py
  SlicerPKDIALib/pkdia/nets/block.py
  SlicerPKDIALib/pkdia/nets/swinv2Unet.py
  SlicerPKDIALib/pkdia/utils/__init__.py
  SlicerPKDIALib/pkdia/utils/modality.py
  SlicerPKDIALib/pkdia/utils.utils.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  SlicerPKDIALib/pkdia/weights/PKDIAv1-weights.pth
  SlicerPKDIALib/pkdia/weights/PKDIAv2-weights.pth
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
