cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerPKDIA)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/conze/slicer-pkdia")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Pierre-Henri Conze (IMT Atlantique), Jonathan Bouyer (Kitware SAS)")
set(EXTENSION_DESCRIPTION "This extension allows the use of the PKDIA neural network in 3DSlicer.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SlicerPKDIA.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/conze/slicer-pkdia/blob/main/Screenshot_01.png?raw=true")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerPKDIA)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
