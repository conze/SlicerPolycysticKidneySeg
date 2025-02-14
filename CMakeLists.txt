cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(PolycysticKidneySeg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/conze/SlicerPKDIA")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Pierre-Henri Conze (IMT Atlantique), Jonathan Bouyer (Kitware SAS)")
set(EXTENSION_DESCRIPTION "This extension allows the use of the PKDIA neural network in 3DSlicer.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/conze/SlicerPKDIA/main/PolycysticKidneySeg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/conze/SlicerPKDIA/main/Screenshot_01.png")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PolycysticKidneySeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
