cmake_minimum_required(VERSION 3.5)

project(PathPlanning)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/PathPlanning")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Rachel Sparks (King's College London)")
set(EXTENSION_DESCRIPTION "PathPlanning UI and helper scripts")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/PathPlanning.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/PathPlanning/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PathPlanner)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
