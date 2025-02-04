# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(VideoReview)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV 4.10 REQUIRED)

# Find Tesseract package
set(TESSERACT_INCLUDE_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/include")
set(TESSERACT_LIBRARY_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/lib")
include_directories(${TESSERACT_INCLUDE_DIR})
link_directories(${TESSERACT_LIBRARY_DIR})

# Install the tessdata folder with the shared library
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tessdata
        DESTINATION ${CMAKE_BINARY_DIR})

# Add the shared library
add_library(video_review SHARED main.cpp)

# Link OpenCV libraries
target_link_libraries(video_review ${OpenCV_LIBS} tesseract)

# Include OpenCV headers
target_include_directories(video_review PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set the output directory for the shared library
set_target_properties(video_review PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
