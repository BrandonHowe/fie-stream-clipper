# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(VideoReview)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Find OpenCV package
if(WIN32)
    set(OpenCV_DIR "D:\\OpenCV\\opencv\\build\\x64\\vc16\\lib\\")
endif()
find_package(OpenCV 4.10 REQUIRED)

# Install the tessdata folder with the shared library
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tessdata
        DESTINATION ${CMAKE_BINARY_DIR})

# Find Tesseract package
if(WIN32)
    set(Tesseract_DIR "D:/vcpkg/installed/x64-windows/share/tesseract")
    find_package(Tesseract CONFIG REQUIRED)
elseif(APPLE)
    set(TESSERACT_INCLUDE_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/include")
    set(TESSERACT_LIBRARY_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/lib")
endif()

# Add the shared library
add_library(video_review SHARED main.cpp)

# Find ffmpeg

find_package(FFMPEG REQUIRED)
target_include_directories(video_review PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(video_review PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(video_review PRIVATE ${FFMPEG_LIBRARIES})

# Include OpenCV headers
target_include_directories(video_review PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link OpenCV and Tesseract
if(WIN32)
    target_link_libraries(video_review PRIVATE ${OpenCV_LIBS} Tesseract::libtesseract)

    set(OPENCV_DLL_PATH "${OpenCV_DIR}../bin/opencv_world4110.dll")
    if(EXISTS ${OPENCV_DLL_PATH})
        message(STATUS "Copying OpenCV DLL: ${OPENCV_DLL_PATH} to ${CMAKE_BINARY_DIR}/Release")
        add_custom_command(
            TARGET video_review
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${OPENCV_DLL_PATH}"
            "${CMAKE_BINARY_DIR}/Release"
        )
    else()
        message(WARNING "OpenCV DLL not found: ${OPENCV_DLL_PATH}")
    endif()
elseif(APPLE)
    include_directories(${TESSERACT_INCLUDE_DIR})
    link_directories(${TESSERACT_LIBRARY_DIR})
    target_link_libraries(video_review PRIVATE ${OpenCV_LIBS} tesseract)
endif()

# Set the output directory for the shared library
set_target_properties(video_review PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
