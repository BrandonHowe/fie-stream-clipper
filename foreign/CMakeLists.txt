# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(VideoReview)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "/MT")
set(CMAKE_MSVC_RUNTIME_LIBRARY "Multi-threaded(/MT)") # Optional, but recommended
set(CMAKE_GENERATOR_PLATFORM "x64")
set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_BUILD_TYPE Release)

# Find OpenCV package
if(WIN32)
    set(OpenCV_DIR "D:\\OpenCV\\opencv\\build\\")
endif()
find_package(OpenCV 4.10 REQUIRED)

# Install the tessdata folder with the shared library
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tessdata
        DESTINATION ${CMAKE_BINARY_DIR})

# Add the shared library
add_library(video_review SHARED main.cpp)

# Include OpenCV headers
target_include_directories(video_review PRIVATE ${OpenCV_INCLUDE_DIRS})

# Find Tesseract package
if(WIN32)
    # set(SW_BUILD_SHARED_LIBS 1)
    # set(DEPENDENCIES
    #     org.sw.demo.google.tesseract.libtesseract-master
    #     org.sw.demo.intel.opencv.imgproc
    #     org.sw.demo.intel.opencv.highgui
    # )

    # find_package(SW REQUIRED)
    # sw_add_package(${DEPENDENCIES})
    # sw_execute()
    set(TESSERACT_INCLUDE_DIR "D:/vcpkg/installed/x64-windows-static/include")
    set(Tesseract_DIR "D:/vcpkg/installed/x64-windows-static/share/tesseract")
    # target_include_directories(video_review PUBLIC ${TESSERACT_INCLUDE_DIR})
    # target_link_libraries(video_review PRIVATE ${OpenCV_LIBS} ${DEPENDENCIES})

    find_package(Tesseract CONFIG REQUIRED)
    IF (${Tesseract_FOUND})
        MESSAGE(STATUS "Found Tesseract.")
    ELSE (${Tesseract_FOUND})
        MESSAGE(STATUS "Could not locate Tesseract.")
    ENDIF (${Tesseract_FOUND})
    # target_include_directories(video_review PUBLIC "D:/vcpkg/installed/x64-windows-static/include")
    
    message(STATUS "Tesseract Include Dir: ${Tesseract_INCLUDE_DIRS}")
    message(STATUS "Tesseract Library: ${Tesseract_LIBRARIES}")
    message(STATUS "Tesseract Binary Dir: ${Tesseract_BINARY_DIR}")
    message(STATUS "Tesseract Version: ${Tesseract_VERSION}")
    # target_link_libraries(video_review PRIVATE ${OpenCV_LIBS} "D:\\vcpkg\\installed\\x64-windows-static\\lib\\tesseract55.lib")
    target_link_libraries(video_review PRIVATE ${OpenCV_LIBS} Tesseract::libtesseract)
elseif(APPLE)
    set(TESSERACT_INCLUDE_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/include")
    set(TESSERACT_LIBRARY_DIR "/opt/homebrew/Cellar/tesseract/5.5.0/lib")
    include_directories(${TESSERACT_INCLUDE_DIR})
    link_directories(${TESSERACT_LIBRARY_DIR})
    target_link_libraries(video_review ${OpenCV_LIBS} tesseract)
endif()

# Set the output directory for the shared library
set_target_properties(video_review PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
