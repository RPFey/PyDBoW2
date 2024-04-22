cmake_minimum_required(VERSION 2.8)
project(_C)

add_subdirectory(thirdparty/pybind11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h  
  DBoW2/ORBextractor.h   
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp 
  DBoW2/ORBextractor.cpp    
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp
  ext.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

# Find OpenCV Package
set(OpenCV_DIR "/home/wen/Projects/pyDBoW3/install/opencv3/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

pybind11_add_module(${PROJECT_NAME} ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

