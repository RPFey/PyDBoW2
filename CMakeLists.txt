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
  DBoW2/BRIEFextractor.h
  DBoW2/FBrief.h
  DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp 
  DBoW2/BRIEFextractor.cpp
  DBoW2/FBrief.cpp
  DBoW2/ORBextractor.cpp    
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp
  ext.cpp)

# Find OpenCV Package
find_package(OpenCV REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdparty/DLib/include
)

link_directories(
  ${PROJECT_SOURCE_DIR}/thirdparty/DLib/build
)

pybind11_add_module(${PROJECT_NAME} ${SRCS_DBOW2})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} DLib )
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

