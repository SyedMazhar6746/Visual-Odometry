cmake_minimum_required(VERSION 3.10)

project(visualodometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set the path to matplotlib-cpp header
# set(MATPLOTLIB_CPP_INCLUDE_DIR "../matplotlib-cpp")

find_package(OpenCV REQUIRED)

# Find Python
find_package(PythonLibs REQUIRED)

add_subdirectory(matplotplusplus)

include_directories(${PYTHON_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}) #${MATPLOTLIB_CPP_INCLUDE_DIR}

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
                    ${OpenCV_LIBS}
                    ${PYTHON_LIBRARIES}
                    matplot)