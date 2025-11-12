cmake_minimum_required(VERSION 3.10)
project(HandwrittenDigitRecognition)

set(CMAKE_CXX_STANDARD 11)

# Ручное указание путей OpenCV
set(OPENCV_PATH "D:/opencv")
set(OPENCV_INCLUDE_DIR "${OPENCV_PATH}/build/include")
set(OPENCV_LIB_DIR "${OPENCV_PATH}/build/x64/vc16/lib")

# Включение директорий
include_directories(${OPENCV_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

# Создание исполняемого файла
add_executable(digit_recognition
    src/main.cpp
    src/neural_network.cpp
    src/image_processor.cpp
    src/drawing_interface.cpp  # Добавляем новый файл
)

# Линковка
target_link_libraries(digit_recognition ${OPENCV_LIB_DIR}/opencv_world4110.lib)