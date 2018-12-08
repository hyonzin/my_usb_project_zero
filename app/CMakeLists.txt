# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
set(PROJECT_PATH /Users/hyeonjin/workspace/android/my_usb_project_zero)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(INCLUDE_PATH ${PROJECT_PATH}/app/src/main/include)

include_directories(${INCLUDE_PATH})
# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
add_library(obs-lib SHARED src/main/cpp/Obs_util.cpp)

add_library(mssd-lib SHARED IMPORTED)
set_target_properties(mssd-lib PROPERTIES IMPORTED_LOCATION ${PROJECT_PATH}/app/src/main/JniLibs/arm64-v8a/libMSSD.so)

add_library(detect-lib SHARED src/main/jni/detect.cpp)
#SET_TARGET_PROPERTIES(detect-lib PROPERTIES LINKER_LANGUAGE CXX)


# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                        detect-lib
                        mssd-lib
                        obs-lib
                       # Links the target library to the log library
                       # included in the NDK.
                        ${log-lib})
