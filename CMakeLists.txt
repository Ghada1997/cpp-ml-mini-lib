cmake_minimum_required(VERSION 3.16)
project(cpp_ml_mini_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

add_library(ml
    src/dataset.cpp
    src/logistic_regression.cpp
    src/naive_bayes.cpp
    src/kmeans.cpp
    src/metrics.cpp
)

target_include_directories(ml PUBLIC include)

# Example apps
add_executable(train_logistic apps/train_logistic.cpp)
target_link_libraries(train_logistic PRIVATE ml)

add_executable(train_naive_bayes apps/train_naive_bayes.cpp)
target_link_libraries(train_naive_bayes PRIVATE ml)

add_executable(run_kmeans apps/run_kmeans.cpp)
target_link_libraries(run_kmeans PRIVATE ml)


# GoogleTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Tests
add_executable(tests_logistic tests/test_logistic_regression.cpp)
target_link_libraries(tests_logistic PRIVATE ml GTest::gtest_main)

add_executable(tests_naive_bayes tests/test_naive_bayes.cpp)
target_link_libraries(tests_naive_bayes PRIVATE ml GTest::gtest_main)

add_executable(tests_kmeans tests/test_kmeans.cpp)
target_link_libraries(tests_kmeans PRIVATE ml GTest::gtest_main)

add_executable(tests_csv_loader tests/test_csv_loader.cpp)
target_link_libraries(tests_csv_loader PRIVATE ml GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(tests_logistic)
gtest_discover_tests(tests_naive_bayes)
gtest_discover_tests(tests_kmeans)
gtest_discover_tests(tests_csv_loader)
