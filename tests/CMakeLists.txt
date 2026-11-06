add_executable(test_module_0 test_module_0.cpp)
target_link_libraries(test_module_0 PRIVATE tensorlite_core)
add_test(NAME test_module_0 COMMAND test_module_0)

add_executable(test_module_1 test_module_1.cpp)
target_link_libraries(test_module_1 PRIVATE tensorlite_core)
add_test(NAME test_module_1 COMMAND test_module_1)

add_executable(test_module_2 test_module_2.cpp)
target_link_libraries(test_module_2 PRIVATE tensorlite_core)
add_test(NAME test_module_2 COMMAND test_module_2 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_module_3 test_module_3.cpp)
target_link_libraries(test_module_3 PRIVATE tensorlite_core)
add_test(NAME test_module_3 COMMAND test_module_3)

add_executable(test_module_4 test_module_4.cpp)
target_link_libraries(test_module_4 PRIVATE tensorlite_core)
add_test(NAME test_module_4 COMMAND test_module_4)

add_executable(test_module_5 test_module_5.cpp)
target_link_libraries(test_module_5 PRIVATE tensorlite_core)
add_test(NAME test_module_5 COMMAND test_module_5)

add_executable(test_module_6 test_module_6.cpp)
target_link_libraries(test_module_6 PRIVATE tensorlite_core)
add_test(NAME test_module_6 COMMAND test_module_6 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_module_7 test_module_7.cpp)
target_link_libraries(test_module_7 PRIVATE tensorlite_core)
add_test(NAME test_module_7 COMMAND test_module_7)

add_executable(test_module_8 test_module_8.cpp)
target_link_libraries(test_module_8 PRIVATE tensorlite_core)
add_test(NAME test_module_8 COMMAND test_module_8)

add_executable(test_module_9 test_module_9.cpp)
target_link_libraries(test_module_9 PRIVATE tensorlite_core)
add_test(NAME test_module_9 COMMAND test_module_9 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_module_10 test_module_10.cpp)
target_link_libraries(test_module_10 PRIVATE tensorlite_core)
add_test(NAME test_module_10 COMMAND test_module_10 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
