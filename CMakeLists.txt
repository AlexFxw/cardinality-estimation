project(guess-sql)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
        src/histogram.cpp
        src/table.cpp
        )

include_directories(src)
include_directories(src/sql-parser/src)
add_subdirectory(src/sql-parser)

add_executable(main src/main.cpp ${SOURCES})
target_link_libraries(main sqlparser)