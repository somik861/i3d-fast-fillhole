cmake_minimum_required(VERSION 3.20)

project(i3d_fast_fillhole)
set(CMAKE_CXX_STANDARD 20)

if(WIN32)
  link_directories(${CMAKE_SOURCE_DIR}/i3dlibs/lib)
  include_directories(${CMAKE_SOURCE_DIR}/i3dlibs/include)
  file(COPY ${CMAKE_SOURCE_DIR}/i3dlibs/bin/cbia.lib.i3dcore.dyn.rel.x64.16.dll
       DESTINATION ${CMAKE_BINARY_DIR})
  file(COPY ${CMAKE_SOURCE_DIR}/i3dlibs/bin/cbia.lib.i3dalgo.dyn.rel.x64.16.dll
       DESTINATION ${CMAKE_BINARY_DIR})
  set(LIBS cbia.lib.i3dcore.dyn.rel.x64.16.lib
           cbia.lib.i3dalgo.dyn.rel.x64.16.lib)

else(WIN32)
  set(LIBS i3dalgo i3dcore)
endif(WIN32)

add_executable(fast_fillhole main.cpp)
target_link_libraries(fast_fillhole ${LIBS})
