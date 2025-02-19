@echo off
rmdir /s /q foreign\build
mkdir foreign\build
cd foreign\build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_PREFIX_PATH=D:/vcpkg/installed/x64-windows -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..\..

copy foreign\build\Release\ windows\runner\resources\dlls
