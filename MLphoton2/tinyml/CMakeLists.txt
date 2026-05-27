set(TINYML_LIB_ROOT "${CMAKE_CURRENT_SOURCE_FILE_DIR}/lib/cmsis-dsp")

# 2. Add the include directories so CMake can find "arm_math.h" and "dsp/none.h"
target_include_directories(${PROJECT_NAME} PUBLIC 
    "${TINYML_LIB_ROOT}/Include"
    "${TINYML_LIB_ROOT}/Include/dsp"
)

# 3. Ensure the Cortex-M4/M33 DSP define is active
target_compile_definitions(${PROJECT_NAME} PUBLIC ARM_MATH_CM4)