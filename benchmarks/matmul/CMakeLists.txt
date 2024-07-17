set(TARGET_NAME matmul)
set(SOURCE_FILE matmul.cpp)

#
# SECTION 1
# This section defines rules to create a cpu-gpu make target
# This can safely be removed if your project is only targetting FPGAs
#

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>
add_executable(${TARGET_NAME} ${SOURCE_FILE})
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

#
# End of SECTION 1
#

#
# SECTION 2
# This section defines rules to create the fpga_emu, report and fpga make targets
# This can safely be removed if your project is only targetting CPUs/GPUs
#

# FPGA device selection
if(NOT DEFINED FPGA_DEVICE)
    set(FPGA_DEVICE "Agilex7")
    message(STATUS "FPGA_DEVICE was not specified.\
                \nConfiguring the design to target the default FPGA family (Intel Agilex® 7). \
                \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA device ${FPGA_DEVICE}")
endif()


set(EMULATOR_TARGET ${TARGET_NAME}.fpga_emu)
set(FPGA_TARGET ${TARGET_NAME}.fpga)



# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "-fintelfpga")
set(HARDWARE_COMPILE_FLAGS "-fintelfpga")
set(HARDWARE_LINK_FLAGS "-fintelfpga")
set(FPGA_FLAGS "-fintelfpga -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation



set(CPU_KERNEL cpu_kernel)
set(CPU_KERNEL_SRC cpu_kernel.cpp)
add_library(CPU_KERNEL STATIC ${CPU_KERNEL_SRC})

set(FPGA_KERNEL_SRC fpga_kernel.cpp)

set(FPGA_EMU_KERNEL fgpa_emu_kernel)
add_library(FPGA_EMU_KERNEL STATIC ${FPGA_KERNEL_SRC})
set_target_properties(FPGA_EMU_KERNEL PROPERTIES COMPILE_FLAGS "-fintelfpga")

set(FPGA_KERNEL fpga_kernel.a)
add_library(FPGA_KERNEL_OBJ OBJECT ${FPGA_KERNEL_SRC})
set_target_properties(FPGA_KERNEL_OBJ PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")

set(FPGA_IMAGE ${BENCHMARK}_fpga.a)
set(FPGA_LINK_FLAGS "-fsycl ${FPGA_FLAGS}")
separate_arguments(FPGA_LINK_FLAGS)
# Custom linking command to use icpx with FPGA arguments
add_custom_command(OUTPUT ${FPGA_IMAGE}
    COMMAND ${CMAKE_CXX_COMPILER} ${FPGA_LINK_FLAGS} -fsycl-link=image -o ${CMAKE_BINARY_DIR}/libFPGA_KERNEL.a $<TARGET_OBJECTS:FPGA_KERNEL_OBJ> 
    DEPENDS FPGA_KERNEL_OBJ
    COMMENT "Linking FPGA_KERNEL with icpx"
)

add_custom_target(fpga_image DEPENDS ${FPGA_IMAGE})


###############################################################################
### FPGA Emulator
###############################################################################
# CMake executes:
#    [compile cpu kernel] icpx -fsycl -Wall cpu_kernel.cpp -o cpu_kernel.a
#    [compile fpga kernel] icpx -fsycl -fintelfpga -Wall fpga_kernel.cpp -o fpga_emu_kernel.a
#    [compile and link]    icpx -fsycl -fintelfpga -Wall -DFPGA_EMULATOR matadd.cpp cpu_kernel.a fpga_emu_kernel.a -o matadd.fpga_emu

add_executable(${EMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
target_link_libraries(${EMULATOR_TARGET} FPGA_EMU_KERNEL CPU_KERNEL)
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early <file>.cpp -o <file>_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${FPGA_KERNEL_SRC})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${FPGA_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> <file>.cpp -o <file>.fpga
# CMake executes:
#   [compile] icpx -fsycl -fintelfpga -o <file>.cpp.o -c <file>.cpp
#   [link]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> <file>.cpp.o -o <file>.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET} FPGA_KERNEL)
target_link_libraries(${FPGA_TARGET} ${CMAKE_BINARY_DIR}/libFPGA_KERNEL.a CPU_KERNEL)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS}")

#
# End of SECTION 2
#

