set(TARGET_NAME "rap")
set(SOURCE_FILE "rap.cpp")
set(KERNELS_DIR "kernels")
set(CPU_KERNEL_SRC "${KERNELS_DIR}/cpu_kernel_${KERNEL_VERSION}_version.cpp")
set(FPGA_KERNEL_SRC "${KERNELS_DIR}/fpga_kernel_${KERNEL_VERSION}_version.cpp")

#
# SECTION 1
# This section defines rules to create the fpga_emu, report and fpga make targets
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


set(EMULATOR_TARGET "${TARGET_NAME}.fpga_emu")
set(FPGA_TARGET "${TARGET_NAME}.fpga")



# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.


set(FPGA_COMPILE_FLAGS "-fintelfpga")
set(FPGA_LINK_FLAGS "-fintelfpga")
set(EMULATOR_COMPILE_FLAGS "${FPGA_COMPILE_FLAGS} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "${FPGA_LINK_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "${FPGA_COMPILE_FLAGS}")
set(FPGA_BACKEND_FLAGS "${FPGA_LINK_FLAGS} -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")


# Compile the CPU kernel
add_library(CPU_KERNEL OBJECT ${CPU_KERNEL_SRC})

# Compile the FPGA kernel
add_library(FPGA_KERNEL OBJECT ${FPGA_KERNEL_SRC})
set_target_properties(FPGA_KERNEL PROPERTIES
    COMPILE_FLAGS "${FPGA_COMPILE_FLAGS}"
)


###############################################################################
### FPGA Emulator
###############################################################################
# CMake executes:
#    [compile cpu kernel]  icpx -fsycl cpu_kernel.cpp -c -o cpu_kernel.o
#    [compile fpga kernel] icpx -fsycl -fintelfpga fpga_kernel.cpp -c -o fpga_emu_kernel.o
#    [compile host code]   icpx -fsycl -DFPGA_EMULATOR rap.cpp -c -o rap.o
#    [link fat binary]     icpx -fsycl -fintelfpga rap.o cpu_kernel.o fpga_emu_kernel.o -o rap.fpga_emu

add_executable(${EMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
target_link_libraries(${EMULATOR_TARGET} FPGA_KERNEL CPU_KERNEL)
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})


###############################################################################
### Generate Report
###############################################################################
# Cmake executes:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_kernel.cpp -o rap_report.a

set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${FPGA_KERNEL_SRC})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} )
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${FPGA_BACKEND_FLAGS} -fsycl-link=early")


###############################################################################
### FPGA Hardware
###############################################################################
# CMake executes:
#   [compile cpu kernel]  icpx -fsycl cpu_kernel.cpp -c -o cpu_kernel.o
#   [compile host code]   icpx -fsycl -fintelfpga rap.cpp -c -o rap.o
#   [create fpga kernel,
#    create fpga image 
#    and link fat binary] icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -reuse-exe=rap.fpga rap.o fpga_kernel.cpp cpu_kernel.o -o rap.fpga
list(GET INCLUDE_DIRS 0 IDIR0)
list(GET INCLUDE_DIRS 1 IDIR1)
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
target_link_libraries(${FPGA_TARGET} CPU_KERNEL)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "-I${IDIR0} -I${IDIR1} ${FPGA_BACKEND_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/${FPGA_TARGET} ${CMAKE_SOURCE_DIR}/benchmarks/${BENCHMARK}/${FPGA_KERNEL_SRC}")

#
# End of SECTION 1
#

