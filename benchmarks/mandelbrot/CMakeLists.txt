set(TARGET_NAME mandelbrot)
set(SOURCE_FILE mandelbrot.cpp)
set(CPU_KERNEL_SRC cpu_kernel.cpp)
set(FPGA_KERNEL_SRC fpga_kernel.cpp)

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
set(FPGA_COMPILE_FLAGS "-fintelfpga")
set(FPGA_LINK_FLAGS "-fintelfpga")
set(EMULATOR_COMPILE_FLAGS "${FPGA_COMPILE_FLAGS} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "${FPGA_LINK_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "${FPGA_COMPILE_FLAGS}")
set(HARDWARE_LINK_FLAGS "${FPGA_LINK_FLAGS}")
set(FPGA_IMAGE_FLAGS "${FPGA_LINK_FLAGS} -Xshardware -Xstarget=${FPGA_DEVICE} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation


# Compile the CPU kernel
add_library(CPU_KERNEL OBJECT ${CPU_KERNEL_SRC})

# Compile the FPGA kernel for emulator device
set(FPGA_EMU_KERNEL fgpa_emu_kernel)
add_library(FPGA_EMU_KERNEL OBJECT ${FPGA_KERNEL_SRC})
set_target_properties(FPGA_EMU_KERNEL PROPERTIES 
    COMPILE_FLAGS "${FPGA_COMPILE_FLAGS}"
)

# Compile and link the FPGA image
add_executable(FPGA_IMAGE.a EXCLUDE_FROM_ALL ${FPGA_KERNEL_SRC})
set_target_properties(FPGA_IMAGE.a PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(FPGA_IMAGE.a PROPERTIES LINK_FLAGS "${FPGA_IMAGE_FLAGS} -fsycl-link=image")
add_custom_target(fpga_image_library DEPENDS FPGA_IMAGE.a)

# Add the custom library of the FPGA image 
add_library(FPGA_IMAGE STATIC IMPORTED GLOBAL)
set_target_properties(FPGA_IMAGE PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/FPGA_IMAGE.a)


###############################################################################
### FPGA Emulator
###############################################################################
# CMake executes:
#    [compile cpu kernel]  icpx -fsycl -Wall cpu_kernel.cpp -c -o cpu_kernel.o
#    [compile fpga kernel] icpx -fsycl -fintelfpga -Wall fpga_kernel.cpp -c -o fpga_emu_kernel.o
#    [compile host code]   icpx -fsycl -Wall -DFPGA_EMULATOR mandelbrot.cpp -c -o mandelbrot.o
#    [link fat binary]     icpx -fsycl -fintelfpga -lpthread mandelbrot.o cpu_kernel.o fpga_emu_kernel.a -o mandelbrot.fpga_emu

add_executable(${EMULATOR_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
target_link_libraries(${EMULATOR_TARGET} FPGA_EMU_KERNEL CPU_KERNEL)
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})





###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> -fsycl-link=early fpga_kernel.cpp -o mandelbrot_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${FPGA_KERNEL_SRC})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE} )
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${FPGA_IMAGE_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®




###############################################################################
### FPGA Hardware
###############################################################################
# CMake executes:
#   [compile cpu kernel]  icpx -fsycl -Wall cpu_kernel.cpp -c -o cpu_kernel.o
#   [compile fpga kernel] icpx -fsycl -fintelfpga -Wall fpga_kernel.cpp -c -o fpga_kernel.o
#   [link fpga kernel]    icpx -fsycl -fintelfpga -Xshardware -Xstarget=<FPGA_DEVICE> fpga_kernel.o -o fpga_image.a
#   [compile host code]   icpx -fsycl -Wall -fintelfpga mandelbrot.cpp -c -o mandelbrot.o
#   [link fat binary]     icpx -fsycl -fintelfpga -Wall mandelbrot.o fpga_image.a cpu_kernel.o -o mandelbrot.fpga
add_executable(${FPGA_TARGET} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
add_dependencies(${FPGA_TARGET} fpga_image_library)
target_link_libraries(${FPGA_TARGET} FPGA_IMAGE CPU_KERNEL)
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS}")
#
# End of SECTION 2
#

