cmake_minimum_required(VERSION 3.20)
project(athleteview_firmware C)
set(CMAKE_C_STANDARD 11)

add_executable(smartpatch_fw
    src/main.c
    src/camera/imx577_driver.c
    src/camera/encoder.c
    src/sensors/max86141.c
    src/sensors/max30208.c
    src/sensors/bme280.c
    src/sensors/icm42688p.c
    src/connectivity/srt_stream.c
    src/dsp/ppg_filter.c
    src/dsp/imu_fusion.c
    src/power/battery_mgmt.c
)
target_include_directories(smartpatch_fw PRIVATE include)
