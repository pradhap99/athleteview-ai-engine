# AthleteView SmartPatch Firmware

Embedded firmware for the Rockchip RV1106-based SmartPatch wearable camera.

## Hardware
- SoC: Rockchip RV1106G2 (Cortex-A7 + 0.5 TOPS NPU)
- Camera: Sony IMX577 (12MP, 4K@30fps via MIPI CSI-2)
- PPG: Analog Devices MAX86141 (dual-channel, SPI)
- Temperature: Maxim MAX30208 (I2C)
- Humidity: Bosch BME280 (I2C)
- IMU: TDK ICM-42688-P (SPI)
- Microphone: InvenSense ICS-43434 (I2S)
- Wi-Fi: Realtek RTL8852BE (PCIe)
- BLE: Nordic nRF5340 (SPI)

## Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-rv1106.cmake
make -j$(nproc)
```
