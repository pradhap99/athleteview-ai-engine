/**
 * AthleteView SmartPatch Firmware — Main Entry Point
 * 
 * Rockchip RV1106 | Sony IMX577 | MAX86141 | MAX30208 | BME280 | ICM-42688-P
 * 
 * Boot sequence: Init sensors → Calibrate → Connect → Stream
 */
#include "athleteview.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static biometric_packet_t g_bio_packet;
static stream_status_t    g_stream_status = STREAM_IDLE;
static volatile bool      g_running = true;

static void sensor_task(void) {
    uint32_t ppg_raw[MAX86141_FIFO_DEPTH];
    uint8_t  ppg_count = 0;
    float    accel[3], gyro[3];
    float    temp, humidity, pressure;

    /* Read PPG sensor FIFO */
    if (max86141_read_fifo(ppg_raw, &ppg_count) == 0 && ppg_count > 0) {
        ppg_process(ppg_raw, ppg_count, &g_bio_packet);
    }

    /* Read temperature */
    if (max30208_read_temp(&temp) == 0) {
        g_bio_packet.body_temp = temp;
    }

    /* Read environment */
    if (bme280_read(&temp, &humidity, &pressure) == 0) {
        g_bio_packet.humidity = humidity;
        g_bio_packet.pressure = pressure;
    }

    /* Read IMU */
    if (icm42688_read(accel, gyro) == 0) {
        memcpy(g_bio_packet.accel, accel, sizeof(accel));
        memcpy(g_bio_packet.gyro, gyro, sizeof(gyro));
        imu_fusion_update(accel, gyro, 1.0f / IMU_SAMPLE_RATE);
    }
}

int main(void) {
    printf("AthleteView SmartPatch v%s booting...\n", SMARTPATCH_VERSION);

    /* Initialize all sensors */
    if (imx577_init() != 0) { printf("ERROR: IMX577 init failed\n"); return -1; }
    if (max86141_init() != 0) { printf("ERROR: MAX86141 init failed\n"); return -1; }
    if (icm42688_init() != 0) { printf("ERROR: ICM-42688 init failed\n"); return -1; }

    printf("All sensors initialized\n");

    /* Start camera capture */
    if (imx577_start_capture(3840, 2160, 30) != 0) {
        printf("ERROR: Camera capture start failed\n");
        return -1;
    }
    printf("Camera: 4K@30fps capture started\n");

    /* Connect to streaming server */
    if (srt_stream_init("ingest.athleteview.ai", 9000) == 0) {
        g_stream_status = STREAM_LIVE;
        printf("SRT stream: CONNECTED\n");
    } else {
        g_stream_status = STREAM_ERROR;
        printf("WARNING: SRT connection failed, buffering locally\n");
    }

    /* Main loop */
    printf("Entering main loop (sensor rate: %d Hz)\n", PPG_SAMPLE_RATE);
    while (g_running) {
        sensor_task();

        /* Check battery */
        int battery = battery_get_level();
        if (battery < 5) {
            printf("CRITICAL: Battery < 5%%, initiating safe shutdown\n");
            g_running = false;
        }

        usleep(1000000 / PPG_SAMPLE_RATE);  /* Sensor sampling interval */
    }

    printf("SmartPatch shutting down\n");
    return 0;
}
