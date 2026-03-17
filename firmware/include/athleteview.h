#ifndef ATHLETEVIEW_H
#define ATHLETEVIEW_H

#include <stdint.h>
#include <stdbool.h>

/* SmartPatch Hardware Configuration */
#define SMARTPATCH_VERSION      "2.0.0"
#define IMX577_MIPI_LANES       4
#define IMX577_MAX_WIDTH        4056
#define IMX577_MAX_HEIGHT       3040
#define IMX577_DEFAULT_FPS      30

/* MAX86141 PPG Sensor */
#define MAX86141_SPI_SPEED      4000000  /* 4 MHz */
#define MAX86141_FIFO_DEPTH     128
#define PPG_SAMPLE_RATE         100      /* Hz */
#define PPG_LED_CURRENT_MA      20       /* Default LED drive current */

/* MAX30208 Temperature */
#define MAX30208_I2C_ADDR       0x50
#define TEMP_RESOLUTION         0.005f   /* °C per LSB */

/* BME280 Environment */
#define BME280_I2C_ADDR         0x76
#define BME280_SAMPLE_RATE      10       /* Hz */

/* ICM-42688-P IMU */
#define ICM42688_SPI_SPEED      8000000  /* 8 MHz */
#define IMU_SAMPLE_RATE         200      /* Hz */
#define IMU_ACCEL_RANGE         16       /* ±16g */
#define IMU_GYRO_RANGE          2000     /* ±2000 °/s */

/* Biometric Data Structures */
typedef struct {
    uint16_t heart_rate;     /* BPM */
    float    spo2;           /* % */
    float    hrv_rmssd;      /* ms */
    float    body_temp;      /* °C */
    float    humidity;        /* % RH */
    float    pressure;        /* hPa */
    float    accel[3];        /* m/s² (x, y, z) */
    float    gyro[3];         /* °/s (x, y, z) */
    uint64_t timestamp_us;
} biometric_packet_t;

/* Stream Status */
typedef enum {
    STREAM_IDLE = 0,
    STREAM_CONNECTING,
    STREAM_LIVE,
    STREAM_PAUSED,
    STREAM_ERROR
} stream_status_t;

/* Function prototypes */
int  imx577_init(void);
int  imx577_start_capture(uint32_t width, uint32_t height, uint32_t fps);
int  max86141_init(void);
int  max86141_read_fifo(uint32_t *data, uint8_t *count);
int  max30208_read_temp(float *temp);
int  bme280_read(float *temp, float *humidity, float *pressure);
int  icm42688_init(void);
int  icm42688_read(float *accel, float *gyro);
int  srt_stream_init(const char *server, uint16_t port);
int  srt_stream_send(const uint8_t *data, uint32_t len);
void ppg_process(const uint32_t *raw, uint8_t count, biometric_packet_t *out);
void imu_fusion_update(const float *accel, const float *gyro, float dt);
int  battery_get_level(void);

#endif /* ATHLETEVIEW_H */
