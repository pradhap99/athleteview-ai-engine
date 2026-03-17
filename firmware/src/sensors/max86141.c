/**
 * MAX86141 PPG/SpO2 Sensor Driver
 * 
 * Analog Devices MAX86141 — Dual-channel optical AFE
 * Interface: SPI @ 4 MHz
 * Package: 2.048 × 1.848 × 0.4mm WLP
 * Features: 19-bit ADC, 89dB dynamic range, 128-word FIFO
 * 
 * Datasheet: https://www.analog.com/media/en/technical-documentation/data-sheets/max86140-max86141.pdf
 */
#include "athleteview.h"
#include <string.h>

/* MAX86141 Register Map */
#define REG_INT_STATUS1      0x00
#define REG_INT_STATUS2      0x01
#define REG_INT_ENABLE1      0x02
#define REG_INT_ENABLE2      0x03
#define REG_FIFO_WRITE_PTR   0x04
#define REG_FIFO_READ_PTR    0x05
#define REG_OVERFLOW_CTR     0x06
#define REG_FIFO_DATA_CTR    0x07
#define REG_FIFO_DATA        0x08
#define REG_FIFO_CONFIG1     0x09
#define REG_FIFO_CONFIG2     0x0A
#define REG_SYSTEM_CTRL      0x0D
#define REG_PPG_CONFIG1      0x0E
#define REG_PPG_CONFIG2      0x0F
#define REG_PPG_CONFIG3      0x10
#define REG_LED_SEQ1         0x20
#define REG_LED_SEQ2         0x21
#define REG_LED1_PA          0x23  /* LED1 (Green) pulse amplitude */
#define REG_LED2_PA          0x24  /* LED2 (IR) pulse amplitude */
#define REG_LED3_PA          0x25  /* LED3 (Red) pulse amplitude */
#define REG_PART_ID          0xFF

/* Expected Part ID */
#define MAX86141_PART_ID     0x24

/* Extern SPI functions (platform-specific) */
extern int spi_write_reg(uint8_t reg, uint8_t val);
extern int spi_read_reg(uint8_t reg, uint8_t *val);
extern int spi_read_burst(uint8_t reg, uint8_t *buf, uint16_t len);

int max86141_init(void) {
    uint8_t part_id;

    /* Software reset */
    spi_write_reg(REG_SYSTEM_CTRL, 0x01);
    usleep(10000);  /* 10ms reset delay */

    /* Verify Part ID */
    spi_read_reg(REG_PART_ID, &part_id);
    if (part_id != MAX86141_PART_ID) {
        printf("MAX86141: Part ID mismatch (got 0x%02X, expected 0x%02X)\n", part_id, MAX86141_PART_ID);
        return -1;
    }

    /* Configure PPG: 100 sps, 4 pulses per sample, 117.3us pulse width */
    spi_write_reg(REG_PPG_CONFIG1, 0x23);  /* ADC range: 32uA, Sample rate: 100 sps */
    spi_write_reg(REG_PPG_CONFIG2, 0x06);  /* Sample averaging: 4, Pulse width: 117.3us */
    spi_write_reg(REG_PPG_CONFIG3, 0x40);  /* LED settling: 12us */

    /* LED sequence: LED1 (Green), LED2 (IR), LED3 (Red) for HR + SpO2 */
    spi_write_reg(REG_LED_SEQ1, 0x21);     /* Seq1: LED1, Seq2: LED2 */
    spi_write_reg(REG_LED_SEQ2, 0x03);     /* Seq3: LED3 */

    /* Set LED pulse amplitudes */
    spi_write_reg(REG_LED1_PA, PPG_LED_CURRENT_MA * 2);  /* Green: 20mA */
    spi_write_reg(REG_LED2_PA, PPG_LED_CURRENT_MA * 2);  /* IR: 20mA */
    spi_write_reg(REG_LED3_PA, PPG_LED_CURRENT_MA * 2);  /* Red: 20mA */

    /* FIFO config: almost full at 120 samples, rollover enabled */
    spi_write_reg(REG_FIFO_CONFIG1, 0x78);  /* A_FULL threshold = 120 */
    spi_write_reg(REG_FIFO_CONFIG2, 0x02);  /* FIFO rollover enabled */

    /* Enable interrupts: FIFO almost full, new data ready */
    spi_write_reg(REG_INT_ENABLE1, 0xC0);

    /* Enter active mode */
    spi_write_reg(REG_SYSTEM_CTRL, 0x04);  /* LP mode, PPG active */

    printf("MAX86141: Initialized (100 sps, 3-LED, FIFO enabled)\n");
    return 0;
}

int max86141_read_fifo(uint32_t *data, uint8_t *count) {
    uint8_t fifo_count;
    spi_read_reg(REG_FIFO_DATA_CTR, &fifo_count);

    if (fifo_count == 0) {
        *count = 0;
        return 0;
    }

    uint8_t max_read = (fifo_count > MAX86141_FIFO_DEPTH) ? MAX86141_FIFO_DEPTH : fifo_count;
    uint8_t raw[MAX86141_FIFO_DEPTH * 3];  /* 3 bytes per sample (19-bit ADC) */

    spi_read_burst(REG_FIFO_DATA, raw, max_read * 3);

    for (uint8_t i = 0; i < max_read; i++) {
        data[i] = ((uint32_t)raw[i * 3] << 16) | ((uint32_t)raw[i * 3 + 1] << 8) | raw[i * 3 + 2];
        data[i] &= 0x07FFFF;  /* 19-bit mask */
    }

    *count = max_read;
    return 0;
}
