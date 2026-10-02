#ifndef I2C_TIME_SYNC_H
#define I2C_TIME_SYNC_H

#include <stdint.h>

// Brings up I2C1 (PC1=SDA, PC2=SCL, REV2 only) as a 100kHz master. Call once during
// setup, after setup_gpio() has enabled the GPIOC/AFIO clocks.
void i2c_time_sync_init(void);

// Call from the main loop with the current board time. Broadcasts a fresh time-sync
// frame over I2C general call at most once every 100ms; a cheap no-op otherwise.
void i2c_time_sync_poll(uint32_t board_time_ms);

#endif
