#ifndef I2C_TIME_SYNC_H
#define I2C_TIME_SYNC_H

#include <stdint.h>

// Brings up I2C1 (PC1=SDA, PC2=SCL, REV2 only) as a 100kHz master. Call once during
// setup, after setup_gpio() has enabled the GPIOC/AFIO clocks. wrap_ms is the value at
// which board time wraps to 0.
void i2c_time_sync_init(uint32_t wrap_ms);

// Call from the main loop with the live board-time counter. Broadcasts a time-sync frame
// over I2C general call at most once every 100ms, starting it the moment the counter
// takes the value the frame carries; a cheap no-op otherwise. When a broadcast is due it
// blocks for up to 1ms, waiting for the next increment.
void i2c_time_sync_poll(const volatile uint32_t *board_time_ms);

#endif
