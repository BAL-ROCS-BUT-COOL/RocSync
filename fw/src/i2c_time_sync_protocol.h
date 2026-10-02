#ifndef I2C_TIME_SYNC_PROTOCOL_H
#define I2C_TIME_SYNC_PROTOCOL_H

// Wire format for RocSync's I2C board-time broadcast. See I2C_PROTOCOL.md for the
// full specification. Kept free of any MCU/register dependency so it can be built
// and unit-tested on the host, independently of the I2C1 peripheral driver that uses it.

#include <stddef.h>
#include <stdint.h>

#define I2C_TIME_SYNC_GENERAL_CALL_ADDR 0x00
#define I2C_TIME_SYNC_FRAME_MARKER 0xA5
#define I2C_TIME_SYNC_PAYLOAD_LEN 6 // marker(1) + board_time_ms, little-endian(4) + crc8(1)

#ifdef __cplusplus
extern "C"
{
#endif

    // CRC-8/CCITT: poly 0x07, init 0x00, no reflection, no final XOR.
    uint8_t i2c_time_sync_crc8(const uint8_t *data, size_t len);

    // Packs one broadcast frame (marker + board_time_ms LE + crc8) into out[I2C_TIME_SYNC_PAYLOAD_LEN].
    void i2c_time_sync_pack(uint32_t board_time_ms, uint8_t out[I2C_TIME_SYNC_PAYLOAD_LEN]);

    // Next board_time_ms value, wrapping to 0 at wrap_ms. No division, so it is cheap
    // enough to call from a millisecond tick interrupt.
    static inline uint32_t i2c_time_sync_advance(uint32_t board_time_ms, uint32_t wrap_ms)
    {
        uint32_t next = board_time_ms + 1;
        return (next >= wrap_ms) ? 0 : next;
    }

#ifdef __cplusplus
}
#endif

#endif
