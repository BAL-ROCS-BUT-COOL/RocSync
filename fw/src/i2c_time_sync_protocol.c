#include "i2c_time_sync_protocol.h"

uint8_t i2c_time_sync_crc8(const uint8_t *data, size_t len)
{
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++)
    {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
    return crc;
}

void i2c_time_sync_pack(uint32_t board_time_ms, uint8_t out[I2C_TIME_SYNC_PAYLOAD_LEN])
{
    out[0] = I2C_TIME_SYNC_FRAME_MARKER;
    out[1] = (uint8_t)(board_time_ms >> 0);
    out[2] = (uint8_t)(board_time_ms >> 8);
    out[3] = (uint8_t)(board_time_ms >> 16);
    out[4] = (uint8_t)(board_time_ms >> 24);
    out[5] = i2c_time_sync_crc8(out, 5);
}
