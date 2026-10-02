// Host-side tests for the MCU-independent parts of the I2C time-sync wire format.
// Run with `pio test -e native` from fw/. See I2C_PROTOCOL.md for the format itself.
// Vectors cross-checked against an independent Python CRC-8 implementation.

#include "i2c_time_sync_protocol.h"
#include <unity.h>

void setUp(void) {}
void tearDown(void) {}

static void test_crc8_empty(void)
{
    TEST_ASSERT_EQUAL_UINT8(0x00, i2c_time_sync_crc8(NULL, 0));
}

static void test_crc8_single_byte(void)
{
    uint8_t marker = I2C_TIME_SYNC_FRAME_MARKER;
    TEST_ASSERT_EQUAL_UINT8(0x72, i2c_time_sync_crc8(&marker, 1));
}

static void test_pack_zero(void)
{
    uint8_t out[I2C_TIME_SYNC_PAYLOAD_LEN];
    uint8_t expected[] = {0xa5, 0x00, 0x00, 0x00, 0x00, 0x1e};
    i2c_time_sync_pack(0, out);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, out, I2C_TIME_SYNC_PAYLOAD_LEN);
}

static void test_pack_is_little_endian(void)
{
    uint8_t out[I2C_TIME_SYNC_PAYLOAD_LEN];
    uint8_t expected[] = {0xa5, 0x04, 0x03, 0x02, 0x01, 0xd6};
    i2c_time_sync_pack(0x01020304, out);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, out, I2C_TIME_SYNC_PAYLOAD_LEN);
}

static void test_pack_rev2_wrap_minus_one(void)
{
    // 2^20 * 100 - 1: the largest value REV2's board_time_ms ever actually carries.
    uint8_t out[I2C_TIME_SYNC_PAYLOAD_LEN];
    uint8_t expected[] = {0xa5, 0xff, 0xff, 0x3f, 0x06, 0xcc};
    i2c_time_sync_pack(104857599u, out);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, out, I2C_TIME_SYNC_PAYLOAD_LEN);
}

static void test_advance_normal_step(void)
{
    TEST_ASSERT_EQUAL_UINT32(6, i2c_time_sync_advance(5, 104));
}

static void test_advance_wraps_at_boundary(void)
{
    TEST_ASSERT_EQUAL_UINT32(0, i2c_time_sync_advance(103, 104));
}

static void test_advance_wraps_at_rev2_boundary(void)
{
    TEST_ASSERT_EQUAL_UINT32(0, i2c_time_sync_advance(104857599u, 104857600u));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_crc8_empty);
    RUN_TEST(test_crc8_single_byte);
    RUN_TEST(test_pack_zero);
    RUN_TEST(test_pack_is_little_endian);
    RUN_TEST(test_pack_rev2_wrap_minus_one);
    RUN_TEST(test_advance_normal_step);
    RUN_TEST(test_advance_wraps_at_boundary);
    RUN_TEST(test_advance_wraps_at_rev2_boundary);
    return UNITY_END();
}
