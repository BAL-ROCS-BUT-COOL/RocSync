#include "i2c_time_sync.h"

#include "ch32fun.h"
#include "i2c_time_sync_protocol.h"

#define I2C_TIME_SYNC_SDA PC1
#define I2C_TIME_SYNC_SCL PC2
#define I2C_TIME_SYNC_CLOCK_HZ 100000
#define I2C_TIME_SYNC_PERIOD_MS 100

// Bounds every wait in a transaction so a missing or mis-cabled module can never wedge
// the main loop. ~10000 loop iterations is on the order of 1-2ms at this MCU's 24MHz
// clock -- generous next to a single I2C byte time (90us at 100kHz) but far short of
// poll_buttons()'s 50ms cadence. Tune on the bench once real hardware exists.
#define I2C_TIME_SYNC_TIMEOUT_LOOPS 10000

// Upper bound on the wait for the next board-time increment, which is 1ms away at most.
// Far more iterations than 1ms takes at 24MHz; it only guards against a stopped SysTick.
#define I2C_TIME_SYNC_EDGE_WAIT_LOOPS 100000

static uint32_t last_broadcast_ms;
static uint32_t board_time_wrap_ms;

static void i2c1_reset_and_configure(void)
{
    // setup_gpio()'s funGpioInitAll() already enabled the GPIOC/AFIO clocks this needs;
    // only the I2C1 peripheral itself is this module's own to bring up.
    RCC->APB1PRSTR |= RCC_APB1Periph_I2C1;
    RCC->APB1PRSTR &= ~RCC_APB1Periph_I2C1;
    RCC->APB1PCENR |= RCC_APB1Periph_I2C1;

    // I2C pins are open-drain, unlike every push-pull pin setup_gpio() configures --
    // the bus is shared, so a pin driving high here would fight another device's low.
    funPinMode(I2C_TIME_SYNC_SDA, GPIO_Speed_10MHz | GPIO_CNF_OUT_OD_AF);
    funPinMode(I2C_TIME_SYNC_SCL, GPIO_Speed_10MHz | GPIO_CNF_OUT_OD_AF);

    uint16_t ctlr2 = I2C1->CTLR2 & (uint16_t)~I2C_CTLR2_FREQ;
    ctlr2 |= (FUNCONF_SYSTEM_CORE_CLOCK / 1000000) & I2C_CTLR2_FREQ;
    I2C1->CTLR2 = ctlr2;

    // Standard mode (<=100kHz): CCR = Fpclk1 / (2 * Fi2c). No DUTY/FS bits -- those
    // only apply to fast mode.
    I2C1->CKCFGR = (FUNCONF_SYSTEM_CORE_CLOCK / (2 * I2C_TIME_SYNC_CLOCK_HZ)) & I2C_CKCFGR_CCR;

    I2C1->CTLR1 |= I2C_CTLR1_PE;
}

void i2c_time_sync_init(uint32_t wrap_ms)
{
    i2c1_reset_and_configure();
    last_broadcast_ms = 0;
    board_time_wrap_ms = wrap_ms;
}

static int i2c_check_event(uint32_t event_mask)
{
    uint32_t status = I2C1->STAR1 | ((uint32_t)I2C1->STAR2 << 16);
    return (status & event_mask) == event_mask;
}

typedef enum
{
    I2C_TIME_SYNC_OK = 0,
    I2C_TIME_SYNC_NO_FOLLOWER, // general call went unacknowledged: normal, nothing plugged in
    I2C_TIME_SYNC_BUS_ERROR,   // a wait timed out: the bus may be stuck
} broadcast_result_t;

static broadcast_result_t broadcast(const uint8_t *payload, int len)
{
    int32_t timeout;

    timeout = I2C_TIME_SYNC_TIMEOUT_LOOPS;
    while ((I2C1->STAR2 & I2C_STAR2_BUSY) && --timeout)
        ;
    if (!timeout)
        return I2C_TIME_SYNC_BUS_ERROR;

    I2C1->CTLR1 |= I2C_CTLR1_START;
    timeout = I2C_TIME_SYNC_TIMEOUT_LOOPS;
    while (!i2c_check_event(I2C_EVENT_MASTER_MODE_SELECT) && --timeout)
        ;
    if (!timeout)
        return I2C_TIME_SYNC_BUS_ERROR;

    // General call address, write direction. Any follower listening for it ACKs
    // regardless of its own address; one with none configured leaves it unacknowledged.
    I2C1->DATAR = I2C_TIME_SYNC_GENERAL_CALL_ADDR << 1;
    timeout = I2C_TIME_SYNC_TIMEOUT_LOOPS;
    while (!(I2C1->STAR1 & I2C_STAR1_AF) &&
           !i2c_check_event(I2C_EVENT_MASTER_TRANSMITTER_MODE_SELECTED) && --timeout)
        ;
    if (I2C1->STAR1 & I2C_STAR1_AF)
    {
        I2C1->STAR1 &= (uint16_t)~I2C_STAR1_AF;
        I2C1->CTLR1 |= I2C_CTLR1_STOP;
        return I2C_TIME_SYNC_NO_FOLLOWER;
    }
    if (!timeout)
        return I2C_TIME_SYNC_BUS_ERROR;

    for (int i = 0; i < len; i++)
    {
        timeout = I2C_TIME_SYNC_TIMEOUT_LOOPS;
        while (!(I2C1->STAR1 & I2C_STAR1_TXE) && --timeout)
            ;
        if (!timeout)
            return I2C_TIME_SYNC_BUS_ERROR;
        I2C1->DATAR = payload[i];
    }

    timeout = I2C_TIME_SYNC_TIMEOUT_LOOPS;
    while (!i2c_check_event(I2C_EVENT_MASTER_BYTE_TRANSMITTED) && --timeout)
        ;
    I2C1->CTLR1 |= I2C_CTLR1_STOP;
    return timeout ? I2C_TIME_SYNC_OK : I2C_TIME_SYNC_BUS_ERROR;
}

void i2c_time_sync_poll(const volatile uint32_t *board_time_ms)
{
    const uint32_t now = *board_time_ms;

    // Unsigned subtraction: correct across board_time_ms's own wraparound the same way
    // it is correct across a plain uint32_t wraparound, except right at the wrap point
    // itself, where it under-counts the elapsed time and triggers one early broadcast --
    // harmless, and rare enough (once per ~29h on REV2) not to warrant extra handling.
    //
    // The frame carries the next counter value, so the gate opens one tick before PERIOD
    // has elapsed: the frame then lands exactly PERIOD after the previous one. A poll that
    // misses that tick (see i2c_time_sync_poll()'s contract) sends the frame later, not
    // earlier, and the period restarts from wherever it went out.
    if (now - last_broadcast_ms < I2C_TIME_SYNC_PERIOD_MS - 1)
        return;

    // The frame is sent the moment board_time_ms takes the value it carries, so a follower
    // that timestamps it learns that instant to within a few microseconds rather than
    // somewhere in a 1ms window. Packing it beforehand leaves only the spin between the
    // increment and the START condition.
    const uint32_t next = i2c_time_sync_advance(now, board_time_wrap_ms);
    uint8_t payload[I2C_TIME_SYNC_PAYLOAD_LEN];
    i2c_time_sync_pack(next, payload);

    // The increment has to be observed by the spin itself. If it already happened while
    // packing, the frame would go out late by an unknown amount, so retry on the next poll.
    if (*board_time_ms != now)
        return;

    uint32_t current = now;
    for (int32_t spins = I2C_TIME_SYNC_EDGE_WAIT_LOOPS; current == now && spins > 0; spins--)
        current = *board_time_ms;
    if (current != next)
        return;

    if (broadcast(payload, I2C_TIME_SYNC_PAYLOAD_LEN) == I2C_TIME_SYNC_BUS_ERROR)
        i2c1_reset_and_configure();

    last_broadcast_ms = next;
}
