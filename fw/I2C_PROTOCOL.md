# RocSync I2C Time-Broadcast Protocol

RocSync's board time is normally read optically, by decoding the ring and counter LEDs from a
camera frame (see [`board_profiles.py`](../sw/rocsync/board_profiles.py)). Some sync targets have
no camera pointed at them — an external module driving a speaker to encode a timestamp
acoustically, for example — and need the board time some other way. This document specifies the
I2C interface RocSync (REV2) exposes for that purpose.

It is written for two audiences: this repo's own firmware (the implementation), and whoever builds
a follower module (audio-sync, and later ultrasound-sync) — the normative reference for what a
conformant follower must do, independent of any particular module's own codebase.

## Topology

RocSync is the **sole I2C bus master**. Every add-on module is an I2C slave. This avoids all
multi-master bus arbitration, and keeps the more delicate peripheral mode (slave, with its clock
stretching and address-matching edge cases) on the module side, which typically has a more capable
MCU than RocSync's own CH32V003.

RocSync periodically broadcasts the current board time to **all** followers at once, via I2C
**General Call** (address `0x00`, write). RocSync keeps no address table and does no discovery —
any number of modules, of any type, can listen on the same bus with zero firmware or protocol
change on RocSync's side. This is deliberate: a "what time is it" broadcast is exactly as useful to
an audio-sync module as to a future ultrasound-sync module, so a generic, module-agnostic broadcast
is the right level of abstraction, not something built around one module's needs.

The cost of general call is that RocSync has no way to know whether any given module actually
received a particular broadcast. That's acceptable here because nothing on RocSync's side needs
that visibility — whether a given module's data turned out to be usable is a question for that
module's own firmware and for post-processing, exactly like a camera's sync visibility is today.

**Gotcha for follower implementers:** general call is address `0x00`, a reserved SMBus/I2C address
that most I2C stacks ignore unless a device explicitly opts in (e.g. ESP-IDF's I2C slave driver
needs `broadcast_en` set). A generic I2C bus scanner/sniffer also typically won't surface general
call traffic the way it surfaces addressed traffic — don't take "the scanner sees nothing" as "the
bus is idle."

## Hardware requirement (REV2 only)

RocSync's I2C1 peripheral defaults to **PC1 (SDA) / PC2 (SCL)** on the CH32V003, no AFIO remap
needed. On REV2 these pins are unused by the LED/button firmware and are already routed on the PCB
to an existing 4-pin through-hole header, **J4** (silkscreened "OLED" — apparently laid out for a
never-populated status display, and a dead end today). J4 is currently **not populated** (not in
the assembled BOM) and has **no pull-up resistors** on the SDA/SCL net. 

**REV1 does not have this available** — PC1 is REV1's ring-latch pin, already in use. This protocol
as specified targets REV2 only.

J4 has to have a connector fitted to its pads before it can be used. RocSync itself needs no
pull-ups added: I2C needs exactly one pair on the bus, wherever it sits, so they belong on the
follower module or on a pigtail/adapter that plugs into J4.

- **Value:** 2.2–4.7kΩ on each of SDA and SCL. Fit one pair per bus, not one per module — pairs on
  several modules act in parallel and lower the effective resistance.
- **The CH32V003's internal pull-ups cannot stand in.** They are 35–55kΩ (datasheet, RPU), far
  too weak for standard mode's 1µs maximum rise time over a cable, and the reference manual
  disables them whenever a pin is in an output mode — including the alternate-function open-drain
  mode the I2C1 peripheral requires.
- **Pull up to the module's own 3.3V rail, not to J4's VCC.** J4's VCC is unregulated: the
  charger output goes through the power switch straight to VCC (the TPS7A2025 on the board supplies
  only the TCXO), so it follows the single-cell battery (about 3.0–4.2V) and is higher while
  USB-powered — enough to overdrive a 3.3V-logic follower's pins. A 3.3V bus high is read
  correctly by RocSync across that whole range (the CH32V003's minimum input-high level is about
  1.9V at a full battery, 2.2V at 5.5V), and PC1/PC2 are 5V-tolerant.
- **Speed:** start at 100kHz; revisit once the cable length is known.

**Open item, not resolved here — hardware-file changes are out of scope for this document:** the
final connector choice and cable-length validation, which affect the pull-up value above.

## Precision philosophy

This link is a periodic **sample source**, not a synchronized oscillator. It deliberately does not
try to discipline a follower's local clock in real time. Instead, each broadcast is one fresh,
independent `(local_time, board_time_ms)` correspondence point; a follower should keep a rolling
set of these and reconcile against them statistically, the same way
[`timeline.py`](../sw/rocsync/timeline.py) already fits a RANSAC affine map between a camera's
container timestamps and its decoded board-time samples — sparse, occasionally-wrong observations
in, a robust clock model out. A follower doesn't need to reinvent this; the same approach applies
directly.

This is deliberately not engineered for sub-millisecond link precision, because it doesn't need to
be: whatever sits downstream of the I2C read on a module (I2S/DAC/amplifier buffering) will essentially 
always dominate any jitter this link could introduce. Building a tighter real-time clock link here 
would buy nothing.

## Cadence

RocSync broadcasts roughly every 100ms (once per ring rotation). This is sized for fast acquisition
latency on (re)connection — a module that just powered on gets a usable sample within ~100ms —
**not** for oscillator-drift compensation; 100ms is far too short a window for crystal drift to
matter at the precision this system already targets everywhere else.

Broadcast cadence jitter is harmless: a late broadcast still carries an exactly-correct value,
snapshotted fresh at send time. Don't read anything into the spacing between broadcasts beyond
"a follower is stale if it hasn't seen one in a while."

## Wire format

```
START | 0x00 (general call, write) | 0xA5 | time[0..3] (little-endian) | crc8 | STOP
```

| Field | Size | Value |
|---|---|---|
| I2C address + R/W | 1 byte | `0x00` (general call, write) |
| Frame marker | 1 byte | `0xA5` |
| `board_time_ms` | 4 bytes | `uint32_t`, little-endian |
| CRC-8 | 1 byte | over the marker + timestamp bytes (5 bytes) |

`board_time_ms` is on the exact same epoch and wraparound as the value a camera decodes optically:
`counter * PERIOD + ring_index`, wrapping at `2^COUNTER_BITS * PERIOD` ms (currently ≈29.1h on
REV2). A follower's decoded audio timestamp and a camera's decoded video timestamp are therefore
directly comparable numbers, with no separate epoch-reconciliation step.

The frame marker reserves room for future frame types without breaking existing followers: a
follower should silently ignore a broadcast whose marker it doesn't recognize, rather than error.

This deliberately deviates from strict I2C-spec general-call semantics, which reserve the byte
following `0x00`+W for `0x06` (reset and program address) or `0x04` (hardware general call). This
is a private bus, not intended to interoperate with generic SMBus peripherals, so that reservation
doesn't apply here — it's a conscious choice, not an oversight.

### CRC-8

CRC-8/CCITT: polynomial `0x07` (x⁸+x²+x¹+1), initial value `0x00`, no input/output reflection, no
final XOR — the plain textbook form, chosen specifically so a from-scratch implementation on a
follower module can reproduce it byte-for-byte without pulling in a CRC library:

```c
static uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
    return crc;
}
```

A CRC is needed independent of general call vs. addressed delivery — I2C's ACK bit proves a byte
was received, not that its content survived the wire intact — and is doubly justified here since an
audio module carries a switching Class-D amplifier (the Louder-ESP32-S3's TAS5805M) as a nearby EMI
source. A follower should verify the CRC and silently discard a mismatching frame; there is no
NACK/retry channel for RocSync to learn about it.

### Reference vectors

Independently cross-checked (Python and the real `i2c_time_sync_protocol.c`, see
`fw/test/test_i2c_time_sync_protocol/`) — useful for checking a from-scratch follower
implementation against, without either side trusting the other's code:

| `board_time_ms` | Bytes (marker, time LE, crc8) |
|---|---|
| `0` | `a5 00 00 00 00 1e` |
| `1` | `a5 01 00 00 00 08` |
| `100` | `a5 64 00 00 00 13` |
| `0x01020304` | `a5 04 03 02 01 d6` |
| `104857599` (REV2 wrap − 1) | `a5 ff ff 3f 06 cc` |

## Master-side firmware behavior

RocSync's I2C1 peripheral runs in master mode only, at 100kHz standard mode, on PC1 (SDA) / PC2
(SCL) — see [`i2c_time_sync.c`](src/i2c_time_sync.c). The broadcast is issued from the main loop
(not from the SysTick ISR that drives LED timing), as a blocking, timeout-guarded transaction; every
wait is bounded so the main loop can never wedge.

**No follower connected is the normal, expected state**, not a fault — most of the time nothing
will be plugged in. The general-call address goes unacknowledged in that case; RocSync detects this
via the hardware acknowledge-failure flag, aborts the transaction promptly (it does not wait out the
full timeout), and simply tries again on the next cycle. Only a genuine bus problem (a stuck SDA/SCL
line, no acknowledge-failure but no progress either) triggers a full peripheral reset.

## Building a follower module

Not implemented in this repo — this section is guidance for whoever writes it.

- Configure the I2C peripheral as a slave listening for general call (e.g. ESP-IDF's
  `i2c_slave_config_t::broadcast_en`).
- Verify the CRC before trusting a frame; discard silently on mismatch.
- Treat a valid frame as one `(local_time, board_time_ms)` correspondence sample, not as a clock to
  synchronize to directly. Keep a rolling window of recent samples and fit them (see "Precision
  philosophy" above) rather than trusting any single sample.
- Ignore an unrecognized frame marker rather than treating it as an error, so a future RocSync
  firmware update that adds a new frame type doesn't break existing followers.

## Non-goals

- **No delivery confirmation.** General call gives RocSync no per-follower acknowledgment
  visibility. This is intentional (see Topology) — nothing on RocSync's side needs it.
- **No real-time clock discipline.** This is a periodic sample source, not a synchronized
  oscillator (see Precision philosophy).
- **No two-way channel, yet.** Nothing here precludes RocSync later also supporting addressed
  reads/writes to a specific follower (it already owns the bus as master) — that would be an
  additive change, not a breaking one, and doesn't need to be designed now.
