#!/usr/bin/env bash
# Cut a recording down to a benchmark clip via downsampling while preserving the original per-frame timestamps

set -euo pipefail

# One ffprobe value, without the padding ffprobe's csv output puts around it: a trailing
# empty field prints as a comma, and a value that holds one comes back quoted.
probe() {
    local value
    value=$(ffprobe -v error "$@")
    value=${value%%$'\n'*}
    value=${value//\"/}
    printf '%s' "${value%,}"
}

if [ $# -ne 3 ]; then
    echo "Usage: $(basename "$0") <input_video> <fps> <output_video>" >&2
    echo "  fps: frames per second to keep, e.g. 1.9" >&2
    exit 1
fi

input=$1
fps=$2
output=$3

if [ ! -f "$input" ]; then
    echo "No such file: $input" >&2
    exit 1
fi
if [ -e "$output" ]; then
    echo "Refusing to overwrite $output" >&2
    exit 1
fi

# A clip cut from a clip inherits the rate of the recording behind both, not the rate of
# the input, which decimation has already divided down.
source_rate=$(probe -show_entries format_tags=source_frame_rate -of csv=p=0 "$input")
if [ -z "$source_rate" ]; then
    source_rate=$(probe -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$input")
fi
if [ -z "$source_rate" ] || [ "$source_rate" = "0/0" ]; then
    echo "Could not read a frame rate from $input" >&2
    exit 1
fi

ffmpeg -hide_banner -loglevel error -i "$input" \
    -vf "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1/$fps)'" \
    -fps_mode passthrough -enc_time_base -1 \
    -c:v libx264 -crf 18 -preset slow -an \
    -metadata source_frame_rate="$source_rate" -movflags use_metadata_tags \
    "$output"

# The tag is what the benchmark reads, so a muxer that dropped it fails here rather than
# silently loosening every tolerance the clip is later held to.
written=$(probe -show_entries format_tags=source_frame_rate -of csv=p=0 "$output")
if [ "$written" != "$source_rate" ]; then
    echo "$output: source_frame_rate tag reads '$written', expected '$source_rate'" >&2
    rm -f "$output"  # a clip that failed its own check is not one to keep or to retry around
    exit 1
fi

frames=$(probe -select_streams v:0 -count_packets -show_entries stream=nb_read_packets \
    -of csv=p=0 "$output")
echo "$output: $frames frames kept, source_frame_rate=$source_rate"
