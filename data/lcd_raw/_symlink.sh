#!/bin/bash
# Create per-city symlinks pointing into ~/Documents/Weather data/.
# Idempotent — safe to run multiple times.
set -e
SRC="/Users/terrykim/Documents/Weather data"
DST="/Users/terrykim/Documents/SF Weather/data/lcd_raw"

link() {
    local slug="$1" filename="$2"
    if [ -f "${SRC}/${filename}" ]; then
        ln -sf "${SRC}/${filename}" "${DST}/${slug}.csv"
        echo "  ✓ ${slug} → ${filename}"
    else
        echo "  ✗ MISSING source for ${slug}: ${SRC}/${filename}"
    fi
}

link nyc "Central Park NY.csv"
link lax "Los Angeles International Airport.csv"
link mia "Miami International Airport.csv"
link chi "Chicago Midway International Airport.csv"
link phx "Phoenix Sky Harbor International Airport.csv"
link den "Denver International Airport.csv"
link aus "Austin-Bergstrom International Airport.csv"
link atl "Hartsfield-Jackson Atlanta International Airport.csv"
link okc "Will Rogers World OKC Airport.csv"
link bos "Boston Logan International Airport.csv"
link sea "Seattle-tacoma International Airport.csv"
link msp "Minneapolis-Saint Paul International Airport.csv"
link phl "Philadelphia International Airport.csv"
link las "mccarren harry reid international airport las vegas.csv"
link san "San Antonio International Airport.csv"
link dal "Dallas:Forth Worth International Airport.csv"
link dca "Ronald Reagan Washington National Airport.csv"
link msy "Louis Armstrong New Orleans International Airport.csv"
link hou "Houston Hobby Airport.csv"

# SFO uses the existing LCD already in the repo root
ln -sf "../../LCD datas.csv" "${DST}/sfo.csv"
echo "  ✓ sfo → LCD datas.csv (repo root)"
