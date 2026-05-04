# Dataset Build Summary

- Hourly grid: **493,569** rows  (1970-01-01 00:00:00 → 2026-04-22 08:00:00)
- Daily SOD: **20,564** rows

**temp_f provenance:**
- lcd: 477,766 (96.80%)
- isd: 15,152 (3.07%)
- <NA>: 593 (0.12%)
- none: 58 (0.01%)

**Outlier clipping:**

| Field | Range | Nulled | After-non-null |
|---|---|---:|---:|
| temp_f | [10.0, 115.0] | 5 | 492,913 |
| dew_f | [-20.0, 80.0] | 1 | 477,708 |
| wetbulb_f | [0.0, 90.0] | 0 | 415,670 |
| rh | [0.0, 100.0] | 6 | 477,695 |
| slp_inhg | [28.5, 31.0] | 0 | 476,994 |
| vis_mi | [0.0, 30.0] | 10,126 | 416,285 |
| wind_dir | [0.0, 360.0] | 0 | 445,666 |
| wind_speed | [0.0, 80.0] | 2 | 480,610 |
| wind_gust | [0.0, 110.0] | 2 | 39,533 |
| p_change | [-2.0, 2.0] | 0 | 143,591 |
| precip_in | [0.0, 6.0] | 0 | 346,189 |

**Hourly column non-null counts (after merge+clip):**

| Column | Non-null | % |
|---|---:|---:|
| lcd_rt | 480,838 | 97.42% |
| dew_f | 477,708 | 96.79% |
| wetbulb_f | 415,670 | 84.22% |
| rh | 477,695 | 96.78% |
| slp_inhg | 476,994 | 96.64% |
| p_change | 143,591 | 29.09% |
| p_tendency | 143,644 | 29.10% |
| vis_mi | 416,285 | 84.34% |
| wind_dir | 445,666 | 90.29% |
| wind_speed | 480,610 | 97.37% |
| wind_gust | 39,533 | 8.01% |
| precip_in | 346,189 | 70.14% |
| isd_rt | 459,554 | 93.11% |
| isd_qc | 459,554 | 93.11% |
| temp_f | 492,913 | 99.87% |
| temp_source | 492,976 | 99.88% |
| cloud_max | 251,276 | 50.91% |
| sky_overcast | 492,976 | 99.88% |
| sky_obscured | 492,976 | 99.88% |