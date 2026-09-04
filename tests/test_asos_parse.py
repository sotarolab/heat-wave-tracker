"""Parsing of IEM's CSV into the observation frame, including the observed
regime variables (wind, gusts, sky cover, ceiling, pressure)."""
import math

from src.heat.asos import parse_iem_csv

CSV = """#DEBUG: comment line
station,valid,tmpf,dwpf,relh,sknt,gust,drct,mslp,alti,vsby,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes
DCA,2026-09-03 21:52,95.0,72.0,48.0,8.0,M,240.0,1013.2,29.92,10.0,FEW,BKN,M,M,6000.0,12000.0,M,M,M
DCA,2026-09-03 22:52,93.0,71.0,49.0,3.0,17.0,M,M,29.90,6.0,CLR,M,M,M,M,M,M,M,TS
DCA,2026-09-03 23:52,M,70.0,50.0,2.0,M,M,M,29.89,10.0,OVC,M,M,M,2500.0,M,M,M,M
"""


def test_core_columns_and_row_filter():
    df = parse_iem_csv(CSV)
    assert len(df) == 2                       # row with missing tmpf dropped
    assert math.isclose(df.temp_c.iloc[0], (95 - 32) * 5 / 9)
    assert df.wind_spd_kt.iloc[1] == 3.0


def test_gust_dir_pressure():
    df = parse_iem_csv(CSV)
    assert math.isnan(df.wind_gust_kt.iloc[0]) and df.wind_gust_kt.iloc[1] == 17.0
    assert df.wind_dir_deg.iloc[0] == 240.0
    assert math.isclose(df.pressure_hpa.iloc[0], 1013.2)                    # mslp preferred
    assert math.isclose(df.pressure_hpa.iloc[1], 29.90 * 33.8639, rel_tol=1e-6)  # altimeter fallback


def test_sky_cover_and_ceiling():
    df = parse_iem_csv(CSV)
    assert df.sky_cover.iloc[0] == 0.75       # max over FEW + BKN
    assert df.ceiling_ft.iloc[0] == 12000.0   # BKN layer height, not the FEW one
    assert df.sky_cover.iloc[1] == 0.0 and math.isnan(df.ceiling_ft.iloc[1])


def test_wx_codes_missing_is_none():
    df = parse_iem_csv(CSV)
    assert df.wx_codes.iloc[0] is None and df.wx_codes.iloc[1] == "TS"


def test_empty_and_headerless():
    assert parse_iem_csv("").empty
    assert parse_iem_csv("station,foo\nDCA,1").empty
