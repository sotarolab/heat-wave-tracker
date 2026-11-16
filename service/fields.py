"""Field images for the map: one PNG per variable and frame, drawn in Python.

The page places each image on a Web Mercator map by its corner coordinates. The GFS grid
is regular in latitude, and Mercator is not, so the rows are resampled to be evenly spaced
in Mercator y before encoding. Without that step a field drawn between 23 and 51 degrees
north sits tens of kilometres off in the middle latitudes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib
import numpy as np
import numpy.typing as npt
from PIL import Image

OPACITY = 0.72
Floats = npt.NDArray[np.floating[Any]]

# NWS heat index risk categories, lower bound in Fahrenheit. Below the first is no
# elevated risk and is left transparent on the map.
RISK_CATEGORIES_F: list[tuple[float, str, str]] = [
    (80.0, "Caution", "#f5c451"),
    (90.0, "Extreme Caution", "#f2994a"),
    (103.0, "Danger", "#e8592e"),
    (125.0, "Extreme Danger", "#d1382a"),
]
NO_RISK = ("No Elevated Risk", "#0ca30c")


@dataclass(frozen=True)
class Variable:
    id: str
    label: str
    source: str  # the dataset variable it is drawn from, in Celsius
    kind: Literal["ramp", "risk"]
    per: Literal["step", "day"]  # one frame per forecast step, or the maximum over a local day
    cmap: str = ""
    vmin_c: float = 0.0
    vmax_c: float = 0.0
    note: str = ""


VARIABLES: dict[str, Variable] = {
    v.id: v
    for v in [
        # The temperature range covers a January cold outbreak and a July heat wave on one scale.
        Variable("t2m", "2 m temperature", "t2m", "ramp", "step", "RdYlBu_r", -20.0, 45.0,
                 "GFS 2 m air temperature"),
        Variable("hi", "Heat index", "hi", "ramp", "step", "YlOrRd", 20.0, 55.0,
                 "what the air feels like with humidity (NWS Rothfusz regression)"),
        Variable("risk", "Heat risk", "hi", "risk", "day", note="NWS heat index risk category at the day's peak"),
    ]
}


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def ramp_stops(var: Variable, n: int = 9) -> list[str]:
    """The colour scale as n hex stops from vmin to vmax: the legend, and the station dots."""
    cmap = matplotlib.colormaps[var.cmap]
    return [matplotlib.colors.to_hex(cmap(k / (n - 1))) for k in range(n)]


def _mercator_y(lat_deg: Floats | float) -> Floats:
    return np.asarray(np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2)), dtype=float)


def edges(lats: Floats, lons: Floats) -> tuple[float, float, float, float]:
    """(south, west, north, east) of the grid's outer cell edges, half a cell past the centres."""
    dlat = abs(float(lats[1] - lats[0])) / 2
    dlon = abs(float(lons[1] - lons[0])) / 2
    return (float(lats.min()) - dlat, float(lons.min()) - dlon, float(lats.max()) + dlat, float(lons.max()) + dlon)


def mercator_rows(lats: Floats, n_out: int) -> npt.NDArray[np.intp]:
    """For each of n_out image rows evenly spaced in Mercator y, north first, the source row to draw.

    `lats` are the grid's cell centres in either order; the result indexes into them.
    """
    south, _, north, _ = edges(lats, np.array([0.0, 1.0]))
    y_n, y_s = float(_mercator_y(north)), float(_mercator_y(south))
    y = y_n - (np.arange(n_out) + 0.5) / n_out * (y_n - y_s)
    lat = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
    return np.asarray(np.abs(lats[None, :] - lat[:, None]).argmin(axis=1), dtype=np.intp)


def _encode(rgba: npt.NDArray[np.uint8]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render(data_c: Floats, lats: Floats, var: Variable) -> bytes:
    """A 2D field in Celsius (rows follow `lats`, columns west to east) as a PNG, north up.

    Missing cells are transparent. A risk field is transparent below the first category.
    """
    rows = mercator_rows(lats, 2 * len(lats))
    data = data_c[rows]
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    valid = ~np.isnan(data)
    if var.kind == "risk":
        for lower_f, _, color in RISK_CATEGORIES_F:
            mask = valid & (data >= f_to_c(lower_f))
            rgba[mask, :3] = [int(color[k : k + 2], 16) for k in (1, 3, 5)]
            rgba[mask, 3] = int(255 * OPACITY)
    else:
        span = var.vmax_c - var.vmin_c
        unit = np.clip((np.where(valid, data, var.vmin_c) - var.vmin_c) / span, 0.0, 1.0)
        rgba[...] = (matplotlib.colormaps[var.cmap](unit) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, int(255 * OPACITY), 0)
    return _encode(rgba)
