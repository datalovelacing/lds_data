#%% imports

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def tnow():
    return datetime.now().strftime('%H:%M:%S')

print(tnow())

# %%

TITLE_FONT_SIZE = 25
LABEL_FONT_SIZE = 14       # percentage labels printed on each state
COLORBAR_FONT_SIZE = 14
BASE_FONT_SIZE = 14     # hover text and general default
WHITE_TEXT_THRESHOLD = 0.75

MIN_COLOR="#d18892",
MID_COLOR="#4d9ec9",
MAX_COLOR="#083B8F",

#%%
"""
LDS Church Membership & Congregation Change Maps
==================================================
Generates:
  1. Static PNG  - membership change, 2012 -> 2026, by state
  2. Static PNG  - congregation change, 2012 -> 2026, by state
  3. Static PNG  - membership change, 2025 -> 2026, by state
  4. Interactive HTML - animated map of membership change vs. the 2012
     baseline, year by year, 2012-2026 (slider + play button)

All maps: white state borders, percentage changes printed directly on each state
(not just in the legend/colorbar), diverging pink->white->dark-blue color scale
(pink = decline, white = no change, dark blue = growth), Albers USA projection (AK/HI shown as
insets, same convention as US Census / news-outlet maps).

Crowded Northeast states (VT, NH, MA, RI, CT, NJ, DE, MD) have their labels
pulled out to a callout column with a leader line, instead of being crammed
inside their tiny borders.

Requirements:
    pip install pandas plotly kaleido

Input CSV expected columns (exact names):
    "Total Church Membership", "Congregations", "state", "Year"
"""

# Set working directory to the folder containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Optional local TopoJSON for Kaleido/static PNG export.
# Download once if the Plotly CDN is unavailable, then leave this file beside the script.
LOCAL_USA_TOPOJSON = "usa_110m.json"

# Avoid the flaky CDN fetch (Error 525 etc.) entirely by pointing Kaleido
# at a local copy of the topojson instead.
if os.path.exists(LOCAL_USA_TOPOJSON):
    pio.defaults.topojson = os.path.dirname(os.path.abspath(LOCAL_USA_TOPOJSON))

# %%
# CONFIG - edit these two lines for your setup
CSV_PATH = "LDS_state_membership_statistics.csv"
OUTPUT_DIR = "."  # where PNG/HTML files get written

# Full state name -> USPS 2-letter code (Plotly's "USA-states" locationmode
# needs these; if your CSV already uses 2-letter codes, you can skip the
# state_abbr step below and use `state` directly).
STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

# Approximate geographic centroids (lat, lon) - used to place the on-map
# number labels. Small northeastern states sit close together, so their
# labels can crowd/overlap a bit; that's a limitation of state-shaped maps
# generally, not something the code can fully avoid.
STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AK": (64.2, -149.5), "AZ": (34.2, -111.7), "AR": (34.9, -92.4),
    "CA": (37.2, -119.4), "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5),
    "DC": (38.9, -77.0), "FL": (27.6, -81.5), "GA": (32.7, -83.4), "HI": (20.3, -156.4),
    "ID": (43.7, -114.6), "IL": (40.0, -89.2), "IN": (39.9, -86.3), "IA": (42.0, -93.5),
    "KS": (38.5, -98.0), "KY": (37.5, -85.3), "LA": (30.3, -92.0), "ME": (45.4, -69.2),
    "MD": (39.0, -76.7), "MA": (42.3, -71.8), "MI": (42.8, -84.5), "MN": (46.3, -94.3),
    "MS": (32.7, -89.7), "MO": (38.5, -92.5), "MT": (47.0, -109.6), "NE": (41.5, -99.8),
    "NV": (39.3, -117.0), "NH": (43.7, -71.6), "NJ": (40.1, -74.5), "NM": (34.5, -106.1),
    "NY": (42.9, -75.5), "NC": (35.6, -79.4), "ND": (47.5, -100.5), "OH": (40.3, -82.8),
    "OK": (35.5, -97.5), "OR": (44.0, -120.5), "PA": (40.9, -77.7), "RI": (41.7, -71.5),
    "SC": (33.9, -80.9), "SD": (44.4, -100.3), "TN": (35.8, -86.4), "TX": (31.5, -99.3),
    "UT": (39.3, -111.7), "VT": (44.0, -72.7), "VA": (37.5, -78.8), "WA": (47.4, -120.5),
    "WV": (38.6, -80.6), "WI": (44.6, -89.9), "WY": (43.0, -107.5),
}

# ================================================================
# COLOR SCALE DEFAULTS
# ================================================================
# Each chart can override these individually.
#
# Defaults:
#   min/max       = automatically calculated from the data
#   mid color     = halfway between min and max
#   white         = exactly 0
#   white band    = range / 10 on either side of zero
#   min color     = pink
#   mid color     = turquoise
#   max color     = dark blue
#
# Example chart-specific overrides are shown at the bottom of the file.

COLOR_MIN_COLOR = "#d18892"
COLOR_MID_COLOR = "#4d9ec9"
COLOR_MAX_COLOR = "#083B8F"

# Fraction of the full color range used for the white band.
# 0.10 means the white band extends 10% of the full range
# on either side of zero, subject to the actual min/max limits.
COLOR_WHITE_BAND_FRACTION = 0.10

# Colorbar appearance/position.
COLORBAR_X = 0.82   # smaller = closer to the map; tune by eye
COLORBAR_Y = 0.56
COLORBAR_LEN = 0.45

#%%
def build_diverging_colorscale(
    color_min,
    color_max,
    white_band=None,
    min_color=COLOR_MIN_COLOR,
    mid_color=COLOR_MID_COLOR,
    max_color=COLOR_MAX_COLOR,
    mid_at=None,   # <-- NEW: manually place the mid color at this data value
):
    """
    ...
    mid_at lets you override where the mid_color sits. If omitted, it
    defaults to the numeric midpoint of (color_min, color_max), same as
    before. If provided, it must fall strictly between color_min and
    color_max (and can be on either side of zero).
    """

    if color_min >= color_max:
        raise ValueError("color_min must be less than color_max.")

    if color_min > 0 or color_max < 0:
        raise ValueError(
            "The color range must include zero because zero is always white."
        )

    full_range = color_max - color_min

    if white_band is None:
        white_band = full_range * COLOR_WHITE_BAND_FRACTION

    if white_band < 0:
        raise ValueError("white_band must be >= 0.")

    white_low = max(color_min, -white_band)
    white_high = min(color_max, white_band)

    # Mid color placed at mid_at if given, otherwise the numeric midpoint.
    if mid_at is None:
        mid_value = (color_min + color_max) / 2
    else:
        if not (color_min < mid_at < color_max):
            raise ValueError("mid_at must be strictly between color_min and color_max.")
        mid_value = mid_at

    def frac(value):
        return (value - color_min) / full_range

    stops = [
        [0.0, min_color],
        [frac(white_low), "#FFFFFF"],
        [frac(white_high), "#FFFFFF"],
    ]

    if mid_value > white_high:
        stops.append([frac(mid_value), mid_color])
    elif mid_value < white_low:
        stops.insert(1, [frac(mid_value), mid_color])

    stops.extend([
        [1.0, max_color],
    ])

    stops = sorted(stops, key=lambda x: x[0])

    cleaned = []
    for position, color in stops:
        if cleaned and abs(cleaned[-1][0] - position) < 1e-12:
            cleaned[-1][1] = color
        else:
            cleaned.append([position, color])

    return cleaned

#%%

# States too small/close-packed for their label to fit inside their own
# borders - these get pulled out to a callout column instead.
CROWDED_STATES = ["VT", "NH", "MA", "RI", "CT", "NJ", "DE", "MD"]

# Column of callout positions (lat, lon), ordered north -> south to match
# real geography. If the column gets clipped at your map's edge, nudge

CALLOUT_LATS = {
    "VT": 41.8, "NH": 40.6, "MA": 39.4, "RI": 38.3,
    "CT": 37.1, "NJ": 35.8, "DE": 34.6, "MD": 33.4,
}

CALLOUT_LONS = {
    "VT": -69.3, "NH": -69.8, "MA": -70.3, "RI": -70.8,
    "CT": -71.3, "NJ": -71.8, "DE": -72.3, "MD": -72.8,
}

FONT_FAMILY = "Helvetica, Arial, sans-serif"

# %%
# DATA LOADING / PREP


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.query('country=="United States"')
    df = df.query('state!="District Of Columbia"')
    df["state_abbr"] = df["state"].map(STATE_TO_ABBR)
    missing = df.loc[df["state_abbr"].isna(), "state"].unique()
    if len(missing):
        raise ValueError(
            f"These state names in the CSV didn't match STATE_TO_ABBR: {list(missing)}. "
            "Add them to the dict (or fix the spelling in the CSV)."
        )
    return df


def compute_change(df: pd.DataFrame, value_col: str, year_a: int, year_b: int) -> pd.DataFrame:
    """One row per state: value at year_a, value at year_b, absolute change, pct change."""
    sub = df[df["Year"].isin([year_a, year_b])]
    piv = sub.pivot_table(
        index=["state_abbr", "state"], columns="Year", values=value_col
    ).reset_index()

    piv = piv.dropna(subset=[year_a, year_b])
    dropped = set(sub["state"].unique()) - set(piv["state"])
    if dropped:
        print(tnow(), f"Note: dropped from change calc (missing {year_a} or {year_b} data): {dropped}")

    piv["change"] = piv[year_b] - piv[year_a]
    piv["pct_change"] = (piv["change"] / piv[year_a]) * 100
    return piv


# %%
# ON-MAP LABELS (in-place for most states, callout + leader line for the
# crowded Northeast cluster)

def build_label_traces(data, value_col, label_fmt, label_size,
                        white_text_threshold=WHITE_TEXT_THRESHOLD):
    """
    Text-label traces for the map: crowded Northeast states get moved to
    a callout column with a leader line back to their true location;
    every other state is labeled in place.
    """
    labels = data.copy()
    labels["lat"] = labels["state_abbr"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[0])
    labels["lon"] = labels["state_abbr"].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[1])
    labels["label"] = labels[value_col].apply(lambda v: label_fmt.format(v))

    is_crowded = labels["state_abbr"].isin(CROWDED_STATES)
    plain, crowded = labels[~is_crowded], labels[is_crowded]

    traces = []

    # In-place labels for every non-crowded state (unchanged from original)
    dark = plain[value_col] >= white_text_threshold
    for mask, text_color in [(~dark, "black"), (dark, "white")]:
        sub = plain[mask]
        traces.append(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], text=sub["label"], mode="text",
            textfont=dict(size=label_size, color=text_color, family=FONT_FAMILY),
            showlegend=False, hoverinfo="skip",
        ))

    if len(crowded):
        # Leader lines: one segment per crowded state, true centroid -> callout.
        # `None` breaks the list into separate disconnected segments.
        line_lat, line_lon = [], []
        for _, row in crowded.iterrows():
            line_lat += [row["lat"], CALLOUT_LATS[row["state_abbr"]], None]
            line_lon += [row["lon"], CALLOUT_LONS[row["state_abbr"]], None]
        traces.append(go.Scattergeo(
            lat=line_lat, lon=line_lon, mode="lines",
            line=dict(width=1, color="#888888"),
            showlegend=False, hoverinfo="skip",
        ))

        # Callout labels themselves (state abbr + value, sitting over open water)
        callout_lat = [CALLOUT_LATS[s] for s in crowded["state_abbr"]]
        callout_lon = [CALLOUT_LONS[s] for s in crowded["state_abbr"]]
        callout_text = [f"{s}: {lbl}" for s, lbl in zip(crowded["state_abbr"], crowded["label"])]
        traces.append(go.Scattergeo(
            lat=callout_lat, lon=callout_lon, text=callout_text,
            mode="text", textposition="middle right",
            textfont=dict(size=label_size - 2, color="black", family=FONT_FAMILY),
            showlegend=False, hoverinfo="skip",
        ))

    return traces


# %%
# STATIC CHANGE MAP (single snapshot, e.g. 2012 -> 2026 delta)

def make_static_change_map(
    data: pd.DataFrame,
    title: str,
    colorbar_title: str,
    filename: str,
    value_col: str = "pct_change",
    label_fmt: str = "{:+.1f}%",
    label_size: int = LABEL_FONT_SIZE,
    white_band: float = None,
    color_min: float = None,
    color_max: float = None,
    mid_color_at: float = None,   # <-- NEW
    white_text_threshold: float = WHITE_TEXT_THRESHOLD,
    show: bool = True,
    save_html: bool = False,
    min_color=COLOR_MIN_COLOR,
    mid_color=COLOR_MID_COLOR,
    max_color=COLOR_MAX_COLOR,
):
    if color_min is None:
        color_min = float(data[value_col].min())
    if color_max is None:
        color_max = float(data[value_col].max())

    fig = px.choropleth(
        data,
        locations="state_abbr",
        locationmode="USA-states",
        color=value_col,
        scope="usa",
        color_continuous_scale=build_diverging_colorscale(
            color_min=color_min,
            color_max=color_max,
            white_band=white_band,
            min_color=min_color,
            mid_color=mid_color,
            max_color=max_color,
            mid_at=mid_color_at,   # <-- NEW
        ),
        color_continuous_midpoint=0,
        range_color=(color_min, color_max),
        hover_name="state",
        hover_data={value_col: ":.1f", "state_abbr": False},
    )

    fig.update_traces(marker_line_color="#edf1f5", marker_line_width=1.5)

    for trace in build_label_traces(data, value_col, label_fmt, label_size,
                                     white_text_threshold=white_text_threshold):
        fig.add_trace(trace)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY)),
        font=dict(family=FONT_FAMILY),
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="white",
            landcolor="white",
            bgcolor="white",
            showframe=False,
        ),
        coloraxis_colorbar=dict(
            title=dict(text=colorbar_title, font=dict(size=COLORBAR_FONT_SIZE, family=FONT_FAMILY)),
            orientation="h",        # NEW
            thickness=20,
            len=COLORBAR_LEN,
            x=0.6,                  # NEW — center horizontally
            xanchor="center",       # NEW
            y=0.05,                # NEW — negative = below the map; tune this
            yanchor="top",          # NEW
            tickformat=".0f",
            ticksuffix="%",
            tickfont=dict(size=COLORBAR_FONT_SIZE, family=FONT_FAMILY),
            outlinewidth=0,
        ),
        paper_bgcolor="white",
        #margin=dict(l=0, r=0, t=70, b=0),
        margin=dict(l=0, r=0, t=70, b=60)
    )

    if show:
        fig.show()

    if save_html:
        out_path = f"{OUTPUT_DIR}/{filename}"
        fig.write_html(out_path)
        print(tnow(), f"Saved {out_path}")

    return fig


# %%
# INTERACTIVE ANIMATED MAP (year-by-year slider)


def make_interactive_change_map(
    df: pd.DataFrame,
    value_col: str = "Total Church Membership",
    baseline_year: int = 2012,
    filename: str = "membership_change_interactive.html",
    label_fmt: str = "{:+.1f}%",
    color_min: float = None,
    color_max: float = None,
    white_band: float = None,
    min_color: str = COLOR_MIN_COLOR,
    mid_color: str = COLOR_MID_COLOR,
    max_color: str = COLOR_MAX_COLOR,
    white_text_threshold: float = WHITE_TEXT_THRESHOLD,
    mid_color_at=None,
):
    """
    For each year 2012-2026, plots each state's percentage change in `value_col`
    relative to `baseline_year` (so 2012 itself reads as 0 everywhere,
    and later years show cumulative growth/decline from that starting
    point). Includes a play/slider control and on-map number labels
    that update with the animation.
    """
    df = df.sort_values("Year").copy()
    baseline = df[df["Year"] == baseline_year].set_index("state_abbr")[value_col]
    df["baseline_value"] = df["state_abbr"].map(baseline)
    df["change_from_baseline"] = (
        (df[value_col] - df["baseline_value"]) / df["baseline_value"] * 100
    )
    df = df.dropna(subset=["change_from_baseline"])

    # Automatically use the full animated data range unless overridden.
    if color_min is None:
        color_min = float(df["change_from_baseline"].min())
    if color_max is None:
        color_max = float(df["change_from_baseline"].max())

    fig = px.choropleth(
        df,
        locations="state_abbr",
        locationmode="USA-states",
        color="change_from_baseline",
        scope="usa",
        color_continuous_scale=build_diverging_colorscale(
            color_min=color_min,
            color_max=color_max,
            white_band=white_band,
            min_color=min_color,
            mid_color=mid_color,
            max_color=max_color,
            mid_at=mid_color_at,   # <-- NEW
        ),
        color_continuous_midpoint=0,
        range_color=(color_min, color_max),
        hover_name="state",
        hover_data={"change_from_baseline": ":.1f", value_col: ":,.0f", "state_abbr": False},
        animation_frame="Year",
    )
    fig.update_traces(marker_line_color="#edf1f5", marker_line_width=1)

    # on-map text labels that update per animation frame
    def scattergeo_for_year(year):
        sub = df[df["Year"] == year]
        return build_label_traces(sub, "change_from_baseline", label_fmt, LABEL_FONT_SIZE,
                                   white_text_threshold=white_text_threshold)

    years = sorted(df["Year"].unique())
    initial_label_traces = scattergeo_for_year(years[0])
    for trace in initial_label_traces:
        fig.add_trace(trace)
    for frame in fig.frames:
        year = int(float(frame.name))
        frame.data = list(frame.data) + scattergeo_for_year(year)

    fig.update_layout(
        title=dict(
            text=f"LDS Membership Change vs. {baseline_year} Baseline, by State",
            x=0.5, xanchor="center", font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY),
        ),
        font=dict(family=FONT_FAMILY),
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="white",
            landcolor="white",
            bgcolor="white",
            showframe=False,
        ),
        coloraxis_colorbar=dict(
            title=dict(text=f"Percent change vs. {baseline_year}",
                       font=dict(size=COLORBAR_FONT_SIZE, family=FONT_FAMILY)),
            thickness=20,
            len=COLORBAR_LEN,      # was hardcoded 0.7
            x=COLORBAR_X,          # NEW — pulls it left, closer to the map
            xanchor="left",
            y=COLORBAR_Y,          # NEW
            yanchor="middle",
            tickformat=".0f",
            ticksuffix="%",
            tickfont=dict(size=COLORBAR_FONT_SIZE, family=FONT_FAMILY),
            outlinewidth=0,
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=70, b=0),
    )

    out_path = f"{OUTPUT_DIR}/{filename}"
    fig.write_html(out_path)
    print(tnow(), f"Saved {out_path}")
    return fig


# %%
# Load data
df = load_data(CSV_PATH)

# %%
# 1. Membership change, 2012 -> 2026
#    Example manual range: -6% to +45%.
mem_change = compute_change(df, "Total Church Membership", 2012, 2026)

make_static_change_map(
    mem_change,
    title="LDS Church Membership Change, 2012\u20132026",
    colorbar_title="Percent change",
    filename="membership_change_2012_2026.html", save_html=True,
    white_band=0.5,
    mid_color_at=25.0,      # dark blue starts ramping in around +25%
    white_text_threshold=25.0,
)

#%%
# 2. Congregation change, 2012 -> 2026
#    None = automatically use the actual min/max.
cong_change = compute_change(df, "Congregations", 2012, 2026)
make_static_change_map(
    cong_change,
    title="LDS Congregation Change, 2012\u20132026",
    colorbar_title="Percent change",
    filename="congregation_change_2012_2026.html", save_html=True,
    white_band=1.0, 
    #mid_color_at=10.0, 
    #full_saturation_at=35.0,
    white_text_threshold=10.0,
)

#%%
# 3. Membership change, 2025 -> 2026
#    None = automatically use the actual min/max.
mem_change_recent = compute_change(df, "Total Church Membership", 2025, 2026)
make_static_change_map(
    mem_change_recent,
    title="LDS Church Membership Change, 2025\u20132026",
    colorbar_title="Percent change",
    filename="membership_change_2025_2026.html", save_html=True,
    white_band=0, 
    #mid_color_at=0.4, full_saturation_at=3,
    white_text_threshold=0.4,
)

# %%
# 4. Interactive animated map, 2012-2026
anim_fig = make_interactive_change_map(
    df,
    value_col="Total Church Membership",
    baseline_year=2012,
    filename="membership_change_interactive.html",
    color_min=-10,
    color_max=40,
    white_band=0,
    min_color="#d18892",
    mid_color="#4d9ec9",
    max_color="#083B8F",
    mid_color_at=20.0,
    white_text_threshold=20,
)


#%%

# def export_frames_as_pngs(fig, output_dir="frames", width=1600, height=1000, scale=2,
#                            label_font_scale=1.0, title_font_scale=1.0, colorbar_font_scale=1.0):
#     os.makedirs(output_dir, exist_ok=True)

#     # Pull the border styling you set on the base figure, so it carries
#     # through to every extracted frame.
#     base_line_color = fig.data[0].marker.line.color
#     base_line_width = fig.data[0].marker.line.width

#     for i, frame in enumerate(fig.frames):
#         frame_fig = go.Figure(data=frame.data, layout=fig.layout)
#         frame_fig.update_layout(updatemenus=[], sliders=[])

#         frame_fig.update_traces(
#             marker_line_color=base_line_color,
#             marker_line_width=base_line_width,
#             selector=dict(type="choropleth"),
#         )

#         # Scale on-map text labels (Scattergeo text traces)
#         for trace in frame_fig.data:
#             if isinstance(trace, go.Scattergeo) and trace.mode and "text" in trace.mode:
#                 if trace.textfont and trace.textfont.size:
#                     trace.textfont.size = trace.textfont.size * label_font_scale

#         # Scale title
#         if frame_fig.layout.title and frame_fig.layout.title.font and frame_fig.layout.title.font.size:
#             frame_fig.update_layout(
#                 title_font_size=frame_fig.layout.title.font.size * title_font_scale
#             )

#         # Scale colorbar text
#         cb = frame_fig.layout.coloraxis.colorbar if frame_fig.layout.coloraxis else None
#         if cb:
#             if cb.tickfont and cb.tickfont.size:
#                 frame_fig.update_layout(coloraxis_colorbar_tickfont_size=cb.tickfont.size * colorbar_font_scale)
#             if cb.title and cb.title.font and cb.title.font.size:
#                 frame_fig.update_layout(coloraxis_colorbar_title_font_size=cb.title.font.size * colorbar_font_scale)

#         print(tnow(), f"Saving frame {i}")
#         frame_fig.write_image(f"{output_dir}/frame_{i:03d}.png", width=width, height=height, scale=scale)
#     print(tnow(), f"Saved {len(fig.frames)} frames to {output_dir}/")

# export_frames_as_pngs(anim_fig, output_dir="frames", scale=1, label_font_scale=1.8, colorbar_font_scale=1.8)

#%% ============================================================
# REEL VIDEO (1080x1920) - purple background, stripped charts, year on top, intro slide
# Append to the end of US_Membership_Maps8.py
# ============================================================


# ---------- Settings ----------
REEL_DIR     = "reel_frames"
OUT_VIDEO    = "lds_membership_reel.mp4"
BG_PATH      = "purple.png"          # same purple background as the prophet video
REEL_W, REEL_H = 1080, 1920
FPS          = 30

CARD_W, CARD_H = 720, 590           # logical size of each chart card (px)
CARD_TOP, CARD_BOTTOM = 60, 70      # room for year at top / colorbar at bottom
RENDER_SCALE = 2
PURPLE       = "#413a66"

HOOK_TEXT    = "LDS Membership Change \nin the United States"
HANDLE_TEXT  = "@lds_data"

BASELINE_YEAR = 2012
SUBSTEPS      = 1     # smooth steps between each pair of years
HOLD_PER_SUBSTEP = 7   # video frames per step (6 x 3 = 18 frames/year = 0.6 sec)
ANIM_END_HOLD = 1.0    # seconds to hold on the last animated year
STATIC_SECONDS = 2.5     # seconds per static map
FINAL_HOLD    = 3      # seconds on the closing frame
FADE_FRAMES   = 5

ANIM_SUBTITLE = f"LDS Membership Change \nin the United States"
# Same settings as your make_interactive_change_map() call
ANIM_MAP_KWARGS = dict(color_min=-10, color_max=40, white_band=0,
                       mid_color_at=20.0, white_text_threshold=20)

# Static maps. First one is also the opening/closing frame (so the Reel loops).
# Comment out any you don't want. Settings mirror your original calls.
STATIC_SLIDES = [
    dict(name="static_1", data=mem_change,
         subtitle="LDS Membership Growth\nin the United States\n2012-2026",
         period="2012\u20132026",
         kwargs=dict(white_band=0.5, mid_color_at=25.0, white_text_threshold=25.0)),

    dict(name="static_3", data=mem_change_recent, 
         subtitle="LDS Church Membership Change\nin the United States\n2025-2026",
         period="2025\u20132026",
         kwargs=dict(white_band=0, white_text_threshold=0.4)),

    dict(name="static_2", data=cong_change, 
         subtitle="LDS Congregation Change\nin the United States\n2012-2026",
         period="Congregations 2012\u20132026",
         kwargs=dict(white_band=1.0, white_text_threshold=10.0)),
]

INTRO_TEXT = """LDS Membership Statistics

How has LDS membership changed in each U.S. state?

Percent change, 2012\u20132026
"""


INTRO_TEXT = """LDS Membership Statistics

How has LDS membership changed in each U.S. state? Which states grew and which shrank.

Percent change, 2012\u20132026

What about active membership change? Wait for the end.
"""

# Settings (fraction of image width to trim)
CROP_LEFT  = 0.1
CROP_RIGHT = 0.02
X_CENTER = 0.5 + (CROP_LEFT - CROP_RIGHT) / 2

TEST_MODE = False
if TEST_MODE:
    RENDER_SCALE = 0.5       # was 2: 4x fewer pixels per image
    SUBSTEPS = 1            # was 6: 15 animation frames instead of 85
    HOLD_PER_SUBSTEP = 18   # keeps the animation at the same ~0.6 sec/year
    STATIC_SECONDS = 2
    FINAL_HOLD = 1

#%% --- 1. Restyle a map for the reel (strip elements, add year at top) ---

def style_for_reel(fig, subtitle, big_text):
    plot_h = CARD_H - CARD_TOP - CARD_BOTTOM

    def y_from_top(px):  # pixels below top of card -> paper coordinate
        return 1 + (CARD_TOP - px) / plot_h

    fig.update_layout(
        title=dict(text=""),
        showlegend=False,
        updatemenus=[], sliders=[],          # no play button / slider
        width=CARD_W, height=CARD_H,
        margin=dict(l=0, r=0, t=CARD_TOP, b=CARD_BOTTOM),
        paper_bgcolor="white",
        annotations=[
            dict(text=f"<b>{big_text}</b>", xref="paper", yref="paper",
                 x=X_CENTER, y=y_from_top(80), xanchor="center", yanchor="middle",
                 showarrow=False,
                 font=dict(size=46, color=PURPLE, family=FONT_FAMILY)),
        ],
        
        # horizontal colorbar just under the map, no title
        coloraxis_colorbar=dict(
            orientation="h", x=X_CENTER, xanchor="center", y=0, yanchor="top",
            len=0.7, thickness=16, title=dict(text=""),
            tickfont=dict(size=14, family=FONT_FAMILY),
        ),
    )
    return fig

def save_card(fig, path):
    fig.write_image(path, width=CARD_W, height=CARD_H, scale=RENDER_SCALE)

#%% --- 2. Upgraded Intro Slide ---

def create_intro_slide(text, path, width=CARD_W, height=CARD_H, text_color=PURPLE):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("segoeuib.ttf", 44)
        font_body = ImageFont.truetype("segoeui.ttf", 32)
    except OSError:
        font_title = font_body = ImageFont.load_default()

    max_w = width * 0.82
    render_items = []
    is_first_line = True
    
    # Process line by line, keeping blank lines for paragraph breaks
    for para in text.split("\n"):
        if not para.strip():
            render_items.append(("", None, "blank"))
            continue
            
        current_font = font_title if is_first_line else font_body
        words = para.strip().split()
        cur = []
        for w in words:
            test_line = " ".join(cur + [w])
            if not cur or draw.textlength(test_line, font=current_font) <= max_w:
                cur.append(w)
            else:
                render_items.append((" ".join(cur), current_font, "text"))
                cur = [w]
                is_first_line = False
        if cur:
            render_items.append((" ".join(cur), current_font, "text"))
            is_first_line = False

    # Calculate total height including paragraph gaps
    total_h = 0
    line_metrics = []
    for line, font, item_type in render_items:
        if item_type == "blank":
            paragraph_gap = 36  # Larger space for a legit return
            line_metrics.append(("", None, 0, paragraph_gap))
            total_h += paragraph_gap
        else:
            bbox = draw.textbbox((0, 0), line if line else "Hg", font=font)
            lh = bbox[3] - bbox[1]
            spacing = 22 if font == font_title else 14
            line_metrics.append((line, font, lh, spacing))
            total_h += lh + spacing

    if line_metrics:
        total_h -= line_metrics[-1][3]

    # Draw centered text
    y = (height - total_h) // 2
    for line, font, lh, spacing in line_metrics:
        if line:
            w = draw.textlength(line, font=font)
            draw.text(((width - w) // 2, y), line, font=font, fill=text_color)
            y += lh + spacing
        else:
            y += spacing

    img.save(path, "PNG")
    return path

#%% --- 3. Render all card images ---
os.makedirs(REEL_DIR, exist_ok=True)
for old in glob.glob(f"{REEL_DIR}/anim_*.png"):
    os.remove(old)

# 3a. Intro
create_intro_slide(INTRO_TEXT, f"{REEL_DIR}/intro.png")

# 3b. Static maps
for s in STATIC_SLIDES:
    fig = make_static_change_map(s["data"], title="", colorbar_title="", filename="",
                                 show=False, **s["kwargs"])
    style_for_reel(fig, s["subtitle"], s["period"])
    save_card(fig, f"{REEL_DIR}/{s['name']}.png")
    print(f"[{tnow()}] Rendered {s['name']}")

# 3c. Year-by-year animation (values interpolated between years for smooth motion)
pct = df.pivot_table(index=["state_abbr", "state"], columns="Year",
                     values="Total Church Membership")
base = pct[BASELINE_YEAR]
pct = pct.sub(base, axis=0).div(base, axis=0) * 100
pct = pct[[y for y in pct.columns if y >= BASELINE_YEAR]]
#years = list(pct.columns)

# Filter out 2021 from the years list so it gets skipped entirely
years = [y for y in pct.columns if y >= BASELINE_YEAR and y != 2021]

anim_frames = []   # (year label, Series of % change per state)
for a, b in zip(years[:-1], years[1:]):
    for k in range(SUBSTEPS):
        t = k / SUBSTEPS
        e = t * t * (3 - 2 * t)                       # smoothstep easing
        vals = pct[a] * (1 - e) + pct[b] * e
        anim_frames.append((a if t < 0.5 else b, vals))
anim_frames.append((years[-1], pct[years[-1]]))

for i, (yr, vals) in enumerate(anim_frames):
    frame_df = vals.rename("pct_change").reset_index().dropna(subset=["pct_change"])
    fig = make_static_change_map(frame_df, title="", colorbar_title="", filename="",
                                 value_col="pct_change", show=False, **ANIM_MAP_KWARGS)
    label = f"{BASELINE_YEAR}\u2013{yr}" if yr > BASELINE_YEAR else str(BASELINE_YEAR)
    style_for_reel(fig, ANIM_SUBTITLE, label)
    save_card(fig, f"{REEL_DIR}/anim_{i:03d}.png")
    print(f"[{tnow()}] Rendered anim {i+1}/{len(anim_frames)}")

#%% --- 4. Build the video ---

def ease_in_out_cubic(t):
    return 4 * t**3 if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

def load_background():
    raw = cv2.imread(BG_PATH)
    if raw is None:    # fallback gradient so the script still runs
        top, bottom = np.array([163, 79, 106]), np.array([107, 42, 59])  # BGR
        grad = np.linspace(top, bottom, REEL_H)[:, None, :] * np.ones((1, REEL_W, 1))
        return grad.astype(np.uint8)
    return cv2.resize(raw, (REEL_W, REEL_H))


# Settings
TITLE_SCALE = 2.0        
TITLE_THICKNESS = 3      
TITLE_MAX_WIDTH = 1000   
PERSISTENT_TITLE = "LDS Membership Growth\nin the United States\n2012-2026"  # Change to your title

def fit_on_background(img, bg, title=None):
    """Floating white card with smooth rounded corners on the purple background."""
    h, w = img.shape[:2]
    scale = min(REEL_W * 0.92 / w, REEL_H * 0.75 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = bg.copy()
    y0, x0 = (REEL_H - new_h) // 2, (REEL_W - new_w) // 2
    
    # --- ROUNDED CORNERS IMPLEMENTATION ---
    radius = 35  # Adjust this value to make the corners rounder (e.g., 20 to 50)
    
    # Create a blank mask for the rounded rectangle
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    
    # Draw a filled white rounded rectangle on the mask
    # OpenCV's ellipse/rectangle combination or drawing lines/circles can create this, 
    # but an easy way is using cv2.rectangle + cv2.circle or drawing polylines.
    # Alternatively, use a clean corner-radius approach:
    cv2.rectangle(mask, (radius, 0), (new_w - radius, new_h), 255, -1)
    cv2.rectangle(mask, (0, radius), (new_w, new_h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (new_w - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, new_h - radius), radius, 255, -1)
    cv2.circle(mask, (new_w - radius, new_h - radius), radius, 255, -1)
    
    # Optional: Smooth out the edges of the mask slightly for anti-aliasing
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Convert mask to 3-channel (float) for blending
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    # Blend the resized card onto the background using the rounded mask
    roi = canvas[y0:y0 + new_h, x0:x0 + new_w].astype(float)
    blended = (resized.astype(float) * mask_3ch) + (roi * (1.0 - mask_3ch))
    canvas[y0:y0 + new_h, x0:x0 + new_w] = blended.astype(np.uint8)
    # --------------------------------------

    # Draw the persistent title cleanly centered above the card
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = title.split("\n")
        line_gap = int(TITLE_SCALE * 38)
        for k, line in enumerate(lines):
            scale_t = TITLE_SCALE
            tw = cv2.getTextSize(line, font, scale_t, TITLE_THICKNESS)[0][0]
            if tw > TITLE_MAX_WIDTH:
                scale_t *= TITLE_MAX_WIDTH / tw
                tw = cv2.getTextSize(line, font, scale_t, TITLE_THICKNESS)[0][0]
            tx = (REEL_W - tw) // 2
            ty = y0 - 40 - (len(lines) - 1 - k) * line_gap
            cv2.putText(canvas, line, (tx + 3, ty + 3), font, scale_t, (0, 0, 0), TITLE_THICKNESS, cv2.LINE_AA)
            cv2.putText(canvas, line, (tx, ty), font, scale_t, (255, 255, 255), TITLE_THICKNESS, cv2.LINE_AA)
            
    return canvas

def add_branding(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bs = cv2.getTextSize(HANDLE_TEXT, font, 0.9, 2)[0]
    bx, by = (REEL_W - bs[0]) // 2, 1580
    cv2.putText(frame, HANDLE_TEXT, (bx + 2, by + 2), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, HANDLE_TEXT, (bx, by), font, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    return frame

def hold(out, img, seconds):
    frame = add_branding(img.copy())
    for _ in range(int(seconds * FPS)):
        out.write(frame)

def fade(out, a, b, n=FADE_FRAMES):
    for t in np.linspace(0, 1, n):
        alpha = ease_in_out_cubic(t)
        out.write(add_branding(cv2.addWeighted(a, 1 - alpha, b, alpha, 0)))

bg = load_background()

def base(path, title=None):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    if "intro" in path:
        pass
    elif "mid" in path:
        pass
    else:
        # Only crop chart frames, leave the intro slide at full width
        w = img.shape[1]
        # --- CROP CONFIGURATION ---
        # Adjust CROP_TOP (e.g., 0.05 for 5% off the top, or an exact pixel count like 100)
        crop_top_px = int(h * 0.05)  # Change 0.05 to whatever percentage you want to chop off
        # Apply vertical crop from the top, and keep your horizontal cropping as-is
        img = img[crop_top_px:, int(w * CROP_LEFT): w - int(w * CROP_RIGHT)]   
    
    return fit_on_background(img, bg, title)

anim_paths  = sorted(glob.glob(f"{REEL_DIR}/anim_*.png"))
intro_base  = base(f"{REEL_DIR}/intro.png")

# Create the mid-video transition slide image
mid_slide_text = "Want to know if active membership is growing or shrinking?\n\nFollow the congregations..."
create_intro_slide(mid_slide_text, f"{REEL_DIR}/mid_slide.png")

# Load it using your base function (which treats intro/mid slides at full width)
mid_base = base(f"{REEL_DIR}/mid_slide.png")


# Overwrite the persistent title with each slide's specific subtitle if it exists, 
# otherwise fall back to the global PERSISTENT_TITLE
static_bases = [
    base(f"{REEL_DIR}/{s['name']}.png", s.get("subtitle", PERSISTENT_TITLE)) 
    for s in STATIC_SLIDES
]
first_anim, last_anim = base(anim_paths[0], PERSISTENT_TITLE), base(anim_paths[-1], PERSISTENT_TITLE)
hook = static_bases[0]

out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"XVID"), FPS, (REEL_W, REEL_H))
print(f"[{tnow()}] Starting render...")

hold(out, hook, 1)                     
fade(out, hook, intro_base)             
hold(out, intro_base, 3)                
fade(out, intro_base, first_anim)       

# E: year-by-year animation with quick fades between frames
prev = first_anim

for n, p in enumerate(anim_paths, 1):
    print(f"[{tnow()}] Writing animation frame {n}/{len(anim_paths)}")
    current_frame = base(p, PERSISTENT_TITLE)
    
    # Fade from the previous year's frame to the current one
    if n > 1:
        fade(out, prev, current_frame)
    
    # Hold the current frame for your desired substep duration
    frame_with_branding = add_branding(current_frame.copy())
    for _ in range(HOLD_PER_SUBSTEP):
        out.write(frame_with_branding)
        
    prev = current_frame

# ... your animation code runs here ...

hold(out, last_anim, ANIM_END_HOLD)

prev = last_anim                        
# Static slide 1
fade(out, prev, static_bases[0])
hold(out, static_bases[0], STATIC_SECONDS)
prev = static_bases[0]

# Static slide 2
fade(out, prev, static_bases[1])
hold(out, static_bases[1], STATIC_SECONDS)
prev = static_bases[1]

# --- INSERT MID-VIDEO TRANSITION SLIDE HERE ---
fade(out, prev, mid_base)
hold(out, mid_base, 3.5)  # Pause for 3.5 seconds to read
prev = mid_base
# ---------------------------------------------

# Static slide 3 (Final chart)
fade(out, prev, static_bases[2])
hold(out, static_bases[2], STATIC_SECONDS)
prev = static_bases[2]

# Loop back to opening hook
#ade(out, prev, hook)                   
#hold(out, hook, FINAL_HOLD)

out.release()
print(f"[{tnow()}] SUCCESS: '{OUT_VIDEO}' created.")

#%% --- 5. Export reel-sized stills (purple background) ---
STILLS_DIR = "membership_exported_stills"
os.makedirs(STILLS_DIR, exist_ok=True)

cv2.imwrite(f"{STILLS_DIR}/00_intro.png", add_branding(intro_base.copy()))
for s, b in zip(STATIC_SLIDES, static_bases):
    cv2.imwrite(f"{STILLS_DIR}/{s['name']}.png", add_branding(b.copy()))
for p in anim_paths:
    name = os.path.basename(p)
    cv2.imwrite(f"{STILLS_DIR}/{name}", add_branding(base(p)))
print(f"[{tnow()}] Stills saved to '{STILLS_DIR}'.")