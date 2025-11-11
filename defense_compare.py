#!/usr/bin/env python3
"""Create a side-by-side defense stock comparison video.

This script is designed to run on Linux, macOS, Windows, and Android (Termux).
It downloads historical price data with yfinance, converts both tickers into a
shared target currency, and animates the growth of an investment in two
companies.

Example usage (Termux or any shell):

    python defense_compare.py --eu RHM.DE --us LMT \
        --start 2003-01-01 --out rhm_vs_lmt.mp4

Dependencies (install with `pip install ...` if they are missing):
    pandas numpy matplotlib yfinance pillow
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless/back-end friendly (Termux, servers, etc.)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter

import yfinance as yf


@dataclass(frozen=True)
class ConflictEvent:
    date: pd.Timestamp
    label: str


DEFAULT_EVENTS: Tuple[ConflictEvent, ...] = (
    ConflictEvent(pd.Timestamp("2003-03-20"), "Iraq invasion"),
    ConflictEvent(pd.Timestamp("2011-03-19"), "Libya intervention"),
    ConflictEvent(pd.Timestamp("2014-02-20"), "Crimea crisis"),
    ConflictEvent(pd.Timestamp("2015-09-30"), "Russian intervention in Syria"),
    ConflictEvent(pd.Timestamp("2021-08-30"), "Afghanistan withdrawal"),
    ConflictEvent(pd.Timestamp("2022-02-24"), "Full invasion of Ukraine"),
)


CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "CHF": "CHF",
    "JPY": "¥",
    "CAD": "CA$",
    "AUD": "A$",
    "NZD": "NZ$",
    "SEK": "kr",
    "NOK": "kr",
    "DKK": "kr",
    "PLN": "zł",
}


SUPPORTED_CURRENCIES = sorted({
    "USD",
    "EUR",
    "GBP",
    "CHF",
    "JPY",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "PLN",
    "CNY",
    "HKD",
})


COUNTRY_TO_CODE = {
    "united states": "US",
    "usa": "US",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "united kingdom": "GB",
    "uk": "GB",
    "italy": "IT",
    "canada": "CA",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "australia": "AU",
    "spain": "ES",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "poland": "PL",
    "switzerland": "CH",
    "india": "IN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an EU defense stock with a US defense stock and export a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eu", required=True, help="Ticker of the European defense company (e.g. RHM.DE)")
    parser.add_argument("--us", required=True, help="Ticker of the US defense company (e.g. LMT)")
    parser.add_argument("--start", default="2003-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--duration", type=float, default=20.0, help="Video duration in seconds")
    parser.add_argument(
        "--frequency",
        default="M",
        choices=["D", "W", "M", "Q"],
        help="Resample frequency for the chart (daily/weekly/monthly/quarterly)",
    )
    parser.add_argument("--max-frames", type=int, default=360, help="Maximum animation frames to keep video smooth")
    parser.add_argument("--amount", type=float, default=1000.0, help="Investment amount in target currency")
    parser.add_argument("--target-currency", default="USD", help="Target currency for normalization (e.g. USD, EUR)")
    parser.add_argument("--eu-currency", default=None, help="Override detected currency for the European ticker")
    parser.add_argument("--us-currency", default=None, help="Override detected currency for the US ticker")
    parser.add_argument("--eu-country", default=None, help="Optional country label for the European ticker (for flag emoji)")
    parser.add_argument("--us-country", default=None, help="Optional country label for the US ticker (for flag emoji)")
    parser.add_argument("--eu-label", default=None, help="Display label for the European ticker in the video")
    parser.add_argument("--us-label", default=None, help="Display label for the US ticker in the video")
    parser.add_argument("--output-dir", default="defense_videos", help="Directory to place the exported video")
    parser.add_argument("--out", default=None, help="Output filename (mp4). If omitted a descriptive name is generated.")
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable conflict annotations",
    )
    parser.add_argument(
        "--events",
        nargs="*",
        default=None,
        help=(
            "Optional custom events as DATE:LABEL. Example: 2011-03-19:'Libya intervention' "
            "(wrap label with quotes if it contains spaces)."
        ),
    )
    return parser.parse_args()


def currency_symbol(code: str) -> str:
    code = (code or "").upper()
    return CURRENCY_SYMBOLS.get(code, f"{code} ")


def country_to_flag(country: Optional[str]) -> str:
    if not country:
        return ""
    country = country.strip()
    if len(country) == 2 and country.isalpha():
        iso_code = country.upper()
    else:
        iso_code = COUNTRY_TO_CODE.get(country.lower())
        if iso_code is None:
            return country
    base = 0x1F1E6
    try:
        return chr(base + ord(iso_code[0]) - ord("A")) + chr(base + ord(iso_code[1]) - ord("A"))
    except Exception:
        return country


@lru_cache(maxsize=16)
def detect_currency(ticker: str) -> Optional[str]:
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        return None
    for attr in ("fast_info", "info"):
        try:
            data = getattr(ticker_obj, attr)
            if data is None:
                continue
            currency = None
            if hasattr(data, "get"):
                currency = data.get("currency")
            else:
                currency = getattr(data, "currency", None)
            if currency:
                return str(currency).upper()
        except Exception:
            continue
    if ticker.endswith(".DE"):
        return "EUR"
    if ticker.endswith(".L"):
        return "GBP"
    if ticker.endswith(".PA"):
        return "EUR"
    if ticker.endswith(".MI"):
        return "EUR"
    return None


@lru_cache(maxsize=16)
def detect_country(ticker: str) -> Optional[str]:
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        return None
    for attr in ("fast_info", "info"):
        try:
            data = getattr(ticker_obj, attr)
            if data is None:
                continue
            if hasattr(data, "get"):
                country = data.get("country") or data.get("longName")
            else:
                country = getattr(data, "country", None)
            if country:
                return str(country)
        except Exception:
            continue
    suffix_map = {
        ".DE": "Germany",
        ".L": "United Kingdom",
        ".PA": "France",
        ".MI": "Italy",
        ".TO": "Canada",
        ".SW": "Switzerland",
    }
    for suffix, country in suffix_map.items():
        if ticker.endswith(suffix):
            return country
    return None


@lru_cache(maxsize=16)
def detect_company_name(ticker: str) -> Optional[str]:
    try:
        ticker_obj = yf.Ticker(ticker)
    except Exception:
        return None
    for attr in ("fast_info", "info"):
        try:
            data = getattr(ticker_obj, attr)
            if data is None:
                continue
            if hasattr(data, "get"):
                name = data.get("shortName") or data.get("longName")
            else:
                name = getattr(data, "shortName", None)
            if name:
                return str(name)
        except Exception:
            continue
    return None


def download_series(ticker: str, start: str, end: str | None) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'.")
    series = data["Close"] if "Close" in data else data.iloc[:, 0]
    series = series.squeeze()
    series.name = ticker
    return series


def convert_currency(
    series: pd.Series,
    from_currency: Optional[str],
    to_currency: str,
    start: str,
    end: str | None,
) -> pd.Series:
    to_currency = (to_currency or "").upper()
    from_currency = (from_currency or "").upper()
    if not to_currency:
        raise ValueError("Target currency must be provided.")
    if not from_currency or from_currency == to_currency:
        return series
    pair = f"{from_currency}{to_currency}=X"
    invert = False
    try:
        fx = download_series(pair, start, end)
    except ValueError:
        reverse_pair = f"{to_currency}{from_currency}=X"
        try:
            fx = download_series(reverse_pair, start, end)
        except ValueError as exc:
            raise ValueError(
                f"Could not find FX data for {from_currency}->{to_currency}"
            ) from exc
        invert = True
    if invert:
        fx = 1 / fx
    fx = fx.reindex(series.index, method="ffill").dropna()
    aligned = series.loc[fx.index]
    converted = aligned * fx
    converted.name = f"{series.name}-{to_currency}"
    return converted


def resample_series(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "D":
        return series.asfreq("D").ffill().dropna()
    return series.resample(frequency).last().dropna()


def align_series(series_list: Iterable[pd.Series]) -> pd.DataFrame:
    frame = pd.concat(series_list, axis=1, join="inner").dropna()
    return frame


def normalize_investment(frame: pd.DataFrame, amount: float) -> pd.DataFrame:
    first_values = frame.iloc[0]
    normalized = frame / first_values * amount
    return normalized


def limit_frames(data: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    if len(data) <= max_frames:
        return data
    indices = np.linspace(0, len(data) - 1, max_frames).round().astype(int)
    return data.iloc[indices]


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_custom_events(raw_events: List[str]) -> Tuple[ConflictEvent, ...]:
    events: List[ConflictEvent] = []
    for raw in raw_events:
        if ":" not in raw:
            raise ValueError(f"Invalid event format '{raw}'. Use DATE:LABEL.")
        date_str, label = raw.split(":", 1)
        date = pd.to_datetime(date_str)
        events.append(ConflictEvent(date, label.strip().strip("'\"")))
    return tuple(sorted(events, key=lambda e: e.date))


def add_events(ax: plt.Axes, events: Iterable[ConflictEvent], ymin: float, ymax: float) -> None:
    for event in events:
        ax.axvline(event.date, color="#9c27b0", linestyle="--", alpha=0.4, linewidth=1.2)
        ax.annotate(
            event.label,
            xy=(event.date, ymax),
            xytext=(0, 6),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4a148c",
        )


def create_animation(
    data: pd.DataFrame,
    left_label: str,
    right_label: str,
    left_flag: str,
    right_flag: str,
    amount: float,
    currency_code: str,
    duration: float,
    output_path: str,
    events: Tuple[ConflictEvent, ...],
) -> None:
    dates = data.index
    eu_values = data.iloc[:, 0]
    us_values = data.iloc[:, 1]

    fig, ax = plt.subplots(figsize=(7.2, 12.8))  # roughly 9:16 portrait
    fig.patch.set_facecolor("#faf7f2")
    ax.set_facecolor("#fffdf8")

    line_eu, = ax.plot([], [], color="#1b5e20", linewidth=3, label=f"{left_flag} {left_label}".strip())
    line_us, = ax.plot([], [], color="#0d47a1", linewidth=3, label=f"{right_flag} {right_label}".strip())

    ax.set_xlim(dates.min(), dates.max())
    ymax = max(eu_values.max(), us_values.max()) * 1.1
    ax.set_ylim(0, ymax)
    symbol = currency_symbol(currency_code)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{symbol}{x:,.0f}"))
    ax.set_ylabel(f"Portfolio value ({currency_code.upper()})", fontsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(color="#dcd6c8", linestyle="--", linewidth=0.8, alpha=0.6)

    start_year = dates[0].year
    title = ax.set_title(
        f"You invest {symbol}{amount:,.0f} in {left_label} vs. {right_label} (Start: {start_year})",
        pad=20,
        fontsize=16,
        fontweight="bold",
        color="#1a1a1a",
    )

    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.16, right=0.96)

    if left_flag:
        ax.text(0.08, 1.02, left_flag, transform=ax.transAxes, fontsize=28, ha="center", va="bottom")
    if right_flag:
        ax.text(0.92, 1.02, right_flag, transform=ax.transAxes, fontsize=28, ha="center", va="bottom")

    legend = ax.legend(loc="upper left", frameon=False, fontsize=11)

    year_text = ax.text(
        0.5,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=36,
        fontweight="bold",
        ha="center",
        color="#37474f",
    )
    value_text = ax.text(
        0.02,
        0.78,
        "",
        transform=ax.transAxes,
        fontsize=12,
        ha="left",
        va="top",
        color="#263238",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="none", alpha=0.6),
    )

    if events:
        add_events(ax, events, ymin=0, ymax=ymax)

    frames = len(dates)
    fps = max(1, frames / duration)

    def init() -> Tuple:
        line_eu.set_data([], [])
        line_us.set_data([], [])
        year_text.set_text("")
        value_text.set_text("")
        return line_eu, line_us, year_text, value_text

    def update(frame_index: int) -> Tuple:
        current_slice = slice(0, frame_index + 1)
        line_eu.set_data(dates[current_slice], eu_values[current_slice])
        line_us.set_data(dates[current_slice], us_values[current_slice])
        current_date = dates[frame_index]
        year_text.set_text(str(current_date.year))
        left_prefix = f"{left_flag} " if left_flag else ""
        right_prefix = f"{right_flag} " if right_flag else ""
        value_text.set_text(
            f"{left_prefix}{left_label}: {symbol}{eu_values.iloc[frame_index]:,.0f}\n"
            f"{right_prefix}{right_label}: {symbol}{us_values.iloc[frame_index]:,.0f}"
        )
        return line_eu, line_us, year_text, value_text

    interval = duration * 1000.0 / frames
    animation = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False,
    )

    animation.save(output_path, writer="ffmpeg", fps=fps, dpi=200)
    plt.close(fig)


def generate_comparison_video(
    left_ticker: str,
    right_ticker: str,
    start: str,
    end: Optional[str],
    amount: float,
    frequency: str,
    max_frames: int,
    target_currency: str,
    duration: float,
    left_currency_hint: Optional[str] = None,
    right_currency_hint: Optional[str] = None,
    left_country_hint: Optional[str] = None,
    right_country_hint: Optional[str] = None,
    output_dir: str = "defense_videos",
    filename: Optional[str] = None,
    events: Optional[Tuple[ConflictEvent, ...]] = None,
    left_label_override: Optional[str] = None,
    right_label_override: Optional[str] = None,
) -> str:
    ensure_directory(output_dir)

    left_series = download_series(left_ticker, start, end)
    right_series = download_series(right_ticker, start, end)

    left_currency = (left_currency_hint or detect_currency(left_ticker) or target_currency).upper()
    right_currency = (right_currency_hint or detect_currency(right_ticker) or target_currency).upper()
    target_currency = target_currency.upper()

    left_series = convert_currency(left_series, left_currency, target_currency, start, end)
    right_series = convert_currency(right_series, right_currency, target_currency, start, end)

    left_series = resample_series(left_series, frequency)
    right_series = resample_series(right_series, frequency)

    combined = align_series([left_series, right_series])
    normalized = normalize_investment(combined, amount)
    normalized = limit_frames(normalized, max_frames)

    if events is None:
        events = tuple(
            event
            for event in DEFAULT_EVENTS
            if normalized.index.min() <= event.date <= normalized.index.max()
        )
    else:
        events = tuple(
            event
            for event in events
            if normalized.index.min() <= event.date <= normalized.index.max()
        )

    left_country = left_country_hint or detect_country(left_ticker)
    right_country = right_country_hint or detect_country(right_ticker)
    left_flag = country_to_flag(left_country)
    right_flag = country_to_flag(right_country)

    start_stamp = normalized.index[0]
    end_stamp = normalized.index[-1]

    if filename:
        output_name = filename
    else:
        output_name = (
            f"{left_ticker.replace('.', '_')}_vs_{right_ticker.replace('.', '_')}_"
            f"{start_stamp.strftime('%Y%m%d')}_{end_stamp.strftime('%Y%m%d')}.mp4"
        )
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    output_path = os.path.join(output_dir, output_name)

    left_label = left_label_override or detect_company_name(left_ticker) or left_ticker
    right_label = right_label_override or detect_company_name(right_ticker) or right_ticker

    create_animation(
        normalized,
        left_label=left_label,
        right_label=right_label,
        left_flag=left_flag,
        right_flag=right_flag,
        amount=amount,
        currency_code=target_currency,
        duration=duration,
        output_path=output_path,
        events=events,
    )

    return output_path


def main() -> None:
    args = parse_args()

    if args.no_events:
        events: Tuple[ConflictEvent, ...] = ()
    elif args.events:
        events = parse_custom_events(args.events)
    else:
        events = None

    output_path = generate_comparison_video(
        left_ticker=args.eu,
        right_ticker=args.us,
        start=args.start,
        end=args.end,
        amount=args.amount,
        frequency=args.frequency,
        max_frames=args.max_frames,
        target_currency=args.target_currency,
        duration=args.duration,
        left_currency_hint=args.eu_currency,
        right_currency_hint=args.us_currency,
        left_country_hint=args.eu_country,
        right_country_hint=args.us_country,
        output_dir=args.output_dir,
        filename=args.out,
        events=events,
        left_label_override=args.eu_label,
        right_label_override=args.us_label,
    )

    print(f"✔ Done: {output_path}")


if __name__ == "__main__":
    main()
