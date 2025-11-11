#!/usr/bin/env python3
"""Streamlit app for crafting defense & aerospace stock comparison videos."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Tuple

import streamlit as st

from defense_compare import (
    ConflictEvent,
    DEFAULT_EVENTS,
    SUPPORTED_CURRENCIES,
    country_to_flag,
    currency_symbol,
    detect_company_name,
    detect_country,
    detect_currency,
    generate_comparison_video,
    parse_custom_events,
)


def _date_to_string(value: dt.date | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d")


def _parse_event_lines(lines: Iterable[str]) -> Tuple[ConflictEvent, ...]:
    raw = [line.strip() for line in lines if line.strip()]
    if not raw:
        return tuple()
    return parse_custom_events(raw)


def _event_option_label(event: ConflictEvent) -> str:
    return f"{event.date.date()}: {event.label}"


st.set_page_config(page_title="Defense Stock Video Builder", layout="centered")
st.title("Defense & Aerospace Stock Video Builder")
st.caption(
    "Pick any two tickers, currencies, and optional conflict markers to craft an animated comparison video."
)

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        1. Enter two stock tickers from any exchange supported by Yahoo Finance.
        2. Choose the time range, target currency, and investment amount.
        3. Optionally add notable conflicts/events to annotate the chart.
        4. Generate the video, preview it directly in the browser, and download the MP4.
        """
    )

with st.form("video_form"):
    col_left, col_right = st.columns(2)
    with col_left:
        left_ticker = st.text_input("First ticker", value="RHM.DE")
    with col_right:
        right_ticker = st.text_input("Second ticker", value="LMT")

    today = dt.date.today()
    default_start = dt.date(today.year - 20, 1, 1)
    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Start date", value=default_start, max_value=today)
    with col_dates[1]:
        end_enabled = st.checkbox("Set end date", value=False)
        end_date = st.date_input("End date", value=today, max_value=today, disabled=not end_enabled)
        if not end_enabled:
            end_date = None

    col_currency = st.columns(3)
    with col_currency[0]:
        target_currency = st.selectbox("Target currency", SUPPORTED_CURRENCIES, index=SUPPORTED_CURRENCIES.index("USD"))
    with col_currency[1]:
        amount = st.number_input("Investment amount", value=1000.0, min_value=10.0, step=100.0)
    with col_currency[2]:
        duration = st.slider("Video duration (seconds)", min_value=10.0, max_value=45.0, value=20.0, step=1.0)

    col_more = st.columns(2)
    with col_more[0]:
        frequency = st.selectbox("Resample frequency", options=["D", "W", "M", "Q"], index=2, help="Control how granular the price curve should be")
        max_frames = st.slider("Max frames", min_value=120, max_value=720, value=360, step=60)
    with col_more[1]:
        left_currency_override = st.text_input(
            "Currency override (left)",
            value="",
            placeholder=f"Auto: {detect_currency(left_ticker) or 'unknown'}",
        )
        right_currency_override = st.text_input(
            "Currency override (right)",
            value="",
            placeholder=f"Auto: {detect_currency(right_ticker) or 'unknown'}",
        )

    col_country = st.columns(2)
    with col_country[0]:
        left_country_override = st.text_input(
            "Country override (left)",
            value="",
            placeholder=f"Auto: {detect_country(left_ticker) or 'unknown'}",
        )
    with col_country[1]:
        right_country_override = st.text_input(
            "Country override (right)",
            value="",
            placeholder=f"Auto: {detect_country(right_ticker) or 'unknown'}",
        )

    col_labels = st.columns(2)
    with col_labels[0]:
        left_label_override = st.text_input(
            "Display name (left)",
            value=detect_company_name(left_ticker) or left_ticker,
        )
    with col_labels[1]:
        right_label_override = st.text_input(
            "Display name (right)",
            value=detect_company_name(right_ticker) or right_ticker,
        )

    default_events = list(DEFAULT_EVENTS)
    selected_defaults = st.multiselect(
        "Conflict markers",
        options=default_events,
        default=default_events,
        format_func=_event_option_label,
    )

    custom_events_text = st.text_area(
        "Extra events (one per line, format YYYY-MM-DD:Label)",
        placeholder="2016-07-15:Turkey coup attempt",
    )

    output_dir = st.text_input("Output directory", value="defense_videos")
    filename = st.text_input("File name (optional)", value="")

    submitted = st.form_submit_button("Generate video")

if submitted:
    with st.spinner("Downloading price data and rendering video..."):
        try:
            custom_events = _parse_event_lines(custom_events_text.splitlines())
            events = tuple(selected_defaults) + custom_events
            output_path = generate_comparison_video(
                left_ticker=left_ticker.strip(),
                right_ticker=right_ticker.strip(),
                start=_date_to_string(start_date),
                end=_date_to_string(end_date),
                amount=float(amount),
                frequency=frequency,
                max_frames=int(max_frames),
                target_currency=target_currency,
                duration=float(duration),
                left_currency_hint=left_currency_override or None,
                right_currency_hint=right_currency_override or None,
                left_country_hint=left_country_override or None,
                right_country_hint=right_country_override or None,
                output_dir=output_dir.strip() or "defense_videos",
                filename=filename.strip() or None,
                events=events,
                left_label_override=left_label_override.strip() or None,
                right_label_override=right_label_override.strip() or None,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
        else:
            st.success("Video ready!")
            video_path = Path(output_path)
            if video_path.exists():
                st.video(str(video_path))
                with video_path.open("rb") as f:
                    st.download_button(
                        "Download MP4",
                        data=f,
                        file_name=video_path.name,
                        mime="video/mp4",
                    )
            else:
                st.warning("Video file could not be found on disk.")

st.sidebar.header("Auto-detected metadata")
left_name = detect_company_name(left_ticker) or "Unknown"
right_name = detect_company_name(right_ticker) or "Unknown"
left_country = detect_country(left_ticker)
right_country = detect_country(right_ticker)
left_flag = country_to_flag(left_country) or ""
right_flag = country_to_flag(right_country) or ""
left_currency = detect_currency(left_ticker) or "?"
right_currency = detect_currency(right_ticker) or "?"

st.sidebar.markdown(
    f"**{left_ticker}** — {left_flag} {left_name}\n\n"
    f"Currency: {left_currency}\n"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{right_ticker}** — {right_flag} {right_name}\n\n"
    f"Currency: {right_currency}\n"
)

symbol = currency_symbol(target_currency if 'target_currency' in locals() else 'USD')
st.sidebar.markdown(f"Videos normalize a {symbol}{amount if 'amount' in locals() else 1000:,.0f} investment.")
