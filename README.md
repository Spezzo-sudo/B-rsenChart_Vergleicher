# BörsenChart Vergleicher

Create cinematic comparison videos for two defense stocks directly from the command line or an interactive web UI.

## Requirements

Install the Python dependencies (works on Windows, macOS, Linux and Android/Termux):

```bash
pip install pandas numpy matplotlib yfinance pillow
```

FFmpeg must be available in your `PATH` to export MP4 files.

## Usage

Generate a 20 second comparison video for Rheinmetall (Germany) vs. Lockheed Martin (USA):

```bash
python defense_compare.py --eu RHM.DE --us LMT --start 2003-01-01 --out rhm_vs_lmt.mp4
```

The script will create an `defense_videos/` directory (if missing) and write the video there. Use
`--duration` to adjust runtime, `--frequency` to control resampling (daily/weekly/monthly/quarterly),
`--target-currency` to normalize in any currency, `--amount` to pick a different investment, and
`--events` to annotate additional conflicts (format `DATE:Label`). Disable annotations with `--no-events`.
Override detection with `--eu-currency`, `--us-currency`, `--eu-country`, `--us-country`, `--eu-label`,
or `--us-label` when you want explicit control over the metadata that lands in the video.

All monetary values are converted into the target currency and the animation highlights flag emojis
plus notable conflicts that shaped the stocks' trajectories.

## Interactive app

Prefer a no-code workflow or want to drive the process from your phone (Termux, S25 Ultra, etc.)?
Launch the Streamlit app and open the provided URL in your browser:

```bash
streamlit run app.py
```

Inside the app you can:

- Pick any two tickers and optionally rename them for the video.
- Choose start/end dates, resampling frequency, runtime, max frames, and target currency.
- Add conflict annotations (defaults include Iraq, Libya, Crimea, Syria, Afghanistan, Ukraine).
- Override detected currencies or countries when necessary and automatically visualize flag emojis.
- Preview the generated MP4 inline and download it directly to your device.

The app uses the same engine as the CLI (`generate_comparison_video`) so both pathways stay in sync.
