# wiki-words-to-video (wwtv)

Generate short, newspaper-style videos by highlighting a term across real Wikipedia pages.

## Install the `wwtv` command

- Global install (recommended): `pipx install .`
- Dev install from a checkout: `pip install -e .`
- Playwright needs a browser once per machine: `playwright install chromium`

## Use it

Basic run:

```bash
wwtv --term "solar eclipse" --seconds 8 --out eclipse.mp4
```

Handy flags:
- `--zoom` (default 1.5) controls page zoom; combine with `--zoom-in`, `--zoom-out`, or `--zoom-end` for smooth transitions.
- `--max-pages` caps how many Wikipedia pages to visit; fewer pages = faster.
- `--capture-scale` (default 2.0) renders pages at higher resolution before downscaling; drop to 1.5 or 1.0 for faster runs.
- `--fps`, `--width`, `--height`, `--crf`, `--preset` tweak video cadence and quality.
- `--case-sensitive` prefers exact-case matches for the highlighted term.
- `--center-debug` writes centering diagnostics (csv/png) while rendering.

See `wwtv --help` for the full set of options. Videos and intermediate frames are written to the current working directory.
