# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Diamondpeter/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Diamondpeter/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for the launch day of Grand Theft Auto VI on Twitch. Given everything recorded up to the end of the release day's first hour (UTC) — that hour's viewer count, the run-up history, and the hourly curves of dozens of past big game launches — it predicts, for each of hours 2 to 24, five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch concurrent viewers in the "Grand Theft Auto VI" category. The model normalises each comparable launch's release-day curve by its own first hour, aggregates the shapes robustly, and calibrates its spread on the dispersion of past launches, so the forecast follows the typical launch-day arc instead of assuming the first hour repeats all day.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=h68hpwev&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
