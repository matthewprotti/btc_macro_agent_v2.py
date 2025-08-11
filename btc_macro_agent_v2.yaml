name: BTC Macro Tracker
on:
  schedule:           # 23:35 UTC daily (17:35 America/Edmonton)
    - cron: "35 23 * * *"
  workflow_dispatch: {}   # allow manual run

permissions:
  contents: write         # needed to push track.csv back

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy lxml html5lib beautifulsoup4 statsmodels

      - name: List workspace
        run: |
          pwd
          ls -la
          find . -maxdepth 2 -type f | sort

      - name: Run tracker
        env:
          BTC_MACRO_DATA_DIR: ${{ github.workspace }}/data
          BTC_MACRO_TRACK:    ${{ github.workspace }}/track.csv
          BTC_MACRO_LOG:      ${{ github.workspace }}/log.txt
        run: |
          python btc_macro_agent_v2.py

      - name: Commit outputs
        run: |
          git config user.name  "macro-bot"
          git config user.email "bot@users.noreply.github.com"
          git add track.csv data/ log.txt || true
          git commit -m "daily update $(date -u +%F)" || echo "no changes"
          git push
