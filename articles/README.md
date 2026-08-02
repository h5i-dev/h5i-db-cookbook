# Articles

Long-form, standalone write-ups drawn from the cookbook recipes. Unlike a
recipe, an article is meant to be read start to finish by someone who has never
seen h5i-db, so it repeats context the recipes assume and carries its own
vocabulary section.

Same convention as `notebooks/`: the `.py` file (jupytext percent format) is the
source of truth, and the `.ipynb` beside it is generated and executed from it.

| Article | What it covers |
|---|---|
| [Practical backtesting for Polymarket](practical_backtesting_for_polymarket.ipynb) | Order-book mechanics, one breakout strategy written from scratch and traded four ways on a fifteen-event toy market, then the same function on real tick-level Polymarket books |

The Polymarket notebook is around 900 KB, larger than any recipe, because it
embeds two `result.report()` pages as iframes. That is deliberate: the report is
part of what the article is arguing for. The same pages are written to
`data/cache/` as standalone HTML files, which are gitignored.

## Building

```bash
python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from build_notebooks import build
build(Path("articles/practical_backtesting_for_polymarket.py"), execute=True, timeout=1500)
EOF
```

Run it from the repository root, so `import cookbook_utils` resolves and the
relative `data/cache/` paths find their inputs.

Part 3 of the Polymarket article needs the bounded Kaggle sample that recipes
04/04, 04/05 and 05/08 use. If it is absent the notebook prints the exact
`kaggle datasets download` commands and stops. That dataset is CC BY-NC 4.0;
review the licence before using derived work.
