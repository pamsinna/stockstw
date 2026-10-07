"""總成績單：7 個工具（S4/S5/S6/S7/S4∩S7、投信版雷達、外資版雷達）一起評估，做多重比較校正。

同時檢定 7 個工具、每個用 α=0.05：就算全部都沒有優勢，也有約 30% 的機率至少一個
「看起來顯著」。Holm 校正後仍顯著的，才算真的有證據。

主要指標（事先指定，不看結果再挑）：
- S4～S7、S4∩S7：照策略規則出場的報酬 − 同期 0050 含息（規則版本 RULES_VERSION 之後的訊號）
- 兩個雷達：上榜後 20 個交易日報酬 − 同期 0050 含息

  python scripts/scorecard.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import significance as sg  # noqa: E402
import live_scorecard as live  # noqa: E402
import radar_scorecard as radar  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    verdicts: dict[str, sg.Verdict] = {}
    df = live.scorecard()
    if not df.empty:
        for k, v in live.significance(df).items():
            if k != "合計":
                verdicts[k] = v
    for variant, (label, _) in radar.VARIANTS.items():
        r = radar.scorecard(variant)
        if not r.empty:
            verdicts[label.split()[-1] + "雷達"] = sg.verdict(*radar.excess(r, radar.PRIMARY_H))
    if not verdicts:
        print("尚無可評估的資料（凍結後的訊號還沒到期）")
        return
    ok = sg.holm({k: v.p_value for k, v in verdicts.items()})
    print(f"總成績單（{len(verdicts)} 個工具，Holm 多重比較校正）")
    for k, v in sorted(verdicts.items(), key=lambda kv: kv[1].p_value):
        mark = "✅ 校正後仍顯著" if ok[k] and v.mean > 0 else "—"
        print(f"  {k:<8} {mark}｜{sg.fmt(v)}")


if __name__ == "__main__":
    main()
