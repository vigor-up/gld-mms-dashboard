"""
CFTC COT 模組 - 黃金/白銀期貨持倉分析
資料來源：CFTC SOCRATA 公開 API（免費，無需 API key）
"""
import urllib.request, urllib.parse, json

CFTC_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

COMMODITY_MAP = {
    "GC=F": "GOLD - COMMODITY EXCHANGE INC.",
    "SI=F": "SILVER - COMMODITY EXCHANGE INC.",
}

def fetch_cot(commodity_name: str, weeks: int = 10) -> list:
    params = urllib.parse.urlencode({
        "$where": f"market_and_exchange_names='{commodity_name}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": str(weeks),
    })
    url = f"{CFTC_API}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())

def parse_cot(rows: list) -> dict:
    if not rows:
        return {"error": "no data"}

    r = rows[0]
    date = r.get("report_date_as_yyyy_mm_dd", "")[:10]

    comm_long  = int(r.get("comm_positions_long_all",  0) or 0)
    comm_short = int(r.get("comm_positions_short_all", 0) or 0)
    spec_long  = int(r.get("noncomm_positions_long_all",  0) or 0)
    spec_short = int(r.get("noncomm_positions_short_all", 0) or 0)
    open_int   = int(r.get("open_interest_all", 1) or 1)

    comm_net = comm_long - comm_short
    spec_net = spec_long - spec_short
    spec_net_pct = round(spec_net / open_int * 100, 1)
    spec_ls_ratio = round(spec_long / spec_short, 2) if spec_short else 999

    # 週環比趨勢
    trend = "─"
    if len(rows) >= 2:
        prev = rows[1]
        prev_spec_net = (int(prev.get("noncomm_positions_long_all", 0) or 0) -
                         int(prev.get("noncomm_positions_short_all", 0) or 0))
        delta = spec_net - prev_spec_net
        if delta > 2000:    trend = "▲ 投機加多"
        elif delta < -2000: trend = "▼ 投機減多"
        else:               trend = "─ 持平"

    # 信號評分
    if spec_net_pct > 15:
        signal, score_add = "📈 大戶偏多", 10
    elif spec_net_pct < 5:
        signal, score_add = "📉 大戶偏空", -10
    else:
        signal, score_add = "⏳ 中性", 0

    return {
        "date": date,
        "open_interest": open_int,
        "comm_net": comm_net,
        "spec_net": spec_net,
        "spec_net_pct": spec_net_pct,
        "spec_ls_ratio": spec_ls_ratio,
        "spec_long": spec_long,
        "spec_short": spec_short,
        "trend": trend,
        "signal": signal,
        "score_add": score_add,
        "summary": f"{date} | 投機淨多 {spec_net_pct}% OI | {trend} | {signal}",
    }

def get_gold_cot() -> dict:
    return parse_cot(fetch_cot(COMMODITY_MAP["GC=F"], weeks=10))

def get_silver_cot() -> dict:
    return parse_cot(fetch_cot(COMMODITY_MAP["SI=F"], weeks=10))

if __name__ == "__main__":
    print("=== 黃金 COT ===")
    g = get_gold_cot()
    for k, v in g.items(): print(f"  {k}: {v}")
    print("\n=== 白銀 COT ===")
    s = get_silver_cot()
    for k, v in s.items(): print(f"  {k}: {v}")
