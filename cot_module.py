"""
CFTC COT 模組 - 黃金/白銀期貨持倉分析
資料來源：CFTC SOCRATA 公開 API（免費，無需 API key）
更新頻率：每週五 15:30 ET 發布 → 本模組每週五 22:00 UTC 自動刷新快取
快取策略：本地 JSON 快取，有效期 7 天，避免重複打 API
"""
import urllib.request, urllib.parse, json, os
from datetime import datetime, timezone, timedelta

CFTC_API   = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cot_cache.json")
CACHE_TTL_HOURS = 168  # 7 天

COMMODITY_MAP = {
    "GC=F": "GOLD - COMMODITY EXCHANGE INC.",
    "SI=F": "SILVER - COMMODITY EXCHANGE INC.",
}

# ── 快取讀寫 ────────────────────────────────────────────────
def _load_cache() -> dict:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass

def _is_cache_valid(cache: dict, key: str) -> bool:
    if key not in cache:
        return False
    fetched_at = cache[key].get('_fetched_at')
    if not fetched_at:
        return False
    age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
    return age_hours < CACHE_TTL_HOURS

# ── CFTC API 呼叫 ───────────────────────────────────────────
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

# ── 解析 ────────────────────────────────────────────────────
def parse_cot(rows: list) -> dict:
    if not rows:
        return {"error": "no data"}
    r    = rows[0]
    date = r.get("report_date_as_yyyy_mm_dd", "")[:10]

    comm_long  = int(r.get("comm_positions_long_all",  0) or 0)
    comm_short = int(r.get("comm_positions_short_all", 0) or 0)
    spec_long  = int(r.get("noncomm_positions_long_all",  0) or 0)
    spec_short = int(r.get("noncomm_positions_short_all", 0) or 0)
    open_int   = int(r.get("open_interest_all", 1) or 1)

    comm_net      = comm_long - comm_short
    spec_net      = spec_long - spec_short
    spec_net_pct  = round(spec_net / open_int * 100, 1)
    spec_ls_ratio = round(spec_long / spec_short, 2) if spec_short else 999

    trend = "─"
    if len(rows) >= 2:
        prev = rows[1]
        prev_net = (int(prev.get("noncomm_positions_long_all", 0) or 0) -
                    int(prev.get("noncomm_positions_short_all", 0) or 0))
        delta = spec_net - prev_net
        if   delta >  2000: trend = "▲ 投機加多"
        elif delta < -2000: trend = "▼ 投機減多"
        else:               trend = "─ 持平"

    if   spec_net_pct > 15: signal, score_add = "📈 大戶偏多",  10
    elif spec_net_pct <  5: signal, score_add = "📉 大戶偏空", -10
    else:                   signal, score_add = "⏳ 中性",       0

    # 計算距下次發布時間（每週五 22:00 UTC）
    now = datetime.now(timezone.utc)
    days_ahead = (4 - now.weekday()) % 7  # 4 = 週五
    if days_ahead == 0 and now.hour >= 22:
        days_ahead = 7
    next_release = (now + timedelta(days=days_ahead)).replace(
        hour=22, minute=0, second=0, microsecond=0)
    hours_to_next = round((next_release - now).total_seconds() / 3600, 1)

    return {
        "date":           date,
        "open_interest":  open_int,
        "comm_net":       comm_net,
        "spec_net":       spec_net,
        "spec_net_pct":   spec_net_pct,
        "spec_ls_ratio":  spec_ls_ratio,
        "spec_long":      spec_long,
        "spec_short":     spec_short,
        "trend":          trend,
        "signal":         signal,
        "score_add":      score_add,
        "next_release_in_hours": hours_to_next,
        "note":           "CFTC 每週五發布，反映當週二持倉",
        "summary":        f"{date} | 投機淨多 {spec_net_pct}% OI | {trend} | {signal}",
    }

# ── 對外介面（帶快取）──────────────────────────────────────
def get_cot(ticker: str, force_refresh: bool = False) -> dict:
    commodity = COMMODITY_MAP.get(ticker)
    if not commodity:
        return {"error": f"unknown ticker {ticker}"}

    cache = _load_cache()
    cache_key = ticker

    if not force_refresh and _is_cache_valid(cache, cache_key):
        result = cache[cache_key].copy()
        result['_from_cache'] = True
        return result

    try:
        rows = fetch_cot(commodity, weeks=10)
        result = parse_cot(rows)
        result['_fetched_at'] = datetime.now(timezone.utc).isoformat()
        result['_from_cache'] = False
        cache[cache_key] = result
        _save_cache(cache)
        return result
    except Exception as e:
        # API 失敗時降級用快取
        if cache_key in cache:
            result = cache[cache_key].copy()
            result['_from_cache'] = True
            result['_fallback'] = str(e)
            return result
        return {"error": str(e)}

def get_gold_cot(force_refresh=False)   -> dict: return get_cot("GC=F", force_refresh)
def get_silver_cot(force_refresh=False) -> dict: return get_cot("SI=F", force_refresh)

def invalidate_cache():
    """每週五 cron 呼叫，強制刷新"""
    get_gold_cot(force_refresh=True)
    get_silver_cot(force_refresh=True)
    print(f"[COT] 快取已刷新 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

if __name__ == "__main__":
    print("=== 黃金 COT ===")
    g = get_gold_cot()
    for k, v in g.items(): print(f"  {k}: {v}")
    print(f"\n  下次 CFTC 發布：約 {g.get('next_release_in_hours')} 小時後")
