#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLD-MMS 系統 v6.0 - Top 10% 預測模型旗艦版
============================================
改進重點：
1. Enhanced Features — RSI / MACD / Bollinger / ADX / Stochastic / CCI / Williams%R / Momentum
2. Regime Detection — ATR-based 趨勢/區間/高波動 體制分類
3. Multi-Timeframe — 日線方向確認 + 4H 切入點
4. Cross-Asset — DXY / 10Y Treasury / VIX 信號加權
5. Enhanced Tech Score — 6 feature → 16+ feature 標準化評分
6. Backtest System — Sharpe / Max Drawdown / Calmar / Profit Factor
7. Model Ensemble — 純技術分析 XGBoost（共識 fallback）
8. Dynamic Position Sizing — Kelly Criterion / ATR-based
"""

import json, argparse, requests, boto3, os, sys
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# ── 白銀/台股/美股 Simple Asset Fetcher ───────────────────
TW_TICKER = '0050.TW'    # 寫死，不可自選
US_TICKER = 'NVDA'       # 寫死，不可自選
TW_NAME   = '元大台灣50'  # 寫死
US_NAME   = 'NVIDIA'      # 寫死

# ─── Bark 推播：只在狀態改變時推送 ─────────────────────────────────
_S3_STATE_KEY   = 'signal_state.json'
_S3_HISTORY_KEY = 'signal_history.json'

def _load_s3_json(s3_client, bucket, key):
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp['Body'].read().decode('utf-8'))
    except Exception:
        return None

def _save_s3_json(s3_client, bucket, key, data):
    try:
        s3_client.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
    except Exception as e:
        print(f"[WARN] S3 write {key} 失敗: {e}")

def _signal_tier(prob_up, prob_dn):
    """把連續機率映射成離散等級，避免微小波動觸發推播"""
    if prob_up >= 90:  return 'STRONG_BUY'
    if prob_up >= 80:  return 'BUY'
    if prob_dn >= 90:  return 'STRONG_SELL'
    if prob_dn >= 80:  return 'SELL'
    return 'WAIT'

def _should_push(prob_up, prob_dn, signal, s3_client=None, bucket='', state_key=None):
    """
    回傳 (要推播, 推播原因)
    規則：tier 改變才推，同狀態持續靜默；連續 4 小時發一次提醒
    """
    import time
    _key      = state_key or _S3_STATE_KEY
    last      = (_load_s3_json(s3_client, bucket, _key) or {}) if s3_client else {}
    now       = time.time()
    cur_tier  = _signal_tier(prob_up, prob_dn)
    last_tier = last.get('tier', 'UNKNOWN')
    last_ts   = last.get('ts', 0)

    reason = ''
    buy_t, sell_t = ('BUY', 'STRONG_BUY'), ('SELL', 'STRONG_SELL')
    if cur_tier != last_tier:
        if   last_tier in buy_t  and cur_tier in sell_t: reason = '⚠️ 方向反轉：買進→賣出'
        elif last_tier in sell_t and cur_tier in buy_t:  reason = '⚠️ 方向反轉：賣出→買進'
        elif cur_tier == 'STRONG_BUY':                   reason = '🚀 信心突破90%：強力買進'
        elif cur_tier == 'STRONG_SELL':                  reason = '🔴 信心突破90%：強力賣出'
        elif cur_tier == 'BUY':                          reason = '📈 新買進訊號（≥80%）'
        elif cur_tier == 'SELL':                         reason = '📉 新賣出訊號（≥80%）'
        elif cur_tier == 'WAIT' and last_tier not in ('WAIT', 'UNKNOWN'): reason = '⏸ 訊號轉為觀望'
    elif cur_tier not in ('WAIT', 'UNKNOWN') and (now - last_ts) > 4 * 3600:
        reason = f'⏰ 4h摘要 | {signal}'

    new_state = {'tier': cur_tier, 'ts': now if reason else last_ts}
    if s3_client:
        _save_s3_json(s3_client, bucket, _key, new_state)
    return bool(reason), reason


# ─── 實戰紀錄（存 S3，累積後計算真實勝率）──────────────────────────

def _update_signal_history(s3_client, bucket, signal, prob_up, prob_dn, score, price):
    """
    1. 載入歷史紀錄
    2. 回填已到期但未 verify 的舊紀錄（用當前 price 對比）
    3. 追加本次訊號（只記信心 ≥ 80%）
    4. 計算近 20 次 1d 勝率回傳
    5. 存回 S3
    """
    from datetime import datetime, timezone
    history = _load_s3_json(s3_client, bucket, _S3_HISTORY_KEY) or []
    now = datetime.now(timezone.utc)

    for rec in history:
        try:
            rec_ts = datetime.fromisoformat(rec['ts'].replace('Z', '+00:00'))
            age_h  = (now - rec_ts).total_seconds() / 3600
            ep, sig = rec.get('price_at_signal', 0), rec.get('signal', '')
            if not ep:
                continue
            if rec.get('result_1d') is None and 24 <= age_h < 168:
                pct = (price - ep) / ep * 100
                rec['result_1d'] = round(pct, 3)
                rec['win_1d'] = ('BUY' in sig and pct > 0) or ('SELL' in sig and pct < 0)
            if rec.get('result_5d') is None and 120 <= age_h < 360:
                pct = (price - ep) / ep * 100
                rec['result_5d'] = round(pct, 3)
                rec['win_5d'] = ('BUY' in sig and pct > 0) or ('SELL' in sig and pct < 0)
        except Exception:
            continue

    qualified = [r for r in history
                 if r.get('result_1d') is not None
                 and (r.get('prob_up', 0) >= 80 or r.get('prob_dn', 0) >= 80)]
    recent_20 = qualified[-20:]
    win_rate  = round(sum(1 for r in recent_20 if r.get('win_1d')) / len(recent_20) * 100, 1)                 if recent_20 else None

    if prob_up >= 80 or prob_dn >= 80:
        history.append({
            'ts': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'signal': signal,
            'prob_up': round(prob_up, 1), 'prob_dn': round(prob_dn, 1),
            'score': score, 'price_at_signal': round(price, 2),
            'result_1d': None, 'result_5d': None,
            'win_1d': None,    'win_5d': None,
        })

    history = history[-200:]
    _save_s3_json(s3_client, bucket, _S3_HISTORY_KEY, history)
    print(f"[INFO] 實戰紀錄 | 總={len(history)} 已驗={len(qualified)} 近20勝率={win_rate}%")
    return win_rate, len(recent_20)


# ─── Twelve Data API（取代 yfinance，不受 IP 封鎖影響）────────────────────
_TD_MAP = {
    'SI=F': ('XAG/USD', None),
    'GC=F': ('XAU/USD', None),
    'GLD':  ('GLD',     None),
}
def _td_symbol(ticker):
    if ticker in _TD_MAP:
        return _TD_MAP[ticker]
    if ticker.endswith('.TW'):
        return (ticker[:-3], 'TSE')
    return (ticker, None)

def _td_fetch(ticker, td_key):
    if not td_key:
        return None, None, None
    symbol, exchange = _td_symbol(ticker)
    params = {'symbol': symbol, 'interval': '1day', 'outputsize': 32, 'apikey': td_key}
    if exchange:
        params['exchange'] = exchange
    try:
        r = requests.get('https://api.twelvedata.com/time_series', params=params, timeout=12)
        data = r.json()
        if data.get('status') == 'error' or 'values' not in data:
            print(f"[WARN] Twelve Data {ticker}: {data.get('message','err')}")
            return None, None, None
        vals = data['values']
        closes = [float(v['close']) for v in vals]
        closes.reverse()
        latest = closes[-1] if closes else None
        prev   = closes[-2] if len(closes) > 1 else latest
        return closes, latest, prev
    except Exception as e:
        print(f"[WARN] Twelve Data {ticker}: {e}")
        return None, None, None

def get_asset_data(ticker, td_key=None):
    _, latest, prev = _td_fetch(ticker, td_key)
    if latest is not None:
        change = round((latest - prev) / prev * 100, 2) if prev else 0.0
        return {'price': round(latest, 2), 'change': change, 'score': None, 'factors': {}}
    try:
        df = yf.Ticker(ticker).history(period='5d')
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        if 'Close' not in df.columns:
            return None
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        if df.empty:
            return None
        close = float(df['Close'].iloc[-1])
        prev2 = float(df['Close'].iloc[-2]) if len(df) > 1 else close
        change = round((close - prev2) / prev2 * 100, 2) if prev2 else 0.0
        return {'price': round(close, 2), 'change': change, 'score': None, 'factors': {}}
    except Exception:
        return None

# ── COT ─────────────────────────────────────────────────────
try:
    from cot_module import get_gold_cot, get_silver_cot
    _COT_AVAILABLE = True
except Exception:
    _COT_AVAILABLE = False
    def _noop(force=False): return {}
    get_gold_cot   = _noop
    get_silver_cot = _noop

# ── 勝率追蹤（回測）─────────────────────────────────────────
def _load_history(path: str) -> list:
    if not os.path.exists(path): return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d.get('signal_history', []) or d.get('signals', [])
    except Exception:
        return []

def calc_win_rate_20(history: list, lookback: int = 20) -> float | None:
    if not history: return None
    recent = history[-lookback:]
    wins = 0
    for e in recent:
        sig = str(e.get('signal', ''))
        pu  = float(e.get('prob_up', 0))
        pd_ = float(e.get('prob_dn', 0))
        if ('BUY' in sig and pu > 50) or ('SELL' in sig and pd_ > 50):
            wins += 1
    return round(wins / len(recent) * 100, 1) if recent else None

def calc_backtest_metrics(history: list, lookback: int = 100) -> dict:
    if len(history) < 5:
        return {}
    recent = history[-lookback:]
    wins, losses = 0, 0
    total_pnl = 0.0
    win_amounts, loss_amounts = [], []
    peak = 0.0
    max_dd = 0.0
    daily_returns = []

    for i, e in enumerate(recent):
        sig  = str(e.get('signal', ''))
        pu   = float(e.get('prob_up', 0))
        pd_  = float(e.get('prob_dn', 0))
        ret  = float(e.get('return_pct', 0))
        daily_returns.append(ret)

        pnl_is_win = False
        if 'BUY' in sig and pu > 50:  pnl_is_win = (ret > 0); wins += 1
        elif 'SELL' in sig and pd_ > 50: pnl_is_win = (ret < 0); wins += 1
        elif 'BUY' in sig or 'SELL' in sig: wins += 1

        if pnl_is_win:
            win_amounts.append(abs(ret))
        else:
            loss_amounts.append(abs(ret))

        total_pnl += ret
        if i == 0: peak = ret
        else:
            peak = max(peak, ret)
            dd = peak - ret
            max_dd = max(max_dd, dd)

    n = len(recent)
    wr    = round(wins / n * 100, 1) if n else 0
    avg_w = round(np.mean(win_amounts), 4) if win_amounts else 0
    avg_l = round(np.mean(loss_amounts), 4) if loss_amounts else 0
    pf    = round((avg_w * wins) / (avg_l * losses), 2) if (avg_l * losses) > 0 else float('inf') if wins > losses else 0

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret  = np.std(daily_returns, ddof=1)
        sharpe   = round(mean_ret / std_ret * np.sqrt(252), 2) if std_ret > 0 else 0
    else:
        sharpe = 0

    annual_ret = total_pnl / n * 252 if n else 0
    calmar     = round(annual_ret / max_dd, 2) if max_dd > 0 else 0

    return {
        'total_trades': n,
        'win_rate':     wr,
        'avg_win':      avg_w,
        'avg_loss':     avg_l,
        'profit_factor': pf,
        'sharpe_ratio': sharpe,
        'max_drawdown': round(max_dd, 4),
        'calmar_ratio': calmar,
        'total_return_pct': round(total_pnl, 2),
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.int64, np.int32, int)): return int(obj)
        if isinstance(obj, (np.float64, np.float32, float)): return float(obj)
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class GldMmsUpdaterV6:
    def __init__(self, fred_key=None, bark_keys=None,
                 aws_ak=None, aws_sk=None, aws_region='ap-northeast-1'):
        self.fred_key    = fred_key
        self.bark_keys   = [k for k in (bark_keys or []) if k]
        self.aws_region  = aws_region
        self.assets      = {}
        self.daily       = {}
        self.macro       = {}
        self.lb_result   = {}
        self.asset_results = {}
        self.regime      = 'UNKNOWN'

        # lb_client 已移除（Lambda 改由 gld_xgb_ensemble.py 本地執行）
        self.s3_client = boto3.client('s3',    **boto_kwargs)

    def _clean(self, df):
        """強化版：解 yfinance 0.2.5x+ MultiIndex + object dtype 問題"""
        if df is None or df.empty:
            return df
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        for col in list(df.columns):
            if col in ('date', 'datetime', 'index', 'date_full'):
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'close' not in df.columns:
            return df.iloc[0:0]
        df = df.dropna(subset=['close'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'adj close']:
            if col in df.columns:
                try:
                    df[col] = df[col].astype('float64')
                except Exception:
                    pass
        return df

    def invoke_lambda(self):
        """
        四資產 Ensemble 推論：黃金 / 白銀 / 元大台灣50(0050) / 納斯達克(QQQ)
        XGBoost + LightGBM + CatBoost，模型存 Cloudflare R2
        首次執行自動訓練（約 8-12 分鐘），之後 30 天重訓一次
        """
        print("[INFO] 執行四資產 Ensemble 推論...")
        try:
            from gld_xgb_ensemble import run_all_assets
            _td_key    = os.environ.get('TWELVE_DATA_KEY', '')
            _r2_ep     = os.environ.get('R2_ENDPOINT_URL',
                             'https://adb1040c847f4ae4a7d6bfedcccd7b77.r2.cloudflarestorage.com')
            _r2_key    = os.environ.get('R2_ACCESS_KEY_ID', '')
            _r2_sec    = os.environ.get('R2_SECRET_ACCESS_KEY', '')
            _r2_bucket = os.environ.get('R2_BUCKET', 'richtrong-collect')
            _r2 = None
            if _r2_key and _r2_sec:
                import boto3 as _b3
                _r2 = _b3.client('s3',
                    endpoint_url=_r2_ep,
                    aws_access_key_id=_r2_key,
                    aws_secret_access_key=_r2_sec,
                    region_name='auto')
            all_results = run_all_assets(_td_key, _r2, _r2_bucket)
            self.lb_result     = all_results.get('gold', {})
            self.asset_results = all_results
            for key, res in all_results.items():
                print(f"[SUCCESS] {res.get('_asset_emoji','')} {res.get('_asset_name',key)}: "
                      f"{res.get('signal','?')} ({res.get('score','?')}) "
                      f"val={res.get('model',{}).get('val_acc',0):.3f}")
            return True
        except Exception as e:
            print(f"[WARN] 四資產 Ensemble 失敗: {e}")
            import traceback; traceback.print_exc()
        self.lb_result = {}
        self.asset_results = {}
        return False

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l, v, o = 'close','high','low','volume','open'

        # 防禦：強制把 OHLCV 全部轉 float（yfinance 0.2.5x+ 偶爾回 object dtype）
        for _col in (c, h, l, v, o):
            if _col in df.columns:
                df[_col] = pd.to_numeric(df[_col], errors='coerce')
        df = df.dropna(subset=[c]).copy()

        delta = df[c].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        ema12 = df[c].ewm(span=12, adjust=False).mean()
        ema26 = df[c].ewm(span=26, adjust=False).mean()
        df['macd']      = ema12 - ema26
        df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_sig']

        df['bb_mid']   = df[c].rolling(20).mean()
        df['bb_std']   = df[c].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_pos']   = (df[c] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        lo14 = df[l].rolling(14).min()
        hi14 = df[h].rolling(14).max()
        df['stoch_k'] = 100 * (df[c] - lo14) / (hi14 - lo14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        tr  = np.maximum(h - l,
               np.maximum(abs(h - df[c].shift(1)),
                          abs(l - df[c].shift(1))))
        plus_dm  = (df[h].diff().where(df[h].diff() > 0, 0)
                    .rolling(14).mean())
        minus_dm = (-df[l].diff().where(df[l].diff() < 0, 0)
                    .rolling(14).mean())
        atr14    = tr.rolling(14).mean()
        df['adx']        = 100 * abs(plus_dm - minus_dm) / (atr14 + 1e-10)
        df['adx_smooth'] = df['adx'].ewm(span=14).mean()
        df['atr']        = atr14

        tp = (h + l + c) / 3
        df['cci'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())

        df['williams_r'] = -100 * (hi14 - df[c]) / (hi14 - lo14 + 1e-10)

        df['momentum'] = df[c] / df[c].shift(10) - 1

        df['obv']     = (np.sign(df[c].diff()) * v).fillna(0).cumsum()
        df['obv_ma5'] = df['obv'].rolling(5).mean()

        df['vwap']     = ((h+l+c)/3 * v).rolling(5).sum() / v.rolling(5).sum()
        df['vwap_dev'] = (df[c] / df['vwap'] - 1) * 100

        df['atr_low']  = df['atr'] < df['atr'].rolling(50).min() * 1.1

        df['bull_div'] = ((df[c] < df[c].shift(3)) & (df['obv'] > df['obv'].shift(3)))
        df['bear_div'] = ((df[c] > df[c].shift(3)) & (df['obv'] < df['obv'].shift(3)))

        hl   = h - l
        body = abs(df[c] - o)
        df['pin_bar'] = (body < hl * 0.2) & (hl > (hl.rolling(20).mean() * 2))

        adx_now = df['adx'].iloc[-1]
        atr_now = df['atr'].iloc[-1]
        atr_sma = df['atr'].rolling(20).mean().iloc[-1]
        vol_ratio = atr_now / (atr_sma + 1e-10)

        if adx_now > 25 and vol_ratio < 1.2:
            self.regime = 'TRENDING'
        elif adx_now < 20 or vol_ratio > 1.5:
            self.regime = 'VOLATILE'
        else:
            self.regime = 'RANGING'

        return df

    def _fetch(self, ticker, name, period='60d', interval='1h'):
        print(f"[INFO] 獲取 {name} ({ticker}) [{interval}]...")
        try:
            try:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True,
                                 multi_level_index=False)
            except TypeError:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
            df = self._clean(df)
            if df is None or df.empty:
                print(f"[WARN] {ticker}: empty after clean")
                return False
            df.reset_index(inplace=True)
            tc = 'datetime' if 'datetime' in df.columns else ('date' if 'date' in df.columns else df.columns[0])
            df['date_full'] = pd.to_datetime(df[tc], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            df = self._calc_indicators(df)
            self.assets[ticker] = df.tail(100).to_dict('records')
            return True
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            return False

    def _fetch_daily(self, ticker, name, period='90d'):
        print(f"[INFO] 獲取日線 {name} ({ticker})...")
        try:
            try:
                df = yf.download(ticker, period=period, interval='1d',
                                 progress=False, auto_adjust=True,
                                 multi_level_index=False)
            except TypeError:
                df = yf.download(ticker, period=period, interval='1d',
                                 progress=False, auto_adjust=True)
            df = self._clean(df)
            if df is None or df.empty:
                print(f"[WARN] 日線 {ticker}: empty")
                return False
            df.reset_index(inplace=True)
            tc = 'datetime' if 'datetime' in df.columns else ('date' if 'date' in df.columns else df.columns[0])
            df['date_full'] = pd.to_datetime(df[tc], errors='coerce').dt.strftime('%Y-%m-%d')
            df = self._calc_indicators(df)
            self.daily[ticker] = df.tail(30).to_dict('records')
            return True
        except Exception as e:
            print(f"[WARN] 日線 {ticker}: {e}")
            return False

    def _fetch_cross_asset(self):
        tickers = {
            'DX-Y.NYB': 'USD_Index',
            '^TNX':     '10Y_Treasury',
            '^VIX':     'VIX',
        }
        for sym, name in tickers.items():
            try:
                try:
                    df = yf.download(sym, period='30d', interval='1d',
                                     progress=False, auto_adjust=True,
                                     multi_level_index=False)
                except TypeError:
                    df = yf.download(sym, period='30d', interval='1d',
                                     progress=False, auto_adjust=True)
                df = self._clean(df)
                if df is not None and not df.empty:
                    df['rsi'] = self._rsi_fast(df['close'], 14)
                    latest = df.iloc[-1]
                    rsi_val = float(latest['rsi']) if pd.notna(latest.get('rsi', None)) else 50.0
                    close_val = float(latest['close'])
                    self.macro[name] = {
                        'close': close_val,
                        'rsi':   rsi_val,
                        'note':  f"{name}: {close_val:.2f} (RSI {rsi_val:.0f})"
                    }
                    print(f"[INFO] {name}: {close_val:.2f}")
            except Exception as e:
                print(f"[WARN] {name}: {e}")

    def _calc_simple_signal_for_ticker(self, ticker, name, gld_history=None, td_key=None):
        """為任意 ticker 計算簡化版訊號（給自選股用）"""
        try:
            _td_closes, _, _ = _td_fetch(ticker, td_key)
            if _td_closes and len(_td_closes) >= 20:
                hist = pd.DataFrame({'Close': _td_closes, 'High': _td_closes, 'Low': _td_closes, 'Volume': [0]*len(_td_closes)})
                hist.index = pd.RangeIndex(len(hist))
            else:
                try:
                    hist = yf.Ticker(ticker).history(period='90d', interval='1d')
                except Exception as e:
                    print(f"[WARN] 自選股 {ticker} 抓資料失敗: {e}")
                    return None
            if hist is None or hist.empty or len(hist) < 20:
                print(f"[WARN] 自選股 {ticker} 資料不足")
                return None
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [str(c[0]) for c in hist.columns]
            if 'Close' not in hist.columns:
                return None
            hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
            hist = hist.dropna(subset=['Close'])
            if len(hist) < 20:
                return None
            close = hist['Close'].astype(float)

            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - 100 / (1 + rs)).iloc[-1]
            mom_pct = (close.iloc[-1] / close.iloc[-min(10, len(close)-1)] - 1) * 100

            cur = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) > 1 else cur
            change = round((cur - prev) / prev * 100, 2) if prev else 0.0

            corr = None
            if gld_history is not None and len(gld_history) >= 20 and len(close) >= 20:
                gld_ret = pd.Series(gld_history[-30:]).pct_change().dropna()
                this_ret = close[-30:].pct_change().dropna()
                ml = min(len(gld_ret), len(this_ret))
                if ml >= 10:
                    corr = float(pd.Series(this_ret.values[-ml:]).corr(
                        pd.Series(gld_ret.values[-ml:])))

            score = 0
            if not pd.isna(rsi):
                if rsi < 30: score += 30
                elif rsi < 45: score += 15
                elif rsi > 70: score -= 30
                elif rsi > 55: score -= 15
            if ma50 is not None and not pd.isna(ma20):
                if cur > ma20 > ma50: score += 25
                elif cur > ma20: score += 10
                elif cur < ma20 < ma50: score -= 25
                elif cur < ma20: score -= 10
            if not pd.isna(mom_pct):
                if mom_pct > 5: score += 20
                elif mom_pct > 2: score += 10
                elif mom_pct < -5: score -= 20
                elif mom_pct < -2: score -= 10

            if score >= 30:    sig, conf = 'BUY', min(95, 50 + score)
            elif score >= 15:  sig, conf = 'WEAK_BUY', min(85, 50 + score)
            elif score <= -30: sig, conf = 'SELL', min(95, 50 + abs(score))
            elif score <= -15: sig, conf = 'WEAK_SELL', min(85, 50 + abs(score))
            else:              sig, conf = 'WAIT', 50 + abs(score) // 2

            return {
                'ticker': ticker, 'name': name,
                'price': round(cur, 2), 'change': change,
                'signal': sig, 'confidence': int(conf),
                'rsi':  round(float(rsi), 1) if not pd.isna(rsi) else None,
                'ma20': round(float(ma20), 2) if not pd.isna(ma20) else None,
                'ma50': round(float(ma50), 2) if ma50 is not None and not pd.isna(ma50) else None,
                'momentum': round(float(mom_pct), 2) if not pd.isna(mom_pct) else None,
                'correlation': round(corr, 2) if corr is not None and not pd.isna(corr) else None,
                'history': [round(float(x), 2) for x in close.tail(30).tolist()],
                'score': int(score),
            }
        except Exception as e:
            print(f"[WARN] 自選股訊號 {ticker}: {e}")
            return None

    @staticmethod
    def _rsi_fast(series: pd.Series, n: int) -> pd.Series:
        d = series.diff()
        g = d.where(d > 0, 0).rolling(n).mean()
        l = (-d.where(d < 0, 0)).rolling(n).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    def fetch_macro(self):
        print("[INFO] 獲取宏觀 + COT...")
        try:
            df = yf.download('UUP', period='60d', interval='1d', progress=False)
            df = self._clean(df)
            if not df.empty:
                df['rsi'] = self._rsi_fast(df['close'], 14)
                df.reset_index(inplace=True)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                self.macro['dxy'] = df.tail(30).to_dict('records')
        except: pass

        if _COT_AVAILABLE:
            try:
                self.macro['cot_gold']   = get_gold_cot()
                self.macro['cot_silver'] = get_silver_cot()
                g = self.macro['cot_gold']
                print(f"[SUCCESS] COT: {g.get('summary','')}")
            except Exception as e:
                print(f"[WARN] COT: {e}")
                self.macro['cot_gold'] = {}
        return True

    def _calc_tech_score(self, row: dict, daily_row: dict | None = None) -> dict:
        regime = self.regime
        score = 50
        detail = []

        rsi = row.get('rsi', 50)
        if   rsi < 25: score += 10; detail.append(f"RSI超賣{rsi:.0f}")
        elif rsi < 35: score += 5;  detail.append(f"RSI偏低{rsi:.0f}")
        elif rsi > 75: score -= 10; detail.append(f"RSI超買{rsi:.0f}")
        elif rsi > 65: score -= 5;  detail.append(f"RSI偏高{rsi:.0f}")

        mh = row.get('macd_hist', 0)
        if   mh > 0.5:  score += 8; detail.append("MACD看漲")
        elif mh > 0.1:  score += 4; detail.append("MACD偏多")
        elif mh < -0.5: score -= 8; detail.append("MACD看跌")
        elif mh < -0.1: score -= 4; detail.append("MACD偏空")

        bbp = row.get('bb_pos', 0.5)
        if   bbp < 0.1: score += 8; detail.append("BB下軌超賣")
        elif bbp < 0.25: score += 4; detail.append("BB偏低")
        elif bbp > 0.9: score -= 8; detail.append("BB上軌超買")
        elif bbp > 0.75: score -= 4; detail.append("BB偏高")

        sk = row.get('stoch_k', 50)
        sd = row.get('stoch_d', 50)
        if sk < 20 and sd < 30: score += 6; detail.append("Stoch超賣")
        elif sk > 80 and sd > 70: score -= 6; detail.append("Stoch超買")
        elif sk > sd and sk < 50: score += 3; detail.append("Stoch金叉")
        elif sk < sd and sk > 50: score -= 3; detail.append("Stoch死叉")

        adx = row.get('adx', 0)
        if   adx > 30: score += 4; detail.append(f"ADX強趨勢{adx:.0f}")
        elif adx < 15: score -= 2; detail.append(f"ADX盤整{adx:.0f}")

        cci = row.get('cci', 0)
        if   cci < -150: score += 5; detail.append(f"CCI超賣{cci:.0f}")
        elif cci > 150:  score -= 5; detail.append(f"CCI超買{cci:.0f}")
        elif cci > 0:   score += 2; detail.append("CCI偏多")

        wr = row.get('williams_r', -50)
        if   wr < -80: score += 5; detail.append("W%R超賣")
        elif wr > -20: score -= 5; detail.append("W%R超買")

        mom = row.get('momentum', 0)
        if   mom > 0.02: score += 5; detail.append(f"動量正{mom*100:.1f}%")
        elif mom < -0.02: score -= 5; detail.append(f"動量負{mom*100:.1f}%")

        vdev = row.get('vwap_dev', 0)
        if   vdev < -1.5: score += 8; detail.append(f"VWAP低估{vdev:.1f}%")
        elif vdev < -0.5: score += 4; detail.append(f"VWAP偏低{vdev:.1f}%")
        elif vdev > 1.5:  score -= 8; detail.append(f"VWAP高估{vdev:.1f}%")
        elif vdev > 0.5:  score -= 4; detail.append(f"VWAP偏高{vdev:.1f}%")

        obv = row.get('obv', 0)
        obv5 = row.get('obv_ma5', obv)
        if   obv > obv5: score += 5; detail.append("OBV偏多")
        else:            score -= 5; detail.append("OBV偏空")

        if row.get('atr_low'): score += 4; detail.append("ATR擠壓")
        if row.get('bull_div'): score += 10; detail.append("底背離")
        if row.get('bear_div'): score -= 10; detail.append("頂背離")
        if row.get('pin_bar'): score -= 3; detail.append("Pin Bar")

        if regime == 'VOLATILE':
            score = 50 + (score - 50) * 0.5
            detail.append("⚠️高波動體制，倉位減半")
        elif regime == 'TRENDING':
            if 'MACD看漲' in str(detail) or '底背離' in str(detail): score += 5
            if 'MACD看跌' in str(detail) or '頂背離' in str(detail): score -= 5

        if daily_row:
            d_rsi  = daily_row.get('rsi', 50)
            d_macd = daily_row.get('macd_hist', 0)
            d_adx  = daily_row.get('adx', 0)
            if d_rsi < 40 and d_macd > 0 and d_adx > 20:
                score += 6; detail.append("日線確認看多")
            elif d_rsi > 60 and d_macd < 0 and d_adx > 20:
                score -= 6; detail.append("日線確認看空")

        dxy = self.macro.get('USD_Index', {})
        if dxy:
            dxy_rsi = dxy.get('rsi', 50)
            if dxy_rsi > 65: score -= 4; detail.append("DXY超買→黃金壓力")
            elif dxy_rsi < 35: score += 4; detail.append("DXY超賣→黃金支撐")

        score = max(0, min(100, score))
        return {'score': score, 'detail': detail}

    def _pure_tech_ensemble(self, row: dict, daily_row: dict | None = None) -> dict:
        votes = {'bull': 0, 'bear': 0, 'neutral': 0, 'score_contrib': 0}

        adx  = row.get('adx', 0)
        mh   = row.get('macd_hist', 0)
        if adx > 25:
            if mh > 0:   votes['bull'] += 2
            else:        votes['bear'] += 2
        else:
            votes['neutral'] += 1

        rsi = row.get('rsi', 50)
        sk  = row.get('stoch_k', 50)
        wr  = row.get('williams_r', -50)
        if rsi < 40 and sk < 40: votes['bull'] += 2
        elif rsi > 60 and sk > 60: votes['bear'] += 2
        elif rsi < 50: votes['bull'] += 1; votes['bear'] -= 0.5
        else: votes['bear'] += 1; votes['bull'] -= 0.5

        bbp  = row.get('bb_pos', 0.5)
        atr_ = row.get('atr_low', False)
        if bbp < 0.2: votes['bull'] += 2
        elif bbp > 0.8: votes['bear'] += 2
        if atr_: votes['bull'] += 1

        obv  = row.get('obv', 0)
        obv5 = row.get('obv_ma5', obv)
        if obv > obv5: votes['bull'] += 1.5
        else: votes['bear'] += 1.5

        vdev = row.get('vwap_dev', 0)
        if vdev < -1.0: votes['bull'] += 2
        elif vdev > 1.0: votes['bear'] += 2
        elif vdev < 0: votes['bull'] += 0.5
        else: votes['bear'] += 0.5

        if row.get('bull_div'): votes['bull'] += 3
        if row.get('bear_div'): votes['bear'] += 3

        total = votes['bull'] + votes['bear'] + votes['neutral']
        bull_ratio = votes['bull'] / total if total > 0 else 0.5
        raw_score = bull_ratio * 100
        consensus_strength = abs(votes['bull'] - votes['bear']) / (total + 0.1)
        final_score = raw_score * (0.7 + 0.3 * min(consensus_strength, 1))

        signal_map = {
            'STRONG_BUY':  final_score >= 75,
            'BUY':         final_score >= 60,
            'PRE_BUY':     final_score >= 52,
            'HOLD':        45 < final_score < 52,
            'PRE_SELL':    40 <= final_score <= 45,
            'SELL':        final_score <= 35,
            'STRONG_SELL': final_score <= 25,
        }
        signal = 'HOLD'
        for k, v in signal_map.items():
            if v: signal = k; break

        return {
            'score':  round(final_score, 1),
            'signal': signal,
            'module_votes': votes,
            'confidence':   round(consensus_strength * 100, 1),
            'version':      'v6.0 Pure-Tech Ensemble'
        }

    def _calc_position_size(self, combined_score: int,
                             regime: str, atr: float,
                             price: float) -> dict:
        win_r  = self.lb_result.get('performance', {}).get('win_rate', 50) / 100
        rr     = 2.0
        kelly  = (win_r * rr - (1 - win_r)) / rr
        kelly  = max(0.05, min(0.25, kelly))

        if regime == 'VOLATILE': kelly *= 0.5
        elif regime == 'RANGING': kelly *= 0.75

        risk_amount = price * 0.01
        contracts   = max(1, round(risk_amount / (atr + 1e-10)))

        return {
            'kelly_fraction': round(kelly, 3),
            'recommended_contracts': int(contracts),
            'regime': regime,
            'atr_risk_pct': round(atr / price * 100, 2),
        }

    def send_push(self, title, content, is_leading=False):
        grp  = "GLD-LEADING" if is_leading else "GLD-MMS"
        icon = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/gold.png'
        ok   = 0
        for key in self.bark_keys:
            try:
                url = (f"https://api.day.app/{key}/{title}/{content}"
                       f"?icon={icon}&group={grp}&isArchive=1")
                requests.get(url, timeout=10)
                ok += 1
            except Exception as e:
                print(f"[WARN] Push {ok+1}: {e}")
        print(f"[INFO] 推播完成: {ok}/{len(self.bark_keys)}")

    def _lambda_fallback_signal(self) -> dict:
        lb_score = self.lb_result.get('score', None)
        if lb_score is None:
            return {}
        # ── 實戰紀錄 + Bark state-diff 推播 ────────────────────────
        _live_win_rate, _live_samples = None, 0
        _push_s3 = None
        try:
            _aws_key    = os.environ.get('AWS_ACCESS_KEY_ID', '')
            _aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
            _bucket     = 'gld-mms-data-richtrong'
            if _aws_key and _aws_secret:
                _push_s3 = boto3.client('s3', region_name='ap-northeast-1',
                                        aws_access_key_id=_aws_key,
                                        aws_secret_access_key=_aws_secret)
                # 取當前黃金收盤價
                _gp = 0.0
                for _sd in self.signals.values():
                    _p = (_sd or {}).get('short_term', {}).get('price') or                          (_sd or {}).get('feature_vector', {}).get('close')
                    if _p:
                        _gp = float(_p)
                        break
                if not _gp and self.daily.get('GC=F'):
                    _gp = float(self.daily['GC=F'][-1].get('close', 0))
                # 更新實戰紀錄
                _live_win_rate, _live_samples = _update_signal_history(
                    _push_s3, _bucket,
                    signal  = str(lb.get('signal', '')),
                    prob_up = float(lb.get('prob_up', 50)),
                    prob_dn = float(lb.get('prob_dn', 50)),
                    score   = int(lb.get('score', 0)),
                    price   = _gp,
                )
        except Exception as _he:
            print(f"[WARN] 實戰紀錄失敗: {_he}")
        # ────────────────────────────────────────────────────────────

        # 為自選股相關性計算準備 GLD 30 日 close 序列
        gold_history = []
        _td_key = os.environ.get('TWELVE_DATA_KEY', '')
        try:
            _gc_closes, _, _ = _td_fetch('GC=F', _td_key)
            if _gc_closes:
                gold_history = [round(x, 2) for x in _gc_closes[-30:]]
            if not gold_history and 'GC=F' in self.daily and self.daily['GC=F']:
                gold_history = [float(r.get('close', 0)) for r in self.daily['GC=F'][-30:] if r.get('close')]
            if not gold_history:
                _gh = yf.Ticker('GLD').history(period='90d', interval='1d')
                if _gh is not None and not _gh.empty:
                    if isinstance(_gh.columns, pd.MultiIndex):
                        _gh.columns = [str(c[0]) for c in _gh.columns]
                    if 'Close' in _gh.columns:
                        _gh['Close'] = pd.to_numeric(_gh['Close'], errors='coerce')
                        _gh = _gh.dropna(subset=['Close'])
                        gold_history = [round(float(x), 2) for x in _gh['Close'].tail(30).tolist()]
        except Exception as e:
            print(f"[WARN] 黃金歷史走勢抓取失敗: {e}")
        if not gold_history:
            _lp = float(self.lb_result.get('gold', {}).get('price', 0) or 0)
            if _lp > 0:
                gold_history = [_lp]

        # 取最新技術指標快照（供信號品質評分器使用）
        _latest_features = {}
        try:
            if hasattr(self, 'daily') and 'GC=F' in (self.daily or {}):
                _raw_df = pd.DataFrame(self.daily['GC=F']).tail(60)
                if not _raw_df.empty and 'close' in _raw_df.columns:
                    from gld_xgb_ensemble import add_features
                    _feat_df = add_features(_raw_df.rename(columns={
                        'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'
                    }), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
                    _latest_features = _feat_df.iloc[-1].dropna().to_dict()
        except Exception:
            pass
        self.lb_result['_latest_features'] = _latest_features

        cot = self.macro.get('cot_gold', {})
        cot_bias = cot.get('score_add', 0)
        combined = round(max(0, min(100, lb_score + cot_bias)))
        if   combined >= 80: signal = 'STRONG_BUY'
        elif combined >= 65: signal = 'BUY'
        elif combined >= 55: signal = 'PRE_BUY'
        elif combined <= 20: signal = 'STRONG_SELL'
        elif combined <= 35: signal = 'SELL'
        elif combined <= 45: signal = 'PRE_SELL'
        else:                signal = 'NEUTRAL'
        price = self.lb_result.get('gold', {}).get('price', 0)
        print(f'[INFO] Lambda Fallback 信號: {signal} {combined}%')
        return {
            'GC=F': {
                'ticker':  'GC=F',
                'short_term': {
                    'signal':     signal,
                    'confidence': combined,
                    'tech_score': lb_score,
                    'lambda_score': lb_score,
                    'reason':     f'Lambda AI {lb_score}% + COT{cot_bias:+.0f}（技術指標暫時不可用）',
                    'regime':     'UNKNOWN',
                },
                'feature_vector': {
                    'rsi': 50.0, 'macd_hist': 0.0, 'bb_pos': 0.5,
                    'stoch_k': 50.0, 'adx': 0.0, 'cci': 0.0, 'williams_r': -50.0,
                    'momentum_pct': 0.0, 'vwap_dev': 0.0,
                    'cot_score_add': cot_bias,
                    'cot_spec_net_pct': 0,
                    'cot_spec_ls_ratio': 1.0,
                    'regime': 'UNKNOWN', 'daily_rsi': None,
                },
                'radar': {
                    'msg':    '技術指標不可用（yfinance 資料格式異常）',
                    'bull_div': False, 'bear_div': False, 'atr_low': False, 'regime': 'UNKNOWN',
                },
                'smart_money': {
                    **self.lb_result.get('smart_money', {}),
                    'cot_report':  '待更新（技術指標 Fallback）',
                    'cot_detail':  cot,
                },
                'performance': self.lb_result.get('performance', {}),
                'position_sizing': {
                    'atr_risk_pct': 2.0, 'kelly_fraction': 0.5,
                    'recommended_contracts': 1,
                },
                'breakdown':  [
                    f'AI評分: {lb_score}% ({signal})',
                    f'COT大戶偏多: COT{cot_bias:+.0f}',
                    '體制: UNKNOWN（技術指標不可用）',
                    f'Kelly倉位: 50% (1合約) | COT+{cot_bias:+.0f}',
                ],
                'price':   float(price),
                'close':   float(price),
                'vwap_dev': 0.0,
            }
        }

    def calculate_signals(self):
        PUSH = 80
        signals = {}

        lb_score = self.lb_result.get('score', None)
        lb_signal = self.lb_result.get('signal', '')
        smart_money = self.lb_result.get('smart_money', {})
        perf = self.lb_result.get('performance', {})

        cot = self.macro.get('cot_gold', {})
        cot_bias = cot.get('score_add', 0)
        cot_net  = cot.get('spec_net_pct', 0)
        cot_ls   = cot.get('spec_ls_ratio', 1.0)

        if not self.assets and lb_score is not None:
            price = self.lb_result.get('gold', {}).get('price', 0)
            combined = round(max(0, min(100, lb_score + cot_bias)))
            if   combined >= 80: signal = 'STRONG_BUY'
            elif combined >= 65: signal = 'BUY'
            elif combined >= 55: signal = 'PRE_BUY'
            elif combined <= 20: signal = 'STRONG_SELL'
            elif combined <= 35: signal = 'SELL'
            else:                signal = 'NEUTRAL'
            signals['GC=F'] = {
                'ticker':  'GC=F',
                'short_term': {
                    'signal':     signal,
                    'confidence': combined,
                    'tech_score': lb_score,
                    'lambda_score': lb_score,
                    'reason':     f'Lambda AI {lb_score}% + COT{cot_bias:+.0f}（技術指標暫時不可用）',
                    'regime':     self.regime,
                },
                'feature_vector': {
                    'rsi': 50.0, 'macd_hist': 0.0, 'bb_pos': 0.5,
                    'stoch_k': 50.0, 'adx': 0.0, 'cci': 0.0, 'williams_r': -50.0,
                    'momentum_pct': 0.0, 'vwap_dev': 0.0,
                    'cot_score_add': cot_bias,
                    'cot_spec_net_pct': cot_net,
                    'cot_spec_ls_ratio': cot_ls,
                    'regime': self.regime, 'daily_rsi': None,
                },
                'radar': {
                    'msg': '技術指標不可用（yfinance 資料格式異常）',
                    'bull_div': False, 'bear_div': False, 'atr_low': False, 'regime': self.regime,
                },
                'smart_money': {
                    **smart_money,
                    'cot_report': cot.get('summary', '待更新'),
                    'cot_detail': cot,
                },
                'performance': perf,
                'position_sizing': {'atr_risk_pct': 2.0, 'kelly_fraction': 0.5, 'recommended_contracts': 1},
                'breakdown':  [
                    f'AI評分: {lb_score}% ({lb_signal or "N/A"})',
                    f'COT大戶偏多: {cot_net}% OI (LS比 {cot_ls})',
                    f'體制: {self.regime}（技術指標不可用）',
                    f'Kelly倉位: 50% (1合約) | COT+{cot_bias:+.0f}',
                ],
                'price': float(price) if price else 0.0,
                'close': float(price) if price else 0.0,
                'vwap_dev': 0.0,
            }
            print(f'[INFO] Fallback Lambda 信號: {signal} {combined}%')

        if not signals:
            print('[INFO] 進入 Fallback 2 應急模式（yfinance 即時報價）')
            try:
                df_yf = yf.download('GC=F', period='5d', interval='1h', progress=False, auto_adjust=True)
                if df_yf.empty:
                    raise ValueError('yfinance 返回空數據')
                close = float(df_yf['close'].iloc[-1])
                prev_close = float(df_yf['close'].iloc[-2]) if len(df_yf) > 1 else close
                change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                if   change_pct > 0.5:  simple_signal = 'STRONG_BUY'
                elif change_pct > 0.1:  simple_signal = 'BUY'
                elif change_pct < -0.5: simple_signal = 'STRONG_SELL'
                elif change_pct < -0.1: simple_signal = 'SELL'
                else:                   simple_signal = 'NEUTRAL'
                base_score = 50 + change_pct * 5 + cot_bias
                combined = round(max(0, min(100, base_score)))
                signals['GC=F'] = {
                    'ticker':  'GC=F',
                    'short_term': {
                        'signal':     simple_signal,
                        'confidence': combined,
                        'tech_score': round(50 + change_pct * 5),
                        'lambda_score': None,
                        'reason':     f'yfinance 即時 {close:.1f} ({change_pct:+.2f}%) + COT{cot_bias:+.0f}（應急模式）',
                        'regime':     'UNKNOWN',
                    },
                    'feature_vector': {
                        'rsi': 50.0, 'macd_hist': 0.0, 'bb_pos': 0.5,
                        'stoch_k': 50.0, 'adx': 0.0, 'cci': 0.0, 'williams_r': -50.0,
                        'momentum_pct': change_pct * 10, 'vwap_dev': 0.0,
                        'cot_score_add': cot_bias,
                        'cot_spec_net_pct': cot_net,
                        'cot_spec_ls_ratio': cot_ls,
                        'regime': 'UNKNOWN', 'daily_rsi': None,
                    },
                    'radar': {
                        'msg': 'yfinance 即時報價應急模式',
                        'bull_div': False, 'bear_div': False, 'atr_low': False, 'regime': 'UNKNOWN',
                    },
                    'smart_money': {
                        **smart_money,
                        'cot_report': cot.get('summary', '應急'),
                        'cot_detail': cot,
                    },
                    'performance': perf,
                    'position_sizing': {'atr_risk_pct': 2.0, 'kelly_fraction': 0.5, 'recommended_contracts': 1},
                    'breakdown':  [
                        f'即時價格: {close:.2f} ({change_pct:+.2f}%)',
                        f'COT大戶偏多: {cot_net}% OI (LS比 {cot_ls})',
                        '體制: UNKNOWN（應急模式）',
                        f'Kelly倉位: 50% (1合約) | COT+{cot_bias:+.0f}',
                    ],
                    'price': close,
                    'close': close,
                    'vwap_dev': 0.0,
                }
                print(f'[INFO] 應急 Fallback 信號: {simple_signal} {combined}% (close={close})')
                try:
                    df_ag = yf.download('SI=F', period='5d', interval='1h', progress=False, auto_adjust=True)
                    if not df_ag.empty:
                        close_ag = float(df_ag['close'].iloc[-1])
                        prev_ag  = float(df_ag['close'].iloc[-2]) if len(df_ag) > 1 else close_ag
                        chg_ag = ((close_ag - prev_ag) / prev_ag * 100) if prev_ag else 0
                        if   chg_ag > 0.5:  sig_ag = 'STRONG_BUY'
                        elif chg_ag > 0.1:  sig_ag = 'BUY'
                        elif chg_ag < -0.5: sig_ag = 'STRONG_SELL'
                        elif chg_ag < -0.1: sig_ag = 'SELL'
                        else:               sig_ag = 'NEUTRAL'
                        sc_ag = round(max(0, min(100, 50 + chg_ag * 5 + cot_bias)))
                        signals['SI=F'] = {
                            'ticker': 'SI=F',
                            'short_term': {
                                'signal': sig_ag, 'confidence': sc_ag,
                                'tech_score': round(50 + chg_ag * 5), 'lambda_score': None,
                                'reason': f'SI即時 {close_ag:.2f} ({chg_ag:+.2f}%) + COT{cot_bias:+.0f}', 'regime': 'UNKNOWN',
                            },
                            'feature_vector': {
                                'rsi': 50.0, 'macd_hist': 0.0, 'bb_pos': 0.5,
                                'stoch_k': 50.0, 'adx': 0.0, 'cci': 0.0, 'williams_r': -50.0,
                                'momentum_pct': chg_ag * 10, 'vwap_dev': 0.0,
                                'cot_score_add': cot_bias, 'cot_spec_net_pct': 0, 'cot_spec_ls_ratio': 1.0,
                                'regime': 'UNKNOWN', 'daily_rsi': None,
                            },
                            'radar': {'msg': '白銀 yfinance 即時模式', 'bull_div': False, 'bear_div': False, 'atr_low': False, 'regime': 'UNKNOWN'},
                            'smart_money': {**smart_money, 'cot_report': '待更新', 'cot_detail': {}},
                            'performance': perf,
                            'position_sizing': {'atr_risk_pct': 2.0, 'kelly_fraction': 0.5, 'recommended_contracts': 1},
                            'breakdown':  [f'白銀即時: {close_ag:.2f} ({chg_ag:+.2f}%)', f'COT+{cot_bias:+.0f}（應急）'],
                            'price': close_ag, 'close': close_ag, 'vwap_dev': 0.0,
                        }
                        print(f'[INFO] SI=F 應急信號: {sig_ag} {sc_ag}%')
                except Exception as e2:
                    print(f'[WARN] SI=F 抓取失敗: {e2}')
            except Exception as e:
                print(f'[WARN] 應急 Fallback 完全失敗: {e}')

        for ticker, raw_data in self.assets.items():
            df    = pd.DataFrame(raw_data)
            now   = df.iloc[-1].to_dict()

            daily_row = None
            if ticker in self.daily:
                daily_row = pd.DataFrame(self.daily[ticker]).iloc[-1].to_dict()

            tech = self._calc_tech_score(now, daily_row)

            if lb_score is None:
                ensemble = self._pure_tech_ensemble(now, daily_row)
                lb_score  = ensemble['score']
                lb_signal  = ensemble['signal']
                ensemble_note = f"【純技術 Ensemble 共識】{ensemble['confidence']}%共識"
            else:
                ensemble_note = ''

            atr_now = now.get('atr', 0)
            pos = self._calc_position_size(tech['score'], self.regime,
                                            atr_now, now['close'])

            if lb_score is not None:
                combined = round(max(0, min(100,
                    lb_score * 0.55
                    + tech['score'] * 0.35
                    + cot_bias * 1.0)))
                note = (f"LambdaAI {lb_score}%×55%"
                        f" + 技術{tech['score']}%×35%"
                        f" + COT{cot_bias:+.0f}×10%")
            else:
                combined = tech['score']
                note = f"純技術分析 {tech['score']}%"
                if ensemble_note:
                    note = ensemble_note + ' | ' + note

            if   combined >= 80: signal = 'STRONG_BUY'
            elif combined >= 65: signal = 'BUY'
            elif combined >= 55: signal = 'PRE_BUY'
            elif combined <= 20: signal = 'STRONG_SELL'
            elif combined <= 35: signal = 'SELL'
            elif combined <= 45: signal = 'PRE_SELL'
            else:                signal = 'HOLD'

            breakdown = [
                f"AI評分: {lb_score or '?'}% ({lb_signal or 'N/A'})",
                f"技術分: {tech['score']} {' '.join(tech['detail'][:3])}",
                f"COT大戶: {cot_net}% OI (LS比 {cot_ls})",
                f"體制: {self.regime} | ATR風險: {pos['atr_risk_pct']}%",
                f"Kelly倉位: {pos['kelly_fraction']*100:.0f}% ({pos['recommended_contracts']}合約)",
            ]
            if ensemble_note: breakdown.insert(0, ensemble_note)

            radar_msgs = []
            if now.get('bull_div'):  radar_msgs.append("底背離")
            if now.get('bear_div'):  radar_msgs.append("頂背離")
            if now.get('atr_low'):   radar_msgs.append("ATR擠壓")
            if now.get('rsi', 50) < 35: radar_msgs.append("RSI超賣")
            if now.get('rsi', 50) > 65: radar_msgs.append("RSI超買")
            if self.regime == 'VOLATILE': radar_msgs.append("⚠️高波動")
            if self.regime == 'TRENDING': radar_msgs.append("→趨勢")

            if ticker == 'GC=F' and combined > PUSH:
                price = float(now['close'])
                etf   = smart_money.get('etf_flow', '?')
                cot_r = cot.get('summary', '?')
                # 改用 state-diff 推播（Bark 只在狀態改變時響）
                _sp_tier = _signal_tier(float(self.lb_result.get('prob_up', 0)),
                                        float(self.lb_result.get('prob_dn', 0)))
                _sp_last = (_load_s3_json(None, '', _S3_STATE_KEY) or {}).get('tier', 'UNKNOWN') \
                           if False else 'SKIP'  # 此處由 update_html 統一處理，跳過

            signals[ticker] = {
                'short_term': {
                    'signal':     signal,
                    'confidence': combined,
                    'tech_score': tech['score'],
                    'lambda_score': lb_score,
                    'reason':     note,
                    'regime':     self.regime,
                },
                'feature_vector': {
                    'rsi':            round(now.get('rsi', 0), 1),
                    'macd_hist':      round(now.get('macd_hist', 0), 2),
                    'bb_pos':         round(now.get('bb_pos', 0.5), 3),
                    'stoch_k':        round(now.get('stoch_k', 50), 1),
                    'adx':            round(now.get('adx', 0), 1),
                    'cci':            round(now.get('cci', 0), 1),
                    'williams_r':     round(now.get('williams_r', -50), 1),
                    'momentum_pct':   round(now.get('momentum', 0) * 100, 2),
                    'vwap_dev':       round(now.get('vwap_dev', 0), 2),
                    'cot_score_add':  cot_bias,
                    'cot_spec_net_pct': cot_net,
                    'cot_spec_ls_ratio': cot_ls,
                    'regime':         self.regime,
                    'daily_rsi':      round(daily_row.get('rsi', 50), 1) if daily_row else None,
                },
                'radar': {
                    'msg':    ' | '.join(radar_msgs) if radar_msgs else '掃描正常',
                    'bull_div': bool(now.get('bull_div')),
                    'bear_div': bool(now.get('bear_div')),
                    'atr_low':   bool(now.get('atr_low')),
                    'regime':    self.regime,
                },
                'smart_money': {
                    **smart_money,
                    'cot_report':  cot.get('summary', '待更新'),
                    'cot_detail':  cot,
                },
                'performance': perf,
                'position_sizing': pos,
                'breakdown':  breakdown,
                'price':      float(round(now['close'], 2)),
                'vwap_dev':   float(round(now.get('vwap_dev', 0), 2)),
            }

        return signals

    def update_html(self, html_file: str):
        DATA_JSON_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data.json')
        history    = _load_history(DATA_JSON_PATH)
        win_rate   = calc_win_rate_20(history, 20)
        bt_metrics = calc_backtest_metrics(history, 100)

        cot = self.macro.get('cot_gold', {})
        breakdown = [
            f"COT大戶偏多: {cot.get('spec_net_pct', 0)}% OI",
            f"LS比率: {cot.get('spec_ls_ratio', '?')}",
        ]
        if win_rate is not None:
            breakdown.append(f"近20次勝率: {win_rate}%")

        lb_model = self.lb_result.get('model', {})
        perf = self.lb_result.get('performance', {})
        prob_up = self.lb_result.get('prob_up')
        prob_dn = self.lb_result.get('prob_dn')

        data = {
            'timestamp':    datetime.utcnow().isoformat() + 'Z',
            'version':      'v6.0 Top-10%-Model',
            'regime':       self.regime,
            'assets': _build_asset_dict(self.asset_results, gold_history, _td_key),
            'gold_history': gold_history,
            'tickers_meta': {
                'gold':   {'ticker': 'GC=F',    'name': '黃金',       'emoji': '🥇'},
                'silver': {'ticker': 'SI=F',    'name': '白銀',       'emoji': '🥈'},
                'tw':     {'ticker': '0050.TW', 'name': '元大台灣50', 'emoji': '🇹🇼'},
                'us':     {'ticker': '^IXIC',   'name': '納斯達克',   'emoji': '🇺🇸'},
            },
            'daily':        self.daily,
            'macro':        self.macro,
            'signals':      self.calculate_signals() or self._lambda_fallback_signal(),
            'win_rate_20':  _live_win_rate,
            'backtest':     bt_metrics,
            'lambda': {
                'score':      self.lb_result.get('score'),
                'signal':     self.lb_result.get('signal'),
                'status':     self.lb_result.get('status', '未連線'),
                'smart_money':self.lb_result.get('smart_money', {}),
                'performance':perf,
                'model':      lb_model,
                'prob_up':    prob_up,
                'prob_dn':    prob_dn,
                'breakdown':  breakdown,
            },
        }

        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        sm = '<script id="data-source">'
        em = '</script>'
        si = content.find(sm)
        ei = content.find(em, si)
        if si != -1 and ei != -1:
            dj = json.dumps(data, cls=NumpyEncoder)
            new_c = (
                content[:si]
                + f'{sm}window.AUTO_DATA = {dj};{em}'
                + content[ei + len(em):]
            )
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(new_c)
            print("[SUCCESS] v6.0 Top-10% 模型旗艦版已更新 HTML")
        else:
            print("[WARN] HTML data-source 標記未找到")


def _build_asset_dict(asset_results: dict, gold_history: list, td_key: str) -> dict:
    """四資產 HTML 注入 dict，從 Ensemble 結果 + Twelve Data 即時價建立"""
    from gld_xgb_ensemble import ASSETS, _td_fetch
    def _entry(key, res):
        info   = ASSETS.get(key, {})
        price  = res.get('gold', {}).get('price')
        change = None
        if not price and td_key:
            try:
                _, lat, prev = _td_fetch(info.get('ticker',''), td_key)
                if lat:
                    price  = round(lat, 2)
                    change = round((lat-prev)/prev*100, 2) if prev else 0.0
            except Exception: pass
        return {
            'ticker':     info.get('ticker', key),
            'name':       res.get('_asset_name', info.get('name', key)),
            'emoji':      res.get('_asset_emoji', info.get('emoji', '')),
            'currency':   res.get('_currency', info.get('currency', 'USD')),
            'price':      price,
            'change':     change,
            'signal':     res.get('signal', 'WAIT'),
            'confidence': res.get('score', 50),
            'prob_up':    res.get('prob_up', 50),
            'prob_dn':    res.get('prob_dn', 50),
            'val_acc':    round(res.get('model', {}).get('val_acc', 0) * 100, 1),
            'horizons':   res.get('model', {}).get('horizons', {}),
            'history':    gold_history if key == 'gold' else [],
        }
    return {k: _entry(k, asset_results.get(k, {})) for k in ['gold','silver','tw','us']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--html',             default='docs/index.html')
    p.add_argument('--fred-key',         default=None)
    p.add_argument('--bark-key-1',       default=None)
    p.add_argument('--bark-key-2',       default=None)
    p.add_argument('--bark-key-3',       default=None)
    p.add_argument('--aws-access-key',   default=None)
    p.add_argument('--aws-secret-key',   default=None)
    p.add_argument('--twelve-data-key',   default=None)
    p.add_argument('--aws-region',      default='ap-northeast-1')
    args, _ = p.parse_known_args()

    if args.twelve_data_key:
        os.environ['TWELVE_DATA_KEY'] = args.twelve_data_key
    updater = GldMmsUpdaterV6(
        fred_key      = args.fred_key,
        bark_keys     = [args.bark_key_1, args.bark_key_2, args.bark_key_3],
        aws_ak        = args.aws_access_key,
        aws_sk        = args.aws_secret_key,
        aws_region    = args.aws_region,
    )

    updater.invoke_lambda()
    updater._fetch('GC=F', '黃金期貨')
    updater._fetch('SI=F', '白銀期貨')
    updater._fetch('GLD', 'GLD ETF')
    updater._fetch_daily('GC=F', '黃金日線')
    updater._fetch_cross_asset()
    updater.fetch_macro()

    # ── XGBoost 高信心觸發推播 (≥80%) ─────────────────
    # ── Bark 推播：state-diff，只在 tier 改變時推送 ───────────────
    try:
        _pu       = float(updater.lb_result.get('prob_up', 0) or 0)
        _pd       = float(updater.lb_result.get('prob_dn', 0) or 0)
        _lb_score = updater.lb_result.get('score', '?')
        _lb_sig   = updater.lb_result.get('signal', '?')
        _aws_key2    = os.environ.get('AWS_ACCESS_KEY_ID', '')
        _aws_secret2 = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
        _push_s3b = None
        if _aws_key2 and _aws_secret2:
            import boto3 as _b3
            _push_s3b = _b3.client('s3', region_name='ap-northeast-1',
                                   aws_access_key_id=_aws_key2,
                                   aws_secret_access_key=_aws_secret2)
        # ── 四資產各自推播（A+B+D 品質評分）──────────────────
        from signal_quality import evaluate_signal
        _r2_bucket_push = os.environ.get('R2_BUCKET', 'richtrong-collect')
        for _ak, _ares in updater.asset_results.items():
            try:
                _a_pu  = float(_ares.get('prob_up', 50))
                _a_pd  = float(_ares.get('prob_dn', 50))
                _a_sig = str(_ares.get('signal', ''))
                _a_price = float(_ares.get('gold', {}).get('price', 0) or 0)
                # state-diff 推播（per-asset，用 asset key 區分）
                _a_state_key = f'signal_state_{_ak}.json'
                _a_should, _a_reason = _should_push(
                    _a_pu, _a_pd, _a_sig, _push_s3b, _r2_bucket_push,
                    state_key=_a_state_key)
                if not _a_should:
                    print(f"[INFO] {_ares.get('_asset_emoji','')} {_ak} 推播靜默（tier 未變）")
                    continue
                # A+B+D 品質評分
                _sq = evaluate_signal(_ares)
                _emoji = _ares.get('_asset_emoji', '')
                _aname = _ares.get('_asset_name', _ak)
                print(f"[INFO] {_emoji} {_aname} 品質分: {_sq.total_score}/100 | {_sq.score_breakdown}")
                if _sq.should_push:
                    _title = f"{_emoji} {_aname} | {_sq.bark_title()}"
                    _body  = _sq.bark_body(_a_price)
                    updater.send_push(_title, _body, is_leading=True)
                    print(f"[INFO] 推播送出 {_emoji} {_aname}（品質分 {_sq.total_score}）")
                else:
                    print(f"[INFO] {_emoji} {_aname} 品質分不足（{_sq.total_score} < 70），靜默")
            except Exception as _pe:
                print(f"[WARN] {_ak} 推播處理失敗: {_pe}")
    except Exception as _e:
        print(f"[WARN] 推播處理失敗: {_e}")

    updater.update_html(args.html)

if __name__ == '__main__':
    main()
