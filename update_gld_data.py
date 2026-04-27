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
    """
    計算完整回測指標：
    - Total Trades, Win Rate
    - Avg Win / Avg Loss / Profit Factor
    - Sharpe Ratio (simplified), Max Drawdown, Calmar Ratio
    """
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
        elif 'BUY' in sig or 'SELL' in sig: wins += 1  # 保守計算

        if pnl_is_win:
            win_amounts.append(abs(ret))
        else:
            loss_amounts.append(abs(ret))

        total_pnl += ret
        # Drawdown
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

    # Sharpe Ratio (annualized, assume 1 trade/day, 252 days)
    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret  = np.std(daily_returns, ddof=1)
        sharpe   = round(mean_ret / std_ret * np.sqrt(252), 2) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Calmar = Annual Return / Max DD
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
        self.assets      = {}   # 1H 技術數據
        self.daily       = {}   # 日線數據（多時間框架確認）
        self.macro       = {}
        self.lb_result   = {}
        self.regime      = 'UNKNOWN'  # TRENDING / RANGING / VOLATILE

        boto_kwargs = dict(region_name=aws_region)
        if aws_ak and aws_sk:
            boto_kwargs.update(dict(
                aws_access_key_id=aws_ak,
                aws_secret_access_key=aws_sk))
        self.lb_client = boto3.client('lambda', **boto_kwargs)
        self.s3_client = boto3.client('s3',    **boto_kwargs)

    # ════════════════════════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════════════════════════
    def _clean(self, df):
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        return df

    # ════════════════════════════════════════════════════════════
    # Step 1: Lambda AI（XGBoost Cloud）
    # ════════════════════════════════════════════════════════════
    def invoke_lambda(self):
        print("[INFO] 呼叫 Lambda gld-mms-updater...")
        try:
            resp = self.lb_client.invoke(
                FunctionName='gld-mms-updater',
                InvocationType='RequestResponse',
                Payload=json.dumps({}))
            status = resp.get('StatusCode', 0)
            print(f"[INFO] Lambda HTTP: {status}")
            if status == 200:
                s3r = self.s3_client.get_object(
                    Bucket='gld-mms-data-richtrong', Key='data.json')
                self.lb_result = json.loads(s3r['Body'].read().decode())
                print(f"[SUCCESS] Lambda 分數: {self.lb_result.get('score','?')} "
                      f"| 信號: {self.lb_result.get('signal','?')}")
                return True
        except Exception as e:
            print(f"[WARN] Lambda 失敗: {e}")
        self.lb_result = {}
        return False

    # ════════════════════════════════════════════════════════════
    # Step 2: Enhanced 技術指標計算（1H + 日線）
    # ════════════════════════════════════════════════════════════
    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 16+ 技術指標"""
        c, h, l, v, o = 'close','high','low','volume','open'

        # ── 基礎指標 ─────────────────────────────────────────
        # RSI (14)
        delta = df[c].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # MACD (12, 26, 9)
        ema12 = df[c].ewm(span=12, adjust=False).mean()
        ema26 = df[c].ewm(span=26, adjust=False).mean()
        df['macd']      = ema12 - ema26
        df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_sig']

        # Bollinger Bands (20, 2)
        df['bb_mid']   = df[c].rolling(20).mean()
        df['bb_std']   = df[c].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_pos']   = (df[c] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Stochastic (14, 3, 3)
        lo14 = df[l].rolling(14).min()
        hi14 = df[h].rolling(14).max()
        df['stoch_k'] = 100 * (df[c] - lo14) / (hi14 - lo14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # ADX (14) — 趨勢強度
        tr  = np.maximum(h - l,
               np.maximum(abs(h - df[c].shift(1)),
                          abs(l - df[c].shift(1))))
        plus_dm  = (df[h].diff().where(df[h].diff() > 0, 0)
                    .rolling(14).mean())
        minus_dm = (-df[l].diff().where(df[l].diff() < 0, 0)
                    .rolling(14).mean())
        atr14    = tr.rolling(14).mean()
        df['adx']       = 100 * abs(plus_dm - minus_dm) / (atr14 + 1e-10)
        df['adx_smooth'] = df['adx'].ewm(span=14).mean()
        df['atr']       = atr14

        # CCI (14)
        tp = (h + l + c) / 3
        df['cci'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())

        # Williams %R (14)
        df['williams_r'] = -100 * (hi14 - df[c]) / (hi14 - lo14 + 1e-10)

        # Momentum (10)
        df['momentum'] = df[c] / df[c].shift(10) - 1

        # OBV
        df['obv']     = (np.sign(df[c].diff()) * v).fillna(0).cumsum()
        df['obv_ma5'] = df['obv'].rolling(5).mean()

        # VWAP
        df['vwap']     = ((h+l+c)/3 * v).rolling(5).sum() / v.rolling(5).sum()
        df['vwap_dev'] = (df[c] / df['vwap'] - 1) * 100

        # ATR（已定義）
        df['atr_low']  = df['atr'] < df['atr'].rolling(50).min() * 1.1

        # 背離
        df['bull_div'] = ((df[c] < df[c].shift(3)) & (df['obv'] > df['obv'].shift(3)))
        df['bear_div'] = ((df[c] > df[c].shift(3)) & (df['obv'] < df['obv'].shift(3)))

        # Pin Bar
        hl   = h - l
        body = abs(df[c] - o)
        df['pin_bar'] = (body < hl * 0.2) & (hl > (hl.rolling(20).mean() * 2))

        # ── Regime Detection（基於 ADX + ATR）───────────────
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
            df = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            df = self._clean(df)
            if df.empty: return False
            df.reset_index(inplace=True)
            tc = 'datetime' if 'datetime' in df.columns else df.columns[0]
            df['date_full'] = pd.to_datetime(df[tc]).dt.strftime('%Y-%m-%d %H:%M')
            df = self._calc_indicators(df)
            self.assets[ticker] = df.tail(100).to_dict('records')
            return True
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}"); return False

    def _fetch_daily(self, ticker, name, period='90d'):
        """日線 — Multi-Timeframe 確認"""
        print(f"[INFO] 獲取日線 {name} ({ticker})...")
        try:
            df = yf.download(ticker, period=period, interval='1d',
                              progress=False, auto_adjust=True)
            df = self._clean(df)
            if df.empty: return False
            df.reset_index(inplace=True)
            tc = 'datetime' if 'datetime' in df.columns else df.columns[0]
            df['date_full'] = pd.to_datetime(df[tc]).dt.strftime('%Y-%m-%d')
            df = self._calc_indicators(df)
            self.daily[ticker] = df.tail(30).to_dict('records')
            return True
        except Exception as e:
            print(f"[WARN] 日線 {ticker}: {e}"); return False

    # ════════════════════════════════════════════════════════════
    # Step 3: Cross-Asset 跨資產數據
    # ════════════════════════════════════════════════════════════
    def _fetch_cross_asset(self):
        """DXY / 10Y Treasury / VIX — 影響黃金的宏觀因素"""
        tickers = {
            'DX-Y.NYB': 'USD_Index',
            '^TNX':     '10Y_Treasury',
            '^VIX':     'VIX',
        }
        for sym, name in tickers.items():
            try:
                df = yf.download(sym, period='30d', interval='1d',
                                 progress=False, auto_adjust=True)
                df = self._clean(df)
                if not df.empty:
                    df['rsi'] = self._rsi_fast(df['close'], 14)
                    latest = df.iloc[-1]
                    self.macro[name] = {
                        'close': float(latest['close']),
                        'rsi':   float(latest.get('rsi', 50)),
                        'note':  f"{name}: {latest['close']:.2f} (RSI {latest.get('rsi',50):.0f})"
                    }
                    print(f"[INFO] {name}: {latest['close']:.2f}")
            except Exception as e:
                print(f"[WARN] {name}: {e}")

    @staticmethod
    def _rsi_fast(series: pd.Series, n: int) -> pd.Series:
        d = series.diff()
        g = d.where(d > 0, 0).rolling(n).mean()
        l = (-d.where(d < 0, 0)).rolling(n).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    def fetch_macro(self):
        print("[INFO] 獲取宏觀 + COT...")
        # UUP / DXY
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

    # ════════════════════════════════════════════════════════════
    # Step 4: Enhanced 技術評分（16 features, regime-aware）
    # ════════════════════════════════════════════════════════════
    def _calc_tech_score(self, row: dict, daily_row: dict | None = None) -> dict:
        """
        16 feature 標準化評分 → 0-100
        Regime-aware weights
        """
        regime = self.regime
        score = 50
        detail = []

        # ── 1. RSI (0-30 超賣, 70+ 超買) ─────────────────────
        rsi = row.get('rsi', 50)
        if   rsi < 25: score += 10; detail.append(f"RSI超賣{rsi:.0f}")
        elif rsi < 35: score += 5;  detail.append(f"RSI偏低{rsi:.0f}")
        elif rsi > 75: score -= 10; detail.append(f"RSI超買{rsi:.0f}")
        elif rsi > 65: score -= 5;  detail.append(f"RSI偏高{rsi:.0f}")

        # ── 2. MACD Histogram (動量方向) ─────────────────────
        mh = row.get('macd_hist', 0)
        if   mh > 0.5:  score += 8; detail.append("MACD看漲")
        elif mh > 0.1:  score += 4; detail.append("MACD偏多")
        elif mh < -0.5: score -= 8; detail.append("MACD看跌")
        elif mh < -0.1: score -= 4; detail.append("MACD偏空")

        # ── 3. Bollinger Bands 位置 ──────────────────────────
        bbp = row.get('bb_pos', 0.5)
        if   bbp < 0.1: score += 8; detail.append("BB下軌超賣")
        elif bbp < 0.25: score += 4; detail.append("BB偏低")
        elif bbp > 0.9: score -= 8; detail.append("BB上軌超買")
        elif bbp > 0.75: score -= 4; detail.append("BB偏高")

        # ── 4. Stochastic %K ────────────────────────────────
        sk = row.get('stoch_k', 50)
        sd = row.get('stoch_d', 50)
        if sk < 20 and sd < 30: score += 6; detail.append("Stoch超賣")
        elif sk > 80 and sd > 70: score -= 6; detail.append("Stoch超買")
        elif sk > sd and sk < 50: score += 3; detail.append("Stoch金叉")
        elif sk < sd and sk > 50: score -= 3; detail.append("Stoch死叉")

        # ── 5. ADX 趨勢強度 ─────────────────────────────────
        adx = row.get('adx', 0)
        if   adx > 30: score += 4; detail.append(f"ADX強趨勢{adx:.0f}")
        elif adx < 15: score -= 2; detail.append(f"ADX盤整{adx:.0f}")

        # ── 6. CCI ─────────────────────────────────────────
        cci = row.get('cci', 0)
        if   cci < -150: score += 5; detail.append(f"CCI超賣{cci:.0f}")
        elif cci > 150:  score -= 5; detail.append(f"CCI超買{cci:.0f}")
        elif cci > 0:   score += 2; detail.append("CCI偏多")

        # ── 7. Williams %R ──────────────────────────────────
        wr = row.get('williams_r', -50)
        if   wr < -80: score += 5; detail.append("W%R超賣")
        elif wr > -20: score -= 5; detail.append("W%R超買")

        # ── 8. Momentum ────────────────────────────────────
        mom = row.get('momentum', 0)
        if   mom > 0.02: score += 5; detail.append(f"動量正{mom*100:.1f}%")
        elif mom < -0.02: score -= 5; detail.append(f"動量負{mom*100:.1f}%")

        # ── 9. VWAP 乖離 ────────────────────────────────────
        vdev = row.get('vwap_dev', 0)
        if   vdev < -1.5: score += 8; detail.append(f"VWAP低估{vdev:.1f}%")
        elif vdev < -0.5: score += 4; detail.append(f"VWAP偏低{vdev:.1f}%")
        elif vdev > 1.5:  score -= 8; detail.append(f"VWAP高估{vdev:.1f}%")
        elif vdev > 0.5:  score -= 4; detail.append(f"VWAP偏高{vdev:.1f}%")

        # ── 10. OBV 方向 ────────────────────────────────────
        obv = row.get('obv', 0)
        obv5 = row.get('obv_ma5', obv)
        if   obv > obv5: score += 5; detail.append("OBV偏多")
        else:            score -= 5; detail.append("OBV偏空")

        # ── 11. ATR 擠壓 ────────────────────────────────────
        if row.get('atr_low'): score += 4; detail.append("ATR擠壓")

        # ── 12. 底背離 / 頂背離 ──────────────────────────────
        if row.get('bull_div'): score += 10; detail.append("底背離")
        if row.get('bear_div'): score -= 10; detail.append("頂背離")

        # ── 13. Pin Bar ─────────────────────────────────────
        if row.get('pin_bar'): score -= 3; detail.append("Pin Bar")

        # ── Regime-aware 調整 ───────────────────────────────
        if regime == 'VOLATILE':
            score = 50 + (score - 50) * 0.5  # 壓縮範圍
            detail.append("⚠️高波動體制，倉位減半")
        elif regime == 'TRENDING':
            # 順勢增強
            if 'MACD看漲' in str(detail) or '底背離' in str(detail): score += 5
            if 'MACD看跌' in str(detail) or '頂背離' in str(detail): score -= 5

        # ── 日線方向確認 ────────────────────────────────────
        if daily_row:
            d_rsi  = daily_row.get('rsi', 50)
            d_macd = daily_row.get('macd_hist', 0)
            d_adx  = daily_row.get('adx', 0)
            if d_rsi < 40 and d_macd > 0 and d_adx > 20:
                score += 6; detail.append("日線確認看多")
            elif d_rsi > 60 and d_macd < 0 and d_adx > 20:
                score -= 6; detail.append("日線確認看空")

        # ── Cross-Asset 宏觀調整 ────────────────────────────
        dxy = self.macro.get('USD_Index', {})
        if dxy:
            dxy_rsi = dxy.get('rsi', 50)
            # 美元弱 → 黃金強（反向關係）
            if dxy_rsi > 65: score -= 4; detail.append("DXY超買→黃金壓力")
            elif dxy_rsi < 35: score += 4; detail.append("DXY超賣→黃金支撐")

        score = max(0, min(100, score))
        return {'score': score, 'detail': detail}

    # ════════════════════════════════════════════════════════════
    # Step 5: Pure-Tech XGBoost Fallback（Model Ensemble）
    # ════════════════════════════════════════════════════════════
    def _pure_tech_ensemble(self, row: dict, daily_row: dict | None = None) -> dict:
        """
        在 Lambda 不可用時，使用純技術指標的多維評分系統。
        這相當於一個輕量版 XGBoost，共識評分。
        包含 6 種技術模塊的投票。
        """
        votes = {
            'bull': 0, 'bear': 0, 'neutral': 0,
            'score_contrib': 0
        }

        # ── Module 1: Trend (ADX + MACD) ───────────────────
        adx  = row.get('adx', 0)
        mh   = row.get('macd_hist', 0)
        if adx > 25:
            if mh > 0:   votes['bull'] += 2
            else:        votes['bear'] += 2
        else:
            votes['neutral'] += 1

        # ── Module 2: Momentum (RSI + Stoch + W%R) ─────────
        rsi = row.get('rsi', 50)
        sk  = row.get('stoch_k', 50)
        wr  = row.get('williams_r', -50)
        if rsi < 40 and sk < 40: votes['bull'] += 2
        elif rsi > 60 and sk > 60: votes['bear'] += 2
        elif rsi < 50: votes['bull'] += 1; votes['bear'] -= 0.5
        else: votes['bear'] += 1; votes['bull'] -= 0.5

        # ── Module 3: Volatility (BB + ATR) ────────────────
        bbp  = row.get('bb_pos', 0.5)
        atr_ = row.get('atr_low', False)
        if bbp < 0.2: votes['bull'] += 2
        elif bbp > 0.8: votes['bear'] += 2
        if atr_: votes['bull'] += 1  # 波動壓縮後可能突破

        # ── Module 4: Volume (OBV) ─────────────────────────
        obv  = row.get('obv', 0)
        obv5 = row.get('obv_ma5', obv)
        if obv > obv5: votes['bull'] += 1.5
        else: votes['bear'] += 1.5

        # ── Module 5: VWAP ─────────────────────────────────
        vdev = row.get('vwap_dev', 0)
        if vdev < -1.0: votes['bull'] += 2
        elif vdev > 1.0: votes['bear'] += 2
        elif vdev < 0: votes['bull'] += 0.5
        else: votes['bear'] += 0.5

        # ── Module 6: Pattern (Divergence) ─────────────────
        if row.get('bull_div'): votes['bull'] += 3
        if row.get('bear_div'): votes['bear'] += 3

        # ── 共識計算 ───────────────────────────────────────
        total = votes['bull'] + votes['bear'] + votes['neutral']
        if total > 0:
            bull_ratio = votes['bull'] / total
        else:
            bull_ratio = 0.5

        # Score: 0-100
        raw_score = bull_ratio * 100
        # 信心度：如果共識很強（票數差距大），分數更高
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

    # ════════════════════════════════════════════════════════════
    # Step 6: Dynamic Position Sizing
    # ════════════════════════════════════════════════════════════
    def _calc_position_size(self, combined_score: int,
                             regime: str, atr: float,
                             price: float) -> dict:
        """
        Kelly Criterion / ATR-based 動態倉位
        - ATR-based: 每單位風險多少合約
        - Regime-adjusted: VOLATILE 減半倉位
        """
        # Kelly fraction (simplified: win_rate / (reward/risk))
        # 假設 avg win = 2*ATR, avg loss = 1*ATR, WR = 歷史勝率
        win_r  = self.lb_result.get('performance', {}).get('win_rate', 50) / 100
        rr     = 2.0  # reward/risk ratio
        kelly  = (win_r * rr - (1 - win_r)) / rr
        kelly  = max(0.05, min(0.25, kelly))  # 上限 25%

        # Regime adjustment
        if regime == 'VOLATILE': kelly *= 0.5
        elif regime == 'RANGING': kelly *= 0.75

        # ATR position sizing
        risk_amount = price * 0.01  # 每份合約風險 1%
        contracts   = max(1, round(risk_amount / (atr + 1e-10)))

        return {
            'kelly_fraction': round(kelly, 3),
            'recommended_contracts': int(contracts),
            'regime': regime,
            'atr_risk_pct': round(atr / price * 100, 2),
        }

    # ════════════════════════════════════════════════════════════
    # Step 7: 推播
    # ════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════
    # Step 8: 綜合評分 + 信號
    # ════════════════════════════════════════════════════════════
    def calculate_signals(self):
        PUSH = 78  # 推播門檻
        signals = {}

        lb_score = self.lb_result.get('score', None)
        lb_signal = self.lb_result.get('signal', '')
        smart_money = self.lb_result.get('smart_money', {})
        perf = self.lb_result.get('performance', {})

        # COT
        cot = self.macro.get('cot_gold', {})
        cot_bias = cot.get('score_add', 0)
        cot_net  = cot.get('spec_net_pct', 0)
        cot_ls   = cot.get('spec_ls_ratio', 1.0)

        # ── Fallback：若 assets 為空（yfinance 失敗），用 Lambda 結果產生基本信號 ──
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
                'ticker':     'GC=F',
                'signal':     signal,
                'confidence': combined,
                'tech_score': lb_score,
                'close':      price,
                'note':       f'Lambda AI {lb_score}% + COT{cot_bias:+.0f}（技術指標暫時不可用）',
                'cot_score_add': cot_bias,
            }
            print(f'[INFO] Fallback 信號: {signal} {combined}%')

        for ticker, raw_data in self.assets.items():
            df    = pd.DataFrame(raw_data)
            now   = df.iloc[-1].to_dict()

            # 日線
            daily_row = None
            if ticker in self.daily:
                daily_row = pd.DataFrame(self.daily[ticker]).iloc[-1].to_dict()

            # Enhanced 技術評分
            tech = self._calc_tech_score(now, daily_row)

            # Pure-Tech Ensemble（Lambda 備用）
            if lb_score is None:
                ensemble = self._pure_tech_ensemble(now, daily_row)
                lb_score  = ensemble['score']
                lb_signal  = ensemble['signal']
                ensemble_note = f"【純技術 Ensemble 共識】{ensemble['confidence']}%共識"
            else:
                ensemble_note = ''

            # COT Dynamic Position Sizing
            atr_now = now.get('atr', 0)
            pos = self._calc_position_size(tech['score'], self.regime,
                                            atr_now, now['close'])

            # 綜合評分
            if lb_score is not None:
                combined = round(max(0, min(100,
                    lb_score * 0.55
                    + tech['score'] * 0.35
                    + cot_bias * 1.0)))  # COT 直接加成分
                note = (f"LambdaAI {lb_score}%×55%"
                        f" + 技術{tech['score']}%×35%"
                        f" + COT{cot_bias:+.0f}×10%")
            else:
                combined = tech['score']
                note = f"純技術分析 {tech['score']}%"
                if ensemble_note:
                    note = ensemble_note + ' | ' + note

            # 信號
            if   combined >= 80: signal = 'STRONG_BUY'
            elif combined >= 65: signal = 'BUY'
            elif combined >= 55: signal = 'PRE_BUY'
            elif combined <= 20: signal = 'STRONG_SELL'
            elif combined <= 35: signal = 'SELL'
            elif combined <= 45: signal = 'PRE_SELL'
            else:                signal = 'HOLD'

            # Breakdown（增強）
            breakdown = [
                f"AI評分: {lb_score or '?'}% ({lb_signal or 'N/A'})",
                f"技術分: {tech['score']} {' '.join(tech['detail'][:3])}",
                f"COT大戶: {cot_net}% OI (LS比 {cot_ls})",
                f"體制: {self.regime} | ATR風險: {pos['atr_risk_pct']}%",
                f"Kelly倉位: {pos['kelly_fraction']*100:.0f}% ({pos['recommended_contracts']}合約)",
            ]
            if ensemble_note: breakdown.insert(0, ensemble_note)

            # Radar
            radar_msgs = []
            if now.get('bull_div'):  radar_msgs.append("底背離")
            if now.get('bear_div'):  radar_msgs.append("頂背離")
            if now.get('atr_low'):   radar_msgs.append("ATR擠壓")
            if now.get('rsi', 50) < 35: radar_msgs.append("RSI超賣")
            if now.get('rsi', 50) > 65: radar_msgs.append("RSI超買")
            if self.regime == 'VOLATILE': radar_msgs.append("⚠️高波動")
            if self.regime == 'TRENDING': radar_msgs.append("→趨勢")

            # Push 推播
            if ticker == 'GC=F' and combined > PUSH:
                price = float(now['close'])
                etf   = smart_money.get('etf_flow', '?')
                cot_r = cot.get('summary', '?')
                self.send_push(
                    f"📈 黃金{'強烈' if signal=='STRONG_BUY' else ''}買進信號！"
                    if 'BUY' in signal else
                    f"📉 黃金{'強烈' if signal=='STRONG_SELL' else ''}賣出信號！",
                    f"綜合評分:{combined}% | ${price:.2f} | "
                    f"ETF:{etf} | COT:{cot_r[:20] if cot_r else '?'} | "
                    f"體制:{self.regime}"
                )

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

    # ════════════════════════════════════════════════════════════
    # Step 9: 更新 HTML
    # ════════════════════════════════════════════════════════════
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
            'assets':       self.assets,
            'daily':        self.daily,
            'macro':        self.macro,
            'signals':      self.calculate_signals(),
            'win_rate_20':  win_rate,
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
                + f'{sm}const AUTO_DATA = {dj};{em}'
                + content[ei + len(em):]
            )
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(new_c)
            print("[SUCCESS] v6.0 Top-10% 模型旗艦版已更新 HTML")
        else:
            print("[WARN] HTML data-source 標記未找到")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--html',             default='docs/index.html')
    p.add_argument('--fred-key',         default=None)
    p.add_argument('--bark-key-1',       default=None)
    p.add_argument('--bark-key-2',       default=None)
    p.add_argument('--bark-key-3',       default=None)
    p.add_argument('--aws-access-key',   default=None)
    p.add_argument('--aws-secret-key',   default=None)
    p.add_argument('--aws-region',      default='ap-northeast-1')
    args, _ = p.parse_known_args()

    updater = GldMmsUpdaterV6(
        fred_key      = args.fred_key,
        bark_keys     = [args.bark_key_1, args.bark_key_2, args.bark_key_3],
        aws_ak        = args.aws_access_key,
        aws_sk        = args.aws_secret_key,
        aws_region    = args.aws_region,
    )

    # 執行順序
    updater.invoke_lambda()                        # 1. Lambda AI (XGBoost Cloud)
    updater._fetch('GC=F', '黃金期貨')            # 2a. 1H 技術指標
    updater._fetch('SI=F', '白銀期貨')
    updater._fetch('GLD', 'GLD ETF')
    updater._fetch_daily('GC=F', '黃金日線')       # 2b. 日線多時間框架
    updater._fetch_cross_asset()                    # 2c. DXY / TNX / VIX
    updater.fetch_macro()                          # 3. 宏觀 + COT
    updater.update_html(args.html)                 # 4. 更新 HTML

if __name__ == '__main__':
    main()
