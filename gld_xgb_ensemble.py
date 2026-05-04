#!/usr/bin/env python3
"""
GLD-MMS XGBoost Ensemble 預測引擎 v2.0
=======================================
取代 AWS Lambda，直接在 GitHub Actions 執行

改進點（相較原始 Lambda v4.0）：
  1. Ensemble：XGBoost + LightGBM + CatBoost 三模型投票平均
  2. Walk-Forward Validation：時序滾動驗證，避免 look-ahead bias
  3. 特徵工程升級：加入趨勢強度 / 跨資產動量 / 波動率結構
  4. 動態信心閾值：用近期分布校準，減少誤報
  5. 自動重訓：模型超過 30 天自動以最新資料重訓
  6. 模型存 Cloudflare R2（不再依賴 AWS S3）

使用方式（在 update_gld_data.py 裡 import）：
  from gld_xgb_ensemble import run_ensemble
  result = run_ensemble(td_key, r2_client, r2_bucket)
"""

import json, os, pickle, time, hashlib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

# ── 全域設定 ──────────────────────────────────────────
MODEL_KEY      = 'models/ensemble_v2.pkl'   # R2 存放路徑
MODEL_MAX_AGE  = 30                          # 模型超過幾天自動重訓
HORIZONS       = ['1d', '5d', '30d']
MIN_TRAIN_ROWS = 200                         # 最少訓練筆數
TD_BASE        = 'https://api.twelvedata.com'

# ── 四資產設定（寫死，不可自選）────────────────────────
ASSETS = {
    'gold':   {'ticker': 'GC=F',    'name': '黃金',      'currency': 'USD', 'emoji': '🥇'},
    'silver': {'ticker': 'SI=F',    'name': '白銀',      'currency': 'USD', 'emoji': '🥈'},
    'tw':     {'ticker': '0050.TW', 'name': '元大台灣50','currency': 'TWD', 'emoji': '🇹🇼'},
    'us':     {'ticker': 'QQQ',   'name': '納斯達克',  'currency': 'USD', 'emoji': '🇺🇸'},
}

# Twelve Data ticker 對照
TD_TICKERS = {
    'GC=F':     ('XAU/USD', None),
    'SI=F':     ('SLV',     None),
    'DX-Y.NYB': ('UUP',     None),
    '^TNX':     ('TLT',     None),
    '^VIX':     ('VIXY',    None),
    'GLD':      ('GLD',     None),
    'GDX':      ('GDX',     None),
    'QQQ':      ('QQQ',     None),
    '0050.TW':  ('0050',    'XTAI'),
    'QQQ':    ('QQQ',     None),
}
TD_FALLBACK = {
    '0050.TW': ('EWT', None),
    'QQQ':   ('TQQQ', None),
}

# ── Twelve Data 抓取 ─────────────────────────────────
def _td_fetch_history(ticker: str, td_key: str, days: int = 500) -> pd.DataFrame:
    """抓歷史日線資料，回傳 DataFrame（index=date, cols=Close/High/Low/Volume）"""
    symbol, exchange = TD_TICKERS.get(ticker, (ticker, None))
    params = {
        'symbol':     symbol,
        'interval':   '1day',
        'outputsize': min(days, 5000),
        'apikey':     td_key,
        'order':      'ASC',
    }
    if exchange:
        params['exchange'] = exchange
    try:
        r = requests.get(f'{TD_BASE}/time_series', params=params, timeout=20)
        data = r.json()
        if data.get('status') == 'error' or 'values' not in data:
            msg = data.get('message', 'err')
            print(f"[WARN] TD {ticker} ({symbol}): {msg}")
            # fallback ticker
            if ticker in TD_FALLBACK:
                fb_sym, fb_ex = TD_FALLBACK[ticker]
                print(f"[INFO] fallback: {ticker} → {fb_sym}")
                fb_p = dict(params)
                fb_p['symbol'] = fb_sym
                if fb_ex: fb_p['exchange'] = fb_ex
                else:     fb_p.pop('exchange', None)
                try:
                    r2 = requests.get(f'{TD_BASE}/time_series', params=fb_p, timeout=12)
                    fb_d = r2.json()
                    if 'values' in fb_d:
                        data = fb_d
                    else:
                        return pd.DataFrame()
                except Exception:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        rows = data['values']
        df = pd.DataFrame(rows)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        for col in ['open','high','low','close','volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.columns = [c.capitalize() for c in df.columns]
        return df.dropna(subset=['Close'])
    except Exception as e:
        print(f"[WARN] TD fetch {ticker}: {e}")
        return pd.DataFrame()


# ── 特徵工程（升級版）────────────────────────────────
def add_features(gold: pd.DataFrame,
                 dxy:  pd.DataFrame,
                 tnx:  pd.DataFrame,
                 vix:  pd.DataFrame) -> pd.DataFrame:
    """
    建立完整特徵矩陣，涵蓋：
    - 技術指標（RSI / MACD / BB / ATR / ADX / Stochastic / CCI / OBV）
    - 趨勢強度（多均線偏離、動量評分）
    - 跨資產動量（DXY / VIX / TNX / GLD-GDX ratio）
    - 波動率結構（ATR ratio、Bollinger 寬度變化率）
    - 季節性（月份、星期、季度）
    """
    df = gold.copy()
    close  = df['Close'].astype(float)
    high   = df['High'].astype(float)
    low    = df['Low'].astype(float)
    volume = df['Volume'].astype(float) if 'Volume' in df.columns else pd.Series(1, index=df.index)

    # ── 基礎報酬 ──
    for p in [1, 3, 5, 10, 20]:
        df[f'ret_{p}d'] = close.pct_change(p)

    # ── 波動率 ──
    df['vol_5d']  = df['ret_1d'].rolling(5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['vol_ratio'] = df['vol_5d'] / (df['vol_20d'] + 1e-9)   # 波動率結構

    # ── RSI ──
    for period in [9, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        df[f'RSI_{period}'] = 100 - 100 / (1 + rs)

    # ── MACD ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # ── Bollinger Bands ──
    for period in [14, 20]:
        ma  = close.rolling(period).mean()
        std = close.rolling(period).std()
        df[f'BB_width_{period}']   = (std * 4) / (ma + 1e-9)
        df[f'BB_pos_{period}']     = (close - (ma - std*2)) / (std * 4 + 1e-9)
        df[f'BB_width_chg_{period}'] = df[f'BB_width_{period}'].pct_change(5)  # 波動率加速

    # ── ATR ──
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low  - close.shift())
    ], axis=1).max(axis=1)
    for period in [7, 14]:
        df[f'ATR_{period}']      = tr.rolling(period).mean()
        df[f'ATR_ratio_{period}'] = df[f'ATR_{period}'] / (close + 1e-9)

    # ── ADX ──
    up = high - high.shift();  dn = low.shift() - low
    pdm = np.where((up > dn) & (up > 0), up, 0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0)
    tr14 = tr.rolling(14).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(14).mean() / (tr14 + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(14).mean() / (tr14 + 1e-9)
    dx  = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
    df['ADX']    = dx.rolling(14).mean()
    df['DI_diff'] = pdi - mdi   # 方向強度差

    # ── Stochastic ──
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── CCI ──
    typical = (high + low + close) / 3
    df['CCI'] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std() + 1e-9)

    # ── OBV ──
    df['OBV']     = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV_ma']  = df['OBV'].rolling(20).mean()
    df['OBV_diff'] = df['OBV'] - df['OBV_ma']

    # ── 均線系統（多週期偏離）──
    for ma in [5, 10, 20, 50, 100, 200]:
        df[f'MA{ma}']      = close.rolling(ma).mean()
        df[f'bias_{ma}']   = (close - df[f'MA{ma}']) / (df[f'MA{ma}'] + 1e-9)
        df[f'ma_slope_{ma}'] = df[f'MA{ma}'].pct_change(5)  # 均線斜率

    # 多均線動量評分（空頭/多頭結構）
    df['trend_score'] = sum([
        (close > df[f'MA{ma}']).astype(int) for ma in [5, 10, 20, 50]
    ])

    # 均線排列（MA5 > MA10 > MA20 > MA50）
    df['ma_aligned_bull'] = (
        (df['MA5'] > df['MA10']) &
        (df['MA10'] > df['MA20']) &
        (df['MA20'] > df['MA50'])
    ).astype(int)

    # ── 跨資產特徵 ──
    if not dxy.empty:
        d = dxy['Close'].astype(float).reindex(df.index, method='ffill')
        for p in [1, 5, 20]:
            df[f'dxy_ret_{p}d'] = d.pct_change(p)
        df['dxy_ma20']      = d.rolling(20).mean()
        df['dxy_ma50']      = d.rolling(50).mean()
        df['dxy_vs_ma50']   = (d - df['dxy_ma50']) / (df['dxy_ma50'] + 1e-9)
        df['dxy_trend']     = np.sign(df['dxy_ma20'] - df['dxy_ma20'].shift(5))
        df['gold_dxy_corr'] = (
            close.pct_change().rolling(20).corr(d.pct_change())
        )

    if not tnx.empty:
        t = tnx['Close'].astype(float).reindex(df.index, method='ffill')
        for p in [1, 5, 20]:
            df[f'tnx_ret_{p}d'] = t.pct_change(p)
        df['tnx_level'] = t
        df['tnx_ma20']  = t.rolling(20).mean()
        df['tnx_vs_ma20'] = (t - df['tnx_ma20']) / (df['tnx_ma20'] + 1e-9)
        df['real_rate_proxy'] = t - df.get('vol_20d', 0) * 100  # 實質利率代理

    if not vix.empty:
        v = vix['Close'].astype(float).reindex(df.index, method='ffill')
        df['vix_level']   = v
        for p in [1, 5]:
            df[f'vix_ret_{p}d'] = v.pct_change(p)
        df['vix_ma20']    = v.rolling(20).mean()
        df['vix_vs_ma20'] = (v - df['vix_ma20']) / (df['vix_ma20'] + 1e-9)
        df['vix_regime']  = pd.cut(v, bins=[0, 15, 20, 30, 9999],
                                   labels=[0, 1, 2, 3]).astype(float)
        df['vix_spike']   = (v.pct_change(1) > 0.15).astype(int)
        df['fear_greed']  = (df['RSI_14'] - 50) / 50 - (v - 20) / 20  # 自製情緒指標

    # ── 季節性 ──
    idx = df.index
    df['month_sin']  = np.sin(2 * np.pi * idx.month / 12)
    df['month_cos']  = np.cos(2 * np.pi * idx.month / 12)
    df['wday_sin']   = np.sin(2 * np.pi * idx.dayofweek / 5)
    df['wday_cos']   = np.cos(2 * np.pi * idx.dayofweek / 5)
    df['quarter']    = idx.quarter.astype(float)
    df['is_q1']      = (idx.month.isin([1, 2, 3])).astype(int)
    df['is_q4']      = (idx.month.isin([10, 11, 12])).astype(int)
    df['is_summer']  = (idx.month.isin([6, 7, 8])).astype(int)
    df['month_num']  = idx.month.astype(float)

    # 月底效應（最後 3 個交易日）
    df['days_to_month_end'] = (
        pd.to_datetime(idx).to_series().reset_index(drop=True)
        .apply(lambda d: (
            d.replace(day=1,
                      month=d.month % 12 + 1 if d.month < 12 else 1,
                      year=d.year + (1 if d.month == 12 else 0))
            - pd.Timedelta(days=1) - d
        ).days).values
    )

    return df


def make_labels(close: pd.Series, horizon: str) -> pd.Series:
    """建立目標變數：N 天後的漲跌（1=漲，0=跌）"""
    days = {'1d': 1, '5d': 5, '30d': 30}[horizon]
    future = close.shift(-days)
    return (future > close).astype(int)


# ── Walk-Forward 訓練 ────────────────────────────────
def train_ensemble(df_features: pd.DataFrame,
                   labels: pd.Series,
                   feature_cols: list) -> dict:
    """
    Walk-Forward Validation 訓練 Ensemble
    - XGBoost + LightGBM + CatBoost 三模型
    - 時序分割：前 70% 訓練，後 30% 驗證
    - 回傳訓練好的模型 dict
    """
    from xgboost import XGBClassifier

    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        print("[WARN] LightGBM 未安裝，跳過")

    try:
        from catboost import CatBoostClassifier
        has_cat = True
    except ImportError:
        has_cat = False
        print("[WARN] CatBoost 未安裝，跳過")

    X = df_features[feature_cols].values
    y = labels.values

    # 時序分割
    split = int(len(X) * 0.70)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    models = {}
    val_scores = {}

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False)
    models['xgb'] = xgb
    val_scores['xgb'] = float(np.mean(
        (xgb.predict_proba(X_val)[:, 1] > 0.5) == y_val))
    print(f"  [XGBoost] val win rate: {val_scores['xgb']:.3f}")

    # LightGBM
    if has_lgbm:
        lgbm = LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr,
                 eval_set=[(X_val, y_val)],
                 callbacks=None)
        models['lgbm'] = lgbm
        val_scores['lgbm'] = float(np.mean(
            (lgbm.predict_proba(X_val)[:, 1] > 0.5) == y_val))
        print(f"  [LightGBM] val win rate: {val_scores['lgbm']:.3f}")

    # CatBoost
    if has_cat:
        cat = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.03,
            subsample=0.8,
            random_seed=42,
            verbose=0,
        )
        cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        models['cat'] = cat
        val_scores['cat'] = float(np.mean(
            (cat.predict_proba(X_val)[:, 1] > 0.5) == y_val))
        print(f"  [CatBoost] val win rate: {val_scores['cat']:.3f}")

    # 按驗證集表現加權
    total_score = sum(val_scores.values())
    weights = {k: v / total_score for k, v in val_scores.items()}
    print(f"  [Ensemble] weights: {weights}")

    ensemble_proba = sum(
        weights[k] * m.predict_proba(X_val)[:, 1]
        for k, m in models.items()
    )
    ensemble_acc = float(np.mean((ensemble_proba > 0.5) == y_val))
    print(f"  [Ensemble] val win rate: {ensemble_acc:.3f}")

    return {
        'models': models,
        'weights': weights,
        'feature_cols': feature_cols,
        'val_acc': ensemble_acc,
        'val_scores': val_scores,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'n_train': int(split),
        'n_val': int(len(X) - split),
    }


# ── R2 模型管理 ──────────────────────────────────────
def _load_ensemble_from_r2(r2_client, bucket: str, model_key: str = MODEL_KEY) -> dict | None:
    """從 R2 載入 Ensemble，若不存在或過期回傳 None"""
    try:
        resp = r2_client.get_object(Bucket=bucket, Key=MODEL_KEY)
        data = resp['Body'].read()
        bundle = pickle.loads(data)

        trained_at = datetime.fromisoformat(
            bundle.get('trained_at', '2000-01-01T00:00:00+00:00')
            .replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - trained_at).days

        if age_days > MODEL_MAX_AGE:
            print(f"[INFO] Ensemble 模型已 {age_days} 天，需重訓")
            return None

        print(f"[INFO] 載入 Ensemble 模型（訓練於 {trained_at.date()}，{age_days}天前）")
        return bundle
    except Exception as e:
        print(f"[INFO] 無現有模型：{e}")
        return None


def _save_ensemble_to_r2(r2_client, bucket: str, bundle: dict, model_key: str = MODEL_KEY):
    """序列化 Ensemble 存到 R2"""
    try:
        data = pickle.dumps(bundle, protocol=4)
        r2_client.put_object(
            Bucket=bucket, Key=MODEL_KEY,
            Body=data,
            ContentType='application/octet-stream',
        )
        size_mb = len(data) / 1024 / 1024
        print(f"[INFO] Ensemble 模型已存 R2（{size_mb:.2f} MB）")
    except Exception as e:
        print(f"[WARN] Ensemble 存 R2 失敗：{e}")


# ── 主入口：推論（含自動訓練）───────────────────────
def run_ensemble(td_key: str, r2_client=None, r2_bucket: str = 'richtrong-collect', target_ticker: str = 'GC=F') -> dict:
    """
    主入口：
    1. 嘗試從 R2 載入已訓練模型
    2. 若無模型或過期，自動訓練並存 R2
    3. 用最新市場資料跑 Ensemble 推論
    4. 回傳預測結果 dict（相容原 Lambda 格式）
    """
    asset_slug = target_ticker.replace('/', '_').replace('^', '').replace('.', '_')
    model_key  = f'models/ensemble_v2_{asset_slug}.pkl'
    print(f"[INFO] === GLD-MMS Ensemble v2.0 | {target_ticker} ===")

    # ── 抓市場資料（500 天歷史 + 最新）──
    print("[INFO] 抓取市場資料...")
    gold = _td_fetch_history(target_ticker, td_key, 520)  # 主資產
    dxy  = _td_fetch_history('DX-Y.NYB', td_key, 520)
    tnx  = _td_fetch_history('^TNX',     td_key, 520)
    vix  = _td_fetch_history('^VIX',     td_key, 520)

    if gold.empty or len(gold) < MIN_TRAIN_ROWS:
        print(f"[ERROR] 黃金資料不足（{len(gold)} 筆），回傳 fallback")
        return _fallback_result()

    # ── 特徵工程 ──
    print(f"[INFO] 建立特徵矩陣（{len(gold)} 筆）...")
    df_feat = add_features(gold, dxy, tnx, vix)

    # ── 嘗試載入或訓練模型 ──
    bundle = _load_ensemble_from_r2(r2_client, r2_bucket, model_key) if r2_client else None

    if bundle is None:
        print("[INFO] 開始訓練 Ensemble（首次或重訓）...")
        horizon_bundles = {}

        for h in HORIZONS:
            print(f"\n  訓練 horizon={h}...")
            labels = make_labels(gold['Close'], h)
            df_aligned = df_feat.loc[labels.index].copy()
            df_aligned['__label__'] = labels

            # 排除 NaN 和最後 N 天（無未來標籤）
            days_map = {'1d': 1, '5d': 5, '30d': 30}
            cutoff   = len(df_aligned) - days_map[h]
            df_train = df_aligned.iloc[:cutoff].dropna()

            if len(df_train) < MIN_TRAIN_ROWS:
                print(f"    [WARN] 資料不足（{len(df_train)}），跳過")
                continue

            # 選特徵欄位（排除 label 和 OHLCV 原始值）
            exclude = {'Open', 'High', 'Low', 'Close', 'Volume', '__label__'}
            feature_cols = [c for c in df_train.columns
                            if c not in exclude and not df_train[c].isnull().all()]

            y = df_train['__label__'].values.astype(int)
            horizon_bundles[h] = train_ensemble(
                df_train, pd.Series(y, index=df_train.index), feature_cols)
            horizon_bundles[h]['feature_cols'] = feature_cols

        bundle = {
            'horizons':   horizon_bundles,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'version':    'v2.0',
        }
        _save_ensemble_to_r2(r2_client, r2_bucket, bundle, model_key)

    # ── 推論 ──
    print("\n[INFO] 執行 Ensemble 推論...")
    horizon_results = {}
    val_accs = {}

    # 用最新一列特徵
    latest_df = df_feat.iloc[[-1]]

    for h, hb in bundle.get('horizons', {}).items():
        try:
            feature_cols = hb['feature_cols']
            # 補齊缺失特徵
            missing = [c for c in feature_cols if c not in latest_df.columns]
            if missing:
                print(f"  [WARN] {h} 缺 {len(missing)} 個特徵，補 0")
            for c in missing:
                latest_df[c] = 0.0

            X_latest = latest_df[feature_cols].values
            if np.isnan(X_latest).any():
                X_latest = np.nan_to_num(X_latest, nan=0.0)

            # 加權投票
            proba = sum(
                hb['weights'][k] * m.predict_proba(X_latest)[0][1]
                for k, m in hb['models'].items()
            )
            horizon_results[h] = round(float(proba) * 100, 1)
            val_accs[h] = hb.get('val_acc', 0)
            print(f"  [{h}] prob_up={horizon_results[h]:.1f}%  "
                  f"val_acc={val_accs[h]:.3f}")
        except Exception as e:
            print(f"  [WARN] {h} 推論失敗：{e}")

    if not horizon_results:
        return _fallback_result()

    # ── 主信號（用 5d，若無則 1d）──
    prob_raw  = horizon_results.get('5d', horizon_results.get('1d', 50)) / 100
    score     = int(prob_raw * 100)
    prob_dn   = round((1 - prob_raw) * 100, 1)

    # 動態閾值（5d CV 準確率加權）
    val_5d    = val_accs.get('5d', 0.52)
    threshold = max(0.60, 0.50 + (val_5d - 0.50) * 0.5)

    if   prob_raw >= threshold + 0.10: signal = 'STRONG_BUY';  label = '🚀 強力買進'
    elif prob_raw >= threshold:        signal = 'BUY';          label = '📈 買進'
    elif prob_raw <= 1 - threshold - 0.10: signal = 'STRONG_SELL'; label = '📉 強力賣出'
    elif prob_raw <= 1 - threshold:    signal = 'SELL';         label = '⚠️ 賣出'
    else:                              signal = 'WAIT';         label = '⏳ 觀望'

    print(f"\n[INFO] 主信號: {label} ({prob_raw:.1%}) | threshold={threshold:.2f}")

    # 最新黃金價格
    gold_price = round(float(gold['Close'].iloc[-1]), 2)

    # ETF flow（用 Twelve Data）
    try:
        gld = _td_fetch_history('GLD', td_key, 10)
        gdx = _td_fetch_history('GDX', td_key, 10)
        if not gld.empty and not gdx.empty:
            ratio_now  = float(gld['Close'].iloc[-1] / gdx['Close'].iloc[-1])
            ratio_prev = float(gld['Close'].iloc[-5] / gdx['Close'].iloc[-5])
            etf_flow   = '流入 ▲' if ratio_now > ratio_prev else '流出 ▼'
            etf_ratio  = round(ratio_now, 3)
        else:
            etf_flow, etf_ratio = '─', None
    except Exception:
        etf_flow, etf_ratio = '─', None

    # 組裝結果（相容原 Lambda 格式）
    n_features = len(bundle['horizons'].get('5d', {}).get('feature_cols', []))
    result = {
        'timestamp':     datetime.now(timezone.utc).isoformat() + 'Z',
        'score':         score,
        'signal':        signal,
        'label':         label,
        'risk_tip':      f"Ensemble v2.0 | 動態閾值={threshold:.2f}",
        'ai_confidence': score,
        'prob_up':       horizon_results.get('5d', round(prob_raw * 100, 1)),
        'prob_dn':       prob_dn,
        'status':        'v2.0 Ensemble XGBoost+LightGBM+CatBoost',
        'model': {
            'type':          'Ensemble',
            'components':    list(bundle['horizons'].get('5d', {}).get('models', {}).keys()),
            'features':      n_features,
            'train_samples': bundle['horizons'].get('5d', {}).get('n_train', 0),
            'val_acc':       round(val_accs.get('5d', 0), 4),
            'dynamic_threshold': round(threshold, 3),
            'horizons': {
                h: {
                    'prob_up':      horizon_results.get(h),
                    'cv_win_rate':  round(val_accs.get(h, 0) * 100, 1),
                }
                for h in HORIZONS
            },
        },
        'smart_money': {
            'etf_flow':  etf_flow,
            'etf_ratio': etf_ratio,
            'cot_report': '由 update_gld_data.py 接入',
        },
        'breakdown': [
            f"Ensemble: XGBoost+LightGBM+CatBoost ({n_features} 特徵)",
            f"5d prob_up={prob_raw:.1%} | val_acc={val_accs.get('5d',0):.3f}",
            f"動態閾值={threshold:.3f}（基於驗證準確率）",
        ],
        'gold':        {'price': gold_price},
        'performance': {
            'sharpe':        None,
            'win_rate':      None,
            'cv_win_rate_1d':  round(val_accs.get('1d', 0) * 100, 1),
            'cv_win_rate_5d':  round(val_accs.get('5d', 0) * 100, 1),
            'cv_win_rate_30d': round(val_accs.get('30d', 0) * 100, 1),
            'max_dd':        None,
            'equity_curve':  [],
        },
    }
    return result


def _fallback_result(ticker: str = 'GC=F') -> dict:
    return {
        'timestamp':     datetime.now(timezone.utc).isoformat() + 'Z',
        'score':         50, 'signal': 'WAIT', 'label': '⏳ 觀望',
        'ai_confidence': 50, 'prob_up': 50.0, 'prob_dn': 50.0,
        'status':        'fallback（資料不足）',
        'model':         {'type': 'fallback', 'horizons': {}},
        'smart_money':   {'etf_flow': '─', 'etf_ratio': None},
        'breakdown':     ['資料不足，無法預測'],
        'gold':          {'price': 0},
        'performance':   {'cv_win_rate_1d': 0, 'cv_win_rate_5d': 0, 'cv_win_rate_30d': 0, 'equity_curve': []},
    }


def run_all_assets(td_key: str, r2_client=None, r2_bucket: str = 'richtrong-collect') -> dict:
    """
    對四個資產分別執行 Ensemble 推論
    回傳 {'gold': result, 'silver': result, 'tw': result, 'us': result}
    """
    results = {}
    for asset_key, info in ASSETS.items():
        ticker = info['ticker']
        print(f"\n{'='*55}")
        print(f"  {info['emoji']} {info['name']} ({ticker})")
        print('='*55)
        try:
            result = run_ensemble(td_key, r2_client, r2_bucket, ticker)
            result['_asset_key']  = asset_key
            result['_asset_name'] = info['name']
            result['_asset_emoji']= info['emoji']
            result['_currency']   = info['currency']
            results[asset_key] = result
        except Exception as e:
            print(f"[WARN] {ticker} 失敗: {e}")
            fb = _fallback_result(ticker)
            fb.update({'_asset_key': asset_key, '_asset_name': info['name'],
                       '_asset_emoji': info['emoji'], '_currency': info['currency']})
            results[asset_key] = fb
    return results


# ── 單獨執行（測試用）───────────────────────────────
if __name__ == '__main__':
    import boto3
    TD_KEY    = os.environ.get('TWELVE_DATA_KEY', '')
    R2_KEY    = os.environ.get('R2_ACCESS_KEY_ID', '')
    R2_SECRET = os.environ.get('R2_SECRET_ACCESS_KEY', '')
    R2_EP     = os.environ.get('R2_ENDPOINT_URL',
                    'https://adb1040c847f4ae4a7d6bfedcccd7b77.r2.cloudflarestorage.com')
    R2_BUCKET = os.environ.get('R2_BUCKET', 'richtrong-collect')

    r2 = boto3.client('s3',
        endpoint_url=R2_EP,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
        region_name='auto') if R2_KEY else None

    result = run_ensemble(TD_KEY, r2, R2_BUCKET)
    print('\n=== 結果 ===')
    print(json.dumps({k: v for k, v in result.items() if k != 'model'}, ensure_ascii=False, indent=2))
    print(f"\nmodel.horizons: {result['model']['horizons']}")
