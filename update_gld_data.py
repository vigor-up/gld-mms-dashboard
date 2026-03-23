#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLD-MMS 系統 v4.1 - 信心度門檻版
1. 信心度 > 80% 才觸發買進/賣出推播。
2. 新增 SELL / STRONG_SELL 信號。
3. 三台 iOS 設備同步推送。
"""

import json
import argparse
import requests
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred
import pandas as pd
import numpy as np
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.int64, np.int32, int)): return int(obj)
        if isinstance(obj, (np.float64, np.float32, float)): return float(obj)
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class LeadingIndicatorUpdater:
    def __init__(self, fred_api_key=None, bark_keys=None):
        self.fred_api_key = fred_api_key
        self.bark_keys = [k for k in (bark_keys or []) if k]
        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
        self.assets = {}
        self.macro = {}

    def _clean_df(self, df):
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower() for col in df.columns]
        return df

    def _detect_pin_bar(self, df):
        if df.empty: return pd.Series([], dtype=bool)
        high_low = df['high'] - df['low']
        atr_20 = high_low.rolling(window=20).mean()
        return high_low > (atr_20 * 3)

    def fetch_asset_data(self, ticker, asset_name, period='60d', interval='1h'):
        print(f"[INFO] 正在獲取 {asset_name} ({ticker}) 數據...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            df = self._clean_df(df)
            if df.empty: return False
            df.reset_index(inplace=True)
            time_col = 'datetime' if 'datetime' in df.columns else df.columns[0]
            df['date_full'] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M')
            if 'adj close' in df.columns: df['close'] = df['adj close']

            # --- 核心指標計算 ---
            # 1. OBV & OBV MA
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ma5'] = df['obv'].rolling(window=5).mean()
            # 2. VWAP
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (df['tp'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            df['vwap_dev'] = (df['close'] / df['vwap'] - 1) * 100
            # 3. ATR 波動率擠壓
            df['tr'] = np.maximum(df['high'] - df['low'],
                       np.maximum(abs(df['high'] - df['close'].shift(1)),
                                 abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(window=20).mean()
            df['atr_low'] = df['atr'] < df['atr'].rolling(window=50).min() * 1.1
            # 4. 能量背離
            df['price_down'] = df['close'] < df['close'].shift(3)
            df['obv_up'] = df['obv'] > df['obv'].shift(3)
            df['bull_div'] = df['price_down'] & df['obv_up']
            # 5. 頂背離（用於賣出）
            df['price_up'] = df['close'] > df['close'].shift(3)
            df['obv_down'] = df['obv'] < df['obv'].shift(3)
            df['bear_div'] = df['price_up'] & df['obv_down']

            df['is_pin_bar'] = self._detect_pin_bar(df).astype(bool)
            self.assets[ticker] = df.tail(100).to_dict('records')
            return True
        except Exception as e:
            print(f"[ERROR] {asset_name} 失敗: {e}")
            return False

    def fetch_macro_data(self):
        print("[INFO] 正在獲取宏觀數據...")
        try:
            uup = yf.download('UUP', period='60d', progress=False)
            uup = self._clean_df(uup)
            if not uup.empty:
                uup['sma20'] = uup['close'].rolling(20).mean()
                uup.reset_index(inplace=True)
                uup['date'] = pd.to_datetime(uup['date']).dt.strftime('%Y-%m-%d')
                self.macro['dxy'] = uup.tail(30).to_dict('records')
            return True
        except: return False

    def send_push(self, title, content, ticker, is_leading=False):
        """推播至所有已設定的設備（最多3台）"""
        icon = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/gold.png'
        group = "GLD-LEADING" if is_leading else "GLD-MMS"
        sent = 0
        for key in self.bark_keys:
            url = f"https://api.day.app/{key}/{title}/{content}?icon={icon}&group={group}&isArchive=1"
            try:
                requests.get(url, timeout=10)
                sent += 1
            except Exception as e:
                print(f"[WARN] Bark 推播失敗 (設備 {sent+1}): {e}")
        print(f"[INFO] 推播完成：{sent}/{len(self.bark_keys)} 台設備")

    def calculate_signals(self):
        signals = {}
        CONFIDENCE_THRESHOLD = 80  # ✅ 信心度門檻：>80% 才推播

        for ticker, data in self.assets.items():
            df = pd.DataFrame(data)
            latest = df.iloc[-1]

            # --- 1. 前瞻雷達 ---
            is_squeeze = bool(latest['atr_low'])
            is_div = bool(latest['bull_div'])
            is_bear_div = bool(latest['bear_div'])
            is_oversold = bool(latest['vwap_dev'] < -1.2)
            is_overbought = bool(latest['vwap_dev'] > 1.2)

            radar_msg = []
            if is_squeeze: radar_msg.append("波動擠壓(即將變盤)")
            if is_div: radar_msg.append("能量底背離(聰明錢進場)")
            if is_oversold: radar_msg.append("乖離過大(預期回抽)")
            if is_bear_div: radar_msg.append("能量頂背離(主力出貨)")
            if is_overbought: radar_msg.append("超買乖離(預期回落)")

            # --- 2. 決策邏輯 ---
            # 趨勢買進
            if latest['close'] > latest['vwap'] and latest['obv'] > latest['obv_ma5']:
                s_sig = 'STRONG_BUY' if latest['vwap_dev'] < 1.0 else 'BUY'
                s_conf = 85 if s_sig == 'STRONG_BUY' else 70
                s_reason = "右側趨勢確認：量價共振向上"
            # 趨勢賣出
            elif latest['close'] < latest['vwap'] and latest['obv'] < latest['obv_ma5']:
                if is_bear_div and is_overbought:
                    s_sig = 'STRONG_SELL'
                    s_conf = 88
                elif is_bear_div or is_overbought:
                    s_sig = 'SELL'
                    s_conf = 82
                else:
                    s_sig = 'SELL'
                    s_conf = 72
                s_reason = "賣出確認：量價共振向下" + (" | " + " & ".join([m for m in radar_msg if "頂" in m or "超買" in m]) if any("頂" in m or "超買" in m for m in radar_msg) else "")
            # 左側預判買入
            elif (is_div or is_oversold) and not bool(latest['is_pin_bar']):
                s_sig = 'PRE_BUY'
                s_conf = 75 if is_div and is_oversold else 65
                s_reason = "偵測到左側交易機會：" + " & ".join(radar_msg) if radar_msg else "左側預判"
            else:
                s_sig = 'HOLD'
                s_conf = 50
                s_reason = "等待變盤信號或趨勢確認"

            signals[ticker] = {
                'short_term': {'signal': s_sig, 'confidence': s_conf, 'reason': s_reason},
                'radar': {
                    'squeeze': is_squeeze,
                    'divergence': is_div,
                    'bear_divergence': is_bear_div,
                    'oversold': is_oversold,
                    'overbought': is_overbought,
                    'msg': " | ".join(radar_msg) if radar_msg else "雷達掃描中..."
                },
                'price': float(round(latest['close'], 2)),
                'vwap_dev': float(round(latest['vwap_dev'], 2)),
                'anomaly': {'pin': bool(latest['is_pin_bar'])}
            }

            # ✅ 推播邏輯：信心度 > 80% 才觸發，三台設備同步
            if ticker == 'GC=F' and s_conf > CONFIDENCE_THRESHOLD:
                if s_sig == 'PRE_BUY':
                    self.send_push(
                        "🚨 變盤預警：準備買入！",
                        f"信心: {s_conf}% | {s_reason} | 預計24H內發動",
                        ticker, is_leading=True
                    )
                elif s_sig in ['BUY', 'STRONG_BUY']:
                    self.send_push(
                        f"📈 黃金買進信號！({'強烈' if s_sig == 'STRONG_BUY' else '普通'})",
                        f"信心: {s_conf}% | 價格: ${latest['close']:.2f} | {s_reason}",
                        ticker
                    )
                elif s_sig in ['SELL', 'STRONG_SELL']:
                    self.send_push(
                        f"📉 黃金賣出信號！({'強烈' if s_sig == 'STRONG_SELL' else '普通'})",
                        f"信心: {s_conf}% | 價格: ${latest['close']:.2f} | {s_reason}",
                        ticker
                    )

        return signals

    def update_html(self, html_file):
        data = {
            'timestamp': datetime.now().isoformat(),
            'assets': self.assets,
            'macro': self.macro,
            'signals': self.calculate_signals()
        }
        with open(html_file, 'r', encoding='utf-8') as f: content = f.read()
        start_marker = '<script id="data-source">'
        end_marker = '</script>'
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if start_idx != -1 and end_idx != -1:
            data_json = json.dumps(data, cls=NumpyEncoder)
            new_script = f'{start_marker}const AUTO_DATA = {data_json};{end_marker}'
            new_content = content[:start_idx] + new_script + content[end_idx + len(end_marker):]
            with open(html_file, 'w', encoding='utf-8') as f: f.write(new_content)
            print(f"[SUCCESS] v4.1 數據已更新")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', default='docs/index.html')
    parser.add_argument('--fred-key', default=None)
    parser.add_argument('--bark-key-1', default=None)
    parser.add_argument('--bark-key-2', default=None)
    parser.add_argument('--bark-key-3', default=None)
    args, _ = parser.parse_known_args()
    updater = LeadingIndicatorUpdater(
        fred_api_key=args.fred_key,
        bark_keys=[args.bark_key_1, args.bark_key_2, args.bark_key_3]
    )
    updater.fetch_asset_data('GC=F', 'Gold Futures (24H)')
    updater.fetch_asset_data('SI=F', 'Silver Futures (24H)')
    updater.fetch_asset_data('GLD', 'GLD ETF (Ref)')
    updater.fetch_macro_data()
    updater.update_html(args.html)

if __name__ == '__main__':
    main()
