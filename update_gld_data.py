#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLD-MMS 系統 v5.0 - Lambda XGBoost 整合版
- 呼叫 AWS Lambda "gld-mms-updater" 取得 AI 評分 (XGBoost)
- 整合技術指標 (OBV/VWAP/ATR/背離)
- 綜合評分：Lambda AI 60% + 技術指標 40%
- 信心度 > 80% 才推播至三台設備
"""

import json
import argparse
import requests
import boto3
from datetime import datetime
import yfinance as yf
from fredapi import Fred
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cot_module import get_gold_cot, get_silver_cot
    _COT_AVAILABLE = True
except Exception:
    _COT_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, (np.int64, np.int32, int)): return int(obj)
        if isinstance(obj, (np.float64, np.float32, float)): return float(obj)
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

class GldMmsUpdater:
    def __init__(self, fred_api_key=None, bark_keys=None,
                 aws_access_key=None, aws_secret_key=None, aws_region='ap-northeast-1'):
        self.fred_api_key = fred_api_key
        self.bark_keys = [k for k in (bark_keys or []) if k]
        self.aws_region = aws_region
        self.assets = {}
        self.macro = {}
        self.lambda_result = {}

        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)

        # AWS Lambda / S3 client
        boto_kwargs = dict(region_name=aws_region)
        if aws_access_key and aws_secret_key:
            boto_kwargs['aws_access_key_id'] = aws_access_key
            boto_kwargs['aws_secret_access_key'] = aws_secret_key
        self.lambda_client = boto3.client('lambda', **boto_kwargs)
        self.s3_client = boto3.client('s3', **boto_kwargs)

    # ── 資料清洗 ──────────────────────────────────────────────
    def _clean_df(self, df):
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower() for col in df.columns]
        return df

    def _detect_pin_bar(self, df):
        if df.empty: return pd.Series([], dtype=bool)
        hl = df['high'] - df['low']
        return hl > (hl.rolling(20).mean() * 3)

    # ── Step 1: 呼叫 Lambda 取得 AI 評分 ─────────────────────
    def invoke_lambda(self):
        """呼叫 gld-mms-updater Lambda，取得 XGBoost AI 評分"""
        print("[INFO] 呼叫 Lambda gld-mms-updater...")
        try:
            response = self.lambda_client.invoke(
                FunctionName='gld-mms-updater',
                InvocationType='RequestResponse',
                Payload=json.dumps({})
            )
            status = response.get('StatusCode', 0)
            print(f"[INFO] Lambda HTTP 狀態碼: {status}")

            if status == 200:
                # Lambda 執行完後從 S3 讀取最新 data.json
                s3_resp = self.s3_client.get_object(
                    Bucket='gld-mms-data-richtrong',
                    Key='data.json'
                )
                self.lambda_result = json.loads(s3_resp['Body'].read().decode())
                score = self.lambda_result.get('score', 50)
                signal = self.lambda_result.get('signal', '未知')
                print(f"[SUCCESS] Lambda AI 評分: {score} | 信號: {signal}")
                return True
            else:
                print(f"[WARN] Lambda 回應異常: {status}")
                return False
        except Exception as e:
            print(f"[WARN] Lambda 呼叫失敗，使用純技術分析: {e}")
            self.lambda_result = {}
            return False

    # ── Step 2: 抓取技術指標數據 ──────────────────────────────
    def fetch_asset_data(self, ticker, asset_name, period='60d', interval='1h'):
        print(f"[INFO] 獲取 {asset_name} ({ticker})...")
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            df = self._clean_df(df)
            if df.empty: return False
            df.reset_index(inplace=True)
            time_col = 'datetime' if 'datetime' in df.columns else df.columns[0]
            df['date_full'] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:%M')
            if 'adj close' in df.columns: df['close'] = df['adj close']

            # OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_ma5'] = df['obv'].rolling(5).mean()
            # VWAP
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (df['tp'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            df['vwap_dev'] = (df['close'] / df['vwap'] - 1) * 100
            # ATR
            df['tr'] = np.maximum(df['high'] - df['low'],
                       np.maximum(abs(df['high'] - df['close'].shift(1)),
                                 abs(df['low'] - df['close'].shift(1))))
            df['atr'] = df['tr'].rolling(20).mean()
            df['atr_low'] = df['atr'] < df['atr'].rolling(50).min() * 1.1
            # 底背離 / 頂背離
            df['bull_div'] = (df['close'] < df['close'].shift(3)) & (df['obv'] > df['obv'].shift(3))
            df['bear_div'] = (df['close'] > df['close'].shift(3)) & (df['obv'] < df['obv'].shift(3))
            df['is_pin_bar'] = self._detect_pin_bar(df).astype(bool)

            self.assets[ticker] = df.tail(100).to_dict('records')
            return True
        except Exception as e:
            print(f"[ERROR] {ticker} 失敗: {e}")
            return False

    def fetch_macro_data(self):
        print("[INFO] 獲取宏觀數據...")
        try:
            uup = yf.download('UUP', period='60d', progress=False)
            uup = self._clean_df(uup)
            if not uup.empty:
                uup['sma20'] = uup['close'].rolling(20).mean()
                uup.reset_index(inplace=True)
                uup['date'] = pd.to_datetime(uup['date']).dt.strftime('%Y-%m-%d')
                self.macro['dxy'] = uup.tail(30).to_dict('records')
        except: pass

        # COT 報告
        if _COT_AVAILABLE:
            try:
                print("[INFO] 獲取 CFTC COT 報告...")
                self.macro['cot_gold']  = get_gold_cot()
                self.macro['cot_silver'] = get_silver_cot()
                gold_cot = self.macro['cot_gold']
                print(f"[SUCCESS] COT 黃金: {gold_cot.get('summary','')}")
            except Exception as e:
                print(f"[WARN] COT 獲取失敗: {e}")
                self.macro['cot_gold'] = {"error": str(e)}
        return True

    # ── Step 3: 推播 ──────────────────────────────────────────
    def send_push(self, title, content, is_leading=False):
        group = "GLD-LEADING" if is_leading else "GLD-MMS"
        icon = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/128/color/gold.png'
        sent = 0
        for key in self.bark_keys:
            url = f"https://api.day.app/{key}/{title}/{content}?icon={icon}&group={group}&isArchive=1"
            try:
                requests.get(url, timeout=10)
                sent += 1
            except Exception as e:
                print(f"[WARN] Bark 推播失敗 (設備{sent+1}): {e}")
        print(f"[INFO] 推播 {sent}/{len(self.bark_keys)} 台設備")

    # ── Step 4: 計算技術分 (0-100) ────────────────────────────
    def _calc_tech_score(self, latest):
        score = 50
        # OBV 方向 (±20)
        if latest.get('obv', 0) > latest.get('obv_ma5', 0): score += 20
        else: score -= 20
        # VWAP 乖離 (±15)
        vdev = latest.get('vwap_dev', 0)
        if vdev < -1.2: score += 15      # 超跌，看多
        elif vdev > 1.2: score -= 15     # 超漲，看空
        elif vdev > 0.5: score += 5
        # 底背離 (±15)
        if latest.get('bull_div'): score += 15
        if latest.get('bear_div'): score -= 15
        # ATR 擠壓 (+5 中性)
        if latest.get('atr_low'): score += 5
        # Pin bar 壓制 (-5)
        if latest.get('is_pin_bar'): score -= 5
        return max(0, min(100, score))

    # ── Step 5: 綜合評分 + 信號 ───────────────────────────────
    def calculate_signals(self):
        PUSH_THRESHOLD = 80
        signals = {}
        lambda_score = self.lambda_result.get('score', None)
        smart_money = self.lambda_result.get('smart_money', {})
        performance = self.lambda_result.get('performance', {})

        for ticker, data in self.assets.items():
            df = pd.DataFrame(data)
            latest = df.iloc[-1]

            # 技術分
            tech_score = self._calc_tech_score(latest)

            # 綜合評分
            if lambda_score is not None:
                combined = round(lambda_score * 0.6 + tech_score * 0.4)
                score_note = f"Lambda AI {lambda_score}% × 60% + 技術 {tech_score}% × 40%"
            else:
                combined = tech_score
                score_note = f"純技術分析 {tech_score}%（Lambda 未連線）"

            # 信號判斷
            if combined >= 80:
                signal = 'STRONG_BUY'
                reason = f"🚀 AI + 技術雙重確認看多 | {score_note}"
            elif combined >= 65:
                signal = 'BUY'
                reason = f"📈 多方趨勢形成 | {score_note}"
            elif combined >= 55:
                signal = 'PRE_BUY'
                reason = f"👀 左側預判機會 | {score_note}"
            elif combined <= 20:
                signal = 'STRONG_SELL'
                reason = f"🔴 AI + 技術雙重確認看空 | {score_note}"
            elif combined <= 35:
                signal = 'SELL'
                reason = f"📉 空方趨勢形成 | {score_note}"
            else:
                signal = 'HOLD'
                reason = f"⏳ 信號不明確，觀望 | {score_note}"

            # 雷達
            radar = {
                'squeeze': bool(latest.get('atr_low')),
                'divergence': bool(latest.get('bull_div')),
                'bear_divergence': bool(latest.get('bear_div')),
                'oversold': bool(latest.get('vwap_dev', 0) < -1.2),
                'overbought': bool(latest.get('vwap_dev', 0) > 1.2),
            }
            radar_msgs = []
            if radar['squeeze']: radar_msgs.append("波動擠壓")
            if radar['divergence']: radar_msgs.append("底背離")
            if radar['bear_divergence']: radar_msgs.append("頂背離")
            if radar['oversold']: radar_msgs.append("超跌")
            if radar['overbought']: radar_msgs.append("超買")
            radar['msg'] = " | ".join(radar_msgs) if radar_msgs else "掃描中..."

            signals[ticker] = {
                'short_term': {
                    'signal': signal,
                    'confidence': combined,
                    'tech_score': tech_score,
                    'lambda_score': lambda_score,
                    'reason': reason
                },
                'radar': radar,
                'smart_money': {
                    **smart_money,
                    'cot_report': self.macro.get('cot_gold', {}).get('summary', '待更新'),
                    'cot_detail': self.macro.get('cot_gold', {}),
                },
                'performance': performance,
                'price': float(round(latest['close'], 2)),
                'vwap_dev': float(round(latest.get('vwap_dev', 0), 2)),
                'anomaly': {'pin': bool(latest.get('is_pin_bar'))}
            }

            # 推播：信心度 > 80%，GC=F 主標的
            if ticker == 'GC=F' and combined > PUSH_THRESHOLD:
                price = float(round(latest['close'], 2))
                etf = smart_money.get('etf_flow', '未知')
                cot = smart_money.get('cot_report', '未知')
                if signal in ['BUY', 'STRONG_BUY']:
                    self.send_push(
                        f"📈 黃金買進！{'強烈' if signal == 'STRONG_BUY' else ''}信號",
                        f"AI綜合評分:{combined}% | 價格:${price:.2f} | ETF:{etf} | COT:{cot}"
                    )
                elif signal in ['SELL', 'STRONG_SELL']:
                    self.send_push(
                        f"📉 黃金賣出！{'強烈' if signal == 'STRONG_SELL' else ''}信號",
                        f"AI綜合評分:{combined}% | 價格:${price:.2f} | ETF:{etf} | COT:{cot}"
                    )

        return signals

    # ── Step 6: 更新 HTML ──────────────────────────────────────
    def update_html(self, html_file):
        data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',  # 明確標示 UTC
            'assets': self.assets,
            'macro': self.macro,
            'signals': self.calculate_signals(),
            'lambda': {
                'score': self.lambda_result.get('score'),
                'signal': self.lambda_result.get('signal'),
                'status': self.lambda_result.get('status', '未連線'),
                'smart_money': self.lambda_result.get('smart_money', {}),
                'performance': self.lambda_result.get('performance', {}),
                'model': self.lambda_result.get('model', {}),
                'prob_up': self.lambda_result.get('prob_up'),
                'prob_dn': self.lambda_result.get('prob_dn'),
            }
        }
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        start_marker = '<script id="data-source">'
        end_marker = '</script>'
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if start_idx != -1 and end_idx != -1:
            data_json = json.dumps(data, cls=NumpyEncoder)
            new_content = (content[:start_idx]
                + f'{start_marker}const AUTO_DATA = {data_json};{end_marker}'
                + content[end_idx + len(end_marker):])
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[SUCCESS] v5.0 Lambda XGBoost 整合版資料已更新")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--html', default='docs/index.html')
    parser.add_argument('--fred-key', default=None)
    parser.add_argument('--bark-key-1', default=None)
    parser.add_argument('--bark-key-2', default=None)
    parser.add_argument('--bark-key-3', default=None)
    parser.add_argument('--aws-access-key', default=None)
    parser.add_argument('--aws-secret-key', default=None)
    parser.add_argument('--aws-region', default='ap-northeast-1')
    args, _ = parser.parse_known_args()

    updater = GldMmsUpdater(
        fred_api_key=args.fred_key,
        bark_keys=[args.bark_key_1, args.bark_key_2, args.bark_key_3],
        aws_access_key=args.aws_access_key,
        aws_secret_key=args.aws_secret_key,
        aws_region=args.aws_region
    )

    # 執行順序
    updater.invoke_lambda()                          # 1. 呼叫 Lambda AI
    updater.fetch_asset_data('GC=F', '黃金期貨')    # 2. 技術指標
    updater.fetch_asset_data('SI=F', '白銀期貨')
    updater.fetch_asset_data('GLD', 'GLD ETF')
    updater.fetch_macro_data()                       # 3. 宏觀數據
    updater.update_html(args.html)                   # 4. 更新 HTML

if __name__ == '__main__':
    main()
