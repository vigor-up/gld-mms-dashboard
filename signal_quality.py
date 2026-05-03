#!/usr/bin/env python3
"""
GLD-MMS 信號品質評分器 v1.0
================================
搭配 Ensemble v2.0，決定是否值得推送 Bark 通知

評分維度：
  A. 三模型共識（XGBoost + LightGBM + CatBoost）—— 最高 40 分
  B. 技術買賣點偵測（RSI / MACD / Bollinger）—— 最高 35 分
  D. 多時框共識（1d + 5d + 30d 同向）—— 最高 25 分

推播閾值：總分 ≥ 70 才發 Bark
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass, field
from typing import Optional

# ── 閾值設定 ──────────────────────────────────────────
PUSH_THRESHOLD   = 70    # 總分門檻（滿分 100）
RSI_OVERSOLD     = 35    # RSI 超賣線
RSI_OVERBOUGHT   = 65    # RSI 超買線
MACD_HIST_MIN    = 0.5   # MACD hist 絕對值最小門檻（過濾微小交叉）
BB_BREAKOUT_PCT  = 0.995 # 布林通道突破比例（收盤 > 上軌 * 0.995）

@dataclass
class SignalQualityResult:
    total_score:    int
    should_push:    bool
    direction:      str          # 'BUY' / 'SELL' / 'WAIT'
    reasons:        list[str]    # 推播說明
    score_breakdown: dict[str, int]
    tech_snapshot:  dict         # 技術指標快照（用於 Bark 內文）

    def bark_title(self) -> str:
        emoji = {'BUY': '📈', 'SELL': '📉', 'WAIT': '⏳'}[self.direction]
        dir_zh = {'BUY': '買進', 'SELL': '賣出', 'WAIT': '觀望'}[self.direction]
        return f"{emoji} 黃金{dir_zh}信號 | 品質分 {self.total_score}/100"

    def bark_body(self, gold_price: float) -> str:
        parts = []
        parts.append(' · '.join(self.reasons[:3]))   # 最多列 3 個原因
        snap = self.tech_snapshot
        indicators = []
        if snap.get('rsi'):
            indicators.append(f"RSI {snap['rsi']:.0f}")
        if snap.get('macd_hist') is not None:
            h = snap['macd_hist']
            indicators.append(f"MACD {'▲' if h > 0 else '▼'}{abs(h):.2f}")
        if snap.get('adx'):
            indicators.append(f"ADX {snap['adx']:.0f}")
        if indicators:
            parts.append(' | '.join(indicators))
        parts.append(f"${gold_price:,.1f}")
        action = {'BUY': '建議分批進場', 'SELL': '建議減倉觀望', 'WAIT': '靜待更明確訊號'}[self.direction]
        parts.append(action)
        return '\n'.join(parts)


def evaluate_signal(
    ensemble_result: dict,
    latest_features: Optional[dict] = None,
    prev_directions: Optional[list[str]] = None,
) -> SignalQualityResult:
    """
    主評分函式

    Parameters
    ----------
    ensemble_result  : run_ensemble() 回傳的 dict
    latest_features  : 最新一列的技術指標值（從 DataFrame 轉成 dict）
    prev_directions  : 最近 N 次的方向列表（用於趨勢連續性）
    """
    score_a = _score_A_model_consensus(ensemble_result)
    score_b, tech_snap = _score_B_technical(ensemble_result, latest_features or {})
    score_d = _score_D_timeframe(ensemble_result)

    total = score_a + score_b + score_d

    # 決定方向
    prob_up = ensemble_result.get('prob_up', 50)
    prob_dn = ensemble_result.get('prob_dn', 50)
    signal  = ensemble_result.get('signal', 'WAIT')

    if 'BUY' in signal:
        direction = 'BUY'
    elif 'SELL' in signal:
        direction = 'SELL'
    else:
        direction = 'WAIT'

    # 整理原因說明（由高分到低分）
    reasons = []
    breakdown = {'A_consensus': score_a, 'B_technical': score_b, 'D_timeframe': score_d}

    if score_a >= 30:
        models = ensemble_result.get('model', {}).get('components', [])
        reasons.append(f"三模型共識 {prob_up:.0f}%" if direction == 'BUY' else f"三模型共識看跌 {prob_dn:.0f}%")
    elif score_a >= 15:
        reasons.append(f"模型多數共識 {prob_up:.0f}%")

    reasons.extend(tech_snap.get('_reasons', []))

    if score_d >= 20:
        reasons.append("1d+5d+30d 三框架同向")
    elif score_d >= 12:
        reasons.append("多時框同向")

    if not reasons:
        reasons = [ensemble_result.get('label', '訊號不明')]

    return SignalQualityResult(
        total_score     = min(100, total),
        should_push     = total >= PUSH_THRESHOLD and direction != 'WAIT',
        direction       = direction,
        reasons         = reasons,
        score_breakdown = breakdown,
        tech_snapshot   = tech_snap,
    )


# ── A: 三模型共識 ────────────────────────────────────
def _score_A_model_consensus(result: dict) -> int:
    """
    三個模型對同一方向的一致程度
    全部同意 + 高機率 → 40分
    大多數同意 → 20分
    意見分歧 → 0分
    """
    model_info = result.get('model', {})
    prob_up    = result.get('prob_up', 50) / 100
    prob_dn    = result.get('prob_dn', 50) / 100
    components = model_info.get('components', [])
    val_acc    = model_info.get('val_acc', 0)

    dominant = max(prob_up, prob_dn)

    # 沒有 Ensemble（只有單模型）
    if len(components) <= 1:
        if dominant >= 0.85: return 25
        if dominant >= 0.75: return 15
        return 5

    # 三模型 Ensemble
    # 用加權機率作為共識代理（三模型平均後的結果）
    if dominant >= 0.90 and val_acc >= 0.54:   return 40
    if dominant >= 0.85 and val_acc >= 0.53:   return 32
    if dominant >= 0.80:                        return 22
    if dominant >= 0.72:                        return 12
    return 0


# ── B: 技術買賣點偵測 ─────────────────────────────────
def _score_B_technical(result: dict, feat: dict) -> tuple[int, dict]:
    """
    偵測技術指標的買賣點，回傳（分數, 技術指標快照）
    快照裡的 _reasons 會被加進推播說明
    """
    score   = 0
    reasons = []
    snap    = {}

    prob_up  = result.get('prob_up', 50) / 100
    prob_dn  = result.get('prob_dn', 50) / 100
    is_bull  = prob_up > prob_dn

    # 從特徵提取指標值
    rsi     = feat.get('RSI_14') or feat.get('RSI14')
    macd    = feat.get('MACD')
    macd_h  = feat.get('MACD_hist')
    adx     = feat.get('ADX')
    bb_pos  = feat.get('BB_pos_20')        # 0=下軌, 1=上軌
    bb_w    = feat.get('BB_width_20')
    stoch_k = feat.get('Stoch_K')
    stoch_d = feat.get('Stoch_D')
    cci     = feat.get('CCI')

    if rsi is not None:
        snap['rsi'] = round(float(rsi), 1)

        # RSI 超賣反彈（買進信號）
        if is_bull and rsi <= RSI_OVERSOLD:
            score += 18
            reasons.append(f"RSI超賣反彈（{rsi:.0f}）")
        # RSI 超買（賣出信號）
        elif not is_bull and rsi >= RSI_OVERBOUGHT:
            score += 18
            reasons.append(f"RSI超買（{rsi:.0f}）")
        # RSI 中性區但方向對
        elif (is_bull and 35 < rsi < 55) or (not is_bull and 45 < rsi < 65):
            score += 6
        # RSI 背離方向
        elif (is_bull and rsi > 70) or (not is_bull and rsi < 30):
            score -= 5   # 扣分（超買還看多是風險）

    if macd_h is not None:
        snap['macd_hist'] = round(float(macd_h), 3)

        # MACD 金叉確認（hist 從負轉正）
        if is_bull and macd_h > MACD_HIST_MIN:
            score += 10
            reasons.append("MACD多頭動能")
        # MACD 死叉確認
        elif not is_bull and macd_h < -MACD_HIST_MIN:
            score += 10
            reasons.append("MACD空頭動能")
        # 方向不一致（扣分）
        elif (is_bull and macd_h < -MACD_HIST_MIN * 2) or \
             (not is_bull and macd_h > MACD_HIST_MIN * 2):
            score -= 5

    if adx is not None:
        snap['adx'] = round(float(adx), 1)

        # ADX > 25 代表趨勢明顯（加分）
        if adx >= 30:
            score += 7
            reasons.append(f"趨勢強勁（ADX {adx:.0f}）")
        elif adx >= 25:
            score += 4

    if bb_pos is not None and bb_w is not None:
        snap['bb_pos'] = round(float(bb_pos), 3)

        # 布林通道超賣區（下軌附近）→ 買進
        if is_bull and bb_pos <= 0.15:
            score += 8
            reasons.append("布林下軌支撐")
        # 布林通道突破上軌 → 強勢
        elif is_bull and bb_pos >= BB_BREAKOUT_PCT:
            score += 5
            reasons.append("布林上軌突破")
        # 布林超買 → 賣出
        elif not is_bull and bb_pos >= 0.85:
            score += 8
            reasons.append("布林上軌壓力")

    if stoch_k is not None and stoch_d is not None:
        # Stochastic 超賣（KD < 20 且 K > D）
        if is_bull and stoch_k < 25 and stoch_k > stoch_d:
            score += 5
            reasons.append("KD超賣黃金交叉")
        elif not is_bull and stoch_k > 75 and stoch_k < stoch_d:
            score += 5
            reasons.append("KD超買死亡交叉")

    if cci is not None:
        if is_bull and cci < -100:
            score += 4
            reasons.append(f"CCI超賣（{cci:.0f}）")
        elif not is_bull and cci > 100:
            score += 4
            reasons.append(f"CCI超買（{cci:.0f}）")

    snap['_reasons'] = reasons[:3]   # 最多保留 3 個技術原因
    return max(0, min(35, score)), snap


# ── D: 多時框共識 ─────────────────────────────────────
def _score_D_timeframe(result: dict) -> int:
    """
    1d / 5d / 30d 三個 horizon 的方向一致程度
    全部同向 > 80% → 25分
    兩個同向 → 12分
    分歧 → 0分
    """
    horizons = result.get('model', {}).get('horizons', {})
    if not horizons:
        return 0

    THRESHOLD = 60   # 超過 60% 算「看多」，低於 40% 算「看空」

    directions = []
    for h in ['1d', '5d', '30d']:
        info = horizons.get(h)
        if info and info.get('prob_up') is not None:
            p = info['prob_up']
            if p >= THRESHOLD:
                directions.append('BUY')
            elif p <= 100 - THRESHOLD:
                directions.append('SELL')
            else:
                directions.append('NEUTRAL')

    if len(directions) < 2:
        return 0

    buy_count  = directions.count('BUY')
    sell_count = directions.count('SELL')

    # 三框架都同向
    if buy_count == 3:
        # 看看各 prob 的強度
        avg_prob = sum(
            horizons[h]['prob_up']
            for h in ['1d', '5d', '30d']
            if horizons.get(h, {}).get('prob_up') is not None
        ) / 3
        if avg_prob >= 75:  return 25
        if avg_prob >= 65:  return 20
        return 15

    if sell_count == 3:
        avg_prob = sum(
            100 - horizons[h]['prob_up']
            for h in ['1d', '5d', '30d']
            if horizons.get(h, {}).get('prob_up') is not None
        ) / 3
        if avg_prob >= 75:  return 25
        if avg_prob >= 65:  return 20
        return 15

    # 兩框架同向
    if buy_count == 2 or sell_count == 2:
        return 12

    return 0


# ── 單獨測試用 ───────────────────────────────────────
if __name__ == '__main__':
    # 模擬 Ensemble 結果測試
    mock_result = {
        'signal':  'STRONG_BUY', 'label': '🚀 強力買進',
        'prob_up': 88.5, 'prob_dn': 11.5,
        'model': {
            'components': ['xgb', 'lgbm', 'cat'],
            'val_acc': 0.571,
            'horizons': {
                '1d':  {'prob_up': 72.0, 'cv_win_rate': 55.0},
                '5d':  {'prob_up': 88.5, 'cv_win_rate': 57.1},
                '30d': {'prob_up': 81.0, 'cv_win_rate': 64.2},
            }
        }
    }
    mock_feat = {
        'RSI_14': 38.5, 'MACD_hist': 1.2, 'ADX': 28.0,
        'BB_pos_20': 0.12, 'Stoch_K': 22.0, 'Stoch_D': 18.0, 'CCI': -95.0
    }

    result = evaluate_signal(mock_result, mock_feat)
    print(f"總分: {result.total_score}/100")
    print(f"是否推播: {result.should_push}")
    print(f"方向: {result.direction}")
    print(f"原因: {result.reasons}")
    print(f"分項: {result.score_breakdown}")
    print(f"Bark 標題: {result.bark_title()}")
    print(f"Bark 內文:\n{result.bark_body(4618.5)}")
