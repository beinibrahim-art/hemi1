"""
🚀 ML-Enhanced Backtest - مقارنة Rule-Based vs ML
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import databento as db
import pandas as pd
import numpy as np
import glob
import os
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

class MLEnhancedBacktest:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("🤖 ML-Enhanced Backtest - مقارنة الأداء")
        print("="*100)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.data_folder = self.config['data']['folder']
        self.symbol = self.config['data']['symbol']
        self.initial_capital = self.config['capital']['initial']
        
        self.output_folder = self.config['output']['folder']
        self.ml_folder = os.path.join(self.output_folder, 'ml_models')
        
        # تحميل ML Model
        print("\n📥 تحميل ML Model...")
        model_file = os.path.join(self.ml_folder, 'XGBoost_model.pkl')
        if not os.path.exists(model_file):
            model_file = os.path.join(self.ml_folder, 'RandomForest_model.pkl')
        
        if os.path.exists(model_file):
            self.ml_model = joblib.load(model_file)
            print(f"✅ تم تحميل: {os.path.basename(model_file)}")
        else:
            print("❌ ML Model غير موجود! شغل ml_trainer.py أولاً")
            sys.exit(1)
        
        self.capital_rule = self.initial_capital
        self.capital_ml = self.initial_capital
        
        self.trades_rule = []
        self.trades_ml = []
        
        self.daily_stats_rule = []
        self.daily_stats_ml = []
        
        self.dbn_files = sorted(glob.glob(os.path.join(self.data_folder, "*.dbn.zst")))
        print(f"✅ {len(self.dbn_files)} ملف")
    
    def load_daily_data(self, file_path):
        try:
            store = db.DBNStore.from_file(file_path)
            df = store.to_df()
            
            symbols_to_try = ['ESZ4', 'ESH5', 'ESM5', 'ESU5', 'ESZ5', 'ESH6', 'ESM6']
            for sym in symbols_to_try:
                if sym in df['symbol'].values:
                    df = df[df['symbol'] == sym].copy()
                    if len(df) > 0:
                        return df
            
            es_symbols = [s for s in df['symbol'].unique() if s.startswith('ES')]
            if es_symbols:
                df = df[df['symbol'] == es_symbols[0]].copy()
                return df
            
            return None
        except:
            return None
    
    def create_ohlc(self, df):
        try:
            df['delta'] = np.where(df['side'] == 'A', df['size'], -df['size'])
            
            ohlc = df.groupby(pd.Grouper(freq='5min')).agg({
                'price': ['first', 'max', 'min', 'last'],
                'size': 'sum',
                'delta': 'sum'
            })
            
            ohlc.columns = ['open', 'high', 'low', 'close', 'volume', 'delta']
            ohlc = ohlc.dropna()
            
            return ohlc
        except:
            return None
    
    def find_swing_points(self, ohlc, period=3):
        swings = []
        for i in range(period, len(ohlc) - period):
            if ohlc['high'].iloc[i] == ohlc['high'].iloc[i-period:i+period+1].max():
                swings.append({'time': ohlc.index[i], 'type': 'High', 'price': ohlc['high'].iloc[i]})
            if ohlc['low'].iloc[i] == ohlc['low'].iloc[i-period:i+period+1].min():
                swings.append({'time': ohlc.index[i], 'type': 'Low', 'price': ohlc['low'].iloc[i]})
        return pd.DataFrame(swings) if swings else pd.DataFrame()
    
    def find_order_blocks(self, ohlc):
        obs = []
        min_strength = 8
        lookback = 20
        
        for i in range(lookback, len(ohlc)):
            if ohlc['close'].iloc[i] > ohlc['close'].iloc[i-1]:
                for j in range(i-1, max(0, i-lookback), -1):
                    if ohlc['close'].iloc[j] < ohlc['open'].iloc[j]:
                        move = ohlc['high'].iloc[i] - ohlc['low'].iloc[j]
                        if move >= min_strength:
                            obs.append({'time': ohlc.index[j], 'type': 'Bullish',
                                       'high': ohlc['high'].iloc[j], 'low': ohlc['low'].iloc[j], 'strength': move})
                        break
            elif ohlc['close'].iloc[i] < ohlc['close'].iloc[i-1]:
                for j in range(i-1, max(0, i-lookback), -1):
                    if ohlc['close'].iloc[j] > ohlc['open'].iloc[j]:
                        move = ohlc['high'].iloc[j] - ohlc['low'].iloc[i]
                        if move >= min_strength:
                            obs.append({'time': ohlc.index[j], 'type': 'Bearish',
                                       'high': ohlc['high'].iloc[j], 'low': ohlc['low'].iloc[j], 'strength': move})
                        break
        
        return pd.DataFrame(obs) if obs else pd.DataFrame()
    
    def get_killzone(self, hour):
        if 7 <= hour < 10:
            return 'London', 10
        elif 13 <= hour < 16:
            return 'NY_AM', 10
        elif 18 <= hour < 21:
            return 'NY_PM', 9
        return None, 0
    
    def find_smart_target(self, entry, direction, swings, current_time):
        targets = []
        st = {'min_target': 10, 'max_target': 100, 'default_target': 15}
        
        if len(swings) > 0:
            future_swings = swings[swings['time'] > current_time]
            
            if direction == 'BUY':
                highs = future_swings[future_swings['type'] == 'High']
                if len(highs) > 0:
                    for idx, (_, swing) in enumerate(highs.head(3).iterrows()):
                        distance = swing['price'] - entry
                        if st['min_target'] <= distance <= st['max_target']:
                            targets.append({'price': swing['price'], 'distance': distance, 'priority': 10 - idx})
            else:
                lows = future_swings[future_swings['type'] == 'Low']
                if len(lows) > 0:
                    for idx, (_, swing) in enumerate(lows.head(3).iterrows()):
                        distance = entry - swing['price']
                        if st['min_target'] <= distance <= st['max_target']:
                            targets.append({'price': swing['price'], 'distance': distance, 'priority': 10 - idx})
        
        if len(targets) == 0:
            default_target = st['default_target']
            if direction == 'BUY':
                return entry + default_target, default_target
            else:
                return entry - default_target, default_target
        
        targets_df = pd.DataFrame(targets)
        targets_df = targets_df.sort_values(['priority', 'distance'], ascending=[False, True])
        best = targets_df.iloc[0]
        
        return best['price'], best['distance']
    
    def find_daily_setups(self, date, ohlc, obs, swings):
        setups = []
        used_obs = {}
        
        for _, ob in obs.iterrows():
            session, priority = self.get_killzone(ob['time'].hour)
            if priority < 9:
                continue
            
            ob_key = f"{ob['time']}_{ob['type']}_{ob['low']:.2f}_{ob['high']:.2f}"
            if ob_key in used_obs:
                continue
            
            future = ohlc[(ohlc.index > ob['time']) & (ohlc.index <= ob['time'] + pd.Timedelta(hours=2))]
            
            found_setup = False
            
            for t, c in future.iterrows():
                if found_setup:
                    break
                
                entry, sl, direction = None, None, None
                
                if ob['type'] == 'Bullish' and c['low'] <= ob['high'] and c['close'] > ob['low']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['low'] - 2.0
                    direction = 'BUY'
                
                elif ob['type'] == 'Bearish' and c['high'] >= ob['low'] and c['close'] < ob['high']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['high'] + 2.0
                    direction = 'SELL'
                
                if entry:
                    tp, target_distance = self.find_smart_target(entry, direction, swings, t)
                    
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    
                    if risk <= 4.5 and reward >= 8.0 and reward/risk >= 2.0:
                        setups.append({
                            'date': date, 'time': t, 'session': session, 'type': direction,
                            'entry': entry, 'sl': sl, 'tp': tp, 'risk': risk, 'target': reward,
                            'rr': reward/risk, 'strength': ob['strength'], 'priority': priority
                        })
                        
                        used_obs[ob_key] = True
                        found_setup = True
        
        return setups
    
    def select_top_3_rule_based(self, setups):
        """الطريقة القديمة (Rule-Based)"""
        if len(setups) == 0:
            return []
        
        df = pd.DataFrame(setups)
        df = df.sort_values(['priority', 'target', 'rr'], ascending=[False, False, False])
        
        top_3 = []
        sessions_used = set()
        
        for _, s in df.iterrows():
            if len(top_3) >= 3:
                break
            if s['session'] not in sessions_used:
                top_3.append(s)
                sessions_used.add(s['session'])
        
        for _, s in df.iterrows():
            if len(top_3) >= 3:
                break
            already_added = any(x['time'] == s['time'] for x in top_3)
            if not already_added:
                top_3.append(s)
        
        return top_3
    
    def select_top_3_ml(self, setups):
        """الطريقة الجديدة (ML-Enhanced)"""
        if len(setups) == 0:
            return []
        
        if len(setups) <= 3:
            return setups
        
        # استخراج Features
        features = []
        for s in setups:
            type_num = 1 if s['type'] == 'BUY' else 0
            session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
            session_num = session_map[s['session']]
            hour = s['time'].hour
            day_of_week = s['time'].dayofweek
            
            features.append([
                type_num, s['strength'], s['risk'], s['target'],
                s['rr'], s['priority'], session_num, hour, day_of_week
            ])
        
        # التنبؤ بالاحتماليات
        probabilities = self.ml_model.predict_proba(features)[:, 1]
        
        # اختر أعلى 3 احتماليات
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        
        return [setups[i] for i in top_3_indices]
    
    def simulate_trade(self, trade, ohlc):
        future = ohlc[ohlc.index > trade['time']]
        
        if len(future) == 0:
            return None
        
        for t, candle in future.iterrows():
            if trade['type'] == 'BUY':
                if candle['low'] <= trade['sl']:
                    return {'result': 'LOSS', 'exit_price': trade['sl'], 'exit_time': t,
                           'pnl_points': -(trade['risk']), 'duration': (t - trade['time']).total_seconds() / 60}
                if candle['high'] >= trade['tp']:
                    return {'result': 'WIN', 'exit_price': trade['tp'], 'exit_time': t,
                           'pnl_points': trade['target'], 'duration': (t - trade['time']).total_seconds() / 60}
            else:
                if candle['high'] >= trade['sl']:
                    return {'result': 'LOSS', 'exit_price': trade['sl'], 'exit_time': t,
                           'pnl_points': -(trade['risk']), 'duration': (t - trade['time']).total_seconds() / 60}
                if candle['low'] <= trade['tp']:
                    return {'result': 'WIN', 'exit_price': trade['tp'], 'exit_time': t,
                           'pnl_points': trade['target'], 'duration': (t - trade['time']).total_seconds() / 60}
        
        return None
    
    def run_backtest(self):
        print("\n" + "="*100)
        print("🚀 بدء المقارنة: Rule-Based vs ML-Enhanced")
        print("="*100)
        
        total_files = len(self.dbn_files)
        processed = 0
        
        for file_idx, file_path in enumerate(self.dbn_files, 1):
            filename = os.path.basename(file_path)
            
            # استخراج التاريخ بطريقة آمنة
            date_str = filename.split('.')[0]
            if '-' in date_str:
                date_str = date_str.split('-')[-1]
            if '_' in date_str:
                date_str = date_str.split('_')[0]
            
            # تحقق من صحة التاريخ
            if not date_str.isdigit() or len(date_str) != 8:
                continue
            
            if file_idx % 20 == 0:
                print(f"\n[{file_idx}/{total_files}] معالجة...")
            
            df = self.load_daily_data(file_path)
            if df is None or len(df) < 50:
                continue
            
            ohlc = self.create_ohlc(df)
            if ohlc is None or len(ohlc) < 10:
                continue
            
            swings = self.find_swing_points(ohlc)
            obs = self.find_order_blocks(ohlc)
            
            if len(obs) == 0:
                continue
            
            # تحويل التاريخ بصيغة واضحة
            try:
                date = pd.to_datetime(date_str, format='%Y%m%d')
            except:
                continue
                
            setups = self.find_daily_setups(date, ohlc, obs, swings)
            
            if len(setups) == 0:
                continue
            
            # Rule-Based
            top_3_rule = self.select_top_3_rule_based(setups)
            daily_pnl_rule = 0
            wins_rule = 0
            losses_rule = 0
            
            for trade in top_3_rule:
                result = self.simulate_trade(trade, ohlc)
                if result:
                    pnl_usd = result['pnl_points'] * 1 * 50
                    self.capital_rule += pnl_usd
                    daily_pnl_rule += pnl_usd
                    if result['result'] == 'WIN':
                        wins_rule += 1
                    else:
                        losses_rule += 1
                    
                    trade_record = {**trade, **result}
                    trade_record['pnl_usd'] = pnl_usd
                    trade_record['capital_after'] = self.capital_rule
                    self.trades_rule.append(trade_record)
            
            self.daily_stats_rule.append({
                'date': date_str, 'trades': len(top_3_rule),
                'wins': wins_rule, 'losses': losses_rule,
                'pnl': daily_pnl_rule, 'capital': self.capital_rule
            })
            
            # ML-Enhanced
            top_3_ml = self.select_top_3_ml(setups)
            daily_pnl_ml = 0
            wins_ml = 0
            losses_ml = 0
            
            for trade in top_3_ml:
                result = self.simulate_trade(trade, ohlc)
                if result:
                    pnl_usd = result['pnl_points'] * 1 * 50
                    self.capital_ml += pnl_usd
                    daily_pnl_ml += pnl_usd
                    if result['result'] == 'WIN':
                        wins_ml += 1
                    else:
                        losses_ml += 1
                    
                    trade_record = {**trade, **result}
                    trade_record['pnl_usd'] = pnl_usd
                    trade_record['capital_after'] = self.capital_ml
                    self.trades_ml.append(trade_record)
            
            self.daily_stats_ml.append({
                'date': date_str, 'trades': len(top_3_ml),
                'wins': wins_ml, 'losses': losses_ml,
                'pnl': daily_pnl_ml, 'capital': self.capital_ml
            })
            
            processed += 1
        
        self.generate_comparison()
        print(f"\n✅ تم معالجة {processed} يوم")
    
    def generate_comparison(self):
        print("\n" + "="*100)
        print("📊 المقارنة النهائية")
        print("="*100)
        
        if len(self.trades_rule) == 0:
            print("\n❌ لا توجد صفقات")
            return
        
        # Rule-Based
        total_rule = len(self.trades_rule)
        wins_rule = sum(1 for t in self.trades_rule if t['result'] == 'WIN')
        wr_rule = wins_rule / total_rule * 100
        roi_rule = (self.capital_rule - self.initial_capital) / self.initial_capital * 100
        
        # ML-Enhanced
        total_ml = len(self.trades_ml)
        wins_ml = sum(1 for t in self.trades_ml if t['result'] == 'WIN')
        wr_ml = wins_ml / total_ml * 100
        roi_ml = (self.capital_ml - self.initial_capital) / self.initial_capital * 100
        
        print(f"\n{'المؤشر':<30} | {'Rule-Based':<20} | {'ML-Enhanced':<20} | {'الفرق':<20}")
        print("="*100)
        print(f"{'إجمالي الصفقات':<30} | {total_rule:<20} | {total_ml:<20} | {total_ml - total_rule:<20}")
        print(f"{'Win Rate':<30} | {wr_rule:<20.2f}% | {wr_ml:<20.2f}% | {wr_ml - wr_rule:<20.2f}%")
        print(f"{'ROI':<30} | {roi_rule:<20.2f}% | {roi_ml:<20.2f}% | {roi_ml - roi_rule:<20.2f}%")
        print(f"{'رأس المال النهائي':<30} | ${self.capital_rule:<19,.0f} | ${self.capital_ml:<19,.0f} | ${self.capital_ml - self.capital_rule:<19,.0f}")
        
        # رسم المقارنة
        self.plot_comparison()
    
    def plot_comparison(self):
        print("\n📊 إنشاء رسومات المقارنة...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Equity Curves
        ax1 = axes[0, 0]
        daily_rule = pd.DataFrame(self.daily_stats_rule)
        daily_ml = pd.DataFrame(self.daily_stats_ml)
        
        ax1.plot(range(len(daily_rule)), daily_rule['capital'], 'b-', linewidth=2, label='Rule-Based')
        ax1.plot(range(len(daily_ml)), daily_ml['capital'], 'r-', linewidth=2, label='ML-Enhanced')
        ax1.axhline(self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Day', fontsize=12)
        ax1.set_ylabel('Capital ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win Rate
        ax2 = axes[0, 1]
        trades_rule_df = pd.DataFrame(self.trades_rule)
        trades_ml_df = pd.DataFrame(self.trades_ml)
        
        wr_rule = (trades_rule_df['result'] == 'WIN').sum() / len(trades_rule_df) * 100
        wr_ml = (trades_ml_df['result'] == 'WIN').sum() / len(trades_ml_df) * 100
        
        ax2.bar(['Rule-Based', 'ML-Enhanced'], [wr_rule, wr_ml], color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_ylim([0, 100])
        for i, v in enumerate([wr_rule, wr_ml]):
            ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. ROI
        ax3 = axes[1, 0]
        roi_rule = (self.capital_rule - self.initial_capital) / self.initial_capital * 100
        roi_ml = (self.capital_ml - self.initial_capital) / self.initial_capital * 100
        
        ax3.bar(['Rule-Based', 'ML-Enhanced'], [roi_rule, roi_ml], color=['blue', 'red'], alpha=0.7)
        ax3.set_title('ROI Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('ROI (%)', fontsize=12)
        for i, v in enumerate([roi_rule, roi_ml]):
            ax3.text(i, v + 5, f'{v:.1f}%', ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Cumulative P&L
        ax4 = axes[1, 1]
        cumsum_rule = daily_rule['pnl'].cumsum()
        cumsum_ml = daily_ml['pnl'].cumsum()
        
        ax4.plot(range(len(cumsum_rule)), cumsum_rule, 'b-', linewidth=2, label='Rule-Based')
        ax4.plot(range(len(cumsum_ml)), cumsum_ml, 'r-', linewidth=2, label='ML-Enhanced')
        ax4.axhline(0, color='black', linewidth=1)
        ax4.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trading Day', fontsize=12)
        ax4.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_file = os.path.join(self.ml_folder, 'rule_vs_ml_comparison.png')
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ تم حفظ: rule_vs_ml_comparison.png")

if __name__ == "__main__":
    bt = MLEnhancedBacktest()
    bt.run_backtest()

