"""
🎯 نظام ICT Backtest الشامل - سنة كاملة
يحلل جميع بيانات السنة دفعة واحدة
"""

import databento as db
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FullYearICTBacktest:
    def __init__(self, data_folder, initial_capital=50000):
        """
        data_folder: مجلد يحتوي على ملفات .dbn.zst
        initial_capital: رأس المال الابتدائي
        """
        self.data_folder = data_folder
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.all_trades = []
        self.daily_stats = []
        
        print("="*120)
        print("🎯 نظام ICT Backtest - السنة الكاملة")
        print("="*120)
        print(f"📂 مجلد البيانات: {data_folder}")
        print(f"💰 رأس المال الابتدائي: ${initial_capital:,.0f}")
        
        # البحث عن جميع الملفات
        self.dbn_files = sorted(glob.glob(os.path.join(data_folder, "*.dbn.zst")))
        print(f"✅ وجدنا {len(self.dbn_files)} ملف")
        
        if len(self.dbn_files) == 0:
            raise ValueError("❌ لم يتم العثور على ملفات .dbn.zst في المجلد المحدد")
    
    def load_daily_data(self, file_path, symbol='ESH5'):
        """تحميل بيانات يوم واحد"""
        try:
            store = db.DBNStore.from_file(file_path)
            df = store.to_df()
            
            # تجربة رموز متعددة
            symbols_to_try = [symbol, 'ESZ5', 'ESM5', 'ESU5']
            for sym in symbols_to_try:
                if sym in df['symbol'].values:
                    df = df[df['symbol'] == sym].copy()
                    break
            
            if len(df) == 0:
                # جرب أي رمز ES
                es_symbols = [s for s in df['symbol'].unique() if s.startswith('ES')]
                if es_symbols:
                    df = df[df['symbol'] == es_symbols[0]].copy()
            
            return df
        except Exception as e:
            print(f"      ⚠️  خطأ في قراءة الملف: {e}")
            return None
    
    def create_ohlc(self, df):
        """إنشاء OHLC من tick data"""
        if df is None or len(df) == 0:
            return None
        
        df['delta'] = np.where(df['side'] == 'A', df['size'], -df['size'])
        
        ohlc = df.groupby(pd.Grouper(freq='5T')).agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum',
            'delta': 'sum'
        })
        
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume', 'delta']
        ohlc = ohlc.dropna()
        
        return ohlc
    
    def find_order_blocks(self, ohlc, min_strength=8):
        """اكتشاف Order Blocks"""
        obs = []
        lookback = 20
        
        for i in range(lookback, len(ohlc)):
            # Bullish OB
            if ohlc['close'].iloc[i] > ohlc['close'].iloc[i-1]:
                for j in range(i-1, max(0, i-lookback), -1):
                    if ohlc['close'].iloc[j] < ohlc['open'].iloc[j]:
                        move = ohlc['high'].iloc[i] - ohlc['low'].iloc[j]
                        if move >= min_strength:
                            obs.append({
                                'time': ohlc.index[j],
                                'type': 'Bullish',
                                'high': ohlc['high'].iloc[j],
                                'low': ohlc['low'].iloc[j],
                                'strength': move
                            })
                        break
            
            # Bearish OB
            elif ohlc['close'].iloc[i] < ohlc['close'].iloc[i-1]:
                for j in range(i-1, max(0, i-lookback), -1):
                    if ohlc['close'].iloc[j] > ohlc['open'].iloc[j]:
                        move = ohlc['high'].iloc[j] - ohlc['low'].iloc[i]
                        if move >= min_strength:
                            obs.append({
                                'time': ohlc.index[j],
                                'type': 'Bearish',
                                'high': ohlc['high'].iloc[j],
                                'low': ohlc['low'].iloc[j],
                                'strength': move
                            })
                        break
        
        return pd.DataFrame(obs) if obs else pd.DataFrame()
    
    def find_swing_points(self, ohlc, period=3):
        """إيجاد نقاط التأرجح (Swing Points)"""
        swings = []
        
        for i in range(period, len(ohlc) - period):
            # Swing High
            if ohlc['high'].iloc[i] == ohlc['high'].iloc[i-period:i+period+1].max():
                swings.append({
                    'time': ohlc.index[i],
                    'type': 'High',
                    'price': ohlc['high'].iloc[i]
                })
            
            # Swing Low
            if ohlc['low'].iloc[i] == ohlc['low'].iloc[i-period:i+period+1].min():
                swings.append({
                    'time': ohlc.index[i],
                    'type': 'Low',
                    'price': ohlc['low'].iloc[i]
                })
        
        return pd.DataFrame(swings).sort_values('time') if swings else pd.DataFrame()
    
    def find_fvgs(self, ohlc, min_gap=2.0):
        """إيجاد Fair Value Gaps"""
        fvgs = []
        
        for i in range(2, len(ohlc)):
            # Bullish FVG
            if ohlc['low'].iloc[i] > ohlc['high'].iloc[i-2]:
                gap = ohlc['low'].iloc[i] - ohlc['high'].iloc[i-2]
                if gap >= min_gap:
                    fvgs.append({
                        'time': ohlc.index[i],
                        'type': 'Bullish',
                        'top': ohlc['low'].iloc[i],
                        'bottom': ohlc['high'].iloc[i-2],
                        'size': gap
                    })
            
            # Bearish FVG
            elif ohlc['high'].iloc[i] < ohlc['low'].iloc[i-2]:
                gap = ohlc['low'].iloc[i-2] - ohlc['high'].iloc[i]
                if gap >= min_gap:
                    fvgs.append({
                        'time': ohlc.index[i],
                        'type': 'Bearish',
                        'top': ohlc['low'].iloc[i-2],
                        'bottom': ohlc['high'].iloc[i],
                        'size': gap
                    })
        
        return pd.DataFrame(fvgs) if fvgs else pd.DataFrame()
    
    def get_killzone(self, hour):
        """تحديد جلسة التداول"""
        # UTC times
        if 2 <= hour < 5:
            return 'London', 10
        elif 7 <= hour < 10:
            return 'NY_AM', 10
        elif 13 <= hour < 17:
            return 'NY_PM', 9
        return None, 0
    
    def find_smart_target(self, entry, direction, swings, obs, fvgs, current_time):
        """إيجاد هدف ذكي بناءً على هيكل السوق"""
        targets = []
        min_target = 10
        max_target = 100
        
        # Swing Points
        if len(swings) > 0:
            future_swings = swings[swings['time'] > current_time]
            
            if direction == 'BUY':
                highs = future_swings[future_swings['type'] == 'High']
                if len(highs) > 0:
                    for idx, (_, swing) in enumerate(highs.head(3).iterrows()):
                        distance = swing['price'] - entry
                        if min_target <= distance <= max_target:
                            targets.append({
                                'type': f'Swing High #{idx+1}',
                                'price': swing['price'],
                                'distance': distance,
                                'priority': 10 - idx
                            })
            else:
                lows = future_swings[future_swings['type'] == 'Low']
                if len(lows) > 0:
                    for idx, (_, swing) in enumerate(lows.head(3).iterrows()):
                        distance = entry - swing['price']
                        if min_target <= distance <= max_target:
                            targets.append({
                                'type': f'Swing Low #{idx+1}',
                                'price': swing['price'],
                                'distance': distance,
                                'priority': 10 - idx
                            })
        
        # Order Blocks
        if len(obs) > 0:
            future_obs = obs[obs['time'] > current_time]
            
            if direction == 'BUY':
                bearish_obs = future_obs[future_obs['type'] == 'Bearish']
                if len(bearish_obs) > 0:
                    ob = bearish_obs.iloc[0]
                    distance = ob['low'] - entry
                    if min_target <= distance <= max_target:
                        targets.append({
                            'type': 'Opposing OB',
                            'price': ob['low'],
                            'distance': distance,
                            'priority': 7
                        })
            else:
                bullish_obs = future_obs[future_obs['type'] == 'Bullish']
                if len(bullish_obs) > 0:
                    ob = bullish_obs.iloc[0]
                    distance = entry - ob['high']
                    if min_target <= distance <= max_target:
                        targets.append({
                            'type': 'Opposing OB',
                            'price': ob['high'],
                            'distance': distance,
                            'priority': 7
                        })
        
        # FVGs
        if len(fvgs) > 0:
            future_fvgs = fvgs[fvgs['time'] > current_time]
            
            if direction == 'BUY':
                bearish_fvgs = future_fvgs[future_fvgs['type'] == 'Bearish']
                if len(bearish_fvgs) > 0:
                    fvg = bearish_fvgs.iloc[0]
                    distance = fvg['bottom'] - entry
                    if min_target <= distance <= max_target:
                        targets.append({
                            'type': 'FVG Fill',
                            'price': fvg['bottom'],
                            'distance': distance,
                            'priority': 5
                        })
            else:
                bullish_fvgs = future_fvgs[future_fvgs['type'] == 'Bullish']
                if len(bullish_fvgs) > 0:
                    fvg = bullish_fvgs.iloc[0]
                    distance = entry - fvg['top']
                    if min_target <= distance <= max_target:
                        targets.append({
                            'type': 'FVG Fill',
                            'price': fvg['top'],
                            'distance': distance,
                            'priority': 5
                        })
        
        # Default target
        if len(targets) == 0:
            default_dist = 15
            if direction == 'BUY':
                return entry + default_dist, default_dist, 'Fixed 15pts'
            else:
                return entry - default_dist, default_dist, 'Fixed 15pts'
        
        # اختر أفضل هدف
        targets_df = pd.DataFrame(targets)
        targets_df = targets_df.sort_values(['priority', 'distance'], ascending=[False, True])
        best = targets_df.iloc[0]
        
        return best['price'], best['distance'], best['type']
    
    def find_daily_setups(self, ohlc, obs, swings, fvgs, date):
        """البحث عن Setups في يوم واحد"""
        setups = []
        used_obs = set()
        
        for _, ob in obs.iterrows():
            session, priority = self.get_killzone(ob['time'].hour)
            if priority < 9:
                continue
            
            # تتبع OB المستخدمة
            ob_key = f"{ob['time']}_{ob['type']}_{ob['low']:.2f}_{ob['high']:.2f}"
            if ob_key in used_obs:
                continue
            
            # ابحث عن Retest
            future = ohlc[(ohlc.index > ob['time']) & 
                         (ohlc.index <= ob['time'] + pd.Timedelta(hours=2))]
            
            found_setup = False
            
            for t, c in future.iterrows():
                if found_setup:
                    break
                
                entry, sl, direction = None, None, None
                
                # Bullish Setup
                if ob['type'] == 'Bullish' and c['low'] <= ob['high'] and c['close'] > ob['low']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['low'] - 2.0
                    direction = 'BUY'
                
                # Bearish Setup
                elif ob['type'] == 'Bearish' and c['high'] >= ob['low'] and c['close'] < ob['high']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['high'] + 2.0
                    direction = 'SELL'
                
                if entry:
                    # إيجاد هدف ذكي
                    tp, target_distance, target_type = self.find_smart_target(
                        entry, direction, swings, obs, fvgs, t
                    )
                    
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    
                    # فلترة
                    if risk <= 4.5 and reward >= 8.0 and reward/risk >= 2.0:
                        setups.append({
                            'date': date,
                            'time': t,
                            'session': session,
                            'type': direction,
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'risk': risk,
                            'target': reward,
                            'rr': reward/risk,
                            'target_type': target_type,
                            'strength': ob['strength'],
                            'priority': priority
                        })
                        
                        used_obs.add(ob_key)
                        found_setup = True
        
        return setups
    
    def select_top_3(self, setups):
        """اختيار أفضل 3 صفقات"""
        if len(setups) == 0:
            return []
        
        df = pd.DataFrame(setups)
        df = df.sort_values(['priority', 'target', 'rr'], ascending=[False, False, False])
        
        # اختر من جلسات مختلفة
        top_3 = []
        sessions_used = set()
        
        for _, s in df.iterrows():
            if len(top_3) >= 3:
                break
            if s['session'] not in sessions_used:
                top_3.append(s)
                sessions_used.add(s['session'])
        
        # أكمل لـ 3
        for _, s in df.iterrows():
            if len(top_3) >= 3:
                break
            already_added = any(x['time'] == s['time'] for x in top_3)
            if not already_added:
                top_3.append(s)
        
        return top_3
    
    def simulate_trade(self, trade, ohlc):
        """محاكاة نتيجة الصفقة"""
        future = ohlc[ohlc.index > trade['time']]
        
        if len(future) == 0:
            return None
        
        for t, candle in future.iterrows():
            if trade['type'] == 'BUY':
                # تحقق من SL
                if candle['low'] <= trade['sl']:
                    return {
                        'result': 'LOSS',
                        'exit_price': trade['sl'],
                        'exit_time': t,
                        'pnl_points': -(trade['risk']),
                        'duration_min': (t - trade['time']).total_seconds() / 60
                    }
                # تحقق من TP
                if candle['high'] >= trade['tp']:
                    return {
                        'result': 'WIN',
                        'exit_price': trade['tp'],
                        'exit_time': t,
                        'pnl_points': trade['target'],
                        'duration_min': (t - trade['time']).total_seconds() / 60
                    }
            else:  # SELL
                if candle['high'] >= trade['sl']:
                    return {
                        'result': 'LOSS',
                        'exit_price': trade['sl'],
                        'exit_time': t,
                        'pnl_points': -(trade['risk']),
                        'duration_min': (t - trade['time']).total_seconds() / 60
                    }
                if candle['low'] <= trade['tp']:
                    return {
                        'result': 'WIN',
                        'exit_price': trade['tp'],
                        'exit_time': t,
                        'pnl_points': trade['target'],
                        'duration_min': (t - trade['time']).total_seconds() / 60
                    }
        
        return None
    
    def calculate_position_size(self, risk_points, risk_pct=0.01):
        """حساب حجم المركز (1% risk)"""
        risk_amount = self.capital * risk_pct
        contracts = int(risk_amount / (risk_points * 50))
        return max(1, contracts)
    
    def process_single_day(self, file_path, file_num, total_files):
        """معالجة يوم واحد"""
        filename = os.path.basename(file_path)
        date_str = filename.split('.')[0].split('-')[-1]  # extract date
        
        print(f"\n[{file_num}/{total_files}] 📅 {date_str}... ", end='')
        
        # تحميل البيانات
        df = self.load_daily_data(file_path)
        if df is None or len(df) < 1000:
            print("⏭️  بيانات غير كافية")
            return
        
        # إنشاء OHLC
        ohlc = self.create_ohlc(df)
        if ohlc is None or len(ohlc) < 50:
            print("⏭️  OHLC غير كافي")
            return
        
        # التحليل
        obs = self.find_order_blocks(ohlc)
        swings = self.find_swing_points(ohlc)
        fvgs = self.find_fvgs(ohlc)
        
        # البحث عن Setups
        setups = self.find_daily_setups(ohlc, obs, swings, fvgs, date_str)
        
        if len(setups) == 0:
            print("⏭️  لا توجد setups")
            self.daily_stats.append({
                'date': date_str,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl_usd': 0,
                'capital': self.capital
            })
            return
        
        # اختيار أفضل 3
        top_3 = self.select_top_3(setups)
        
        # محاكاة الصفقات
        daily_pnl = 0
        wins = 0
        losses = 0
        
        for trade in top_3:
            result = self.simulate_trade(trade, ohlc)
            
            if result:
                contracts = self.calculate_position_size(trade['risk'])
                pnl_usd = result['pnl_points'] * contracts * 50
                
                self.capital += pnl_usd
                daily_pnl += pnl_usd
                
                if result['result'] == 'WIN':
                    wins += 1
                else:
                    losses += 1
                
                # حفظ الصفقة
                trade_record = {**trade, **result}
                trade_record['contracts'] = contracts
                trade_record['pnl_usd'] = pnl_usd
                trade_record['capital_after'] = self.capital
                self.all_trades.append(trade_record)
        
        # إحصائيات اليوم
        self.daily_stats.append({
            'date': date_str,
            'trades': len(top_3),
            'wins': wins,
            'losses': losses,
            'pnl_usd': daily_pnl,
            'capital': self.capital
        })
        
        print(f"✅ {len(setups)} setups → {len(top_3)} trades | W:{wins} L:{losses} | ${daily_pnl:+,.0f} | Balance: ${self.capital:,.0f}")
    
    def run_full_backtest(self, max_days=None):
        """تشغيل backtest على جميع الأيام"""
        print("\n" + "="*120)
        print("🚀 بدء Backtest الشامل...")
        print("="*120)
        
        files_to_process = self.dbn_files[:max_days] if max_days else self.dbn_files
        total_files = len(files_to_process)
        
        for i, file_path in enumerate(files_to_process, 1):
            try:
                self.process_single_day(file_path, i, total_files)
            except Exception as e:
                print(f"❌ خطأ: {e}")
                continue
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """إنشاء تقرير شامل"""
        print("\n" + "="*120)
        print("📊 النتائج النهائية - السنة الكاملة")
        print("="*120)
        
        if len(self.all_trades) == 0:
            print("❌ لا توجد صفقات")
            return
        
        trades_df = pd.DataFrame(self.all_trades)
        daily_df = pd.DataFrame(self.daily_stats)
        
        # إحصائيات عامة
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['result'] == 'WIN'])
        losses = len(trades_df[trades_df['result'] == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl_usd'].sum()
        final_capital = self.capital
        roi = ((final_capital - self.initial_capital) / self.initial_capital * 100)
        
        avg_win = trades_df[trades_df['result'] == 'WIN']['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['result'] == 'LOSS']['pnl_usd'].mean() if losses > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum() / 
                           trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum()) if losses > 0 else float('inf')
        
        avg_rr = trades_df['rr'].mean()
        avg_target = trades_df['target'].mean()
        avg_duration = trades_df['duration_min'].mean()
        
        # Max Drawdown
        daily_df['cumulative'] = daily_df['capital']
        daily_df['peak'] = daily_df['cumulative'].expanding().max()
        daily_df['drawdown'] = (daily_df['cumulative'] - daily_df['peak']) / daily_df['peak'] * 100
        max_drawdown = daily_df['drawdown'].min()
        
        # Trading Days
        trading_days = len(daily_df[daily_df['trades'] > 0])
        
        print(f"\n📈 ملخص الأداء:")
        print(f"   {'='*100}")
        print(f"   عدد أيام التداول:      {trading_days}")
        print(f"   إجمالي الصفقات:        {total_trades}")
        print(f"   الصفقات الرابحة:       {wins} ({win_rate:.1f}%)")
        print(f"   الصفقات الخاسرة:       {losses} ({100-win_rate:.1f}%)")
        
        print(f"\n💰 النتائج المالية:")
        print(f"   {'='*100}")
        print(f"   رأس المال الابتدائي:   ${self.initial_capital:,.0f}")
        print(f"   رأس المال النهائي:     ${final_capital:,.0f}")
        print(f"   الربح/الخسارة الصافي:   ${total_pnl:+,.0f}")
        print(f"   العائد (ROI):           {roi:+.2f}%")
        print(f"   Max Drawdown:           {max_drawdown:.2f}%")
        print(f"   Profit Factor:          {profit_factor:.2f}")
        
        print(f"\n📊 متوسطات:")
        print(f"   {'='*100}")
        print(f"   متوسط الربح:           ${avg_win:+,.0f}")
        print(f"   متوسط الخسارة:         ${avg_loss:,.0f}")
        print(f"   نسبة الربح/الخسارة:    {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   N/A")
        print(f"   متوسط R:R:             1:{avg_rr:.2f}")
        print(f"   متوسط الهدف:           {avg_target:.1f} نقاط")
        print(f"   متوسط مدة الصفقة:      {avg_duration:.0f} دقيقة")
        
        # تحليل الجلسات
        print(f"\n🎪 تحليل الجلسات:")
        print(f"   {'='*100}")
        for session in ['London', 'NY_AM', 'NY_PM']:
            session_trades = trades_df[trades_df['session'] == session]
            if len(session_trades) > 0:
                s_wins = len(session_trades[session_trades['result'] == 'WIN'])
                s_total = len(session_trades)
                s_winrate = (s_wins / s_total * 100)
                s_pnl = session_trades['pnl_usd'].sum()
                print(f"   {session:12} → {s_total:4} صفقات | WR: {s_winrate:5.1f}% | P&L: ${s_pnl:+12,.0f}")
        
        # أفضل شهر
        trades_df['month'] = pd.to_datetime(trades_df['date'], format='%Y%m%d').dt.to_period('M')
        monthly = trades_df.groupby('month')['pnl_usd'].sum().sort_values(ascending=False)
        
        print(f"\n📅 أفضل 5 شهور:")
        print(f"   {'='*100}")
        for month, pnl in monthly.head(5).items():
            print(f"   {month} → ${pnl:+,.0f}")
        
        print(f"\n📅 أسوأ 5 شهور:")
        print(f"   {'='*100}")
        for month, pnl in monthly.tail(5).items():
            print(f"   {month} → ${pnl:+,.0f}")
        
        # حفظ النتائج
        output_folder = os.path.join(os.path.dirname(self.data_folder), 'ICT_Results')
        os.makedirs(output_folder, exist_ok=True)
        
        trades_output = os.path.join(output_folder, 'full_year_trades.csv')
        daily_output = os.path.join(output_folder, 'full_year_daily.csv')
        summary_output = os.path.join(output_folder, 'full_year_summary.txt')
        
        trades_df.to_csv(trades_output, index=False)
        daily_df.to_csv(daily_output, index=False)
        
        # حفظ ملخص نصي
        with open(summary_output, 'w', encoding='utf-8') as f:
            f.write(f"نتائج ICT Backtest - السنة الكاملة\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"إجمالي الصفقات: {total_trades}\n")
            f.write(f"معدل النجاح: {win_rate:.1f}%\n")
            f.write(f"العائد: {roi:+.2f}%\n")
            f.write(f"رأس المال النهائي: ${final_capital:,.0f}\n")
            f.write(f"الربح الصافي: ${total_pnl:+,.0f}\n")
            f.write(f"Max Drawdown: {max_drawdown:.2f}%\n")
            f.write(f"Profit Factor: {profit_factor:.2f}\n")
        
        print(f"\n✅ تم حفظ النتائج في:")
        print(f"   📁 {output_folder}")
        print(f"   📄 full_year_trades.csv ({len(trades_df)} صفقة)")
        print(f"   📄 full_year_daily.csv ({len(daily_df)} يوم)")
        print(f"   📄 full_year_summary.txt")
        print("="*120)

# ===========================
# التشغيل
# ===========================
if __name__ == "__main__":
    # المسار إلى مجلد البيانات
    data_folder = r"C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N"
    
    # إنشاء Backtester
    backtester = FullYearICTBacktest(
        data_folder=data_folder,
        initial_capital=50000
    )
    
    # تشغيل Backtest
    # استخدم max_days للاختبار السريع، أو اتركه None للسنة كاملة
    backtester.run_full_backtest(max_days=None)  # None = جميع الأيام
    
    print("\n🎉 اكتمل Backtest الشامل!")

