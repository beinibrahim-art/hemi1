import databento as db
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

"""
🎯 نظام Backtesting الكامل
يختبر استراتيجية ICT على سنة كاملة من البيانات
"""

class ICTBacktester:
    def __init__(self, file_path, initial_capital=50000):
        """
        file_path: ملف DBN
        initial_capital: رأس المال الابتدائي ($50,000 مثلاً)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades_history = []
        self.daily_stats = []
        
        print("📂 جاري تحميل البيانات...")
        print(f"   رأس المال الابتدائي: ${self.initial_capital:,.0f}")
        
        try:
            store = db.DBNStore.from_file(file_path)
            self.df = store.to_df()
            self.df = self.df[self.df['symbol'] == 'ESH5'].copy()
            
            print(f"✅ تم تحميل {len(self.df):,} صفقة")
            print(f"   من: {self.df.index[0]}")
            print(f"   إلى: {self.df.index[-1]}")
            
            # معلومات الوقت
            self.df['date'] = self.df['ts_event'].dt.date
            self.df['hour'] = self.df['ts_event'].dt.hour
            
            # Delta
            self.df['delta'] = np.where(self.df['side'] == 'A', 
                                        self.df['size'], 
                                        -self.df['size'])
            
            # الأيام الفريدة
            self.unique_days = sorted(self.df['date'].unique())
            print(f"   عدد أيام التداول: {len(self.unique_days)}")
            
        except Exception as e:
            print(f"❌ خطأ في تحميل الملف: {e}")
            raise
    
    def create_daily_ohlc(self, date):
        """إنشاء OHLC ليوم محدد"""
        day_data = self.df[self.df['date'] == date]
        
        if len(day_data) == 0:
            return None
        
        ohlc = day_data.groupby(pd.Grouper(key='ts_event', freq='5T')).agg({
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
    
    def get_killzone(self, hour):
        """تحديد الجلسة"""
        if 2 <= hour < 5: return 'London', 10
        elif 7 <= hour < 10: return 'NY_AM', 10
        elif 13 <= hour < 17: return 'NY_PM', 9
        return None, 0
    
    def find_daily_setups(self, date, ohlc, order_blocks):
        """البحث عن Setups في يوم محدد"""
        setups = []
        
        if len(order_blocks) == 0:
            return setups
        
        for _, ob in order_blocks.iterrows():
            session, priority = self.get_killzone(ob['time'].hour)
            if priority < 9:
                continue
            
            # ابحث عن Retest
            future = ohlc[(ohlc.index > ob['time']) & 
                         (ohlc.index <= ob['time'] + pd.Timedelta(hours=1))]
            
            for t, c in future.iterrows():
                entry, sl, tp = None, None, None
                
                # Bullish Setup
                if ob['type'] == 'Bullish' and c['low'] <= ob['high'] and c['close'] > ob['low']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['low'] - 2.0
                    tp = entry + 8.0
                
                # Bearish Setup
                elif ob['type'] == 'Bearish' and c['high'] >= ob['low'] and c['close'] < ob['high']:
                    entry = (ob['high'] + ob['low']) / 2
                    sl = ob['high'] + 2.0
                    tp = entry - 8.0
                
                if entry:
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    
                    # الفلاتر
                    if risk <= 4.0 and reward >= 7.0 and reward/risk >= 2.0:
                        setups.append({
                            'date': date,
                            'time': t,
                            'session': session,
                            'type': 'BUY' if tp > entry else 'SELL',
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'risk': risk,
                            'target': reward,
                            'rr': reward/risk,
                            'strength': ob['strength']
                        })
        
        return setups
    
    def select_top_3(self, setups):
        """اختيار أفضل 3 صفقات من اليوم"""
        if len(setups) == 0:
            return []
        
        df = pd.DataFrame(setups)
        df = df.sort_values(['strength', 'rr'], ascending=[False, False])
        
        # اختر 3 فقط
        selected = []
        sessions_used = set()
        
        for _, s in df.iterrows():
            if len(selected) < 3:
                if s['session'] not in sessions_used or len(selected) == 0:
                    selected.append(s)
                    sessions_used.add(s['session'])
        
        # أكمل لـ 3
        if len(selected) < 3:
            for _, s in df.iterrows():
                if len(selected) >= 3:
                    break
                if not any(x['time'] == s['time'] for x in selected):
                    selected.append(s)
        
        return selected
    
    def simulate_trade(self, trade, ohlc):
        """محاكاة الصفقة"""
        # ابحث عن الشموع بعد الدخول
        future = ohlc[ohlc.index > trade['time']]
        
        if len(future) == 0:
            return None  # لا توجد بيانات
        
        for t, candle in future.iterrows():
            # تحقق من SL
            if trade['type'] == 'BUY':
                if candle['low'] <= trade['sl']:
                    return {
                        'result': 'LOSS',
                        'exit_price': trade['sl'],
                        'exit_time': t,
                        'pnl_points': -(trade['risk']),
                        'duration': (t - trade['time']).total_seconds() / 60
                    }
                # تحقق من TP
                if candle['high'] >= trade['tp']:
                    return {
                        'result': 'WIN',
                        'exit_price': trade['tp'],
                        'exit_time': t,
                        'pnl_points': trade['target'],
                        'duration': (t - trade['time']).total_seconds() / 60
                    }
            
            else:  # SELL
                if candle['high'] >= trade['sl']:
                    return {
                        'result': 'LOSS',
                        'exit_price': trade['sl'],
                        'exit_time': t,
                        'pnl_points': -(trade['risk']),
                        'duration': (t - trade['time']).total_seconds() / 60
                    }
                if candle['low'] <= trade['tp']:
                    return {
                        'result': 'WIN',
                        'exit_price': trade['tp'],
                        'exit_time': t,
                        'pnl_points': trade['target'],
                        'duration': (t - trade['time']).total_seconds() / 60
                    }
        
        # لم يصل لـ SL ولا TP (نادر)
        return None
    
    def calculate_position_size(self, risk_points, risk_pct=0.01):
        """حساب حجم المركز (1% risk)"""
        risk_amount = self.capital * risk_pct
        contracts = int(risk_amount / (risk_points * 50))
        return max(1, contracts)  # على الأقل عقد واحد
    
    def run_backtest(self, max_days=None):
        """تشغيل Backtest كامل"""
        print("\n" + "="*100)
        print("🚀 بدء Backtesting...")
        print("="*100)
        
        days_to_test = self.unique_days[:max_days] if max_days else self.unique_days
        
        for day_num, date in enumerate(days_to_test, 1):
            print(f"\n[{day_num}/{len(days_to_test)}] يوم {date}...")
            
            # إنشاء OHLC
            ohlc = self.create_daily_ohlc(date)
            if ohlc is None or len(ohlc) < 50:
                print("   ⏭️  بيانات غير كافية")
                continue
            
            # اكتشاف OBs
            obs = self.find_order_blocks(ohlc)
            
            # إيجاد Setups
            setups = self.find_daily_setups(date, ohlc, obs)
            
            if len(setups) == 0:
                print("   ⏭️  لا توجد Setups")
                self.daily_stats.append({
                    'date': date,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0,
                    'capital': self.capital
                })
                continue
            
            # اختيار أفضل 3
            top_3 = self.select_top_3(setups)
            
            print(f"   ✅ وجدنا {len(setups)} Setup، اخترنا {len(top_3)}")
            
            # محاكاة الصفقات
            daily_pnl = 0
            wins = 0
            losses = 0
            
            for trade in top_3:
                result = self.simulate_trade(trade, ohlc)
                
                if result:
                    # حساب حجم المركز
                    contracts = self.calculate_position_size(trade['risk'])
                    pnl_usd = result['pnl_points'] * contracts * 50
                    
                    self.capital += pnl_usd
                    daily_pnl += pnl_usd
                    
                    if result['result'] == 'WIN':
                        wins += 1
                        print(f"      ✅ WIN: +{result['pnl_points']:.1f} نقاط = ${pnl_usd:,.0f}")
                    else:
                        losses += 1
                        print(f"      ❌ LOSS: {result['pnl_points']:.1f} نقاط = ${pnl_usd:,.0f}")
                    
                    # حفظ الصفقة
                    trade_record = {**trade, **result}
                    trade_record['contracts'] = contracts
                    trade_record['pnl_usd'] = pnl_usd
                    trade_record['capital_after'] = self.capital
                    self.trades_history.append(trade_record)
            
            # إحصائيات اليوم
            self.daily_stats.append({
                'date': date,
                'trades': len(top_3),
                'wins': wins,
                'losses': losses,
                'pnl': daily_pnl,
                'capital': self.capital
            })
            
            print(f"   💰 P&L اليوم: ${daily_pnl:+,.0f} | الرصيد: ${self.capital:,.0f}")
        
        self.generate_report()
    
    def generate_report(self):
        """إنشاء تقرير شامل"""
        print("\n" + "="*100)
        print("📊 نتائج Backtesting")
        print("="*100)
        
        if len(self.trades_history) == 0:
            print("❌ لا توجد صفقات")
            return
        
        trades_df = pd.DataFrame(self.trades_history)
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
        
        print(f"\n📈 الملخص العام:")
        print(f"   عدد الأيام:        {len(daily_df)}")
        print(f"   عدد الصفقات:      {total_trades}")
        print(f"   الصفقات الرابحة:  {wins} ({win_rate:.1f}%)")
        print(f"   الصفقات الخاسرة:  {losses} ({100-win_rate:.1f}%)")
        print(f"\n💰 النتائج المالية:")
        print(f"   رأس المال الابتدائي:  ${self.initial_capital:,.0f}")
        print(f"   رأس المال النهائي:    ${final_capital:,.0f}")
        print(f"   الربح/الخسارة:         ${total_pnl:+,.0f}")
        print(f"   العائد (ROI):          {roi:+.2f}%")
        print(f"\n📊 متوسطات:")
        print(f"   متوسط الربح:      ${avg_win:+,.0f}")
        print(f"   متوسط الخسارة:    ${avg_loss:,.0f}")
        print(f"   نسبة الربح/الخسارة: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   نسبة الربح/الخسارة: N/A")
        
        # حفظ النتائج
        trades_df.to_csv('/mnt/user-data/outputs/backtest_trades.csv', index=False)
        daily_df.to_csv('/mnt/user-data/outputs/backtest_daily.csv', index=False)
        
        print(f"\n✅ تم حفظ النتائج:")
        print(f"   - backtest_trades.csv (كل الصفقات)")
        print(f"   - backtest_daily.csv (إحصائيات يومية)")
        print("="*100)

# ===========================
# التشغيل
# ===========================
if __name__ == "__main__":
    print("🎯 نظام Backtesting - ICT Strategy")
    print("="*100)
    
    # المسار (عدّله حسب ملفك)
    file_path = '/mnt/user-data/uploads/glbx-mdp3-20250306_trades_dbn.zst'
    
    # إنشاء Backtester
    backtester = ICTBacktester(file_path, initial_capital=50000)
    
    # تشغيل (حدد max_days للاختبار السريع، أو اتركه None للسنة كاملة)
    backtester.run_backtest(max_days=10)  # ابدأ بـ 10 أيام للاختبار
    
    print("\n🎉 اكتمل Backtesting!")
