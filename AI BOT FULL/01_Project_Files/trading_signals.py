import databento as db
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ICTTradingSignals:
    """
    نظام إشارات التداول حسب استراتيجية ICT
    الهدف: 3 صفقات عالية الدقة يومياً
    """
    
    def __init__(self, ohlc_5m, ohlc_15m, order_blocks, fvgs, sweeps):
        self.ohlc_5m = ohlc_5m
        self.ohlc_15m = ohlc_15m
        self.order_blocks = order_blocks
        self.fvgs = fvgs
        self.sweeps = sweeps
        self.signals = []
    
    def is_killzone(self, timestamp):
        """
        تحديد ما إذا كان الوقت ضمن Killzone مهمة
        """
        hour = timestamp.hour
        
        # London Killzone: 02:00-05:00 UTC
        if 2 <= hour < 5:
            return 'London', 10  # أولوية عالية
        
        # NY AM Killzone: 07:00-10:00 UTC
        elif 7 <= hour < 10:
            return 'NY_AM', 10
        
        # NY PM Killzone: 13:00-17:00 UTC
        elif 13 <= hour < 17:
            return 'NY_PM', 9
        
        # London Close: 10:00-12:00 UTC
        elif 10 <= hour < 12:
            return 'London_Close', 8
        
        return None, 0
    
    def check_order_block_retest(self, current_time, current_price):
        """
        التحقق من إعادة اختبار Order Block
        """
        if len(self.order_blocks) == 0:
            return None
        
        # ابحث عن Order Blocks الحديثة (آخر 4 ساعات)
        recent_obs = self.order_blocks[
            self.order_blocks['time'] > (current_time - pd.Timedelta(hours=4))
        ]
        
        for idx, ob in recent_obs.iterrows():
            # Bullish OB: السعر يلامس المنطقة من الأعلى
            if ob['type'] == 'Bullish':
                if ob['low'] <= current_price <= ob['high']:
                    return {
                        'type': 'BUY',
                        'reason': 'Order Block Retest (Bullish)',
                        'entry': current_price,
                        'stop_loss': ob['low'] - 2.0,  # 2 نقطة تحت OB
                        'take_profit': current_price + (current_price - ob['low']) * 2,  # R:R 1:2
                        'strength': ob['strength']
                    }
            
            # Bearish OB: السعر يلامس المنطقة من الأسفل
            elif ob['type'] == 'Bearish':
                if ob['low'] <= current_price <= ob['high']:
                    return {
                        'type': 'SELL',
                        'reason': 'Order Block Retest (Bearish)',
                        'entry': current_price,
                        'stop_loss': ob['high'] + 2.0,
                        'take_profit': current_price - (ob['high'] - current_price) * 2,
                        'strength': ob['strength']
                    }
        
        return None
    
    def check_fvg_fill(self, current_time, current_price):
        """
        التحقق من ملء Fair Value Gap
        """
        if len(self.fvgs) == 0:
            return None
        
        # ابحث عن FVGs الحديثة (آخر 2 ساعة)
        recent_fvgs = self.fvgs[
            self.fvgs['time'] > (current_time - pd.Timedelta(hours=2))
        ]
        
        for idx, fvg in recent_fvgs.iterrows():
            # Bullish FVG: السعر يدخل الفجوة للأسفل
            if fvg['type'] == 'Bullish':
                if fvg['bottom'] <= current_price <= fvg['top']:
                    return {
                        'type': 'BUY',
                        'reason': 'Fair Value Gap Fill (Bullish)',
                        'entry': current_price,
                        'stop_loss': fvg['bottom'] - 2.0,
                        'take_profit': current_price + (fvg['size'] * 2),
                        'strength': fvg['size_pct']
                    }
            
            # Bearish FVG: السعر يدخل الفجوة للأعلى
            elif fvg['type'] == 'Bearish':
                if fvg['bottom'] <= current_price <= fvg['top']:
                    return {
                        'type': 'SELL',
                        'reason': 'Fair Value Gap Fill (Bearish)',
                        'entry': current_price,
                        'stop_loss': fvg['top'] + 2.0,
                        'take_profit': current_price - (fvg['size'] * 2),
                        'strength': fvg['size_pct']
                    }
        
        return None
    
    def check_liquidity_sweep_reversal(self, current_time):
        """
        التحقق من انعكاس بعد Liquidity Sweep
        """
        if len(self.sweeps) == 0:
            return None
        
        # ابحث عن Sweeps الحديثة جداً (آخر 15 دقيقة)
        recent_sweeps = self.sweeps[
            self.sweeps['time'] > (current_time - pd.Timedelta(minutes=15))
        ]
        
        if len(recent_sweeps) > 0:
            latest_sweep = recent_sweeps.iloc[-1]
            
            # Buy-side Sweep → توقع هبوط
            if latest_sweep['type'] == 'Buy-side Sweep':
                return {
                    'type': 'SELL',
                    'reason': 'Liquidity Sweep Reversal (Buy-side)',
                    'entry': latest_sweep['close'],
                    'stop_loss': latest_sweep['level'] + 3.0,
                    'take_profit': latest_sweep['close'] - (latest_sweep['level'] - latest_sweep['close']) * 1.5,
                    'strength': 8
                }
            
            # Sell-side Sweep → توقع صعود
            elif latest_sweep['type'] == 'Sell-side Sweep':
                return {
                    'type': 'BUY',
                    'reason': 'Liquidity Sweep Reversal (Sell-side)',
                    'entry': latest_sweep['close'],
                    'stop_loss': latest_sweep['level'] - 3.0,
                    'take_profit': latest_sweep['close'] + (latest_sweep['close'] - latest_sweep['level']) * 1.5,
                    'strength': 8
                }
        
        return None
    
    def check_delta_divergence(self, current_idx):
        """
        التحقق من تباعد Delta (Order Flow)
        """
        if current_idx < 10:
            return None
        
        recent_candles = self.ohlc_5m.iloc[current_idx-10:current_idx+1]
        
        # السعر يصعد لكن Delta سالب → ضعف صعودي
        if recent_candles['close'].iloc[-1] > recent_candles['close'].iloc[0]:
            if recent_candles['cumulative_delta'].iloc[-1] < recent_candles['cumulative_delta'].iloc[-5]:
                return {
                    'type': 'SELL',
                    'reason': 'Bearish Delta Divergence',
                    'entry': recent_candles['close'].iloc[-1],
                    'stop_loss': recent_candles['high'].iloc[-1] + 2.0,
                    'take_profit': recent_candles['close'].iloc[-1] - 10.0,
                    'strength': 7
                }
        
        # السعر يهبط لكن Delta موجب → ضعف هبوطي
        elif recent_candles['close'].iloc[-1] < recent_candles['close'].iloc[0]:
            if recent_candles['cumulative_delta'].iloc[-1] > recent_candles['cumulative_delta'].iloc[-5]:
                return {
                    'type': 'BUY',
                    'reason': 'Bullish Delta Divergence',
                    'entry': recent_candles['close'].iloc[-1],
                    'stop_loss': recent_candles['low'].iloc[-1] - 2.0,
                    'take_profit': recent_candles['close'].iloc[-1] + 10.0,
                    'strength': 7
                }
        
        return None
    
    def generate_signals(self):
        """
        توليد إشارات التداول
        """
        print("\n🎯 جاري توليد إشارات التداول...")
        
        daily_signals = []
        
        for idx in range(20, len(self.ohlc_5m)):
            candle = self.ohlc_5m.iloc[idx]
            current_time = candle.name
            current_price = candle['close']
            
            # 1. التحقق من Killzone
            session, priority = self.is_killzone(current_time)
            if priority < 8:  # فقط الجلسات المهمة
                continue
            
            # 2. البحث عن Setup
            signal = None
            
            # الأولوية الأولى: Liquidity Sweep Reversal
            signal = self.check_liquidity_sweep_reversal(current_time)
            
            # الأولوية الثانية: Order Block Retest
            if signal is None:
                signal = self.check_order_block_retest(current_time, current_price)
            
            # الأولوية الثالثة: FVG Fill
            if signal is None:
                signal = self.check_fvg_fill(current_time, current_price)
            
            # الأولوية الرابعة: Delta Divergence
            if signal is None:
                signal = self.check_delta_divergence(idx)
            
            # إضافة الإشارة
            if signal is not None:
                signal['time'] = current_time
                signal['session'] = session
                signal['priority'] = priority
                
                # حساب Risk/Reward
                risk = abs(signal['entry'] - signal['stop_loss'])
                reward = abs(signal['take_profit'] - signal['entry'])
                signal['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
                
                # فلترة: فقط R:R > 1.5
                if signal['risk_reward'] >= 1.5:
                    daily_signals.append(signal)
        
        # ترتيب حسب الأولوية والقوة
        signals_df = pd.DataFrame(daily_signals)
        if len(signals_df) > 0:
            signals_df = signals_df.sort_values(['priority', 'strength'], ascending=[False, False])
        
        self.signals = signals_df
        
        print(f"✅ تم توليد {len(signals_df)} إشارة تداول")
        
        return signals_df
    
    def get_top_3_signals(self):
        """
        الحصول على أفضل 3 صفقات لليوم
        """
        if len(self.signals) == 0:
            print("❌ لا توجد إشارات متاحة")
            return None
        
        # اختيار أفضل 3 (موزعة على الجلسات)
        top_signals = []
        sessions_used = set()
        
        for idx, signal in self.signals.iterrows():
            # تجنب تكرار الجلسة
            if signal['session'] not in sessions_used or len(top_signals) < 3:
                top_signals.append(signal)
                sessions_used.add(signal['session'])
            
            if len(top_signals) >= 3:
                break
        
        top_df = pd.DataFrame(top_signals)
        
        print("\n" + "="*100)
        print("🎯 أفضل 3 صفقات لليوم")
        print("="*100)
        
        for i, (idx, signal) in enumerate(top_df.iterrows(), 1):
            print(f"\n📍 صفقة #{i} - {signal['type']}")
            print(f"   الوقت: {signal['time']}")
            print(f"   الجلسة: {signal['session']}")
            print(f"   السبب: {signal['reason']}")
            print(f"   الدخول: {signal['entry']:.2f}")
            print(f"   Stop Loss: {signal['stop_loss']:.2f}")
            print(f"   Take Profit: {signal['take_profit']:.2f}")
            print(f"   Risk/Reward: 1:{signal['risk_reward']:.2f}")
            print(f"   القوة: {signal['strength']:.2f}")
        
        print("\n" + "="*100)
        
        return top_df

# تحميل البيانات المحللة
if __name__ == "__main__":
    print("📂 تحميل نتائج التحليل...")
    
    ohlc_5m = pd.read_csv('/home/claude/ohlc_5min.csv', index_col=0, parse_dates=True)
    ohlc_15m = pd.read_csv('/home/claude/ohlc_15min.csv', index_col=0, parse_dates=True)
    
    try:
        order_blocks = pd.read_csv('/home/claude/order_blocks.csv', parse_dates=['time'])
    except:
        order_blocks = pd.DataFrame()
    
    try:
        fvgs = pd.read_csv('/home/claude/fair_value_gaps.csv', parse_dates=['time'])
    except:
        fvgs = pd.DataFrame()
    
    try:
        sweeps = pd.read_csv('/home/claude/liquidity_sweeps.csv', parse_dates=['time'])
    except:
        sweeps = pd.DataFrame()
    
    # إنشاء نظام الإشارات
    signal_system = ICTTradingSignals(ohlc_5m, ohlc_15m, order_blocks, fvgs, sweeps)
    
    # توليد الإشارات
    all_signals = signal_system.generate_signals()
    
    # الحصول على أفضل 3 صفقات
    top_3 = signal_system.get_top_3_signals()
    
    # حفظ النتائج
    if top_3 is not None:
        top_3.to_csv('/home/claude/top_3_trades.csv', index=False)
        print("\n✅ تم حفظ أفضل 3 صفقات في top_3_trades.csv")
