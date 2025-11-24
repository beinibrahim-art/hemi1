"""
🎯 Live Trading Integration - استخدام ML في التداول الحي
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

class LiveTradingML:
    """
    نظام التداول الحي مع ML
    
    ملاحظة: هذا كود توضيحي
    يحتاج integration مع منصة التداول (NinjaTrader, TradingView, etc)
    """
    
    def __init__(self, model_path=None):
        print("="*80)
        print("🎯 Live Trading System - ML Enhanced")
        print("="*80)
        
        # تحميل ML Model
        print(f"\n📥 تحميل ML Model...")
        
        # إذا لم يُحدد المسار، استخدم المسار الافتراضي
        if model_path is None:
            base_dir = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models'
            model_path = os.path.join(base_dir, 'XGBoost_ForwardTested_model.pkl')
            
            # إذا لم يوجد، استخدم Balanced
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'XGBoost_Balanced_model.pkl')
            
            # إذا لم يوجد، استخدم الأصلي
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'XGBoost_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✅ تم تحميل: {os.path.basename(model_path)}")
        
        # إعدادات
        self.min_probability = 0.70  # الحد الأدنى للثقة
        self.max_daily_trades = 3    # أقصى عدد صفقات يومياً
        self.max_daily_loss = 1000   # أقصى خسارة يومية ($)
        
        # متتبعات
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.active_trade = None
        
        print(f"\n⚙️  الإعدادات:")
        print(f"   Min Probability: {self.min_probability*100:.0f}%")
        print(f"   Max Daily Trades: {self.max_daily_trades}")
        print(f"   Max Daily Loss: ${self.max_daily_loss}")
    
    def is_killzone(self, current_time):
        """
        تحقق من Killzone الحالي
        current_time: datetime object (UTC)
        """
        hour = current_time.hour
        
        # London: 07:00-10:00 UTC
        if 7 <= hour < 10:
            return 'London', 10
        
        # NY AM: 13:00-16:00 UTC
        elif 13 <= hour < 16:
            return 'NY_AM', 10
        
        # NY PM: 18:00-21:00 UTC
        elif 18 <= hour < 21:
            return 'NY_PM', 9
        
        return None, 0
    
    def extract_setup_features(self, setup):
        """
        استخراج features من Setup
        
        setup = {
            'type': 'BUY' or 'SELL',
            'entry': 5000.25,
            'sl': 4996.00,
            'tp': 5015.75,
            'ob_strength': 12.5,
            'session': 'London',
            'priority': 10,
            'time': datetime object
        }
        """
        # حساب
        risk = abs(setup['entry'] - setup['sl'])
        target = abs(setup['tp'] - setup['entry'])
        rr = target / risk if risk > 0 else 0
        
        # تحويل
        type_num = 1 if setup['type'] == 'BUY' else 0
        session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
        session_num = session_map.get(setup['session'], 0)
        hour = setup['time'].hour
        day_of_week = setup['time'].weekday()
        
        # Features array (نفس الترتيب في التدريب!)
        features = [
            type_num,              # 0
            setup['ob_strength'],  # 1: strength
            risk,                  # 2: risk
            target,                # 3: target
            rr,                    # 4: rr
            setup['priority'],     # 5: priority
            session_num,           # 6: session_num
            hour,                  # 7: hour
            day_of_week            # 8: day_of_week
        ]
        
        return features
    
    def evaluate_setup(self, setup):
        """
        تقييم Setup باستخدام ML
        
        Returns:
            probability (float): احتمالية النجاح (0-1)
            decision (str): 'TAKE' أو 'SKIP'
            reason (str): السبب
        """
        # استخراج features
        features = self.extract_setup_features(setup)
        
        # التنبؤ
        probability = self.model.predict_proba([features])[0][1]
        
        # القرار
        if probability >= self.min_probability:
            decision = 'TAKE'
            reason = f"High confidence ({probability*100:.1f}%)"
        else:
            decision = 'SKIP'
            reason = f"Low confidence ({probability*100:.1f}%)"
        
        # فحوصات إضافية
        if self.daily_trades >= self.max_daily_trades:
            decision = 'SKIP'
            reason = "Max daily trades reached"
        
        if self.daily_pnl <= -self.max_daily_loss:
            decision = 'SKIP'
            reason = "Max daily loss reached"
        
        return probability, decision, reason
    
    def print_setup_analysis(self, setup, probability, decision, reason):
        """طباعة تحليل Setup"""
        print("\n" + "="*80)
        print(f"📊 Setup Analysis - {setup['time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        print(f"\n🎯 Setup Details:")
        print(f"   Type: {setup['type']}")
        print(f"   Entry: {setup['entry']:.2f}")
        print(f"   SL: {setup['sl']:.2f}")
        print(f"   TP: {setup['tp']:.2f}")
        print(f"   Risk: {abs(setup['entry']-setup['sl']):.2f} points")
        print(f"   Target: {abs(setup['tp']-setup['entry']):.2f} points")
        print(f"   R:R: 1:{abs(setup['tp']-setup['entry'])/abs(setup['entry']-setup['sl']):.2f}")
        print(f"   Session: {setup['session']}")
        print(f"   OB Strength: {setup['ob_strength']:.1f}")
        
        print(f"\n🤖 ML Analysis:")
        print(f"   Win Probability: {probability*100:.1f}%")
        
        if probability >= 0.90:
            conf_level = "🔥 VERY HIGH"
        elif probability >= 0.80:
            conf_level = "✅ HIGH"
        elif probability >= 0.70:
            conf_level = "⚠️  MEDIUM"
        else:
            conf_level = "❌ LOW"
        
        print(f"   Confidence: {conf_level}")
        
        print(f"\n📋 Decision: {decision}")
        print(f"   Reason: {reason}")
        
        print(f"\n📊 Daily Status:")
        print(f"   Trades Today: {self.daily_trades}/{self.max_daily_trades}")
        print(f"   Daily P&L: ${self.daily_pnl:+.2f}")
        
        if decision == 'TAKE':
            print(f"\n✅ GO FOR IT!")
        else:
            print(f"\n⏭️  SKIP THIS ONE")
        
        print("="*80)
    
    def example_usage(self):
        """مثال على الاستخدام"""
        print("\n" + "="*80)
        print("📚 مثال على الاستخدام")
        print("="*80)
        
        # مثال Setup 1
        setup1 = {
            'type': 'BUY',
            'entry': 5000.25,
            'sl': 4996.00,
            'tp': 5015.75,
            'ob_strength': 12.5,
            'session': 'London',
            'priority': 10,
            'time': datetime.now().replace(hour=8, minute=30)
        }
        
        prob1, dec1, reason1 = self.evaluate_setup(setup1)
        self.print_setup_analysis(setup1, prob1, dec1, reason1)
        
        # مثال Setup 2
        setup2 = {
            'type': 'SELL',
            'entry': 5000.50,
            'sl': 5004.75,
            'tp': 4988.25,
            'ob_strength': 8.2,
            'session': 'NY_PM',
            'priority': 9,
            'time': datetime.now().replace(hour=19, minute=15)
        }
        
        prob2, dec2, reason2 = self.evaluate_setup(setup2)
        self.print_setup_analysis(setup2, prob2, dec2, reason2)

# دالة مساعدة للاستخدام السريع
def quick_check_setup(entry, sl, tp, trade_type, ob_strength, session='London'):
    """
    فحص سريع لـ Setup
    
    مثال:
    quick_check_setup(5000.25, 4996.00, 5015.75, 'BUY', 12.5, 'London')
    """
    system = LiveTradingML()
    
    setup = {
        'type': trade_type,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'ob_strength': ob_strength,
        'session': session,
        'priority': 10 if session in ['London', 'NY_AM'] else 9,
        'time': datetime.now()
    }
    
    prob, dec, reason = system.evaluate_setup(setup)
    system.print_setup_analysis(setup, prob, dec, reason)
    
    return prob, dec

if __name__ == "__main__":
    system = LiveTradingML()
    system.example_usage()
    
    print("\n" + "="*80)
    print("💡 كيف تستخدمه:")
    print("="*80)
    print("""
    from live_trading_integration import quick_check_setup
    
    # عندما تجد Setup:
    prob, decision = quick_check_setup(
        entry=5000.25,
        sl=4996.00,
        tp=5015.75,
        trade_type='BUY',
        ob_strength=12.5,
        session='London'
    )
    
    if decision == 'TAKE':
        # خذ الصفقة!
        print("✅ ENTER TRADE")
    else:
        # تجنبها
        print("⏭️ SKIP")
    """)

