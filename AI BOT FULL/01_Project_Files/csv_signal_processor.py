"""
🔄 CSV Signal Processor
يقرأ الإشارات من المؤشر (CSV) ويعطي قرارات ML

Workflow:
1. المؤشر يحفظ Setup في signals.csv
2. هذا السكريبت يقرأ signals.csv
3. ML يقيّم كل Setup
4. يكتب القرار في decisions.csv
5. المؤشر يقرأ decisions.csv وينفذ
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

class CSVSignalProcessor:
    def __init__(self, 
                 signals_file='signals.csv',
                 decisions_file='decisions.csv',
                 model_path=None):
        
        print("="*80)
        print("🔄 CSV Signal Processor - معالج إشارات CSV")
        print("="*80)
        
        self.signals_file = signals_file
        self.decisions_file = decisions_file
        
        # تحميل ML Model
        print(f"\n📥 تحميل ML Model...")
        if model_path is None:
            base_dir = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models'
            model_path = os.path.join(base_dir, 'XGBoost_ForwardTested_model.pkl')
            
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'XGBoost_Balanced_model.pkl')
            
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, 'XGBoost_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✅ تم تحميل: {os.path.basename(model_path)}")
        
        # إعدادات
        self.min_probability = 0.70
        self.max_daily_trades = 3
        self.max_daily_loss = 1000
        
        # متتبع
        self.processed_ids = set()
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.current_date = None
        
        print(f"\n⚙️  الإعدادات:")
        print(f"   Signals File: {self.signals_file}")
        print(f"   Decisions File: {self.decisions_file}")
        print(f"   Min Probability: {self.min_probability*100:.0f}%")
        print(f"   Max Daily Trades: {self.max_daily_trades}")
        print(f"   Max Daily Loss: ${self.max_daily_loss}")
    
    def extract_features(self, row):
        """
        استخراج features من signal
        
        row يجب أن يحتوي على:
        - type: 'BUY' أو 'SELL'
        - entry: float
        - sl: float
        - tp: float
        - ob_strength: float
        - session: 'London', 'NY_AM', 'NY_PM'
        - timestamp: datetime string
        """
        # حساب
        risk = abs(row['entry'] - row['sl'])
        target = abs(row['tp'] - row['entry'])
        rr = target / risk if risk > 0 else 0
        
        # تحويل
        type_num = 1 if str(row['type']).upper() == 'BUY' else 0
        session_map = {'London': 2, 'london': 2, 'NY_AM': 1, 'ny_am': 1, 'NY_PM': 0, 'ny_pm': 0}
        session_num = session_map.get(row.get('session', 'London'), 0)
        
        # استخراج الوقت
        try:
            ts = pd.to_datetime(row['timestamp'])
            hour = ts.hour
            day_of_week = ts.weekday()
        except:
            hour = 8
            day_of_week = 1
        
        # Priority
        priority = row.get('priority', 10)
        
        # Features array
        features = [
            type_num,              # 0
            row['ob_strength'],    # 1: strength
            risk,                  # 2: risk
            target,                # 3: target
            rr,                    # 4: rr
            priority,              # 5: priority
            session_num,           # 6: session_num
            hour,                  # 7: hour
            day_of_week            # 8: day_of_week
        ]
        
        return features
    
    def evaluate_signal(self, row):
        """تقييم signal واحد"""
        # استخراج features
        features = self.extract_features(row)
        
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
        
        return {
            'probability': probability,
            'decision': decision,
            'reason': reason
        }
    
    def process_signals(self):
        """
        قراءة signals.csv ومعالجة الإشارات الجديدة
        """
        # التحقق من وجود الملف
        if not os.path.exists(self.signals_file):
            print(f"\n⏳ في انتظار: {self.signals_file}")
            return False
        
        # قراءة الإشارات
        try:
            signals_df = pd.read_csv(self.signals_file)
        except Exception as e:
            print(f"❌ خطأ في قراءة {self.signals_file}: {e}")
            return False
        
        # التحقق من وجود إشارات جديدة
        if len(signals_df) == 0:
            return False
        
        # قراءة القرارات السابقة
        if os.path.exists(self.decisions_file):
            try:
                decisions_df = pd.read_csv(self.decisions_file)
                self.processed_ids = set(decisions_df['signal_id'].values)
            except:
                decisions_df = pd.DataFrame()
        else:
            decisions_df = pd.DataFrame()
        
        # معالجة الإشارات الجديدة
        new_decisions = []
        processed_count = 0
        
        for idx, row in signals_df.iterrows():
            signal_id = row.get('signal_id', idx)
            
            # تخطي المعالجة سابقاً
            if signal_id in self.processed_ids:
                continue
            
            # إعادة تعيين العداد اليومي
            try:
                signal_date = pd.to_datetime(row['timestamp']).date()
                if self.current_date is None or signal_date != self.current_date:
                    self.current_date = signal_date
                    self.daily_trades = 0
                    self.daily_pnl = 0.0
                    print(f"\n📅 يوم جديد: {signal_date}")
            except:
                pass
            
            # تقييم
            print(f"\n🔍 معالجة Signal #{signal_id}...")
            print(f"   Type: {row['type']}")
            print(f"   Entry: {row['entry']:.2f}")
            print(f"   SL: {row['sl']:.2f}")
            print(f"   TP: {row['tp']:.2f}")
            
            result = self.evaluate_signal(row)
            
            print(f"   🤖 ML Probability: {result['probability']*100:.1f}%")
            print(f"   📋 Decision: {result['decision']}")
            print(f"   💬 Reason: {result['reason']}")
            
            # حفظ القرار
            decision_row = {
                'signal_id': signal_id,
                'timestamp': row.get('timestamp', datetime.now().isoformat()),
                'type': row['type'],
                'entry': row['entry'],
                'sl': row['sl'],
                'tp': row['tp'],
                'probability': result['probability'],
                'decision': result['decision'],
                'reason': result['reason'],
                'processed_at': datetime.now().isoformat()
            }
            
            new_decisions.append(decision_row)
            self.processed_ids.add(signal_id)
            processed_count += 1
            
            if result['decision'] == 'TAKE':
                self.daily_trades += 1
        
        # حفظ القرارات
        if len(new_decisions) > 0:
            new_df = pd.DataFrame(new_decisions)
            
            if len(decisions_df) > 0:
                decisions_df = pd.concat([decisions_df, new_df], ignore_index=True)
            else:
                decisions_df = new_df
            
            decisions_df.to_csv(self.decisions_file, index=False)
            print(f"\n✅ تم حفظ {processed_count} قرار في: {self.decisions_file}")
            return True
        
        return False
    
    def monitor_loop(self, interval=5):
        """
        مراقبة مستمرة لـ signals.csv
        interval: الوقت بين كل فحص (ثواني)
        """
        print("\n" + "="*80)
        print("👁️  بدء المراقبة المستمرة...")
        print("="*80)
        print(f"\n📂 مراقبة: {self.signals_file}")
        print(f"⏱️  كل {interval} ثواني")
        print(f"\n💡 اضغط Ctrl+C للإيقاف")
        print("="*80)
        
        try:
            while True:
                processed = self.process_signals()
                
                if not processed:
                    print(f"\r⏳ في انتظار إشارات جديدة... [{datetime.now().strftime('%H:%M:%S')}]", end='', flush=True)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n✅ تم إيقاف المراقبة")
            print(f"📊 تم معالجة {len(self.processed_ids)} إشارة إجمالاً")

def create_example_signals():
    """إنشاء ملف signals.csv تجريبي"""
    signals = [
        {
            'signal_id': 1,
            'timestamp': datetime.now().isoformat(),
            'type': 'BUY',
            'entry': 5000.25,
            'sl': 4996.00,
            'tp': 5015.75,
            'ob_strength': 12.5,
            'session': 'London',
            'priority': 10
        },
        {
            'signal_id': 2,
            'timestamp': datetime.now().isoformat(),
            'type': 'SELL',
            'entry': 5010.50,
            'sl': 5014.75,
            'tp': 4998.25,
            'ob_strength': 8.2,
            'session': 'NY_PM',
            'priority': 9
        }
    ]
    
    df = pd.DataFrame(signals)
    df.to_csv('signals.csv', index=False)
    print("✅ تم إنشاء signals.csv تجريبي")
    return df

if __name__ == "__main__":
    # إنشاء ملف تجريبي إذا لم يوجد
    if not os.path.exists('signals.csv'):
        print("📝 إنشاء ملف signals.csv تجريبي...")
        create_example_signals()
        print()
    
    # تشغيل المعالج
    processor = CSVSignalProcessor()
    
    # خيارات
    print("\n" + "="*80)
    print("اختر الوضع:")
    print("="*80)
    print("  1. معالجة مرة واحدة (Process Once)")
    print("  2. مراقبة مستمرة (Continuous Monitor)")
    print()
    
    choice = input("اختر (1/2): ").strip()
    
    if choice == '2':
        processor.monitor_loop(interval=5)
    else:
        processor.process_signals()
        print("\n✅ تم الانتهاء من المعالجة")

