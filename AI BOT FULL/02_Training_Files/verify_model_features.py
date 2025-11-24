"""
🔍 Model Features Verification
التحقق من أن المؤشر يرسل نفس الـ Features التي تدرب عليها الموديل
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def verify_model_features():
    """
    عرض الـ Features التي تدرب عليها الموديل
    """
    print("="*80)
    print("🔍 Model Features Verification - التحقق من Features")
    print("="*80)
    
    # تحميل الموديل
    model_path = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models\XGBoost_ForwardTested_model.pkl'
    
    if not os.path.exists(model_path):
        model_path = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models\XGBoost_Balanced_model.pkl'
    
    if not os.path.exists(model_path):
        model_path = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models\XGBoost_model.pkl'
    
    print(f"\n📥 تحميل الموديل...")
    print(f"   Path: {os.path.basename(model_path)}")
    
    model = joblib.load(model_path)
    
    print(f"\n✅ تم تحميل الموديل")
    print(f"   Type: {type(model).__name__}")
    
    # عرض الـ Features
    print("\n" + "="*80)
    print("📊 الـ Features التي تدرب عليها الموديل:")
    print("="*80)
    
    feature_names = [
        "type_num",
        "strength", 
        "risk",
        "target",
        "rr",
        "priority",
        "session_num",
        "hour",
        "day_of_week"
    ]
    
    print("\nالترتيب الدقيق (يجب أن يطابق المؤشر هذا بالضبط!):\n")
    
    for i, name in enumerate(feature_names):
        print(f"  [{i}] {name:15s}", end="")
        
        if name == "type_num":
            print(" → 0=SELL, 1=BUY")
        elif name == "strength":
            print(" → قوة Order Block (1-20)")
        elif name == "risk":
            print(" → |entry - sl| بالنقاط")
        elif name == "target":
            print(" → |tp - entry| بالنقاط")
        elif name == "rr":
            print(" → target / risk")
        elif name == "priority":
            print(" → أولوية Setup (9-10)")
        elif name == "session_num":
            print(" → 0=NY_PM, 1=NY_AM, 2=London")
        elif name == "hour":
            print(" → ساعة اليوم (0-23) UTC")
        elif name == "day_of_week":
            print(" → يوم الأسبوع (0=Mon, 6=Sun)")
    
    # مثال على البيانات الصحيحة
    print("\n" + "="*80)
    print("📝 مثال على Setup صحيح من TopStep:")
    print("="*80)
    
    example = {
        'type': 'BUY',
        'entry': 5000.25,
        'sl': 4996.00,
        'tp': 5015.75,
        'ob_strength': 12.5,
        'session': 'London',
        'timestamp': '2025-11-21T08:30:00'
    }
    
    print("\n📥 البيانات من المؤشر:")
    for key, val in example.items():
        print(f"   {key:15s} = {val}")
    
    # حساب الـ Features
    print("\n🔧 تحويل إلى Features:")
    
    type_num = 1 if example['type'] == 'BUY' else 0
    strength = example['ob_strength']
    risk = abs(example['entry'] - example['sl'])
    target = abs(example['tp'] - example['entry'])
    rr = target / risk
    priority = 10
    session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
    session_num = session_map[example['session']]
    ts = pd.to_datetime(example['timestamp'])
    hour = ts.hour
    day_of_week = ts.weekday()
    
    features = [
        type_num,
        strength,
        risk,
        target,
        rr,
        priority,
        session_num,
        hour,
        day_of_week
    ]
    
    print()
    for i, (name, val) in enumerate(zip(feature_names, features)):
        print(f"   [{i}] {name:15s} = {val}")
    
    # اختبار التنبؤ
    print("\n🤖 اختبار التنبؤ:")
    
    features_array = np.array([features])
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0]
    
    print(f"   Prediction: {'WIN' if prediction == 1 else 'LOSS'}")
    print(f"   Probability (WIN): {probability[1]*100:.2f}%")
    print(f"   Probability (LOSS): {probability[0]*100:.2f}%")
    
    if probability[1] >= 0.70:
        print(f"   Decision: ✅ TAKE")
    else:
        print(f"   Decision: ⏭️  SKIP")
    
    # التحذيرات
    print("\n" + "="*80)
    print("⚠️  تحذيرات مهمة:")
    print("="*80)
    print("""
  1. الترتيب يجب أن يكون بالضبط كما هو أعلاه!
  2. type_num: 0=SELL, 1=BUY (ليس العكس!)
  3. session_num: 0=NY_PM, 1=NY_AM, 2=London (ليس عشوائي!)
  4. hour: يجب أن يكون UTC (ليس CST!)
  5. day_of_week: 0=Monday, 6=Sunday (حسب Python)
  6. كل الأسعار يجب أن تكون float (ليس string!)
    """)
    
    # توصيات
    print("="*80)
    print("✅ التوصيات:")
    print("="*80)
    print("""
  1. استخدم csv_signal_processor.py → يتعامل مع التحويل تلقائياً
  2. المؤشر يرسل فقط:
     - type, entry, sl, tp, ob_strength, session, timestamp
  3. Python يحسب باقي الـ Features بشكل صحيح
  4. لا تحاول حساب الـ Features في المؤشر!
    """)
    
    print("="*80)
    
    return model, feature_names

def test_custom_setup():
    """
    اختبار Setup مخصص
    """
    model, feature_names = verify_model_features()
    
    print("\n\n" + "="*80)
    print("🧪 اختبر Setup الخاص بك")
    print("="*80)
    
    try:
        print("\nأدخل تفاصيل Setup:")
        
        trade_type = input("Type (BUY/SELL): ").upper()
        entry = float(input("Entry: "))
        sl = float(input("SL: "))
        tp = float(input("TP: "))
        ob_strength = float(input("OB Strength: "))
        
        print("\nSession:")
        print("  1. London")
        print("  2. NY_AM")
        print("  3. NY_PM")
        session_choice = input("اختر (1/2/3): ")
        session_map = {'1': 'London', '2': 'NY_AM', '3': 'NY_PM'}
        session = session_map.get(session_choice, 'London')
        
        # حساب Features
        type_num = 1 if trade_type == 'BUY' else 0
        risk = abs(entry - sl)
        target = abs(tp - entry)
        rr = target / risk
        priority = 10
        sess_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
        session_num = sess_map[session]
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        features = [
            type_num,
            ob_strength,
            risk,
            target,
            rr,
            priority,
            session_num,
            hour,
            day_of_week
        ]
        
        print("\n🔧 Features المحسوبة:")
        for i, (name, val) in enumerate(zip(feature_names, features)):
            print(f"   [{i}] {name:15s} = {val}")
        
        # التنبؤ
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        print("\n🤖 نتيجة ML:")
        print(f"   Prediction: {'WIN' if prediction == 1 else 'LOSS'}")
        print(f"   WIN Probability: {probability[1]*100:.2f}%")
        
        if probability[1] >= 0.70:
            print(f"\n   ✅ Decision: TAKE")
        else:
            print(f"\n   ⏭️  Decision: SKIP")
        
        print("\n" + "="*80)
        
        # مرة أخرى؟
        again = input("\nاختبار آخر؟ (y/n): ").lower()
        if again == 'y':
            test_custom_setup()
    
    except Exception as e:
        print(f"\n❌ خطأ: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # وضع الاختبار
        test_custom_setup()
    else:
        # عرض المعلومات فقط
        verify_model_features()
        
        print("\n💡 للاختبار التفاعلي:")
        print("   python verify_model_features.py test")

