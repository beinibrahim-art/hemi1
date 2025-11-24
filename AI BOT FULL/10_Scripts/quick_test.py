"""
🎯 اختبار سريع للنظام - 10 أيام فقط
للتأكد من أن كل شيء يعمل قبل تشغيل السنة كاملة
"""

import databento as db
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# استيراد النظام الكامل
from full_year_backtest import FullYearICTBacktest

print("="*120)
print("🧪 اختبار سريع - 10 أيام")
print("="*120)

# المسار إلى مجلد البيانات
data_folder = r"C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N"

print(f"\n📂 البحث في: {data_folder}")

# التحقق من وجود المجلد
if not os.path.exists(data_folder):
    print(f"❌ خطأ: المجلد غير موجود!")
    print(f"   تأكد من المسار الصحيح")
    input("\nاضغط Enter للخروج...")
    exit(1)

# التحقق من وجود ملفات
dbn_files = glob.glob(os.path.join(data_folder, "*.dbn.zst"))
if len(dbn_files) == 0:
    print(f"❌ خطأ: لا توجد ملفات .dbn.zst في المجلد")
    input("\nاضغط Enter للخروج...")
    exit(1)

print(f"✅ وجدنا {len(dbn_files)} ملف")
print(f"\n🎯 سنختبر أول 10 أيام فقط...")

try:
    # إنشاء Backtester
    backtester = FullYearICTBacktest(
        data_folder=data_folder,
        initial_capital=50000
    )
    
    print(f"\n{'='*120}")
    print("🚀 بدء الاختبار...")
    print(f"{'='*120}")
    
    # تشغيل على 10 أيام فقط
    backtester.run_full_backtest(max_days=10)
    
    print(f"\n{'='*120}")
    print("✅ نجح الاختبار!")
    print(f"{'='*120}")
    print("\nالآن يمكنك تشغيل السنة كاملة باستخدام:")
    print("  python full_year_backtest.py")
    print("\nأو:")
    print("  RUN_BACKTEST.bat")
    print(f"{'='*120}")

except Exception as e:
    print(f"\n{'='*120}")
    print(f"❌ خطأ أثناء التشغيل:")
    print(f"{'='*120}")
    print(f"{e}")
    print(f"\n{'='*120}")
    print("💡 نصائح لحل المشكلة:")
    print("   1. تأكد من تثبيت المكتبات: pip install databento pandas numpy")
    print("   2. تأكد من أن البيانات صحيحة")
    print("   3. جرب ملف واحد فقط أولاً")
    print(f"{'='*120}")

input("\nاضغط Enter للخروج...")

