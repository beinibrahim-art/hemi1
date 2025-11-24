"""
🔮 Forward Testing - اختبار على بيانات لم يراها الموديل!
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import json
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

class ForwardTestML:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("🔮 Forward Testing - اختبار حقيقي!")
        print("="*100)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.output_folder = self.config['output']['folder']
        self.ml_folder = os.path.join(self.output_folder, 'ml_models')
        os.makedirs(self.ml_folder, exist_ok=True)
    
    def load_and_split_data(self):
        """تحميل البيانات وتقسيمها زمنياً"""
        print("\n📥 تحميل البيانات...")
        
        trades_file = os.path.join(self.output_folder, 'backtest_trades.csv')
        df = pd.read_csv(trades_file)
        
        print(f"✅ {len(df)} صفقة")
        
        # تحويل التاريخ
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"   الفترة: {df['date'].min().date()} إلى {df['date'].max().date()}")
        
        # تقسيم زمني
        n_total = len(df)
        n_train = int(n_total * 0.70)  # أول 70% للتدريب
        n_val = int(n_total * 0.80)    # 70-80% للـ validation
        # الباقي (80-100%) للاختبار
        
        self.train_df = df.iloc[:n_train].copy()
        self.val_df = df.iloc[n_train:n_val].copy()
        self.test_df = df.iloc[n_val:].copy()
        
        print(f"\n📊 التقسيم الزمني:")
        print(f"   Training:   {len(self.train_df)} صفقات ({self.train_df['date'].min().date()} → {self.train_df['date'].max().date()})")
        print(f"   Validation: {len(self.val_df)} صفقات ({self.val_df['date'].min().date()} → {self.val_df['date'].max().date()})")
        print(f"   Testing:    {len(self.test_df)} صفقات ({self.test_df['date'].min().date()} → {self.test_df['date'].max().date()})")
        
        print(f"\n✅ الموديل لن يرى بيانات Testing أبداً أثناء التدريب!")
        
        return df
    
    def extract_features(self, df):
        """استخراج features"""
        df = df.copy()
        
        df['type_num'] = (df['type'] == 'BUY').astype(int)
        df['result_num'] = (df['result'] == 'WIN').astype(int)
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        
        session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
        df['session_num'] = df['session'].map(session_map)
        
        feature_cols = [
            'type_num', 'strength', 'risk', 'target', 'rr',
            'priority', 'session_num', 'hour', 'day_of_week',
        ]
        
        X = df[feature_cols].values
        y = df['result_num'].values
        
        return X, y, feature_cols
    
    def train_fresh_model(self):
        """تدريب موديل جديد على Training data فقط"""
        print("\n🎓 تدريب موديل جديد (على Training data فقط)...")
        
        X_train, y_train, self.feature_names = self.extract_features(self.train_df)
        X_val, y_val, _ = self.extract_features(self.val_df)
        
        # XGBoost مع Scale Pos Weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   WIN: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   LOSS: {len(y_train)-y_train.sum()} ({(1-y_train.mean())*100:.1f}%)")
        print(f"   Scale Pos Weight: {scale_pos_weight:.2f}")
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # تقييم على Validation
        val_score = self.model.score(X_val, y_val)
        print(f"\n✅ Validation Accuracy: {val_score*100:.2f}%")
        
        return self.model
    
    def forward_test(self):
        """الاختبار الحقيقي على بيانات لم يراها!"""
        print("\n" + "="*100)
        print("🔮 Forward Test - اختبار على بيانات جديدة!")
        print("="*100)
        
        X_test, y_test, _ = self.extract_features(self.test_df)
        
        print(f"\nTest Data:")
        print(f"   Samples: {len(X_test)}")
        print(f"   Period: {self.test_df['date'].min().date()} → {self.test_df['date'].max().date()}")
        print(f"   WIN: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        print(f"   LOSS: {len(y_test)-y_test.sum()} ({(1-y_test.mean())*100:.1f}%)")
        
        # التنبؤ
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # المقاييس
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 النتائج:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}%")
        
        # تقرير مفصل
        print(f"\n📋 تقرير مفصل:")
        print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
        
        # محاكاة Backtest (prob > 70%)
        high_conf = y_proba >= 0.70
        if high_conf.sum() > 0:
            selected_y = y_test[high_conf]
            backtest_wr = selected_y.mean()
            print(f"\n🎯 محاكاة Backtest (probability > 70%):")
            print(f"   Selected Trades: {high_conf.sum()} من {len(y_test)}")
            print(f"   Win Rate: {backtest_wr*100:.1f}% ← النتيجة الحقيقية! 🎉")
            print(f"   Rejected Trades: {(~high_conf).sum()} (تجنبها الموديل)")
        
        # Confusion Matrix
        self.plot_forward_test_results(y_test, y_pred, y_proba, high_conf, backtest_wr)
        
        return accuracy, backtest_wr
    
    def plot_forward_test_results(self, y_test, y_pred, y_proba, high_conf, backtest_wr):
        """رسم نتائج Forward Test"""
        print("\n📊 إنشاء الرسومات...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['LOSS', 'WIN'], yticklabels=['LOSS', 'WIN'])
        ax1.set_title('Confusion Matrix - Forward Test', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # 2. Probability Distribution
        ax2 = axes[0, 1]
        win_probs = y_proba[y_test == 1]
        loss_probs = y_proba[y_test == 0]
        ax2.hist(win_probs, bins=20, alpha=0.5, label='WIN (actual)', color='green')
        ax2.hist(loss_probs, bins=20, alpha=0.5, label='LOSS (actual)', color='red')
        ax2.axvline(0.70, color='black', linestyle='--', linewidth=2, label='Threshold (70%)')
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Probability Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        
        # 3. Win Rate by Confidence Level
        ax3 = axes[1, 0]
        thresholds = np.arange(0.5, 1.0, 0.05)
        win_rates = []
        trade_counts = []
        
        for thresh in thresholds:
            mask = y_proba >= thresh
            if mask.sum() > 0:
                wr = y_test[mask].mean()
                win_rates.append(wr * 100)
                trade_counts.append(mask.sum())
            else:
                win_rates.append(0)
                trade_counts.append(0)
        
        ax3_twin = ax3.twinx()
        ax3.plot(thresholds * 100, win_rates, 'b-o', linewidth=2, label='Win Rate')
        ax3_twin.plot(thresholds * 100, trade_counts, 'r--s', linewidth=2, label='# Trades')
        ax3.axvline(70, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax3.set_xlabel('Confidence Threshold (%)', fontsize=12)
        ax3.set_ylabel('Win Rate (%)', fontsize=12, color='b')
        ax3_twin.set_ylabel('Number of Trades', fontsize=12, color='r')
        ax3.set_title('Win Rate vs Confidence Level', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # 4. Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        summary_text = f"""
Forward Test Summary

Test Period: {self.test_df['date'].min().date()} to {self.test_df['date'].max().date()}
Test Samples: {len(y_test)} trades

Overall Performance:
  Accuracy: {accuracy*100:.2f}%
  
WIN Detection:
  Recall: {report['1']['recall']*100:.1f}%
  Precision: {report['1']['precision']*100:.1f}%
  F1-Score: {report['1']['f1-score']:.2f}

LOSS Detection:
  Recall: {report['0']['recall']*100:.1f}%
  Precision: {report['0']['precision']*100:.1f}%
  F1-Score: {report['0']['f1-score']:.2f}

Backtest Simulation (prob > 70%):
  Selected: {high_conf.sum()} trades
  Win Rate: {backtest_wr*100:.1f}%
  Rejected: {(~high_conf).sum()} trades

Conclusion:
  ✓ Tested on UNSEEN data
  ✓ Real-world performance
  ✓ More reliable than overfitted model
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        forward_file = os.path.join(self.ml_folder, 'forward_test_results.png')
        plt.savefig(forward_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ تم حفظ: forward_test_results.png")
    
    def compare_overfitted_vs_realistic(self):
        """مقارنة بين النتائج المبالغ فيها والواقعية"""
        print("\n" + "="*100)
        print("📊 مقارنة: Overfitted vs Realistic")
        print("="*100)
        
        print(f"\n{'Method':<25} | {'Data Used':<30} | {'Win Rate':<12} | {'Reliability':<15}")
        print("="*100)
        print(f"{'Original Test':<25} | {'Random split (WRONG!)':<30} | {'98.3%':<12} | {'❌ Overfitted':<15}")
        print(f"{'Forward Test':<25} | {'Time-based split (RIGHT!)':<30} | {'??.?%':<12} | {'✅ Realistic':<15}")
        
    def run(self):
        """تشغيل كامل"""
        # 1. تحميل وتقسيم
        self.load_and_split_data()
        
        # 2. تدريب جديد
        self.train_fresh_model()
        
        # 3. Forward test
        accuracy, backtest_wr = self.forward_test()
        
        # 4. مقارنة
        self.compare_overfitted_vs_realistic()
        
        # 5. حفظ الموديل الجديد
        new_model_file = os.path.join(self.ml_folder, 'XGBoost_ForwardTested_model.pkl')
        joblib.dump(self.model, new_model_file)
        print(f"\n💾 تم حفظ الموديل الجديد: XGBoost_ForwardTested_model.pkl")
        
        print("\n" + "="*100)
        print("✅ Forward Testing مكتمل!")
        print("="*100)
        print(f"\n🎯 Win Rate الحقيقي (prob > 70%): {backtest_wr*100:.1f}%")
        print(f"   هذا أكثر واقعية من 98.3%!")
        print(f"\n📊 الرسومات: forward_test_results.png")
        print(f"💾 الموديل الجديد: XGBoost_ForwardTested_model.pkl")

if __name__ == "__main__":
    tester = ForwardTestML()
    tester.run()

