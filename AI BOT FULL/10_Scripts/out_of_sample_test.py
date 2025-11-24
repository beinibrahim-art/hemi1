"""
🔬 Out-of-Sample Test - اختبار على بيانات من فترة مختلفة تماماً
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

class OutOfSampleTest:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("🔬 Out-of-Sample Test - اختبار على فترة مختلفة!")
        print("="*100)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.output_folder = self.config['output']['folder']
        self.ml_folder = os.path.join(self.output_folder, 'ml_models')
    
    def load_and_split_by_period(self):
        """تقسيم حسب الفترة الزمنية"""
        print("\n📥 تحميل البيانات...")
        
        trades_file = os.path.join(self.output_folder, 'backtest_trades.csv')
        df = pd.read_csv(trades_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ {len(df)} صفقة")
        print(f"   الفترة: {df['date'].min().date()} → {df['date'].max().date()}")
        
        # حساب نقطة المنتصف
        date_range = df['date'].max() - df['date'].min()
        mid_date = df['date'].min() + (date_range / 2)
        
        print(f"\n📊 التقسيم حسب الفترة:")
        print(f"   نقطة المنتصف: {mid_date.date()}")
        
        # الطريقة 1: تدريب على الأولى، اختبار على الثانية
        print(f"\n🔄 Scenario 1: Train on EARLY data, Test on LATER data")
        self.early_df = df[df['date'] < mid_date].copy()
        self.later_df = df[df['date'] >= mid_date].copy()
        
        print(f"   Training (Early):  {len(self.early_df)} صفقات ({self.early_df['date'].min().date()} → {self.early_df['date'].max().date()})")
        print(f"   Testing (Later):   {len(self.later_df)} صفقات ({self.later_df['date'].min().date()} → {self.later_df['date'].max().date()})")
        
        # الطريقة 2: تدريب على الثانية، اختبار على الأولى
        print(f"\n🔄 Scenario 2: Train on LATER data, Test on EARLY data")
        print(f"   Training (Later):  {len(self.later_df)} صفقات")
        print(f"   Testing (Early):   {len(self.early_df)} صفقات")
        
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
    
    def train_and_test_scenario(self, train_df, test_df, scenario_name):
        """تدريب واختبار سيناريو معين"""
        print("\n" + "="*100)
        print(f"🎯 {scenario_name}")
        print("="*100)
        
        # استخراج features
        X_train, y_train, feature_names = self.extract_features(train_df)
        X_test, y_test, _ = self.extract_features(test_df)
        
        print(f"\nTraining Data:")
        print(f"   Period: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
        print(f"   Samples: {len(X_train)}")
        print(f"   WIN: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   LOSS: {len(y_train)-y_train.sum()} ({(1-y_train.mean())*100:.1f}%)")
        
        # تدريب
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # اختبار
        print(f"\nTest Data:")
        print(f"   Period: {test_df['date'].min().date()} → {test_df['date'].max().date()}")
        print(f"   Samples: {len(X_test)}")
        print(f"   WIN: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
        print(f"   LOSS: {len(y_test)-y_test.sum()} ({(1-y_test.mean())*100:.1f}%)")
        
        # التنبؤ
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # المقاييس
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 النتائج:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}%")
        
        print(f"\n📋 تقرير مفصل:")
        report = classification_report(y_test, y_pred, target_names=['LOSS', 'WIN'], output_dict=True)
        print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
        
        # محاكاة Backtest
        high_conf = y_proba >= 0.70
        if high_conf.sum() > 0:
            selected_y = y_test[high_conf]
            backtest_wr = selected_y.mean()
            print(f"\n🎯 محاكاة Backtest (probability > 70%):")
            print(f"   Selected: {high_conf.sum()} من {len(y_test)}")
            print(f"   Win Rate: {backtest_wr*100:.1f}%")
            print(f"   Rejected: {(~high_conf).sum()}")
        else:
            backtest_wr = 0
        
        return {
            'scenario': scenario_name,
            'accuracy': accuracy,
            'loss_recall': report['LOSS']['recall'],
            'loss_precision': report['LOSS']['precision'],
            'win_recall': report['WIN']['recall'],
            'win_precision': report['WIN']['precision'],
            'backtest_wr': backtest_wr,
            'selected_trades': high_conf.sum(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def compare_scenarios(self, results):
        """مقارنة السيناريوهات"""
        print("\n" + "="*100)
        print("📊 مقارنة السيناريوهات")
        print("="*100)
        
        print(f"\n{'Scenario':<40} | {'Accuracy':<10} | {'LOSS R':<8} | {'WIN R':<8} | {'Backtest WR':<12}")
        print("="*100)
        
        for r in results:
            print(f"{r['scenario']:<40} | {r['accuracy']*100:>8.2f}% | {r['loss_recall']*100:>6.1f}% | {r['win_recall']*100:>6.1f}% | {r['backtest_wr']*100:>10.1f}%")
        
        # رسم المقارنة
        self.plot_comparison(results)
    
    def plot_comparison(self, results):
        """رسم المقارنة"""
        print("\n📊 إنشاء رسومات المقارنة...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        scenarios = [r['scenario'].replace('Scenario ', 'S') for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]
        ax1.bar(range(len(scenarios)), accuracies, alpha=0.7)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=0, ha='center')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim([70, 100])
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Recall Comparison
        ax2 = axes[0, 1]
        x = np.arange(len(scenarios))
        width = 0.35
        loss_recalls = [r['loss_recall'] * 100 for r in results]
        win_recalls = [r['win_recall'] * 100 for r in results]
        ax2.bar(x - width/2, loss_recalls, width, label='LOSS Recall', alpha=0.7, color='red')
        ax2.bar(x + width/2, win_recalls, width, label='WIN Recall', alpha=0.7, color='green')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios)
        ax2.set_ylabel('Recall (%)')
        ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Backtest Win Rate
        ax3 = axes[1, 0]
        backtest_wrs = [r['backtest_wr'] * 100 for r in results]
        colors = ['green' if w > 90 else 'orange' if w > 80 else 'red' for w in backtest_wrs]
        ax3.bar(range(len(scenarios)), backtest_wrs, alpha=0.7, color=colors)
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios)
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Backtest Win Rate (prob > 70%)', fontsize=14, fontweight='bold')
        ax3.set_ylim([70, 100])
        ax3.axhline(90, color='green', linestyle='--', alpha=0.5, label='90%')
        for i, v in enumerate(backtest_wrs):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        avg_accuracy = np.mean([r['accuracy'] for r in results]) * 100
        avg_backtest_wr = np.mean([r['backtest_wr'] for r in results]) * 100
        
        summary_text = f"""
Out-of-Sample Test Summary

Total Scenarios: {len(results)}

Average Performance:
  Accuracy: {avg_accuracy:.1f}%
  Backtest Win Rate: {avg_backtest_wr:.1f}%

Key Findings:
  
Scenario 1 (Train Early → Test Later):
  - Tests model on recent data
  - Accuracy: {results[0]['accuracy']*100:.1f}%
  - Win Rate: {results[0]['backtest_wr']*100:.1f}%

Scenario 2 (Train Later → Test Early):
  - Tests model on older data
  - Accuracy: {results[1]['accuracy']*100:.1f}%
  - Win Rate: {results[1]['backtest_wr']*100:.1f}%

Conclusion:
  {'✓ Consistent across periods' if abs(results[0]['backtest_wr'] - results[1]['backtest_wr']) < 0.05 else '⚠ Performance varies by period'}
  {'✓ Model generalizes well' if avg_backtest_wr > 90 else '⚠ Model may be period-specific'}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        out_file = os.path.join(self.ml_folder, 'out_of_sample_test.png')
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ تم حفظ: out_of_sample_test.png")
    
    def run(self):
        """تشغيل كامل"""
        # 1. تحميل وتقسيم
        self.load_and_split_by_period()
        
        # 2. السيناريو 1: تدريب على الأولى، اختبار على الثانية
        result1 = self.train_and_test_scenario(
            self.early_df, 
            self.later_df,
            "Scenario 1: Train on Early Period → Test on Later Period"
        )
        
        # 3. السيناريو 2: تدريب على الثانية، اختبار على الأولى
        result2 = self.train_and_test_scenario(
            self.later_df,
            self.early_df,
            "Scenario 2: Train on Later Period → Test on Early Period"
        )
        
        # 4. مقارنة
        self.compare_scenarios([result1, result2])
        
        print("\n" + "="*100)
        print("✅ Out-of-Sample Test مكتمل!")
        print("="*100)
        print(f"\n📊 النتائج:")
        print(f"   Scenario 1 Win Rate: {result1['backtest_wr']*100:.1f}%")
        print(f"   Scenario 2 Win Rate: {result2['backtest_wr']*100:.1f}%")
        print(f"   Average: {(result1['backtest_wr'] + result2['backtest_wr'])/2*100:.1f}%")
        
        if abs(result1['backtest_wr'] - result2['backtest_wr']) < 0.05:
            print(f"\n✅ النموذج ثابت عبر الفترات المختلفة!")
        else:
            print(f"\n⚠️  الأداء يختلف حسب الفترة - قد يكون period-specific")

if __name__ == "__main__":
    tester = OutOfSampleTest()
    tester.run()

