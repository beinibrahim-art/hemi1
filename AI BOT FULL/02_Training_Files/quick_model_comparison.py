"""
⚡ Quick Model Comparison - اختبار سريع لجميع النماذج
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

class QuickModelComparison:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("⚡ Quick Model Comparison")
        print("="*100)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.output_folder = self.config['output']['folder']
        self.ml_folder = os.path.join(self.output_folder, 'ml_models')
        
        # تحميل البيانات
        self.load_data()
        
        # تحميل جميع النماذج
        self.load_all_models()
    
    def load_data(self):
        """تحميل بيانات الباكتست"""
        print("\n📥 تحميل البيانات...")
        
        trades_file = os.path.join(self.output_folder, 'backtest_trades.csv')
        self.trades_df = pd.read_csv(trades_file)
        
        print(f"✅ {len(self.trades_df)} صفقة")
        
        # استخراج features
        df = self.trades_df.copy()
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
        
        self.X = df[feature_cols].values
        self.y = df['result_num'].values
    
    def load_all_models(self):
        """تحميل جميع النماذج"""
        print("\n🤖 تحميل النماذج...")
        
        self.models = {}
        
        model_files = [
            'XGBoost_model.pkl',
            'RandomForest_model.pkl',
            'RandomForest_Balanced_model.pkl',
            'XGBoost_Balanced_model.pkl',
            'XGBoost_SMOTE_model.pkl',
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.ml_folder, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✅ {model_name}")
            else:
                print(f"   ⏭️  {model_file} not found")
    
    def test_all_models(self):
        """اختبار جميع النماذج"""
        print("\n" + "="*100)
        print("🎯 اختبار جميع النماذج")
        print("="*100)
        
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n🔍 {model_name}...")
            
            # التنبؤ
            y_pred = model.predict(self.X)
            y_proba = model.predict_proba(self.X)[:, 1]
            
            # حساب المقاييس
            accuracy = (y_pred == self.y).mean()
            
            # Win trades
            wins = (self.y == 1)
            win_recall = (y_pred[wins] == 1).mean()
            win_precision = (self.y[y_pred == 1] == 1).mean()
            
            # Loss trades
            losses = (self.y == 0)
            loss_recall = (y_pred[losses] == 0).mean()
            loss_precision = (self.y[y_pred == 0] == 0).mean()
            
            # محاكاة Win Rate في باكتست
            # (نأخذ فقط الصفقات التي probability > 70%)
            high_confidence = y_proba >= 0.70
            if high_confidence.sum() > 0:
                backtest_win_rate = self.y[high_confidence].mean()
                backtest_trades = high_confidence.sum()
            else:
                backtest_win_rate = 0
                backtest_trades = 0
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'WIN Recall': win_recall,
                'WIN Precision': win_precision,
                'LOSS Recall': loss_recall,
                'LOSS Precision': loss_precision,
                'Backtest Win Rate': backtest_win_rate,
                'Backtest Trades': backtest_trades
            })
            
            print(f"   Accuracy: {accuracy*100:.2f}%")
            print(f"   WIN Recall: {win_recall*100:.1f}%")
            print(f"   LOSS Recall: {loss_recall*100:.1f}%")
            print(f"   Backtest Win Rate (prob>70%): {backtest_win_rate*100:.1f}%")
            print(f"   Backtest Trades (prob>70%): {backtest_trades}")
        
        return pd.DataFrame(results)
    
    def generate_report(self, results_df):
        """إنشاء تقرير المقارنة"""
        print("\n" + "="*100)
        print("📊 تقرير المقارنة النهائي")
        print("="*100)
        
        # ترتيب حسب Backtest Win Rate
        results_df = results_df.sort_values('Backtest Win Rate', ascending=False)
        
        print(f"\n{'Model':<25} | {'Accuracy':<10} | {'WIN R':<8} | {'LOSS R':<8} | {'Backtest WR':<12} | {'Trades':<8}")
        print("="*100)
        
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<25} | {row['Accuracy']*100:>8.2f}% | {row['WIN Recall']*100:>6.1f}% | {row['LOSS Recall']*100:>6.1f}% | {row['Backtest Win Rate']*100:>10.1f}% | {row['Backtest Trades']:>6.0f}")
        
        print("\n" + "="*100)
        
        # أفضل موديل
        best = results_df.iloc[0]
        print(f"\n🏆 الأفضل: {best['Model']}")
        print(f"   Backtest Win Rate: {best['Backtest Win Rate']*100:.1f}%")
        print(f"   Accuracy: {best['Accuracy']*100:.2f}%")
        print(f"   LOSS Recall: {best['LOSS Recall']*100:.1f}%")
        
        # حفظ النتائج
        results_file = os.path.join(self.ml_folder, 'models_quick_comparison.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ تم حفظ: models_quick_comparison.csv")
    
    def run(self):
        """تشغيل المقارنة"""
        results_df = self.test_all_models()
        self.generate_report(results_df)
        
        print("\n" + "="*100)
        print("✅ المقارنة مكتملة!")
        print("="*100)

if __name__ == "__main__":
    comp = QuickModelComparison()
    comp.run()

