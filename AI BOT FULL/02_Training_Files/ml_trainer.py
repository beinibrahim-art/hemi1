"""
🤖 ML Model Trainer - تدريب الذكاء الاصطناعي على بيانات الباكتست
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

class MLTrainer:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("🤖 ML Trainer - تدريب الذكاء الاصطناعي")
        print("="*100)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.output_folder = self.config['output']['folder']
        self.ml_folder = os.path.join(self.output_folder, 'ml_models')
        os.makedirs(self.ml_folder, exist_ok=True)
        
        print(f"\n📂 مجلد ML: {self.ml_folder}")
    
    def load_backtest_data(self):
        """تحميل نتائج الباكتست"""
        print("\n📥 تحميل بيانات الباكتست...")
        
        trades_file = os.path.join(self.output_folder, 'backtest_trades.csv')
        
        if not os.path.exists(trades_file):
            print(f"❌ لم يتم العثور على: {trades_file}")
            print("⚠️  يجب تشغيل الباكتست أولاً!")
            return None
        
        self.trades_df = pd.read_csv(trades_file)
        print(f"✅ تم تحميل {len(self.trades_df)} صفقة")
        
        return self.trades_df
    
    def extract_features(self):
        """استخراج Features للتعلم"""
        print("\n🔧 استخراج Features...")
        
        df = self.trades_df.copy()
        
        # التحويلات
        df['type_num'] = (df['type'] == 'BUY').astype(int)
        df['result_num'] = (df['result'] == 'WIN').astype(int)
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        
        # Session encoding
        session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
        df['session_num'] = df['session'].map(session_map)
        
        # Features
        feature_cols = [
            'type_num',          # نوع الصفقة (BUY/SELL)
            'strength',          # قوة OB
            'risk',              # المخاطرة بالنقاط
            'target',            # الهدف بالنقاط
            'rr',                # Risk:Reward
            'priority',          # أولوية الـ Killzone
            'session_num',       # نوع الجلسة
            'hour',              # الساعة
            'day_of_week',       # يوم الأسبوع
        ]
        
        self.X = df[feature_cols].values
        self.y = df['result_num'].values
        
        self.feature_names = feature_cols
        
        print(f"✅ Features: {len(feature_cols)}")
        print(f"✅ Samples: {len(self.X)}")
        print(f"✅ Win rate: {self.y.mean()*100:.1f}%")
        
        return self.X, self.y
    
    def train_models(self):
        """تدريب عدة ML Models"""
        print("\n🎓 بدء التدريب...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        self.models = {}
        self.scores = {}
        
        # Model 1: Random Forest
        print("\n🌲 Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        self.models['RandomForest'] = rf
        self.scores['RandomForest'] = rf_score
        print(f"   ✅ Accuracy: {rf_score*100:.2f}%")
        
        # Model 2: XGBoost
        print("\n🚀 XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        self.models['XGBoost'] = xgb_model
        self.scores['XGBoost'] = xgb_score
        print(f"   ✅ Accuracy: {xgb_score*100:.2f}%")
        
        # اختيار الأفضل
        best_name = max(self.scores, key=self.scores.get)
        self.best_model = self.models[best_name]
        self.best_name = best_name
        
        print(f"\n🏆 الأفضل: {best_name} ({self.scores[best_name]*100:.2f}%)")
        
        # Detailed report
        y_pred = self.best_model.predict(X_test)
        print(f"\n📊 تقرير مفصل:")
        print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
        
        # Save models
        for name, model in self.models.items():
            model_file = os.path.join(self.ml_folder, f'{name}_model.pkl')
            joblib.dump(model, model_file)
            print(f"💾 تم حفظ: {name}")
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Feature Importance
        self.plot_feature_importance()
        
        return self.best_model
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """رسم Confusion Matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['LOSS', 'WIN'],
                    yticklabels=['LOSS', 'WIN'])
        plt.title(f'Confusion Matrix - {self.best_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        cm_file = os.path.join(self.ml_folder, 'confusion_matrix.png')
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 تم حفظ: confusion_matrix.png")
    
    def plot_feature_importance(self):
        """رسم أهمية Features"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_imp['feature'], feature_imp['importance'])
            plt.xlabel('Importance', fontsize=12)
            plt.title('Feature Importance', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            imp_file = os.path.join(self.ml_folder, 'feature_importance.png')
            plt.savefig(imp_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 تم حفظ: feature_importance.png")
            
            print(f"\n🔝 أهم Features:")
            for idx, row in feature_imp.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    def run(self):
        """تشغيل كامل"""
        # 1. تحميل البيانات
        if self.load_backtest_data() is None:
            return
        
        # 2. استخراج Features
        self.extract_features()
        
        # 3. التدريب
        self.train_models()
        
        print("\n" + "="*100)
        print("✅ تم التدريب بنجاح!")
        print("="*100)
        print(f"\n📁 النتائج في: {self.ml_folder}")
        print(f"   - {self.best_name}_model.pkl")
        print(f"   - confusion_matrix.png")
        print(f"   - feature_importance.png")
        print("\n🚀 الخطوة التالية: ml_enhanced_backtest.py")

if __name__ == "__main__":
    trainer = MLTrainer()
    trainer.run()

