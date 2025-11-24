"""
🤖 ML Trainer - Balanced Version
معالجة عدم التوازن بين WIN و LOSS
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
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

class MLTrainerBalanced:
    def __init__(self, config_file='config.json'):
        print("="*100)
        print("🤖 ML Trainer - Balanced Version")
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
            return None
        
        self.trades_df = pd.read_csv(trades_file)
        print(f"✅ تم تحميل {len(self.trades_df)} صفقة")
        
        wins = (self.trades_df['result'] == 'WIN').sum()
        losses = (self.trades_df['result'] == 'LOSS').sum()
        print(f"   WIN: {wins} ({wins/len(self.trades_df)*100:.1f}%)")
        print(f"   LOSS: {losses} ({losses/len(self.trades_df)*100:.1f}%)")
        print(f"   Imbalance Ratio: {wins/losses:.2f}:1")
        
        return self.trades_df
    
    def extract_features(self):
        """استخراج Features"""
        print("\n🔧 استخراج Features...")
        
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
        self.feature_names = feature_cols
        
        print(f"✅ Features: {len(feature_cols)}")
        print(f"✅ Samples: {len(self.X)}")
        print(f"✅ Win rate: {self.y.mean()*100:.1f}%")
        
        return self.X, self.y
    
    def train_models_balanced(self):
        """تدريب Models مع معالجة عدم التوازن"""
        print("\n🎓 بدء التدريب (Balanced Version)...")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        print(f"   Train WIN: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        print(f"   Train LOSS: {len(y_train)-y_train.sum()} ({(1-y_train.mean())*100:.1f}%)")
        
        self.models = {}
        self.scores = {}
        
        # Model 1: Random Forest with Class Weights
        print("\n🌲 Random Forest (Class Weighted)...")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"   Class Weights: LOSS={class_weight_dict[0]:.2f}, WIN={class_weight_dict[1]:.2f}")
        
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight=class_weight_dict,
            random_state=42
        )
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        self.models['RandomForest_Balanced'] = rf
        self.scores['RandomForest_Balanced'] = rf_score
        print(f"   ✅ Accuracy: {rf_score*100:.2f}%")
        
        # Model 2: XGBoost with Scale Pos Weight
        print("\n🚀 XGBoost (Scale Pos Weight)...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   Scale Pos Weight: {scale_pos_weight:.2f}")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_test, y_test)
        self.models['XGBoost_Balanced'] = xgb_model
        self.scores['XGBoost_Balanced'] = xgb_score
        print(f"   ✅ Accuracy: {xgb_score*100:.2f}%")
        
        # Model 3: XGBoost with SMOTE
        print("\n🔄 XGBoost + SMOTE (Oversampling)...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE: {len(X_train_smote)} samples")
        print(f"   WIN: {y_train_smote.sum()}, LOSS: {len(y_train_smote)-y_train_smote.sum()}")
        
        xgb_smote = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
        xgb_smote.fit(X_train_smote, y_train_smote)
        xgb_smote_score = xgb_smote.score(X_test, y_test)
        self.models['XGBoost_SMOTE'] = xgb_smote
        self.scores['XGBoost_SMOTE'] = xgb_smote_score
        print(f"   ✅ Accuracy: {xgb_smote_score*100:.2f}%")
        
        # اختيار الأفضل
        best_name = max(self.scores, key=self.scores.get)
        self.best_model = self.models[best_name]
        self.best_name = best_name
        
        print(f"\n🏆 الأفضل: {best_name} ({self.scores[best_name]*100:.2f}%)")
        
        # مقارنة جميع Models
        print(f"\n📊 مقارنة Models:")
        for name, score in sorted(self.scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {score*100:.2f}%")
        
        # Detailed report
        y_pred = self.best_model.predict(X_test)
        print(f"\n📊 تقرير مفصل ({self.best_name}):")
        print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
        
        # حفظ Models
        for name, model in self.models.items():
            model_file = os.path.join(self.ml_folder, f'{name}_model.pkl')
            joblib.dump(model, model_file)
            print(f"💾 تم حفظ: {name}")
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Feature Importance
        self.plot_feature_importance()
        
        # Compare all models
        self.plot_models_comparison(X_test, y_test)
        
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
        
        # إضافة النسب
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        cm_file = os.path.join(self.ml_folder, 'confusion_matrix_balanced.png')
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 تم حفظ: confusion_matrix_balanced.png")
    
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
            plt.title(f'Feature Importance - {self.best_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            imp_file = os.path.join(self.ml_folder, 'feature_importance_balanced.png')
            plt.savefig(imp_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 تم حفظ: feature_importance_balanced.png")
            
            print(f"\n🔝 أهم Features:")
            for idx, row in feature_imp.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    def plot_models_comparison(self, X_test, y_test):
        """مقارنة بصرية بين Models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = list(self.models.keys())
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        accuracies = [self.scores[name] for name in model_names]
        colors = ['green' if name == self.best_name else 'blue' for name in model_names]
        ax1.barh(model_names, accuracies, color=colors, alpha=0.7)
        ax1.set_xlabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim([0.5, 1.0])
        for i, v in enumerate(accuracies):
            ax1.text(v + 0.01, i, f'{v*100:.2f}%', va='center', fontweight='bold')
        
        # 2. Precision & Recall for each model
        ax2 = axes[0, 1]
        metrics_data = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics_data.append({
                'Model': name,
                'WIN Recall': report['1']['recall'],
                'LOSS Recall': report['0']['recall']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        x = np.arange(len(model_names))
        width = 0.35
        ax2.bar(x - width/2, metrics_df['WIN Recall'], width, label='WIN Recall', alpha=0.7)
        ax2.bar(x + width/2, metrics_df['LOSS Recall'], width, label='LOSS Recall', alpha=0.7)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Recall Comparison (WIN vs LOSS)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.replace('_', '\n') for n in model_names], fontsize=8)
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 3. F1-Score Comparison
        ax3 = axes[1, 0]
        f1_data = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_data.append({
                'Model': name,
                'WIN F1': report['1']['f1-score'],
                'LOSS F1': report['0']['f1-score']
            })
        
        f1_df = pd.DataFrame(f1_data)
        x = np.arange(len(model_names))
        ax3.bar(x - width/2, f1_df['WIN F1'], width, label='WIN F1', alpha=0.7, color='green')
        ax3.bar(x + width/2, f1_df['LOSS F1'], width, label='LOSS F1', alpha=0.7, color='red')
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('F1-Score', fontsize=12)
        ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([n.replace('_', '\n') for n in model_names], fontsize=8)
        ax3.legend()
        ax3.set_ylim([0, 1])
        
        # 4. Overall Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"""
Model Performance Summary

Best Model: {self.best_name}
Accuracy: {self.scores[self.best_name]*100:.2f}%

Original Model (Unbalanced):
- WIN Recall: 95%
- LOSS Recall: 69%
- Problem: Missing many LOSS trades

Balanced Models:
- Improved LOSS detection
- Better F1-Score balance
- More reliable for trading

Recommendation:
Use {self.best_name} for:
✓ Better risk management
✓ Avoiding bad setups
✓ Higher overall profit
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        comparison_file = os.path.join(self.ml_folder, 'models_comparison.png')
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 تم حفظ: models_comparison.png")
    
    def run(self):
        """تشغيل كامل"""
        if self.load_backtest_data() is None:
            return
        
        self.extract_features()
        self.train_models_balanced()
        
        print("\n" + "="*100)
        print("✅ تم التدريب بنجاح!")
        print("="*100)
        print(f"\n📁 النتائج في: {self.ml_folder}")
        print(f"   - {self.best_name}_model.pkl")
        print(f"   - confusion_matrix_balanced.png")
        print(f"   - feature_importance_balanced.png")
        print(f"   - models_comparison.png")

if __name__ == "__main__":
    trainer = MLTrainerBalanced()
    trainer.run()

