"""
🎯 ICT ML Trading Dashboard
لوحة تحكم متكاملة للتداول الآلي
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
import threading
import time

app = Flask(__name__)

# متغيرات عامة
SYSTEM_STATUS = {
    'is_connected': False,
    'is_trading': False,
    'account_type': None,
    'account_balance': 0.0,
    'daily_pnl': 0.0,
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'current_model': None,
    'last_update': None
}

ACTIVE_TRADES = []
TRADE_HISTORY = []
PENDING_SIGNALS = []

# مسارات
BASE_DIR = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder'
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
SIGNALS_FILE = r'C:\Users\hemi_\Downloads\ICT_Core_System\signals.csv'
DECISIONS_FILE = r'C:\Users\hemi_\Downloads\ICT_Core_System\decisions.csv'

# ML Model
current_model = None
model_features = ['type_num', 'strength', 'risk', 'target', 'rr', 
                  'priority', 'session_num', 'hour', 'day_of_week']

# ===========================
# ML Functions
# ===========================

def load_model(model_name):
    """تحميل ML Model"""
    global current_model
    
    model_path = os.path.join(ML_MODELS_DIR, model_name)
    
    if not os.path.exists(model_path):
        return False, f"Model not found: {model_name}"
    
    try:
        current_model = joblib.load(model_path)
        SYSTEM_STATUS['current_model'] = model_name
        return True, f"Model loaded: {model_name}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def extract_features(signal):
    """استخراج features من signal"""
    risk = abs(signal['entry'] - signal['sl'])
    target = abs(signal['tp'] - signal['entry'])
    rr = target / risk if risk > 0 else 0
    
    type_num = 1 if signal['type'].upper() == 'BUY' else 0
    session_map = {'London': 2, 'london': 2, 'NY_AM': 1, 'ny_am': 1, 'NY_PM': 0, 'ny_pm': 0}
    session_num = session_map.get(signal.get('session', 'London'), 0)
    
    try:
        ts = pd.to_datetime(signal['timestamp'])
        hour = ts.hour
        day_of_week = ts.weekday()
    except:
        hour = 8
        day_of_week = 1
    
    priority = signal.get('priority', 10)
    
    features = [
        type_num,
        signal['ob_strength'],
        risk,
        target,
        rr,
        priority,
        session_num,
        hour,
        day_of_week
    ]
    
    return features

def evaluate_signal(signal):
    """تقييم signal باستخدام ML"""
    if current_model is None:
        return None, "No model loaded"
    
    try:
        features = extract_features(signal)
        probability = current_model.predict_proba([features])[0][1]
        
        # قرار
        if probability >= 0.70:
            decision = 'TAKE'
            reason = f"High confidence ({probability*100:.1f}%)"
        else:
            decision = 'SKIP'
            reason = f"Low confidence ({probability*100:.1f}%)"
        
        # فحوصات إضافية
        if SYSTEM_STATUS['total_trades'] >= 3:
            decision = 'SKIP'
            reason = "Max daily trades reached (3)"
        
        if SYSTEM_STATUS['daily_pnl'] <= -1000:
            decision = 'SKIP'
            reason = "Max daily loss reached ($1000)"
        
        return {
            'probability': probability,
            'decision': decision,
            'reason': reason
        }, None
    
    except Exception as e:
        return None, str(e)

# ===========================
# Signal Processing
# ===========================

def process_signals():
    """معالجة الإشارات من signals.csv"""
    if not SYSTEM_STATUS['is_trading']:
        return
    
    if not os.path.exists(SIGNALS_FILE):
        return
    
    try:
        signals_df = pd.read_csv(SIGNALS_FILE)
        
        for idx, row in signals_df.iterrows():
            signal_id = row.get('signal_id', idx)
            
            # تخطي المعالجة سابقاً
            processed_ids = [s['signal_id'] for s in TRADE_HISTORY + PENDING_SIGNALS]
            if signal_id in processed_ids:
                continue
            
            # تقييم
            signal = row.to_dict()
            result, error = evaluate_signal(signal)
            
            if error:
                continue
            
            # حفظ
            signal_data = {
                'signal_id': signal_id,
                'timestamp': row.get('timestamp', datetime.now().isoformat()),
                'type': row['type'],
                'entry': row['entry'],
                'sl': row['sl'],
                'tp': row['tp'],
                'ob_strength': row['ob_strength'],
                'session': row.get('session', 'Unknown'),
                'probability': result['probability'],
                'decision': result['decision'],
                'reason': result['reason'],
                'status': 'pending',
                'processed_at': datetime.now().isoformat()
            }
            
            if result['decision'] == 'TAKE':
                PENDING_SIGNALS.append(signal_data)
                # هنا يمكن إرسال أمر للمنصة
            else:
                signal_data['status'] = 'skipped'
                TRADE_HISTORY.append(signal_data)
    
    except Exception as e:
        print(f"Error processing signals: {e}")

def monitor_signals_loop():
    """حلقة مراقبة مستمرة"""
    while True:
        if SYSTEM_STATUS['is_trading']:
            process_signals()
        time.sleep(5)  # كل 5 ثواني

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """حالة النظام"""
    SYSTEM_STATUS['last_update'] = datetime.now().isoformat()
    return jsonify(SYSTEM_STATUS)

@app.route('/api/models')
def get_models():
    """قائمة ML Models المتاحة"""
    if not os.path.exists(ML_MODELS_DIR):
        return jsonify({'models': []})
    
    models = [f for f in os.listdir(ML_MODELS_DIR) if f.endswith('.pkl')]
    return jsonify({'models': models})

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """تحميل ML Model"""
    data = request.json
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'message': 'No model specified'})
    
    success, message = load_model(model_name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """الاتصال بالحساب"""
    data = request.json
    account_type = data.get('account_type')  # 'sim', 'funded', 'combine'
    
    # محاكاة الاتصال (في الواقع، هنا تتصل بـ NinjaTrader API)
    SYSTEM_STATUS['is_connected'] = True
    SYSTEM_STATUS['account_type'] = account_type
    SYSTEM_STATUS['account_balance'] = 50000.0 if account_type == 'sim' else 25000.0
    SYSTEM_STATUS['daily_pnl'] = 0.0
    SYSTEM_STATUS['total_trades'] = 0
    SYSTEM_STATUS['winning_trades'] = 0
    SYSTEM_STATUS['losing_trades'] = 0
    
    return jsonify({
        'success': True, 
        'message': f'Connected to {account_type} account',
        'account_info': {
            'type': account_type,
            'balance': SYSTEM_STATUS['account_balance'],
            'status': 'Active'
        }
    })

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """قطع الاتصال"""
    SYSTEM_STATUS['is_connected'] = False
    SYSTEM_STATUS['is_trading'] = False
    SYSTEM_STATUS['account_type'] = None
    
    return jsonify({'success': True, 'message': 'Disconnected'})

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """بدء التداول الآلي"""
    if not SYSTEM_STATUS['is_connected']:
        return jsonify({'success': False, 'message': 'Not connected to account'})
    
    if current_model is None:
        return jsonify({'success': False, 'message': 'No ML model loaded'})
    
    SYSTEM_STATUS['is_trading'] = True
    return jsonify({'success': True, 'message': 'Auto-trading started'})

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    """إيقاف التداول الآلي"""
    SYSTEM_STATUS['is_trading'] = False
    return jsonify({'success': True, 'message': 'Auto-trading stopped'})

@app.route('/api/active_trades')
def get_active_trades():
    """الصفقات النشطة"""
    return jsonify({'trades': ACTIVE_TRADES})

@app.route('/api/pending_signals')
def get_pending_signals():
    """الإشارات المعلقة"""
    return jsonify({'signals': PENDING_SIGNALS})

@app.route('/api/trade_history')
def get_trade_history():
    """سجل الصفقات"""
    return jsonify({'history': TRADE_HISTORY[-50:]})  # آخر 50 صفقة

@app.route('/api/stats')
def get_stats():
    """إحصائيات التداول"""
    total = SYSTEM_STATUS['total_trades']
    wins = SYSTEM_STATUS['winning_trades']
    losses = SYSTEM_STATUS['losing_trades']
    
    win_rate = (wins / total * 100) if total > 0 else 0
    
    return jsonify({
        'total_trades': total,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': round(win_rate, 2),
        'daily_pnl': SYSTEM_STATUS['daily_pnl'],
        'account_balance': SYSTEM_STATUS['account_balance']
    })

@app.route('/api/test_signal', methods=['POST'])
def api_test_signal():
    """اختبار signal يدوياً"""
    data = request.json
    
    signal = {
        'type': data['type'],
        'entry': float(data['entry']),
        'sl': float(data['sl']),
        'tp': float(data['tp']),
        'ob_strength': float(data['ob_strength']),
        'session': data['session'],
        'timestamp': datetime.now().isoformat()
    }
    
    result, error = evaluate_signal(signal)
    
    if error:
        return jsonify({'success': False, 'message': error})
    
    return jsonify({
        'success': True,
        'probability': result['probability'],
        'decision': result['decision'],
        'reason': result['reason']
    })

# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("="*80)
    print("🎯 ICT ML Trading Dashboard")
    print("="*80)
    print()
    print("🌐 Starting web server...")
    print("📍 URL: http://localhost:5000")
    print()
    print("⚙️  Features:")
    print("   ✅ ML Model Selection")
    print("   ✅ Account Connection (Sim/Funded)")
    print("   ✅ Auto-Trading")
    print("   ✅ Live Monitoring")
    print("   ✅ Trade History")
    print("   ✅ Statistics")
    print()
    print("💡 Press Ctrl+C to stop")
    print("="*80)
    print()
    
    # بدء حلقة مراقبة الإشارات
    monitor_thread = threading.Thread(target=monitor_signals_loop, daemon=True)
    monitor_thread.start()
    
    # بدء Flask
    app.run(host='0.0.0.0', port=5000, debug=False)

