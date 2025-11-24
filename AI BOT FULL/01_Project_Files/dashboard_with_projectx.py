"""
🎯 ICT ML Trading Dashboard مع ProjectX API
Dashboard متكامل مع ربط حقيقي بـ ProjectX
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
import asyncio

# استيراد ProjectX Connector
from projectx_connector import ProjectXConnector, PROJECTX_AVAILABLE

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
    'last_update': None,
    'projectx_environment': None
}

ACTIVE_TRADES = []
TRADE_HISTORY = []
PENDING_SIGNALS = []

# ProjectX Connector
projectx = None
event_loop = None

# ML Model
current_model = None
model_features = ['type_num', 'strength', 'risk', 'target', 'rr', 
                  'priority', 'session_num', 'hour', 'day_of_week']

# مسارات
BASE_DIR = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder'
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
SIGNALS_FILE = r'C:\Users\hemi_\Downloads\ICT_Core_System\signals.csv'
DECISIONS_FILE = r'C:\Users\hemi_\Downloads\ICT_Core_System\decisions.csv'
PROJECTX_CONFIG_FILE = r'C:\Users\hemi_\Downloads\ICT_Core_System\projectx_config.json'

# ===========================
# ProjectX Functions
# ===========================

def init_event_loop():
    """تهيئة Event Loop للـ async functions"""
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

def run_async(coro):
    """تشغيل async function"""
    global event_loop
    if event_loop is None:
        init_event_loop()
    return event_loop.run_until_complete(coro)

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
        
        if probability >= 0.70:
            decision = 'TAKE'
            reason = f"High confidence ({probability*100:.1f}%)"
        else:
            decision = 'SKIP'
            reason = f"Low confidence ({probability*100:.1f}%)"
        
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
            
            processed_ids = [s['signal_id'] for s in TRADE_HISTORY + PENDING_SIGNALS]
            if signal_id in processed_ids:
                continue
            
            signal = row.to_dict()
            result, error = evaluate_signal(signal)
            
            if error:
                continue
            
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
                
                # إذا كان متصل بـ ProjectX، نفذ الصفقة
                if SYSTEM_STATUS['is_connected'] and projectx:
                    execute_trade_on_projectx(signal_data)
            else:
                signal_data['status'] = 'skipped'
                TRADE_HISTORY.append(signal_data)
    
    except Exception as e:
        print(f"Error processing signals: {e}")

def execute_trade_on_projectx(signal):
    """تنفيذ صفقة على ProjectX"""
    global projectx
    
    try:
        # تنفيذ الأمر
        symbol = 'ES'  # E-mini S&P 500
        side = signal['type']
        quantity = 1
        
        print(f"\n🚀 Executing trade on ProjectX:")
        print(f"   Signal ID: {signal['signal_id']}")
        print(f"   {side} {quantity} {symbol} @ {signal['entry']}")
        
        # وضع أمر السوق
        order_id = run_async(
            projectx.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
        )
        
        if order_id[0]:
            signal['order_id'] = order_id[0]
            signal['status'] = 'executed'
            ACTIVE_TRADES.append(signal)
            SYSTEM_STATUS['total_trades'] += 1
            
            print(f"✅ Trade executed: {order_id[0]}")
        else:
            print(f"❌ Trade failed: {order_id[1]}")
    
    except Exception as e:
        print(f"❌ Error executing trade: {e}")

def monitor_signals_loop():
    """حلقة مراقبة مستمرة"""
    while True:
        if SYSTEM_STATUS['is_trading']:
            process_signals()
        time.sleep(5)

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('dashboard_projectx.html')

@app.route('/api/status')
def get_status():
    """حالة النظام"""
    SYSTEM_STATUS['last_update'] = datetime.now().isoformat()
    SYSTEM_STATUS['projectx_available'] = PROJECTX_AVAILABLE
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

@app.route('/api/projectx_connect', methods=['POST'])
def api_projectx_connect():
    """الاتصال بـ ProjectX API"""
    global projectx
    
    if not PROJECTX_AVAILABLE:
        return jsonify({
            'success': False, 
            'message': 'ProjectX API not installed. Run: pip install projectx-api'
        })
    
    data = request.json
    username = data.get('username')
    api_key = data.get('api_key')
    environment = data.get('environment', 'TOPSTEP_X')
    
    if not username or not api_key:
        return jsonify({'success': False, 'message': 'Username and API Key required'})
    
    try:
        # إنشاء connector
        projectx = ProjectXConnector(
            username=username,
            api_key=api_key,
            environment=environment
        )
        
        # الاتصال
        success, message = run_async(projectx.connect())
        
        if success:
            SYSTEM_STATUS['is_connected'] = True
            SYSTEM_STATUS['projectx_environment'] = environment
            
            # الحصول على معلومات الحساب
            account_info, _ = run_async(projectx.get_account_info())
            
            if account_info:
                SYSTEM_STATUS['account_type'] = account_info.get('account_type', 'Unknown')
                SYSTEM_STATUS['account_balance'] = account_info.get('balance', 0)
                
                return jsonify({
                    'success': True,
                    'message': 'Connected to ProjectX',
                    'account_info': account_info
                })
        
        return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/projectx_disconnect', methods=['POST'])
def api_projectx_disconnect():
    """قطع الاتصال من ProjectX"""
    global projectx
    
    if projectx:
        try:
            success, message = run_async(projectx.disconnect())
            
            if success:
                SYSTEM_STATUS['is_connected'] = False
                SYSTEM_STATUS['is_trading'] = False
                projectx = None
                
                return jsonify({'success': True, 'message': 'Disconnected from ProjectX'})
            
            return jsonify({'success': False, 'message': message})
        
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return jsonify({'success': True, 'message': 'Already disconnected'})

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """بدء التداول الآلي"""
    if not SYSTEM_STATUS['is_connected']:
        return jsonify({'success': False, 'message': 'Not connected to ProjectX'})
    
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
    return jsonify({'history': TRADE_HISTORY[-50:]})

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

@app.route('/api/upload_model', methods=['POST'])
def api_upload_model():
    """رفع ML Model"""
    if 'model' not in request.files:
        return jsonify({'success': False, 'message': 'No model file uploaded'})
    
    file = request.files['model']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if not file.filename.endswith('.pkl'):
        return jsonify({'success': False, 'message': 'Only .pkl files allowed'})
    
    try:
        # حفظ الملف مؤقتاً
        temp_path = os.path.join(ML_MODELS_DIR, 'uploaded_model.pkl')
        file.save(temp_path)
        
        # تحميل Model للتحقق
        success, message = load_model('uploaded_model.pkl')
        
        if success:
            # إعادة تسمية الملف
            final_path = os.path.join(ML_MODELS_DIR, file.filename)
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            # إعادة تحميل بالاسم الجديد
            load_model(file.filename)
            
            return jsonify({
                'success': True,
                'message': f'Model uploaded and loaded: {file.filename}'
            })
        else:
            # حذف الملف إذا فشل التحميل
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

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
    print("🎯 ICT ML Trading Dashboard with ProjectX API")
    print("="*80)
    print()
    
    if not PROJECTX_AVAILABLE:
        print("⚠️  ProjectX API not installed!")
        print("📦 Install: pip install projectx-api")
        print("📚 Docs: https://gateway.docs.projectx.com/docs/intro")
        print()
    else:
        print("✅ ProjectX API available")
        print()
    
    print("🌐 Starting web server...")
    print("📍 URL: http://localhost:5000")
    print()
    print("⚙️  Features:")
    print("   ✅ ML Model Selection")
    print("   ✅ ProjectX API Connection (Real)")
    print("   ✅ TopStep, Tradeify, Funding Futures, E8X, FXIFY")
    print("   ✅ Auto-Trading")
    print("   ✅ Live Monitoring")
    print("   ✅ Trade Execution")
    print()
    print("💡 Press Ctrl+C to stop")
    print("="*80)
    print()
    
    # تهيئة Event Loop
    init_event_loop()
    
    # بدء حلقة مراقبة الإشارات
    monitor_thread = threading.Thread(target=monitor_signals_loop, daemon=True)
    monitor_thread.start()
    
    # بدء Flask
    app.run(host='0.0.0.0', port=5000, debug=False)

