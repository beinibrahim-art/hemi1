"""
🚀 REAL Trading Platform
منصة تداول حقيقية متكاملة مع ProjectX API
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import asyncio
import threading
import json
import os
import glob
from datetime import datetime, timedelta
from collections import deque
import joblib
import pandas as pd
import numpy as np

# استيراد ProjectX API الحقيقي
try:
    from projectx_api import ProjectXClient, Environment, LoginKeyCredentials
    PROJECTX_AVAILABLE = True
    print("✅ ProjectX API loaded successfully")
except ImportError as e:
    PROJECTX_AVAILABLE = False
    print(f"❌ ProjectX API not available: {e}")

# استيراد محرك التداول
try:
    from projectx_api_complete import ProjectXAPI
    print("✅ Complete ProjectX API loaded successfully")
except ImportError as e:
    print(f"❌ ProjectX API not available: {e}")
    
try:
    from projectx_trading_engine import ProjectXTradingEngine
    print("✅ Trading Engine (legacy) loaded successfully")
except ImportError as e:
    print(f"❌ Trading Engine not available: {e}")

CACHE_BASE_DIR = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder'
LIVE_DATA_DIR = os.path.join(CACHE_BASE_DIR, 'live_data')
LIVE_BARS_DIR = os.path.join(LIVE_DATA_DIR, 'bars')
LIVE_SIGNALS_DIR = os.path.join(LIVE_DATA_DIR, 'signals')

# مسار ملف signals.csv الرئيسي (للتكامل مع المؤشر)
MAIN_SIGNALS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'signals.csv')

for path in (LIVE_DATA_DIR, LIVE_BARS_DIR, LIVE_SIGNALS_DIR):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ===========================
# Global State
# ===========================
class TradingSystem:
    def __init__(self):
        self.client = None
        self.session_token = None      # Session token من ProjectX
        self.api_endpoint = None       # API endpoint URL
        self.is_connected = False
        self.is_trading = False
        self.account_info = {}
        self.available_accounts = []   # الحسابات المتاحة
        self.selected_account_id = None  # الحساب المختار
        self.trading_engine = None     # محرك التداول الحقيقي
        self.selected_contract = "CON.F.US.EP.U25"  # ES Contract (default)
        self.positions = []
        self.orders = []
        self.market_data = {}
        self.ml_model = None
        self.model_name = None
        
        # Contract/meta info
        self.contract_details = {}
        self.daily_trades = 0
        self.daily_trade_date = None
        self.processed_setup_ids = set()
        self.auto_logs = deque(maxlen=200)
        self.executed_trades = deque(maxlen=200)
        self.skipped_trades = deque(maxlen=200)
        
        # CSV tracking
        self.daily_signals_file = None
        
        # Trading stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'daily_pnl': 0.0,
            'total_pnl': 0.0
        }
        
        # Event loop for async operations
        self.loop = None
        self.loop_thread = None
        
        # Auto Trading thread
        self.auto_trading_thread = None
        self.market_stream_thread = None
        self.last_trade_time = None
        self.max_trades_per_day = 3
        self.max_daily_loss = 1000.0
        self.min_ml_probability = 0.70
        self.tick_size_fallback = 0.25
        
        # Setup detection sensitivity (lower = more sensitive)
        self.setup_threshold_multiplier = 1.5  # Default: 1.5x std deviation
        self.min_candles_for_setup = 20  # Minimum candles needed
        self.min_momentum_ticks = 6  # Minimum ticks for momentum
        self.debug_setup_detection = False  # Enable detailed logging for setup detection
    
    def settings_payload(self):
        return {
            'min_ml_probability': round(self.min_ml_probability * 100, 2),
            'max_trades_per_day': self.max_trades_per_day,
            'max_daily_loss': self.max_daily_loss,
            'setup_threshold_multiplier': round(self.setup_threshold_multiplier, 2),
            'min_candles_for_setup': self.min_candles_for_setup,
            'min_momentum_ticks': self.min_momentum_ticks,
            'debug_setup_detection': self.debug_setup_detection
        }
    
    def update_settings(self, min_probability=None, max_trades=None, max_loss=None,
                       threshold_multiplier=None, min_candles=None, min_momentum=None,
                       debug_setup=None):
        changes = []
        warnings = []
        
        if min_probability is not None:
            original = min_probability
            self.min_ml_probability = min(max(min_probability, 0.4), 0.99)
            if abs(original - self.min_ml_probability) > 0.001:
                warnings.append(f"MinProb adjusted from {original*100:.1f}% to {self.min_ml_probability*100:.1f}% (range: 40-99%)")
            changes.append(f"MinProb={self.min_ml_probability*100:.1f}%")
        
        if max_trades is not None:
            original = max_trades
            self.max_trades_per_day = max(1, int(max_trades))
            if original != self.max_trades_per_day:
                warnings.append(f"MaxTrades adjusted from {original} to {self.max_trades_per_day} (min: 1)")
            changes.append(f"MaxTrades={self.max_trades_per_day}")
        
        if max_loss is not None:
            original = max_loss
            self.max_daily_loss = max(50.0, float(max_loss))
            if abs(original - self.max_daily_loss) > 0.1:
                warnings.append(f"MaxLoss adjusted from ${original:.0f} to ${self.max_daily_loss:.0f} (min: $50)")
            changes.append(f"MaxLoss=${self.max_daily_loss:.0f}")
        
        if threshold_multiplier is not None:
            original = threshold_multiplier
            self.setup_threshold_multiplier = max(0.5, min(float(threshold_multiplier), 3.0))
            if abs(original - self.setup_threshold_multiplier) > 0.01:
                warnings.append(f"ThresholdMult adjusted from {original:.2f}x to {self.setup_threshold_multiplier:.2f}x (range: 0.5-3.0)")
            changes.append(f"ThresholdMult={self.setup_threshold_multiplier:.2f}x")
        
        if min_candles is not None:
            original = min_candles
            self.min_candles_for_setup = max(10, int(min_candles))
            if original != self.min_candles_for_setup:
                warnings.append(f"MinCandles adjusted from {original} to {self.min_candles_for_setup} (min: 10)")
            changes.append(f"MinCandles={self.min_candles_for_setup}")
        
        if min_momentum is not None:
            original = min_momentum
            self.min_momentum_ticks = max(2, int(min_momentum))
            if original != self.min_momentum_ticks:
                warnings.append(f"MinMomentum adjusted from {original} to {self.min_momentum_ticks} ticks (min: 2)")
            changes.append(f"MinMomentum={self.min_momentum_ticks} ticks")
        
        if debug_setup is not None:
            self.debug_setup_detection = bool(debug_setup)
            changes.append(f"DebugSetupDetection={self.debug_setup_detection}")
        
        if changes:
            self.add_log(f"Settings updated: {', '.join(changes)}", "INFO")
            if warnings:
                for warn in warnings:
                    self.add_log(f"⚠️ {warn}", "WARN")
        
    def start_event_loop(self):
        """بدء Event Loop للعمليات الغير متزامنة"""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            print("✅ Event loop started")
    
    def start_auto_trading_loop(self):
        """بدء Auto Trading Loop"""
        if self.auto_trading_thread is None or not self.auto_trading_thread.is_alive():
            self.auto_trading_thread = threading.Thread(target=self._auto_trading_loop, daemon=True)
            self.auto_trading_thread.start()
            print("✅ Auto Trading loop started")
    
    def add_log(self, message, level="INFO"):
        """تسجيل رسائل النظام لإظهارها في الواجهة"""
        entry = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message
        }
        self.auto_logs.append(entry)
        print(f"[{entry['time']}] [{level}] {message}")
        try:
            socketio.emit('log', entry)
        except Exception:
            pass

    def _save_signal_to_csv(self, trade_summary, status='EXECUTED'):
        """حفظ إشارة في ملف CSV يومي + signals.csv الرئيسي"""
        try:
            today = datetime.utcnow().date()
            
            # 1. حفظ في ملف يومي (signals_YYYYMMDD.csv)
            daily_csv_file = os.path.join(LIVE_SIGNALS_DIR, f"signals_{today.strftime('%Y%m%d')}.csv")
            
            # 2. حفظ في ملف signals.csv الرئيسي (للتكامل مع المؤشر)
            main_csv_file = MAIN_SIGNALS_FILE
            
            print(f"📂 [CSV] Main file path: {main_csv_file}")
            print(f"📂 [CSV] Daily file path: {daily_csv_file}")
            print(f"📂 [CSV] Main file exists: {os.path.exists(main_csv_file)}")
            if os.path.exists(main_csv_file):
                print(f"📂 [CSV] Main file size: {os.path.getsize(main_csv_file)} bytes")
            
            # إعداد البيانات
            signal_data = {
                'timestamp': trade_summary.get('timestamp', datetime.utcnow().isoformat()),
                'status': status,  # EXECUTED or SKIPPED
                'type': trade_summary.get('type', 'UNKNOWN'),
                'entry': trade_summary.get('entry', 0),
                'sl': trade_summary.get('sl', 0),
                'tp': trade_summary.get('tp', 0),
                'probability': trade_summary.get('probability', 0),
                'reason': trade_summary.get('reason', ''),
                'contract': trade_summary.get('contract', self.selected_contract or 'N/A'),
                'strength': trade_summary.get('strength', 0),
                'risk': trade_summary.get('risk', 0),
                'target': trade_summary.get('target', 0),
                'rr': trade_summary.get('rr', 0),
                'result': trade_summary.get('result', 'PENDING'),
                'pnl': trade_summary.get('pnl', 0),
                'order_id': trade_summary.get('order_id', ''),
                'session': trade_summary.get('session', ''),
                'saved_at': datetime.utcnow().isoformat()
            }
            
            df = pd.DataFrame([signal_data])
            
            print(f"📊 [CSV] Signal data: {signal_data}")
            print(f"📊 [CSV] DataFrame shape: {df.shape}")
            
            # حفظ في الملف اليومي
            try:
                file_exists = os.path.exists(daily_csv_file)
                file_size_before = os.path.getsize(daily_csv_file) if file_exists else 0
                header_needed = not file_exists or file_size_before == 0
                
                df.to_csv(daily_csv_file, mode='a', header=header_needed, index=False, encoding='utf-8')
                
                file_size_after = os.path.getsize(daily_csv_file) if os.path.exists(daily_csv_file) else 0
                print(f"✅ Daily CSV saved: {daily_csv_file} ({file_size_before} -> {file_size_after} bytes)")
            except Exception as e:
                import traceback
                print(f"⚠️ Daily CSV save error: {e}")
                traceback.print_exc()
            
            # حفظ في signals.csv الرئيسي (يتم تحديثه دائماً)
            try:
                # التأكد من وجود المجلد
                main_csv_dir = os.path.dirname(main_csv_file)
                if main_csv_dir and not os.path.exists(main_csv_dir):
                    os.makedirs(main_csv_dir, exist_ok=True)
                    print(f"📁 Created directory: {main_csv_dir}")
                
                # التحقق من وجود الملف وحجمه قبل الكتابة
                file_exists = os.path.exists(main_csv_file)
                file_size_before = os.path.getsize(main_csv_file) if file_exists else 0
                
                # إذا كان الملف موجوداً لكن فارغاً، نحتاج header
                # إذا كان الملف غير موجود، نحتاج header
                header_needed = not file_exists or file_size_before == 0
                
                print(f"📝 [CSV] Writing to {main_csv_file}")
                print(f"📝 [CSV] File exists: {file_exists}, Size before: {file_size_before}, Header needed: {header_needed}")
                
                # حفظ البيانات
                df.to_csv(main_csv_file, mode='a', header=header_needed, index=False, encoding='utf-8')
                
                # التحقق من الحفظ بعد الكتابة
                file_size_after = os.path.getsize(main_csv_file) if os.path.exists(main_csv_file) else 0
                
                if file_size_after > file_size_before:
                    print(f"✅ Signal saved to {main_csv_file}: {status} {signal_data['type']} @ {signal_data['entry']}")
                    print(f"   Size: {file_size_before} -> {file_size_after} bytes (+{file_size_after - file_size_before})")
                    
                    # محاولة قراءة الملف للتحقق من البيانات
                    try:
                        test_df = pd.read_csv(main_csv_file)
                        print(f"📖 File verified: {len(test_df)} rows, columns: {list(test_df.columns)}")
                    except Exception as read_err:
                        print(f"⚠️ Cannot read file after save: {read_err}")
                else:
                    print(f"⚠️ File size didn't increase: {main_csv_file}")
                    print(f"   Before: {file_size_before} bytes, After: {file_size_after} bytes")
            except Exception as e:
                import traceback
                print(f"⚠️ Main CSV save error: {e}")
                print(f"   File path: {main_csv_file}")
                traceback.print_exc()
            
        except Exception as e:
            print(f"⚠️ CSV save error: {e}")
            import traceback
            traceback.print_exc()
    
    def record_trade_execution(self, trade_summary):
        """تخزين الصفقات المنفذة وبثها للواجهة"""
        try:
            trade_summary.setdefault('timestamp', datetime.utcnow().isoformat())
            self.executed_trades.appendleft(trade_summary)
            socketio.emit('executed_trades', list(self.executed_trades))
            # حفظ في CSV
            print(f"📝 [CSV] Recording EXECUTED trade: {trade_summary.get('type')} @ {trade_summary.get('entry')}")
            self._save_signal_to_csv(trade_summary, status='EXECUTED')
        except Exception as e:
            import traceback
            print(f"⚠️ Trade record error: {e}")
            traceback.print_exc()
    
    def record_skipped_trade(self, trade_summary):
        """تخزين الصفقات التي تم تجاهلها مع السبب"""
        try:
            trade_summary.setdefault('timestamp', datetime.utcnow().isoformat())
            self.skipped_trades.appendleft(trade_summary)
            socketio.emit('skipped_trades', list(self.skipped_trades))
            # حفظ في CSV
            print(f"📝 [CSV] Recording SKIPPED trade: {trade_summary.get('type')} @ {trade_summary.get('entry')}")
            self._save_signal_to_csv(trade_summary, status='SKIPPED')
        except Exception as e:
            import traceback
            print(f"⚠️ Skipped record error: {e}")
            traceback.print_exc()
    
    def reset_daily_counters(self):
        """إعادة تعيين العدادات اليومية"""
        today = datetime.utcnow().date()
        if self.daily_trade_date != today:
            self.daily_trade_date = today
            self.daily_trades = 0
            self.stats['daily_pnl'] = 0.0
            self.add_log("Daily counters reset", "INFO")
    
    def refresh_contract_details(self, contract_id=None):
        """جلب تفاصيل العقد (tickSize, tickValue)"""
        if contract_id is None:
            contract_id = self.selected_contract
        if not contract_id or not self.is_connected or not self.session_token or not self.api_endpoint:
            return
        
        try:
            import requests
            url = f"{self.api_endpoint}/api/Contract/searchById"
            headers = {
                'accept': 'text/plain',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.session_token}'
            }
            response = requests.post(url, headers=headers, json={"contractId": contract_id}, timeout=10)
            result = response.json()
            if result.get('success') and result.get('contract'):
                self.contract_details = result['contract']
                tick = self.contract_details.get('tickSize', self.tick_size_fallback)
                self.add_log(f"Contract {contract_id} specs loaded (tickSize={tick})", "INFO")
            else:
                self.add_log(f"Unable to load contract details: {result.get('errorMessage')}", "WARN")
        except Exception as e:
            self.add_log(f"Contract detail error: {e}", "ERROR")
    
    def get_tick_size(self):
        """إرجاع قيمة tickSize للعقد الحالي"""
        if self.contract_details and self.contract_details.get('tickSize'):
            try:
                return float(self.contract_details.get('tickSize'))
            except (TypeError, ValueError):
                return self.tick_size_fallback
        return self.tick_size_fallback
    
    def _auto_trading_loop(self):
        """Auto Trading Loop - يفحص البيانات ويقيمها بالML وينفذ الصفقات"""
        import time
        import requests
        import pandas as pd
        import numpy as np
        
        while True:
            try:
                if not self.is_trading:
                    time.sleep(5)
                    continue
                
                if not self.is_connected or not self.ml_model or not self.selected_account_id:
                    self.add_log("Auto trading paused - Missing connection/model/account", "WARN")
                    time.sleep(10)
                    continue
                
                self.reset_daily_counters()
                
                self.add_log("Scanning live data for setups...", "INFO")
                df = fetch_recent_bars(self.selected_contract, use_live=False)
                
                if df is None or len(df) < self.min_candles_for_setup:
                    self.add_log(f"Insufficient candles for analysis: {len(df) if df is not None else 0} < {self.min_candles_for_setup}", "WARN")
                    time.sleep(20)
                    continue
                
                tick_size = self.get_tick_size()
                broadcast_chart_data(df, self.selected_contract, timeframe='5m', data_source='SIM')
                
                # Enable debug logging if enabled in settings or every 5th scan
                debug_mode = self.debug_setup_detection or (self.stats.get('total_trades', 0) % 5 == 0)
                
                setups = build_auto_setups_from_bars(
                    df, 
                    tick_size=tick_size,
                    threshold_multiplier=self.setup_threshold_multiplier,
                    min_momentum_ticks=self.min_momentum_ticks,
                    min_candles=self.min_candles_for_setup,
                    debug=debug_mode
                )
                
                if not setups:
                    # Log summary only (detailed logging done in build_auto_setups_from_bars if debug=True)
                    if not debug_mode:  # Only log summary if not in debug mode
                        self.add_log(f"No valid setups detected (analyzed {len(df)} candles, threshold={self.setup_threshold_multiplier:.2f}x)", "INFO")
                    time.sleep(25)
                    continue
                
                trade_executed = False
                
                for setup in setups:
                    setup_id = setup.get('id')
                    if setup_id and setup_id in self.processed_setup_ids:
                        continue
                    
                    evaluation, error = evaluate_setup_with_ml(setup)
                    if error:
                        self.add_log(f"ML evaluation error: {error}", "ERROR")
                        continue
                    
                    probability = evaluation.get('probability', 0)
                    decision = evaluation.get('decision')
                    
                    self.add_log(f"Setup {setup.get('reason')} -> ML {probability*100:.1f}%", "INFO")
                    
                    if decision != 'TAKE':
                        trading_system.record_skipped_trade({
                            'type': setup.get('type'),
                            'entry': setup.get('entry'),
                            'sl': setup.get('sl'),
                            'tp': setup.get('tp'),
                            'probability': probability,
                            'reason': evaluation.get('reason'),
                            'source': 'AUTO',
                            'contract': trading_system.selected_contract,
                            'session': setup.get('session'),
                            'priority': setup.get('priority')
                        })
                        if setup_id:
                            self.processed_setup_ids.add(setup_id)
                        continue
                    
                    result = execute_projectx_trade(setup, source='AUTO', probability=probability, reason=evaluation.get('reason'))
                    if result.get('success'):
                        trade_executed = True
                        if setup_id:
                            self.processed_setup_ids.add(setup_id)
                        self.last_trade_time = datetime.utcnow()
                        break
                    else:
                        self.add_log(f"Trade execution failed: {result.get('message')}", "ERROR")
                        if setup_id:
                            self.processed_setup_ids.add(setup_id)
                
                if not trade_executed:
                    self.add_log("No trades executed this cycle", "INFO")
                
                time.sleep(30)
                
            except Exception as e:
                print(f"❌ [AUTO] Error in auto trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(30)
    
    def start_market_stream(self):
        """تشغيل تدفق بيانات السوق لإرسال التحديثات عبر WebSocket"""
        if self.market_stream_thread is None or not self.market_stream_thread.is_alive():
            self.market_stream_thread = threading.Thread(target=self._market_stream_loop, daemon=True)
            self.market_stream_thread.start()
            self.add_log("Market data stream started", "INFO")
    
    def _market_stream_loop(self):
        """حلقة مستمرة لجلب بيانات الشموع وإرسالها عبر WebSocket"""
        import time
        while True:
            try:
                if not self.is_connected or not self.selected_contract:
                    time.sleep(3)
                    continue
                
                df = fetch_recent_bars(self.selected_contract, hours=2, timeframe=5, limit=120, use_live=False)
                if df is not None and len(df) > 0:
                    broadcast_chart_data(df, self.selected_contract, timeframe='5m', data_source='SIM')
                
                time.sleep(5)
            except Exception as e:
                self.add_log(f"Market stream error: {e}", "ERROR")
                time.sleep(5)
    
    def _run_event_loop(self):
        """تشغيل Event Loop"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_async(self, coro):
        """تشغيل coroutine في Event Loop"""
        if self.loop is None:
            self.start_event_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)

# نظام التداول العالمي
trading_system = TradingSystem()


# ===========================
# Helper Functions
# ===========================

def stats_payload():
    stats = dict(trading_system.stats)
    total = stats.get('total_trades', 0)
    wins = stats.get('winning_trades', 0)
    stats['win_rate'] = round((wins / total) * 100, 2) if total > 0 else 0.0
    return stats

def status_payload():
    return {
        'is_connected': trading_system.is_connected,
        'is_trading': trading_system.is_trading,
        'model_loaded': trading_system.ml_model is not None,
        'model_name': trading_system.model_name,
        'projectx_available': PROJECTX_AVAILABLE,
        'last_update': datetime.now().isoformat()
    }

def broadcast_status():
    socketio.emit('status', status_payload())

def broadcast_account_info(account=None):
    payload = account or trading_system.account_info
    if payload:
        socketio.emit('account', payload)

def broadcast_stats():
    socketio.emit('stats', stats_payload())

def broadcast_positions(positions):
    socketio.emit('positions', {'positions': positions})

def broadcast_orders(orders):
    socketio.emit('orders', {'orders': orders})

def broadcast_executed_trades():
    socketio.emit('executed_trades', list(trading_system.executed_trades))

def broadcast_skipped_trades():
    socketio.emit('skipped_trades', list(trading_system.skipped_trades))

def broadcast_settings():
    socketio.emit('settings', trading_system.settings_payload())

def store_live_bars(df, contract_id, timeframe='5m', data_source='SIM'):
    """حفظ بيانات الشموع الحية في CSV"""
    if df is None or df.empty or 'timestamp' not in df.columns:
        return
    
    try:
        df_to_save = df.copy()
        df_to_save['timestamp'] = pd.to_datetime(df_to_save['timestamp'], errors='coerce').dt.tz_localize(None)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col not in df_to_save.columns:
                df_to_save[col] = np.nan
        df_to_save[numeric_cols] = df_to_save[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df_to_save = df_to_save.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'])
        if df_to_save.empty:
            return
        
        df_to_save = df_to_save.sort_values('timestamp').drop_duplicates(subset='timestamp')
        df_to_save['contract'] = contract_id
        df_to_save['timeframe'] = timeframe
        df_to_save['source'] = data_source
        df_to_save['saved_at'] = datetime.utcnow().isoformat()
        
        file_name = f"{contract_id.replace('.', '_')}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        file_path = os.path.join(LIVE_BARS_DIR, file_name)
        header_needed = not os.path.exists(file_path)
        df_to_save[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'contract', 'timeframe', 'source', 'saved_at']]\
            .to_csv(file_path, mode='a', header=header_needed, index=False, encoding='utf-8')
    except Exception as e:
        trading_system.add_log(f"Live data save error: {e}", "WARN")


def broadcast_chart_data(df, contract_id, timeframe='5m', data_source='SIM', order_blocks=None, executed_trades=None, fvgs=None):
    if df is None or df.empty:
        return
    
    df = df.copy()
    if 'timestamp' not in df.columns:
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['timestamp'], inplace=True)
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp')
    
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)
    if df.empty:
        return
    
    store_live_bars(df, contract_id, timeframe, data_source)
    
    candles = []
    for _, row in df.iterrows():
        try:
            if pd.isna(row['open']) or pd.isna(row['high']) or pd.isna(row['low']) or pd.isna(row['close']):
                continue
            candles.append({
                'time': int(row['timestamp'].timestamp()),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
        except Exception:
            continue
    if not candles:
        return
    
    # استخراج Order Blocks إذا لم يتم توفيرها
    if order_blocks is None:
        lookback_size = min(max(200, len(df) // 2), len(df))
        obs = find_order_blocks_from_bars(df, lookback=lookback_size, min_strength=8)
        
        # تحويل Order Blocks إلى تنسيق للشارت
        order_blocks = []
        for ob in obs:
            try:
                ob_index = ob.get('index')
                if ob_index is not None and ob_index < len(df):
                    ob_time = df.iloc[ob_index]['timestamp']
                else:
                    ob_time = ob.get('time')
                    if ob_time is None or (not isinstance(ob_time, pd.Timestamp) and not isinstance(ob_time, str)):
                        matching = df[(df['high'] >= ob['high'] - 0.5) & (df['high'] <= ob['high'] + 0.5) & 
                                     (df['low'] >= ob['low'] - 0.5) & (df['low'] <= ob['low'] + 0.5)]
                        if len(matching) > 0:
                            ob_time = matching.iloc[0]['timestamp']
                        else:
                            ob_time = df.iloc[len(df)//2]['timestamp']
                
                if isinstance(ob_time, str):
                    ob_time = pd.to_datetime(ob_time)
                elif not isinstance(ob_time, pd.Timestamp):
                    ob_time = pd.to_datetime(ob_time)
                
                ob_timestamp = int(ob_time.timestamp()) if isinstance(ob_time, pd.Timestamp) else int(pd.to_datetime(ob_time).timestamp())
                
                order_blocks.append({
                    'time': ob_timestamp,
                    'type': ob['type'],
                    'high': float(ob['high']),
                    'low': float(ob['low']),
                    'strength': float(ob['strength'])
                })
            except Exception:
                continue
    
    # استخراج FVGs إذا لم يتم توفيرها
    if fvgs is None:
        raw_fvgs = find_fvgs_from_bars(df, min_gap=2.0)
        fvgs = []
        for fvg in raw_fvgs:
            try:
                fvg_time = fvg.get('time')
                if fvg_time is None:
                    continue
                
                if isinstance(fvg_time, str):
                    fvg_time = pd.to_datetime(fvg_time)
                elif not isinstance(fvg_time, pd.Timestamp):
                    fvg_time = pd.to_datetime(fvg_time)
                
                fvg_timestamp = int(fvg_time.timestamp()) if isinstance(fvg_time, pd.Timestamp) else int(pd.to_datetime(fvg_time).timestamp())
                
                fvgs.append({
                    'time': fvg_timestamp,
                    'type': fvg['type'],
                    'top': float(fvg['top']),
                    'bottom': float(fvg['bottom']),
                    'size': float(fvg['size'])
                })
            except Exception:
                continue
    
    # جلب executed_trades إذا لم يتم توفيرها
    if executed_trades is None:
        executed_trades = list(trading_system.executed_trades)[:50]
    
    # التأكد من أن البيانات ليست None
    if order_blocks is None:
        order_blocks = []
    if fvgs is None:
        fvgs = []
    if executed_trades is None:
        executed_trades = []
    
    # التأكد من أن القوائم ليست None ويمكن serializeها
    if order_blocks is None:
        order_blocks = []
    if fvgs is None:
        fvgs = []
    if executed_trades is None:
        executed_trades = []
    
    # تحويل إلى list إذا كانت tuple أو نوع آخر
    order_blocks = list(order_blocks) if order_blocks else []
    fvgs = list(fvgs) if fvgs else []
    executed_trades = list(executed_trades) if executed_trades else []
    
    # التأكد من أن القوائم قابلة للتسلسل
    try:
        # Test serialization
        import json
        test_ob = json.dumps(order_blocks, default=str)
        test_fvg = json.dumps(fvgs, default=str)
        test_trades = json.dumps(executed_trades, default=str)
        print(f"🔍 [SOCKET] Serialization test: OB={len(test_ob)}, FVG={len(test_fvg)}, Trades={len(test_trades)}")
    except Exception as ser_error:
        print(f"⚠️  [SOCKET] Serialization test failed: {ser_error}")
        # Reset to empty lists if serialization fails
        order_blocks = []
        fvgs = []
        executed_trades = []
    
    print(f"🔍 [SOCKET] BEFORE payload creation:")
    print(f"   order_blocks: {order_blocks} (type: {type(order_blocks)}, length: {len(order_blocks)})")
    print(f"   fvgs: {fvgs} (type: {type(fvgs)}, length: {len(fvgs)})")
    print(f"   executed_trades: {executed_trades} (type: {type(executed_trades)}, length: {len(executed_trades)})")
    
    # Create payload with explicit field assignment
    payload = {
        'success': True,
        'contract': contract_id,
        'timeframe': timeframe,
        'data_source': data_source,
        'current_price': float(df['close'].iloc[-1]),
        'candles': candles,
        'latest_timestamp': df['timestamp'].iloc[-1].isoformat()
    }
    
    # Add fields explicitly to ensure they're included
    payload['order_blocks'] = order_blocks if order_blocks else []
    payload['fvgs'] = fvgs if fvgs else []
    payload['executed_trades'] = executed_trades if executed_trades else []
    
    # Double-check that fields are present
    if 'order_blocks' not in payload:
        payload['order_blocks'] = []
    if 'fvgs' not in payload:
        payload['fvgs'] = []
    if 'executed_trades' not in payload:
        payload['executed_trades'] = []
    
    print(f"🔍 [SOCKET] AFTER payload creation:")
    print(f"   payload keys: {list(payload.keys())}")
    print(f"   'order_blocks' in payload: {'order_blocks' in payload}")
    print(f"   'fvgs' in payload: {'fvgs' in payload}")
    print(f"   'executed_trades' in payload: {'executed_trades' in payload}")
    print(f"   payload['order_blocks']: {payload.get('order_blocks')} (length: {len(payload.get('order_blocks', []))})")
    print(f"   payload['fvgs']: {payload.get('fvgs')} (length: {len(payload.get('fvgs', []))})")
    print(f"   payload['executed_trades']: {payload.get('executed_trades')} (length: {len(payload.get('executed_trades', []))})")
    
    print(f"📡 [SOCKET] Broadcasting chart: {len(candles)} candles, {len(order_blocks)} Order Blocks, {len(fvgs)} FVGs, {len(executed_trades)} trades")
    
    # Test JSON serialization before emitting
    try:
        import json
        test_json = json.dumps(payload, default=str)
        print(f"🔍 [SOCKET] JSON serialization test: SUCCESS")
        print(f"🔍 [SOCKET] JSON contains 'order_blocks': {'\"order_blocks\"' in test_json}")
        print(f"🔍 [SOCKET] JSON contains 'fvgs': {'\"fvgs\"' in test_json}")
        print(f"🔍 [SOCKET] JSON contains 'executed_trades': {'\"executed_trades\"' in test_json}")
        # Print a snippet of the JSON to verify
        if '"order_blocks"' in test_json:
            idx = test_json.index('"order_blocks"')
            print(f"🔍 [SOCKET] JSON snippet around order_blocks: {test_json[idx:idx+100]}")
    except Exception as json_error:
        print(f"❌ [SOCKET] JSON serialization test FAILED: {json_error}")
        import traceback
        traceback.print_exc()
    
    try:
        # Force convert to dict to ensure all fields are included
        final_payload = dict(payload)
        # Explicitly ensure all fields are present
        final_payload['order_blocks'] = list(final_payload.get('order_blocks', []))
        final_payload['fvgs'] = list(final_payload.get('fvgs', []))
        final_payload['executed_trades'] = list(final_payload.get('executed_trades', []))
        
        print(f"🔍 [SOCKET] FINAL payload before emit:")
        print(f"   Final payload keys: {list(final_payload.keys())}")
        print(f"   'order_blocks' in final_payload: {'order_blocks' in final_payload}")
        print(f"   'fvgs' in final_payload: {'fvgs' in final_payload}")
        print(f"   'executed_trades' in final_payload: {'executed_trades' in final_payload}")
        print(f"   order_blocks value: {final_payload.get('order_blocks')} (type: {type(final_payload.get('order_blocks'))}, length: {len(final_payload.get('order_blocks', []))})")
        print(f"   fvgs value: {final_payload.get('fvgs')} (type: {type(final_payload.get('fvgs'))}, length: {len(final_payload.get('fvgs', []))})")
        print(f"   executed_trades value: {final_payload.get('executed_trades')} (type: {type(final_payload.get('executed_trades'))}, length: {len(final_payload.get('executed_trades', []))})")
        
        # Emit as a regular dict - Socket.IO will serialize it automatically
        # CRITICAL: Ensure all fields are present and not None
        emit_payload = {
            'success': final_payload.get('success', True),
            'contract': final_payload.get('contract', contract_id),
            'timeframe': final_payload.get('timeframe', timeframe),
            'data_source': final_payload.get('data_source', data_source),
            'current_price': final_payload.get('current_price', float(df['close'].iloc[-1])),
            'candles': final_payload.get('candles', candles),
            'latest_timestamp': final_payload.get('latest_timestamp', df['timestamp'].iloc[-1].isoformat()),
            'order_blocks': final_payload.get('order_blocks', []),
            'fvgs': final_payload.get('fvgs', []),
            'executed_trades': final_payload.get('executed_trades', [])
        }
        
        # Final verification before emit
        print(f"🔍 [SOCKET] EMIT PAYLOAD VERIFICATION:")
        print(f"   emit_payload keys: {list(emit_payload.keys())}")
        print(f"   emit_payload has 'order_blocks': {'order_blocks' in emit_payload}")
        print(f"   emit_payload has 'fvgs': {'fvgs' in emit_payload}")
        print(f"   emit_payload has 'executed_trades': {'executed_trades' in emit_payload}")
        print(f"   emit_payload['order_blocks'] = {emit_payload['order_blocks']} (length: {len(emit_payload['order_blocks'])})")
        print(f"   emit_payload['fvgs'] = {emit_payload['fvgs']} (length: {len(emit_payload['fvgs'])})")
        print(f"   emit_payload['executed_trades'] = {emit_payload['executed_trades']} (length: {len(emit_payload['executed_trades'])})")
        
        # Use JSON serialization to ensure all fields are included
        try:
            import json as json_module
            # Test serialization first
            json_test = json_module.dumps(emit_payload, default=str)
            print(f"🔍 [SOCKET] JSON test length: {len(json_test)}")
            print(f"🔍 [SOCKET] JSON contains order_blocks: {'\"order_blocks\"' in json_test}")
            print(f"🔍 [SOCKET] JSON contains fvgs: {'\"fvgs\"' in json_test}")
            print(f"🔍 [SOCKET] JSON contains executed_trades: {'\"executed_trades\"' in json_test}")
            
            # CRITICAL: Emit as a completely new dict to avoid any filtering
            # Create a fresh dict with all fields explicitly
            # Force empty lists to be sent as empty arrays (not None or filtered out)
            order_blocks_list = list(emit_payload.get('order_blocks', [])) if emit_payload.get('order_blocks') else []
            fvgs_list = list(emit_payload.get('fvgs', [])) if emit_payload.get('fvgs') else []
            executed_trades_list = list(emit_payload.get('executed_trades', [])) if emit_payload.get('executed_trades') else []
            
            # Ensure they are actual lists (not None, not filtered)
            if not isinstance(order_blocks_list, list):
                order_blocks_list = []
            if not isinstance(fvgs_list, list):
                fvgs_list = []
            if not isinstance(executed_trades_list, list):
                executed_trades_list = []
            
            socket_payload = {
                'success': bool(emit_payload.get('success', True)),
                'contract': str(emit_payload.get('contract', contract_id)),
                'timeframe': str(emit_payload.get('timeframe', timeframe)),
                'data_source': str(emit_payload.get('data_source', data_source)),
                'current_price': float(emit_payload.get('current_price', float(df['close'].iloc[-1]))),
                'candles': list(emit_payload.get('candles', candles)),
                'latest_timestamp': str(emit_payload.get('latest_timestamp', df['timestamp'].iloc[-1].isoformat())),
                'order_blocks': order_blocks_list,
                'fvgs': fvgs_list,
                'executed_trades': executed_trades_list
            }
            
            # CRITICAL: Add fields as separate assignments AFTER dict creation
            socket_payload['order_blocks'] = order_blocks_list
            socket_payload['fvgs'] = fvgs_list
            socket_payload['executed_trades'] = executed_trades_list
            
            # Final check before emit
            print(f"🔍 [SOCKET] SOCKET_PAYLOAD BEFORE EMIT:")
            print(f"   socket_payload keys: {list(socket_payload.keys())}")
            print(f"   socket_payload has 'order_blocks': {'order_blocks' in socket_payload}")
            print(f"   socket_payload has 'fvgs': {'fvgs' in socket_payload}")
            print(f"   socket_payload has 'executed_trades': {'executed_trades' in socket_payload}")
            print(f"   socket_payload['order_blocks'] length: {len(socket_payload['order_blocks'])}")
            print(f"   socket_payload['fvgs'] length: {len(socket_payload['fvgs'])}")
            print(f"   socket_payload['executed_trades'] length: {len(socket_payload['executed_trades'])}")
            
            # CRITICAL FIX: Use json.dumps to ensure all fields are serialized correctly
            # Then parse it back to ensure Flask-SocketIO doesn't filter anything
            import json as json_module
            json_str = json_module.dumps(socket_payload, default=str)
            parsed_payload = json_module.loads(json_str)
            
            # Verify all fields are present after JSON round-trip
            print(f"🔍 [SOCKET] AFTER JSON ROUND-TRIP:")
            print(f"   parsed_payload keys: {list(parsed_payload.keys())}")
            print(f"   parsed_payload has 'order_blocks': {'order_blocks' in parsed_payload}")
            print(f"   parsed_payload has 'fvgs': {'fvgs' in parsed_payload}")
            print(f"   parsed_payload has 'executed_trades': {'executed_trades' in parsed_payload}")
            
            # CRITICAL: Flask-SocketIO may filter empty lists, so we send them as objects with a flag
            # Convert empty lists to objects with a 'data' field to ensure they're not filtered
            emit_order_blocks = parsed_payload.get('order_blocks', [])
            emit_fvgs = parsed_payload.get('fvgs', [])
            emit_executed_trades = parsed_payload.get('executed_trades', [])
            
            # Ensure they are lists (not None)
            if emit_order_blocks is None:
                emit_order_blocks = []
            if emit_fvgs is None:
                emit_fvgs = []
            if emit_executed_trades is None:
                emit_executed_trades = []
            
            # Create final emit payload with explicit field assignment
            final_emit_payload = {
                'success': parsed_payload.get('success', True),
                'contract': parsed_payload.get('contract', contract_id),
                'timeframe': parsed_payload.get('timeframe', timeframe),
                'data_source': parsed_payload.get('data_source', data_source),
                'current_price': parsed_payload.get('current_price', float(df['close'].iloc[-1])),
                'candles': parsed_payload.get('candles', candles),
                'latest_timestamp': parsed_payload.get('latest_timestamp', df['timestamp'].iloc[-1].isoformat()),
                'order_blocks': emit_order_blocks,
                'fvgs': emit_fvgs,
                'executed_trades': emit_executed_trades
            }
            
            # Final verification
            print(f"🔍 [SOCKET] FINAL EMIT PAYLOAD:")
            print(f"   Keys: {list(final_emit_payload.keys())}")
            print(f"   order_blocks: {final_emit_payload.get('order_blocks')} (type: {type(final_emit_payload.get('order_blocks'))}, length: {len(final_emit_payload.get('order_blocks', []))})")
            print(f"   fvgs: {final_emit_payload.get('fvgs')} (type: {type(final_emit_payload.get('fvgs'))}, length: {len(final_emit_payload.get('fvgs', []))})")
            print(f"   executed_trades: {final_emit_payload.get('executed_trades')} (type: {type(final_emit_payload.get('executed_trades'))}, length: {len(final_emit_payload.get('executed_trades', []))})")
            
            # Emit the main chart payload
            # CRITICAL: Use namespace=None to ensure broadcast to all clients
            socketio.emit('chart', final_emit_payload, namespace=None)
            print(f"✅ [SOCKET] Chart payload emitted successfully (namespace=None)")
            print(f"🔍 [SOCKET] Emitted payload keys: {list(final_emit_payload.keys())}")
            print(f"🔍 [SOCKET] order_blocks in emitted: {'order_blocks' in final_emit_payload}, value: {final_emit_payload.get('order_blocks')}")
            print(f"🔍 [SOCKET] fvgs in emitted: {'fvgs' in final_emit_payload}, value: {final_emit_payload.get('fvgs')}")
            print(f"🔍 [SOCKET] executed_trades in emitted: {'executed_trades' in final_emit_payload}, value: {final_emit_payload.get('executed_trades')}")
            
            # CRITICAL: Also emit the missing fields as separate events to ensure they're received
            # This is a workaround for Flask-SocketIO potentially filtering empty lists
            socketio.emit('chart_order_blocks', {'order_blocks': emit_order_blocks}, namespace=None)
            socketio.emit('chart_fvgs', {'fvgs': emit_fvgs}, namespace=None)
            socketio.emit('chart_executed_trades', {'executed_trades': emit_executed_trades}, namespace=None)
            print(f"✅ [SOCKET] Separate events emitted (namespace=None): order_blocks={len(emit_order_blocks)}, fvgs={len(emit_fvgs)}, trades={len(emit_executed_trades)}")
        except Exception as json_emit_error:
            print(f"❌ [SOCKET] Error emitting payload: {json_emit_error}")
            import traceback
            traceback.print_exc()
    except Exception as emit_error:
        print(f"❌ [SOCKET] Error emitting payload: {emit_error}")
        import traceback
        traceback.print_exc()

@socketio.on('connect')
def handle_socket_connect():
    emit('status', status_payload())
    if trading_system.account_info:
        emit('account', trading_system.account_info)
    emit('stats', stats_payload())
    if trading_system.auto_logs:
        emit('logs', list(trading_system.auto_logs))
    if trading_system.positions:
        emit('positions', {'positions': trading_system.positions})
    if trading_system.orders:
        emit('orders', {'orders': trading_system.orders})
    if trading_system.executed_trades:
        emit('executed_trades', list(trading_system.executed_trades))
    if trading_system.skipped_trades:
        emit('skipped_trades', list(trading_system.skipped_trades))
    emit('settings', trading_system.settings_payload())

def determine_session_info(timestamp):
    """تحديد الجلسة الحالية بناءً على الوقت"""
    hour = timestamp.hour
    
    if 2 <= hour < 5:
        return 'London', 10
    if 7 <= hour < 10:
        return 'NY_AM', 10
    if 13 <= hour < 17:
        return 'NY_PM', 9
    if 10 <= hour < 12:
        return 'London', 8
    
    # خارج الجلسات الرئيسية
    return ('NY_PM' if hour >= 12 else 'London'), 6


def fetch_recent_bars(contract_id, hours=4, timeframe=5, limit=200, use_live=False, _trying_alternatives=False):
    """جلب أحدث الشموع من ProjectX"""
    print(f"🔍 [BARS] fetch_recent_bars called: contract={contract_id}, live={use_live}, timeframe={timeframe}m")
    print(f"🔍 [BARS] is_connected={trading_system.is_connected}, has_token={bool(trading_system.session_token)}")
    
    if not trading_system.is_connected or not trading_system.session_token:
        print(f"❌ [BARS] Not connected or no token - returning None")
        return None
    
    try:
        import requests
        end_time = datetime.utcnow()
        # استخدام نطاق زمني أوسع (أيام بدلاً من ساعات) للحصول على بيانات أكثر
        # جرب 2 يوم أولاً، ثم 1 يوم إذا فشل
        start_time = end_time - timedelta(days=2)  # آخر يومين
        
        payload = {
            "contractId": contract_id,
            "live": use_live,
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "unit": 2,  # Minutes
            "unitNumber": timeframe,
            "limit": limit,
            "includePartialBar": True
        }
        
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        
        url = f"{trading_system.api_endpoint}/api/History/retrieveBars"
        
        # Log request details for debugging
        print(f"📊 [BARS] Fetching bars: contract={contract_id}, live={use_live}, timeframe={timeframe}m, days=2")
        print(f"📊 [BARS] URL: {url}")
        print(f"📊 [BARS] Time range: {start_time} to {end_time}")
        print(f"📊 [BARS] Payload: {json.dumps(payload, indent=2)}")
        print(f"📊 [BARS] Headers: {dict(headers)}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📊 [BARS] Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"❌ [BARS] {error_msg}")
            trading_system.add_log(f"API error: {error_msg}", "ERROR")
            # Try to load from CSV as fallback
            return _load_bars_from_csv(contract_id)
        
        result = response.json()
        
        print(f"📊 [BARS] API response: success={result.get('success')}, errorCode={result.get('errorCode')}, errorMessage={result.get('errorMessage')}")
        
        # Check if success or errorCode == 0 (like in original working code)
        if not (result.get('success') or result.get('errorCode') == 0):
            # Get error message from multiple possible fields
            error_code = result.get('errorCode')
            error_msg = (result.get('errorMessage') or 
                        result.get('message') or 
                        result.get('detail') or
                        result.get('title') or
                        f'Error code: {error_code}' if error_code else 'API returned success=False without error message')
            
            print(f"❌ [BARS] API returned error: {error_msg}")
            print(f"❌ [BARS] Error code: {error_code}")
            
            # Try with live=True if use_live=False failed
            if not use_live:
                print(f"🔄 [BARS] Retrying with live=True...")
                return fetch_recent_bars(contract_id, hours=hours, timeframe=timeframe, limit=limit, use_live=True)
            
            # Try with shorter time range (1 day instead of 2 days)
            if hours == 4:  # Only try once
                print(f"🔄 [BARS] Trying with 1 day time range...")
                start_time_1day = end_time - timedelta(days=1)
                payload_1day = payload.copy()
                payload_1day['startTime'] = start_time_1day.isoformat() + "Z"
                response_1day = requests.post(url, headers=headers, json=payload_1day, timeout=30)
                result_1day = response_1day.json()
                if (result_1day.get('success') or result_1day.get('errorCode') == 0) and result_1day.get('bars'):
                    bars_1day = result_1day.get('bars', [])
                    if bars_1day and len(bars_1day) > 0:
                        print(f"✅ [BARS] Found data with 1 day range!")
                        df = pd.DataFrame(bars_1day)
                        df['t'] = pd.to_datetime(df['t'])
                        df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                        df = df.sort_values('timestamp')
                        return df
            
            # Try alternative contracts if current one failed (only if not already trying alternatives)
            if not _trying_alternatives and error_code == 1 and (contract_id == 'CON.F.US.EP.U25' or 'EP.U25' in contract_id):
                print(f"🔄 [BARS] Trying alternative ES contracts due to error code 1...")
                alternative_contracts = [
                    "CON.F.US.EP.Z25",  # December 2025
                    "CON.F.US.EP.H25",  # March 2025
                    "CON.F.US.EP.M25",  # June 2025
                    "CON.F.US.EP.U24",  # September 2024 (previous month)
                ]
                
                for alt_contract in alternative_contracts:
                    if alt_contract == contract_id:
                        continue
                    print(f"   🔄 [BARS] Trying {alt_contract} (SIM data)...")
                    # Make direct API call for alternative contract
                    alt_payload = payload.copy()
                    alt_payload['contractId'] = alt_contract
                    alt_payload['live'] = False  # Try SIM first
                    alt_response = requests.post(url, headers=headers, json=alt_payload, timeout=30)
                    alt_result = alt_response.json()
                    
                    if (alt_result.get('success') or alt_result.get('errorCode') == 0) and alt_result.get('bars'):
                        alt_bars = alt_result.get('bars', [])
                        if alt_bars and len(alt_bars) > 0:
                            print(f"   ✅ [BARS] Found data with {alt_contract}!")
                            trading_system.selected_contract = alt_contract
                            df = pd.DataFrame(alt_bars)
                            df['t'] = pd.to_datetime(df['t'])
                            df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                            df = df.sort_values('timestamp')
                            trading_system.add_log(f"Using alternative contract: {alt_contract}", "INFO")
                            return df
                
                print(f"   ❌ [BARS] No data found in any alternative contract")
            
            # Try to load from CSV as fallback
            csv_data = _load_bars_from_csv(contract_id)
            if csv_data is not None and len(csv_data) > 0:
                print(f"✅ [BARS] Loaded {len(csv_data)} bars from CSV fallback")
                return csv_data
            
            # Log the error
            if error_code == 1:
                trading_system.add_log(f"No bars returned: API error code 1 (no data available)", "WARN")
            else:
                trading_system.add_log(f"No bars returned: {error_msg}", "WARN")
            
            return None
        
        bars = result.get('bars', [])
        
        if not bars or len(bars) == 0:
            print(f"⚠️ [BARS] Empty bars array returned for {contract_id}")
            print(f"⚠️ [BARS] success={result.get('success')}, errorCode={result.get('errorCode')}, errorMessage={result.get('errorMessage')}")
            
            # Try with live=True if use_live=False failed
            if not use_live:
                print(f"🔄 [BARS] Retrying with live=True...")
                return fetch_recent_bars(contract_id, hours=hours, timeframe=timeframe, limit=limit, use_live=True)
            
            # Try alternative contracts if current one failed (only if not already trying alternatives)
            if not _trying_alternatives and (contract_id == 'CON.F.US.EP.U25' or 'EP.U25' in contract_id):
                print(f"🔄 [BARS] Trying alternative ES contracts...")
                alternative_contracts = [
                    "CON.F.US.EP.Z25",  # December 2025
                    "CON.F.US.EP.H25",  # March 2025
                    "CON.F.US.EP.M25",  # June 2025
                    "CON.F.US.EP.U24",  # September 2024 (previous month)
                ]
                
                for alt_contract in alternative_contracts:
                    if alt_contract == contract_id:
                        continue
                    print(f"   🔄 [BARS] Trying {alt_contract}...")
                    # Make direct API call for alternative contract (don't recurse to avoid loops)
                    alt_payload = payload.copy()
                    alt_payload['contractId'] = alt_contract
                    alt_payload['live'] = False  # Try SIM first
                    alt_response = requests.post(url, headers=headers, json=alt_payload, timeout=30)
                    alt_result = alt_response.json()
                    
                    if (alt_result.get('success') or alt_result.get('errorCode') == 0) and alt_result.get('bars'):
                        alt_bars = alt_result.get('bars', [])
                        if alt_bars and len(alt_bars) > 0:
                            print(f"   ✅ [BARS] Found data with {alt_contract}!")
                            trading_system.selected_contract = alt_contract
                            df = pd.DataFrame(alt_bars)
                            df['t'] = pd.to_datetime(df['t'])
                            df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
                            df = df.sort_values('timestamp')
                            trading_system.add_log(f"Using alternative contract: {alt_contract}", "INFO")
                            return df
                
                print(f"   ❌ [BARS] No data found in any alternative contract")
            
            # Try to load from CSV as fallback
            csv_data = _load_bars_from_csv(contract_id)
            if csv_data is not None and len(csv_data) > 0:
                print(f"✅ [BARS] Loaded {len(csv_data)} bars from CSV fallback")
                return csv_data
            
            trading_system.add_log(f"No bars in response (array is empty)", "WARN")
            return None
        
        print(f"✅ [BARS] Retrieved {len(bars)} bars")
        df = pd.DataFrame(bars)
        df['t'] = pd.to_datetime(df['t'])
        df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        df = df.sort_values('timestamp')
        return df
    
    except Exception as e:
        trading_system.add_log(f"Error fetching bars: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        # Try to load from CSV as fallback
        csv_data = _load_bars_from_csv(contract_id)
        if csv_data is not None and len(csv_data) > 0:
            print(f"✅ [BARS] Loaded {len(csv_data)} bars from CSV fallback after error")
            return csv_data
        return None

def _load_bars_from_csv(contract_id):
    """تحميل bars من CSV كـ fallback إذا فشل API"""
    try:
        print(f"📂 [CSV] Attempting to load bars from CSV for {contract_id}...")
        print(f"📂 [CSV] LIVE_BARS_DIR: {LIVE_BARS_DIR}")
        print(f"📂 [CSV] Directory exists: {os.path.exists(LIVE_BARS_DIR)}")
        
        # البحث عن ملف CSV للعقد المحدد
        csv_pattern = f"{contract_id.replace('.', '_')}_*.csv"
        csv_path = os.path.join(LIVE_BARS_DIR, csv_pattern)
        print(f"📂 [CSV] Searching for: {csv_path}")
        
        csv_files = glob.glob(csv_path)
        print(f"📂 [CSV] Found {len(csv_files)} CSV file(s)")
        
        if not csv_files:
            # جرب البحث في جميع ملفات CSV في المجلد
            all_csv_files = glob.glob(os.path.join(LIVE_BARS_DIR, "*.csv"))
            print(f"📂 [CSV] Found {len(all_csv_files)} total CSV files in directory")
            if all_csv_files:
                # استخدام أحدث ملف CSV متاح
                latest_csv = max(all_csv_files, key=os.path.getmtime)
                print(f"📂 [CSV] Using latest CSV file: {os.path.basename(latest_csv)}")
                csv_files = [latest_csv]
            else:
                print(f"📂 [CSV] No CSV files found in {LIVE_BARS_DIR}")
                return None
        
        # استخدام أحدث ملف
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"📂 [CSV] Loading from: {latest_csv}")
        print(f"📂 [CSV] File size: {os.path.getsize(latest_csv)} bytes")
        
        df = pd.read_csv(latest_csv)
        print(f"📂 [CSV] CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        if df.empty or 'timestamp' not in df.columns:
            print(f"⚠️ [CSV] CSV file is empty or missing timestamp column")
            return None
        
        # تحويل timestamp إلى datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # إزالة الصفوف المكررة والحفاظ على الأحدث
        df = df.tail(200)  # آخر 200 صف فقط
        
        # التأكد من وجود الأعمدة المطلوبة
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ [CSV] Missing column: {col}")
                return None
        
        print(f"✅ [CSV] Loaded {len(df)} bars from CSV (from {df['timestamp'].min()} to {df['timestamp'].max()})")
        trading_system.add_log(f"Loaded {len(df)} bars from CSV fallback", "INFO")
        return df
        
    except Exception as e:
        print(f"⚠️ [CSV] Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def retrieve_open_positions():
    """قراءة المراكز المفتوحة من ProjectX"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return []
    
    try:
        import requests
        url = f"{trading_system.api_endpoint}/api/Position/searchOpen"
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        payload = {"accountId": trading_system.selected_account_id}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        result = response.json()
        if result.get('success'):
            positions = result.get('positions', [])
            trading_system.positions = positions
            return positions
    except Exception as e:
        trading_system.add_log(f"Positions fetch error: {e}", "ERROR")
    return []

def retrieve_open_orders():
    """قراءة الأوامر المفتوحة من ProjectX"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return []
    
    try:
        import requests
        url = f"{trading_system.api_endpoint}/api/Order/searchOpen"
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        payload = {"accountId": trading_system.selected_account_id}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        result = response.json()
        if result.get('success'):
            orders = result.get('orders', [])
            trading_system.orders = orders
            return orders
    except Exception as e:
        trading_system.add_log(f"Orders fetch error: {e}", "ERROR")
    return []


def find_order_blocks_from_bars(df, lookback=20, min_strength=10):
    """استخراج Order Blocks من بيانات الشموع - بنفس طريقة الباكتست"""
    if df is None or len(df) < 10:
        return []
    
    obs = []
    # استخدام lookback أو كل البيانات المتاحة
    start_idx = max(0, len(df) - lookback)
    recent = df.iloc[start_idx:].copy().reset_index(drop=True)
    
    if len(recent) < 5:
        return []
    
    # البحث عن Bullish OB: شمعة صاعدة قوية
    for i in range(2, len(recent) - 2):
        try:
            c = recent.iloc[i]
            prev = recent.iloc[i-1] if i > 0 else c
            
            # Bullish OB: شمعة صاعدة كبيرة
            if c['close'] > c['open']:
                body = float(c['close'] - c['open'])
                prev_body = abs(float(prev['close'] - prev['open'])) if prev['close'] != prev['open'] else 0.25
                
                # شرط: body أكبر من 1.5x من الشمعة السابقة أو أكبر من min_strength ticks
                if body > prev_body * 1.5 or body >= min_strength * 0.25:
                    # تحقق من أن الشموع التالية لم تكسر الـ OB
                    broken = False
                    for j in range(i+1, min(i+10, len(recent))):  # فحص 10 شموع قادمة
                        if recent.iloc[j]['low'] < c['low']:
                            broken = True
                            break
                    
                    if not broken:
                        # استخدام index من recent ثم التحويل إلى df الأصلي
                        original_idx = start_idx + i
                        if original_idx < len(df):
                            ob_time = df.iloc[original_idx]['timestamp']
                        else:
                            # fallback: البحث عن أقرب candle
                            ob_time = c.get('timestamp')
                            if ob_time is None:
                                ob_time = df.iloc[-1]['timestamp']
                        
                        obs.append({
                            'type': 'Bullish',
                            'high': float(c['high']),
                            'low': float(c['low']),
                            'strength': round(body / 0.25, 1),  # تحويل إلى ticks
                            'time': ob_time,
                            'index': original_idx  # حفظ index للمساعدة في البحث لاحقاً
                        })
            
            # Bearish OB: شمعة هابطة كبيرة
            elif c['close'] < c['open']:
                body = float(c['open'] - c['close'])
                prev_body = abs(float(prev['open'] - prev['close'])) if prev['open'] != prev['close'] else 0.25
                
                if body > prev_body * 1.5 or body >= min_strength * 0.25:
                    broken = False
                    for j in range(i+1, min(i+10, len(recent))):
                        if recent.iloc[j]['high'] > c['high']:
                            broken = True
                            break
                    
                    if not broken:
                        # استخدام index من recent ثم التحويل إلى df الأصلي
                        original_idx = start_idx + i
                        if original_idx < len(df):
                            ob_time = df.iloc[original_idx]['timestamp']
                        else:
                            # fallback: البحث عن أقرب candle
                            ob_time = c.get('timestamp')
                            if ob_time is None:
                                ob_time = df.iloc[-1]['timestamp']
                        
                        obs.append({
                            'type': 'Bearish',
                            'high': float(c['high']),
                            'low': float(c['low']),
                            'strength': round(body / 0.25, 1),
                            'time': ob_time,
                            'index': original_idx  # حفظ index للمساعدة في البحث لاحقاً
                        })
        except Exception as e:
            print(f"⚠️  [OB] Error processing candle {i}: {e}")
            continue
    
    return obs

def find_fvgs_from_bars(df, min_gap=2.0):
    """استخراج Fair Value Gaps من بيانات الشموع - بنفس طريقة الباكتست"""
    if df is None or len(df) < 3:
        return []
    
    fvgs = []
    df = df.copy().reset_index(drop=True)
    
    for i in range(1, len(df) - 1):
        try:
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            next_c = df.iloc[i+1]
            
            # Bullish FVG: gap بين high[i-1] و low[i+1]
            gap_bottom = prev['high']
            gap_top = next_c['low']
            
            if gap_top > gap_bottom:  # يوجد gap
                gap_size = gap_top - gap_bottom
                if gap_size >= min_gap:
                    # تحقق من أن الشمعة الحالية عبرت الـ gap (مع tolerance)
                    if curr['low'] <= gap_bottom + 0.1 and curr['high'] >= gap_top - 0.1:
                        # الحصول على timestamp
                        if 'timestamp' in curr:
                            fvg_time = curr['timestamp']
                        elif hasattr(df, 'index') and isinstance(df.index[i], pd.Timestamp):
                            fvg_time = df.index[i]
                        else:
                            fvg_time = pd.to_datetime(curr.get('time', pd.Timestamp.now()))
                        
                        fvgs.append({
                            'type': 'Bullish',
                            'top': float(gap_top),
                            'bottom': float(gap_bottom),
                            'size': float(gap_size),
                            'time': fvg_time
                        })
            
            # Bearish FVG: gap بين low[i-1] و high[i+1]
            gap_top_bear = prev['low']
            gap_bottom_bear = next_c['high']
            
            if gap_bottom_bear < gap_top_bear:  # يوجد gap
                gap_size = gap_top_bear - gap_bottom_bear
                if gap_size >= min_gap:
                    # تحقق من أن الشمعة الحالية عبرت الـ gap (مع tolerance)
                    if curr['high'] >= gap_top_bear - 0.1 and curr['low'] <= gap_bottom_bear + 0.1:
                        # الحصول على timestamp
                        if 'timestamp' in curr:
                            fvg_time = curr['timestamp']
                        elif hasattr(df, 'index') and isinstance(df.index[i], pd.Timestamp):
                            fvg_time = df.index[i]
                        else:
                            fvg_time = pd.to_datetime(curr.get('time', pd.Timestamp.now()))
                        
                        fvgs.append({
                            'type': 'Bearish',
                            'top': float(gap_top_bear),
                            'bottom': float(gap_bottom_bear),
                            'size': float(gap_size),
                            'time': fvg_time
                        })
        except Exception as e:
            continue
    
    return fvgs

def find_smart_target_simple(entry, direction, swing_high, swing_low, min_target=8.0, min_rr=2.0, sl=None):
    """حساب TP بنفس طريقة الباكتست - Default target عندما لا يوجد swing points
    
    Args:
        entry: سعر الدخول
        direction: 'BUY' أو 'SELL'
        swing_high: أعلى سعر تأرجح (للتوجيه فقط)
        swing_low: أقل سعر تأرجح (للتوجيه فقط)
        min_target: الحد الأدنى للهدف (نقاط)
        min_rr: الحد الأدنى لـ Risk:Reward Ratio
        sl: Stop Loss الفعلي (إذا تم توفيره، سيُستخدم بدلاً من swing_low - 2.0)
    """
    if direction.upper() == 'BUY':
        # إذا تم توفير SL، استخدمه مباشرة لحساب risk
        if sl is not None:
            risk = abs(entry - sl)
        else:
            # استخدام swing_low - 2.0 كـ fallback
            risk = entry - (swing_low - 2.0)
        
        # تحديد target بناءً على risk
        # لكن حد أقصى للـ target: 30 نقطة (120 ticks) لتجنب الأهداف البعيدة جداً
        max_target = 30.0  # حد أقصى معقول
        calculated_target = risk * min_rr
        target = max(min_target, min(calculated_target, max_target))
        
        # إذا كان target أقل من 15 (القيمة الافتراضية في الباكتست)، استخدم 15
        # لكن فقط إذا كان risk معقول (أقل من 10 نقاط)
        if risk <= 10.0 and target < 15.0:
            target = 15.0
        
        tp = entry + target
        return tp, target
    else:  # SELL
        # إذا تم توفير SL، استخدمه مباشرة لحساب risk
        if sl is not None:
            risk = abs(sl - entry)
        else:
            # استخدام swing_high + 2.0 كـ fallback
            risk = (swing_high + 2.0) - entry
        
        # تحديد target بناءً على risk
        max_target = 30.0  # حد أقصى معقول
        calculated_target = risk * min_rr
        target = max(min_target, min(calculated_target, max_target))
        
        # إذا كان target أقل من 15، استخدم 15 (فقط إذا كان risk معقول)
        if risk <= 10.0 and target < 15.0:
            target = 15.0
        
        tp = entry - target
        return tp, target

def build_auto_setups_from_bars(df, tick_size=0.25, threshold_multiplier=1.5, 
                                  min_momentum_ticks=6, min_candles=20, debug=False):
    """تحويل بيانات الشموع إلى Setups يمكن تقييمها - بنفس منطق الباكتست تماماً"""
    setups = []
    
    if df is None or len(df) < min_candles:
        if debug:
            trading_system.add_log(f"❌ Setup detection: df is None or len={len(df) if df is not None else 0} < {min_candles}", "DEBUG")
        return setups
    
    df = df.copy().reset_index(drop=True)
    latest = df.iloc[-1]
    timestamp = latest['timestamp']
    session, priority = determine_session_info(timestamp)
    
    # استخدام نطاق أوسع للبحث عن swing points
    lookback_start = max(0, len(df) - 20)
    lookback_end = max(3, len(df) - 1)
    recent = df.iloc[lookback_start:lookback_end]
    
    if len(recent) < 5:
        if debug:
            trading_system.add_log(f"❌ Setup detection: recent candles={len(recent)} < 5", "DEBUG")
        return setups
    
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    current_price = latest['close']
    
    # حساب momentum على مدى أقصر (3-5 شموع بدل 6)
    momentum_period = min(5, len(df) - 1)
    if momentum_period > 0:
        change = latest['close'] - df['close'].iloc[-momentum_period-1]
    else:
        change = latest['close'] - df['close'].iloc[0]
    
    # حساب threshold بشكل أكثر مرونة - استخدام الحد الأدنى فقط إذا كان threshold_multiplier صغير
    price_std = recent['close'].std() if len(recent) > 1 else tick_size * 10
    std_threshold = price_std * threshold_multiplier
    ticks_threshold = tick_size * min_momentum_ticks
    
    # إذا كان threshold_multiplier صغير (مثل 0.5)، استخدم momentum أصغر (min_momentum_ticks // 2)
    # وإلا استخدم الأكبر بين ticks_threshold و std_threshold
    if threshold_multiplier < 1.0:
        # عندما threshold_multiplier صغير، خفّض momentum threshold أيضاً
        relaxed_momentum = max(1, min_momentum_ticks // 2)  # نصف min_momentum_ticks على الأقل 1
        threshold = tick_size * relaxed_momentum  # threshold أصغر
    else:
        threshold = max(ticks_threshold, std_threshold)  # الأكبر
    
    # تحقق من breakout (أكثر مرونة: بالقرب من swing point يكفي)
    # زد tolerance بناءً على المسافة بين swing high/low
    swing_range = swing_high - swing_low
    # tolerance = 5% من النطاق أو 5 ticks على الأقل، وأقصى 10 ticks
    breakout_tolerance = max(tick_size * 5, min(swing_range * 0.05, tick_size * 10))
    
    strength = round(abs(change) / max(tick_size, 0.25), 2)
    change_ticks = change / tick_size
    
    # Logging تفصيلي
    if debug:
        trading_system.add_log(
            f"🔍 Setup Analysis: Price=${current_price:.2f}, SwingH=${swing_high:.2f}, SwingL=${swing_low:.2f}, "
            f"Change={change_ticks:.1f} ticks, Threshold={threshold:.2f} ({threshold/tick_size:.1f} ticks), "
            f"Std={price_std:.2f}, Tolerance={breakout_tolerance/tick_size:.1f} ticks, "
            f"ThresholdMult={threshold_multiplier:.2f}x, MinMomentum={min_momentum_ticks} ticks", "DEBUG"
        )
    
    # BUY setup - أكثر مرونة مع خيارات متعددة
    # تحقق من القرب من swing high (أعلى أو قريب منه)
    price_above_swing = current_price > (swing_high - breakout_tolerance)
    # تحقق من القرب من swing high (أيضاً إذا كان فوق 70% من النطاق)
    price_in_upper_zone = current_price > (swing_low + swing_range * 0.7)  # في أعلى 30% من النطاق
    price_near_swing_high = abs(current_price - swing_high) < (tick_size * 5)  # قرب swing high
    has_upward_momentum = change > threshold
    has_weak_upward_momentum = change > (tick_size * max(1, min_momentum_ticks // 2))  # momentum أضعف
    
    # استخراج Order Blocks أولاً - بنفس طريقة الباكتست
    obs = find_order_blocks_from_bars(df, lookback=min(20, len(df)), min_strength=8)  # تقليل min_strength لاكتشاف أكثر
    
    # محاولة استخدام Order Blocks (بنفس منطق الباكتست) - PRIORITY #1
    used_ob = False
    if obs:
        # BUY: اختر أقرب Bullish OB
        bullish_obs = [ob for ob in obs if ob['type'] == 'Bullish']
        if bullish_obs:
            # اختر Order Block الذي السعر قريب منه (في نطاق 15 ticks)
            ob_distance = []
            for ob in bullish_obs:
                # المسافة من Order Block
                if current_price < ob['low']:
                    dist = ob['low'] - current_price
                elif current_price > ob['high']:
                    dist = current_price - ob['high']
                else:
                    dist = 0  # داخل Order Block
                ob_distance.append((dist, ob))
            
            # اختر الأقرب
            ob_distance.sort(key=lambda x: x[0])
            ob = ob_distance[0][1]
            dist = ob_distance[0][0]
            
            # نفس منطق الباكتست: entry = (ob['high'] + ob['low']) / 2
            entry = (ob['high'] + ob['low']) / 2
            sl = ob['low'] - 2.0  # نفس منطق الباكتست
            
            # السماح بالسعر في Order Block أو قريب منه (15 ticks = 3.75 points)
            max_distance = tick_size * 15  # نطاق أوسع
            if dist <= max_distance:
                # تمرير SL الفعلي من Order Block لحساب TP بشكل صحيح
                tp, target_distance = find_smart_target_simple(entry, 'BUY', swing_high, swing_low, sl=sl)
                
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                
                # فلترة بنفس طريقة الباكتست
                if risk <= 4.5 and reward >= 8.0 and reward/risk >= 2.0:
                    setups.append({
                        'id': f"{timestamp.isoformat()}_BUY_OB",
                        'type': 'BUY',
                        'entry': round(entry, 2),
                        'sl': round(sl, 2),
                        'tp': round(tp, 2),
                        'risk': risk,
                        'target': reward,
                        'rr': reward/risk,
                        'strength': ob['strength'],  # من Order Block - بنفس طريقة الباكتست
                        'ob_strength': ob['strength'],
                        'priority': priority,
                        'session': session,
                        'timestamp': timestamp.isoformat(),
                        'reason': f'Order Block (dist={dist/tick_size:.1f} ticks, momentum={change_ticks:.1f} ticks)',
                        'source': 'AUTO_OB'
                    })
                    used_ob = True
                    if debug:
                        trading_system.add_log(f"✅ BUY Setup from OB: Entry=${entry:.2f}, SL=${sl:.2f}, TP=${tp:.2f}, Strength={ob['strength']:.1f}, Dist={dist/tick_size:.1f} ticks", "INFO")
    
    # Option 1: Breakout واضح (إذا لم يتم استخدام OB)
    if not used_ob and (price_above_swing or price_in_upper_zone) and has_upward_momentum:
        # استخدام نفس منطق الباكتست: SL عند swing_low - 2.0
        entry = float(current_price)
        sl = swing_low - 2.0  # نفس منطق الباكتست: ob['low'] - 2.0
        # تمرير SL الفعلي لحساب TP بشكل صحيح
        tp, _ = find_smart_target_simple(entry, 'BUY', swing_high, swing_low, sl=sl)
        
        # حساب risk/target/rr بنفس طريقة الباكتست
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        # فلترة بنفس طريقة الباكتست (risk <= 4.5, reward >= 8.0, rr >= 2.0)
        if risk <= 4.5 and reward >= 8.0 and reward/risk >= 2.0:
            setups.append({
                'id': f"{timestamp.isoformat()}_BUY_BREAKOUT",
                'type': 'BUY',
                'entry': round(entry, 2),
                'sl': round(sl, 2),
                'tp': round(tp, 2),
                'risk': risk,
                'target': reward,
                'rr': reward/risk,
                'strength': strength,   # استخدام 'strength' كما في الباكتست
                'ob_strength': strength,  # احتفاظ بـ ob_strength للتوافق
                'priority': priority,
                'session': session,
                'timestamp': timestamp.isoformat(),
                'reason': f'Breakout + momentum ({change_ticks:.1f} ticks)',
                'source': 'AUTO_BREAKOUT'
            })
            if debug:
                trading_system.add_log(f"✅ BUY Setup: Breakout @ ${entry:.2f}, SL=${sl:.2f}, TP=${tp:.2f}, Strength={strength:.1f}", "INFO")
        if debug:
            trading_system.add_log(f"✅ BUY Setup: Breakout @ ${entry:.2f}, Momentum={change_ticks:.1f} ticks", "INFO")
    
    # Option 2: Bounce من Support (near swing low أو في المنطقة السفلية مع momentum صاعد)
    price_near_swing_low = abs(current_price - swing_low) < (tick_size * 4)
    price_in_lower_zone = current_price < (swing_low + swing_range * 0.3)  # في أدنى 30% من النطاق
    if (price_near_swing_low or price_in_lower_zone) and has_weak_upward_momentum and change > 0:
        risk_price = max(abs(current_price - swing_low), tick_size * 4)
        entry = float(current_price)
        sl = entry - risk_price
        tp = entry + (risk_price * 2.5)  # TP أكبر في bounce
        
        setups.append({
            'id': f"{timestamp.isoformat()}_BUY_BOUNCE",
            'type': 'BUY',
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'strength': strength,
            'ob_strength': strength,
            'priority': priority - 1,  # أولوية أقل
            'session': session,
            'timestamp': timestamp.isoformat(),
            'reason': f'Support bounce + weak momentum ({change_ticks:.1f} ticks)',
            'source': 'AUTO_BOUNCE'
        })
        if debug:
            trading_system.add_log(f"✅ BUY Setup: Bounce @ ${entry:.2f}, Momentum={change_ticks:.1f} ticks", "INFO")
    
    # SELL setup - أكثر مرونة مع خيارات متعددة
    price_below_swing = current_price < (swing_low + breakout_tolerance)
    price_in_lower_zone_sell = current_price < (swing_low + swing_range * 0.3)  # في أدنى 30% من النطاق
    price_near_swing_low_sell = abs(current_price - swing_low) < (tick_size * 5)
    has_downward_momentum = change < -threshold
    has_weak_downward_momentum = change < -(tick_size * max(1, min_momentum_ticks // 2))
    
    # محاولة استخدام Bearish Order Block - PRIORITY #1
    used_ob_sell = False
    if obs:
        bearish_obs = [ob for ob in obs if ob['type'] == 'Bearish']
        if bearish_obs:
            # اختر Order Block الذي السعر قريب منه
            ob_distance = []
            for ob in bearish_obs:
                if current_price > ob['high']:
                    dist = current_price - ob['high']
                elif current_price < ob['low']:
                    dist = ob['low'] - current_price
                else:
                    dist = 0  # داخل Order Block
                ob_distance.append((dist, ob))
            
            ob_distance.sort(key=lambda x: x[0])
            ob = ob_distance[0][1]
            dist = ob_distance[0][0]
            
            entry = (ob['high'] + ob['low']) / 2
            sl = ob['high'] + 2.0
            
            # السماح بالسعر في Order Block أو قريب منه (15 ticks)
            max_distance = tick_size * 15
            if dist <= max_distance:
                # تمرير SL الفعلي من Order Block لحساب TP بشكل صحيح
                tp, _ = find_smart_target_simple(entry, 'SELL', swing_high, swing_low, sl=sl)
                
                risk = abs(sl - entry)
                reward = abs(entry - tp)
                
                if risk <= 4.5 and reward >= 8.0 and reward/risk >= 2.0:
                    setups.append({
                        'id': f"{timestamp.isoformat()}_SELL_OB",
                        'type': 'SELL',
                        'entry': round(entry, 2),
                        'sl': round(sl, 2),
                        'tp': round(tp, 2),
                        'risk': risk,
                        'target': reward,
                        'rr': reward/risk,
                        'strength': ob['strength'],
                        'ob_strength': ob['strength'],
                        'priority': priority,
                        'session': session,
                        'timestamp': timestamp.isoformat(),
                        'reason': f'Order Block (dist={dist/tick_size:.1f} ticks, momentum={change_ticks:.1f} ticks)',
                        'source': 'AUTO_OB'
                    })
                    used_ob_sell = True
                    if debug:
                        trading_system.add_log(f"✅ SELL Setup from OB: Entry=${entry:.2f}, SL=${sl:.2f}, TP=${tp:.2f}, Strength={ob['strength']:.1f}, Dist={dist/tick_size:.1f} ticks", "INFO")
    
    # Option 1: Breakdown واضح (إذا لم يتم استخدام OB)
    if not used_ob_sell and (price_below_swing or price_in_lower_zone_sell) and has_downward_momentum:
        risk_price = max(abs(swing_high - current_price), tick_size * 5)
        entry = float(current_price)
        sl = entry + risk_price
        tp = entry - (risk_price * 2)
        
        setups.append({
            'id': f"{timestamp.isoformat()}_SELL_BREAKDOWN",
            'type': 'SELL',
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'strength': strength,
            'ob_strength': strength,
            'priority': priority,
            'session': session,
            'timestamp': timestamp.isoformat(),
            'reason': f'Breakdown + momentum ({abs(change_ticks):.1f} ticks)',
            'source': 'AUTO_BREAKOUT'
        })
        if debug:
            trading_system.add_log(f"✅ SELL Setup: Breakdown @ ${entry:.2f}, Momentum={abs(change_ticks):.1f} ticks", "INFO")
    
    # Option 2: Rejection من Resistance (near swing high أو في المنطقة العلوية مع momentum هابط)
    price_near_swing_high_sell = abs(current_price - swing_high) < (tick_size * 4)
    price_in_upper_zone_sell = current_price > (swing_low + swing_range * 0.7)  # في أعلى 30% من النطاق
    if (price_near_swing_high_sell or price_in_upper_zone_sell) and has_weak_downward_momentum and change < 0:
        risk_price = max(abs(swing_high - current_price), tick_size * 4)
        entry = float(current_price)
        sl = entry + risk_price
        tp = entry - (risk_price * 2.5)
        
        setups.append({
            'id': f"{timestamp.isoformat()}_SELL_REJECTION",
            'type': 'SELL',
            'entry': round(entry, 2),
            'sl': round(sl, 2),
            'tp': round(tp, 2),
            'strength': strength,
            'ob_strength': strength,
            'priority': priority - 1,
            'session': session,
            'timestamp': timestamp.isoformat(),
            'reason': f'Resistance rejection + weak momentum ({abs(change_ticks):.1f} ticks)',
            'source': 'AUTO_REJECTION'
        })
        if debug:
            trading_system.add_log(f"✅ SELL Setup: Rejection @ ${entry:.2f}, Momentum={abs(change_ticks):.1f} ticks", "INFO")
    
    if not setups and debug:
        trading_system.add_log(
            f"⏭️ No setups: PriceAboveSwing={price_above_swing}, PriceBelowSwing={price_below_swing}, "
            f"UpperZone={price_in_upper_zone}, LowerZone={price_in_lower_zone}, "
            f"UpwardMom={has_upward_momentum} ({change_ticks:.1f}>{threshold/tick_size:.1f}), "
            f"DownwardMom={has_downward_momentum} ({change_ticks:.1f}<-{threshold/tick_size:.1f}), "
            f"WeakUp={has_weak_upward_momentum}, WeakDown={has_weak_downward_momentum}", "DEBUG"
        )
    
    return setups


def execute_projectx_trade(setup, source='MANUAL', probability=None, reason=None):
    """تنفيذ الصفقة عبر ProjectX API"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return {'success': False, 'message': 'Not connected or account not selected'}
    
    try:
        import requests
        trading_system.refresh_contract_details()
        contract_id = trading_system.selected_contract
        
        entry_price = setup['entry']
        sl_price = setup['sl']
        tp_price = setup['tp']
        side = 0 if setup['type'].upper() == 'BUY' else 1
        
        trading_system.add_log(f"{source} executing {setup['type']} @ {entry_price:.2f}", "INFO")
        
        order_url = f"{trading_system.api_endpoint}/api/Order/place"
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        
        # Market order
        order_payload = {
            "accountId": trading_system.selected_account_id,
            "contractId": contract_id,
            "type": 2,
            "side": side,
            "size": 1,
            "customTag": f"{source}_MAIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        order_response = requests.post(order_url, headers=headers, json=order_payload, timeout=30)
        order_result = order_response.json()
        
        if not order_result.get('success'):
            error_msg = order_result.get('errorMessage', 'Order placement failed')
            trading_system.add_log(f"Market order failed: {error_msg}", "ERROR")
            return {'success': False, 'message': error_msg}
        
        order_id = order_result.get('orderId')
        trading_system.add_log(f"Market order placed (ID={order_id})", "INFO")
        
        import time
        time.sleep(1)
        
        # Stop Loss
        sl_payload = {
            "accountId": trading_system.selected_account_id,
            "contractId": contract_id,
            "type": 4,
            "side": 1 - side,
            "size": 1,
            "stopPrice": sl_price,
            "customTag": f"{source}_SL_{order_id}_{datetime.now().strftime('%H%M%S')}"
        }
        sl_response = requests.post(order_url, headers=headers, json=sl_payload, timeout=30)
        sl_result = sl_response.json()
        
        # Take Profit
        tp_payload = {
            "accountId": trading_system.selected_account_id,
            "contractId": contract_id,
            "type": 1,
            "side": 1 - side,
            "size": 1,
            "limitPrice": tp_price,
            "customTag": f"{source}_TP_{order_id}_{datetime.now().strftime('%H%M%S')}"
        }
        tp_response = requests.post(order_url, headers=headers, json=tp_payload, timeout=30)
        tp_result = tp_response.json()
        
        trading_system.add_log(f"SL status: {sl_result.get('success')} | TP status: {tp_result.get('success')}", "INFO")
        trading_system.reset_daily_counters()
        trading_system.daily_trades += 1
        trading_system.stats['total_trades'] += 1
        broadcast_stats()
        positions = retrieve_open_positions()
        broadcast_positions(positions)
        orders = retrieve_open_orders()
        broadcast_orders(orders)
        
        trade_summary = {
            'order_id': order_id,
            'timestamp': datetime.utcnow().isoformat(),
            'contract': contract_id,
            'type': setup.get('type'),
            'entry': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'source': source,
            'probability': probability,
            'reason': reason or setup.get('reason') or 'Executed trade',
            'decision': 'TAKE' if probability is None or probability >= trading_system.min_ml_probability else 'EXECUTED',
            'sl_order': sl_result.get('orderId') if sl_result.get('success') else None,
            'tp_order': tp_result.get('orderId') if tp_result.get('success') else None
        }
        trading_system.record_trade_execution(trade_summary)
        
        return {
            'success': True,
            'message': 'Trade executed successfully',
            'order_id': order_id,
            'sl_order': sl_result.get('orderId') if sl_result.get('success') else None,
            'tp_order': tp_result.get('orderId') if tp_result.get('success') else None,
            'probability': probability,
            'decision': 'TAKE'
        }
    
    except Exception as e:
        trading_system.add_log(f"Trade execution error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e)}

# ===========================
# ProjectX API Functions (REAL)
# ===========================

async def connect_to_projectx(username, api_key, environment='TOPSTEP_X'):
    """الاتصال الحقيقي بـ ProjectX API باستخدام REST API مباشرة"""
    import requests
    
    print(f"\n🔐 Connecting to ProjectX...")
    print(f"   Environment: {environment}")
    print(f"   Username: {username}")
    
    # API Endpoints حسب البيئة - URLs الصحيحة من التوثيق
    api_endpoints = {
        'TOPSTEP_X': 'https://api.topstepx.com',  # ✅ بدون .projectx
        'E8X': 'https://api.e8.com',
        'FUNDING_FUTURES': 'https://api.fundingfutures.com',
        'FXIFY_FUTURES': 'https://api.fxifyfutures.com',
        'ALPHA_TICKS': 'https://api.alphaticks.com',
        'BLUE_GUARDIAN': 'https://api.blueguardianfutures.com',
        'BLUSKY': 'https://api.blusky.com',
        'FUTURES_ELITE': 'https://api.futureselite.com',
        'GOAT_FUNDED': 'https://api.goatfundedfutures.com',
        'THE_FUTURES_DESK': 'https://api.thefuturesdesk.com',
        'TICK_TICK_TRADER': 'https://api.tickticktrader.com',
        'TOP_ONE_FUTURES': 'https://api.toponefutures.com',
        'TX3_FUNDING': 'https://api.tx3funding.com'
    }
    
    api_url = api_endpoints.get(environment, api_endpoints['TOPSTEP_X'])
    login_url = f"{api_url}/api/Auth/loginKey"
    
    print(f"   API URL: {login_url}")
    
    # تسجيل دخول حقيقي حسب التوثيق الرسمي
    try:
        # Body حسب التوثيق الرسمي
        payload = {
            "userName": username,
            "apiKey": api_key
        }
        
        print(f"   Sending login request...")
        
        # POST request
        response = requests.post(
            login_url,
            headers={
                'accept': 'text/plain',
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=30
        )
        
        print(f"   Response status: {response.status_code}")
        
        # معالجة الاستجابة
        if response.status_code == 200:
            result = response.json()
            
            print(f"   Response: {result}")
            
            if result.get('success') or result.get('errorCode') == 0:
                # حفظ الـ token
                trading_system.session_token = result.get('token')
                trading_system.api_endpoint = api_url
                trading_system.is_connected = True
                
                print("✅ Connected to ProjectX successfully!")
                print(f"   Session Token: {trading_system.session_token[:20]}...")
                trading_system.add_log(f"Connected to {environment} as {username}", "INFO")
                broadcast_status()
                
                # الحصول على معلومات الحساب
                account_info = await fetch_account_info()
                broadcast_account_info(account_info)
                broadcast_stats()
                
                return True, "Connected successfully"
            else:
                error_msg = result.get('errorMessage', 'Unknown error')
                print(f"❌ Login failed: {error_msg}")
                trading_system.add_log(f"Login failed: {error_msg}", "ERROR")
                raise Exception(f"Login failed: {error_msg}")
        else:
            error_text = response.text
            print(f"❌ HTTP Error {response.status_code}: {error_text}")
            trading_system.add_log(f"HTTP {response.status_code} during login", "ERROR")
            raise Exception(f"HTTP {response.status_code}: {error_text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        trading_system.add_log(f"Connection error: {e}", "ERROR")
        raise e
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        trading_system.add_log(f"Connection exception: {e}", "ERROR")
        raise e

async def search_accounts():
    """البحث عن الحسابات المتاحة"""
    if not trading_system.is_connected or not trading_system.session_token:
        return []
    
    try:
        import requests
        
        print("🔍 Searching for accounts...")
        
        # استدعاء API الحقيقي
        search_url = f"{trading_system.api_endpoint}/api/Account/search"
        
        response = requests.post(
            search_url,
            headers={
                'accept': 'text/plain',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {trading_system.session_token}'
            },
            json={
                'onlyActiveAccounts': True
            },
            timeout=30
        )
        
        print(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                accounts = result.get('accounts', [])
                
                print(f"✅ Found {len(accounts)} account(s)")
                
                for acc in accounts:
                    print(f"   - {acc.get('name')}: ${acc.get('balance'):,.2f} (ID: {acc.get('id')})")
                
                return accounts
            else:
                print(f"⚠️  API error: {result.get('errorMessage')}")
                return []
        else:
            print(f"⚠️  HTTP {response.status_code}")
            return []
    
    except Exception as e:
        print(f"⚠️  Error searching accounts: {e}")
        return []

async def select_account(account_id):
    """اختيار حساب للتداول"""
    if not trading_system.is_connected or not trading_system.session_token:
        return None
    
    try:
        print(f"📊 Selecting account ID: {account_id}")
        
        # حفظ الحساب المختار
        trading_system.selected_account_id = account_id
        
        # تهيئة محرك التداول الكامل (ProjectXAPI)
        from projectx_api_complete import ProjectXAPI
        
        # Extract platform from api_endpoint
        # e.g., https://api.topstepx.com -> topstepx
        platform = trading_system.api_endpoint.split('//')[1].split('.')[1]
        
        trading_system.trading_engine = ProjectXAPI(platform=platform)
        trading_system.trading_engine.session_token = trading_system.session_token
        trading_system.trading_engine.headers['Authorization'] = f'Bearer {trading_system.session_token}'
        
        print(f"✅ Account selected: {account_id}")
        print(f"✅ Trading engine initialized with ProjectXAPI")
        trading_system.add_log(f"Account selected: {account_id}", "INFO")
        trading_system.refresh_contract_details()
        
        return True
    
    except Exception as e:
        print(f"❌ Error selecting account: {e}")
        import traceback
        traceback.print_exc()
        return None

async def fetch_account_info():
    """جلب معلومات الحساب الحقيقية عبر REST API"""
    if not trading_system.is_connected or not trading_system.session_token:
        return None
    
    try:
        # أولاً، ابحث عن الحسابات المتاحة
        accounts = await search_accounts()
        
        if accounts and len(accounts) > 0:
            # استخدام أول حساب نشط
            first_account = accounts[0]
            
            trading_system.selected_account_id = first_account.get('id')
            trading_system.available_accounts = accounts
            
            trading_system.account_info = {
                'account_id': first_account.get('id'),
                'account_name': first_account.get('name'),
                'username': 'Connected',
                'balance': first_account.get('balance', 0.0),
                'equity': first_account.get('balance', 0.0),
                'available_funds': first_account.get('balance', 0.0),
                'can_trade': first_account.get('canTrade', False),
                'is_visible': first_account.get('isVisible', False),
                'status': 'Active' if first_account.get('canTrade') else 'Inactive',
                'account_type': 'Live' if first_account.get('balance') > 25000 else 'Evaluation',
                'last_update': datetime.now().isoformat()
            }
            
            print(f"✅ Account info loaded")
            print(f"   Account: {trading_system.account_info['account_name']}")
            print(f"   Balance: ${trading_system.account_info['balance']:,.2f}")
            print(f"   Can Trade: {trading_system.account_info['can_trade']}")
            
            broadcast_account_info(trading_system.account_info)
            return trading_system.account_info
        else:
            # إذا لم يتم العثور على حسابات
            print(f"⚠️  No accounts found, using demo values")
            trading_system.account_info = {
                'username': 'Connected',
                'balance': 50000.0,
                'equity': 50000.0,
                'status': 'Demo',
                'account_type': 'Demo',
                'last_update': datetime.now().isoformat()
            }
            broadcast_account_info(trading_system.account_info)
            return trading_system.account_info
    
    except Exception as e:
        print(f"⚠️  Error fetching account info: {e}")
        import traceback
        traceback.print_exc()
        # استخدام بيانات افتراضية
        trading_system.account_info = {
            'username': 'Connected',
            'balance': 50000.0,
            'equity': 50000.0,
            'status': 'Demo',
            'account_type': 'Demo',
            'last_update': datetime.now().isoformat()
        }
        broadcast_account_info(trading_system.account_info)
        return trading_system.account_info

async def fetch_positions():
    """جلب الصفقات المفتوحة الحقيقية"""
    if not trading_system.is_connected or not trading_system.client:
        return []
    
    try:
        # استدعاء API الحقيقي للحصول على الصفقات
        # positions = await trading_system.client.get_positions()
        
        # مؤقت
        trading_system.positions = []
        return trading_system.positions
    
    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        return []

async def place_real_order(symbol, side, quantity, order_type='MARKET', limit_price=None):
    """وضع أمر حقيقي عبر ProjectX API"""
    if not trading_system.is_connected or not trading_system.client:
        raise Exception("Not connected to ProjectX")
    
    try:
        print(f"\n📤 Placing REAL order:")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side}")
        print(f"   Quantity: {quantity}")
        print(f"   Type: {order_type}")
        
        # استدعاء API الحقيقي لوضع الأمر
        # order = await trading_system.client.place_order(
        #     symbol=symbol,
        #     side=side,
        #     quantity=quantity,
        #     order_type=order_type,
        #     limit_price=limit_price
        # )
        
        # مؤقت - سيتم استبداله بـ API call حقيقي
        order = {
            'order_id': f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'type': order_type,
            'status': 'FILLED',
            'filled_price': limit_price or 5000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        trading_system.orders.append(order)
        trading_system.stats['total_trades'] += 1
        
        print(f"✅ Order placed: {order['order_id']}")
        
        return order
    
    except Exception as e:
        print(f"❌ Error placing order: {e}")
        raise e

async def fetch_market_data(symbol='ES'):
    """جلب بيانات السوق الحقيقية"""
    if not trading_system.is_connected or not trading_system.client:
        return None
    
    try:
        # استدعاء API الحقيقي للحصول على بيانات السوق
        # market_data = await trading_system.client.get_market_data(symbol)
        
        # مؤقت
        trading_system.market_data[symbol] = {
            'symbol': symbol,
            'bid': 5000.25,
            'ask': 5000.50,
            'last': 5000.25,
            'volume': 125000,
            'timestamp': datetime.now().isoformat()
        }
        
        return trading_system.market_data[symbol]
    
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return None

async def disconnect_from_projectx():
    """قطع الاتصال من ProjectX"""
    if trading_system.client and trading_system.is_connected:
        try:
            await trading_system.client.logout()
            trading_system.is_connected = False
            trading_system.client = None
            print("✅ Disconnected from ProjectX")
            trading_system.add_log("Disconnected from ProjectX", "INFO")
            broadcast_status()
            return True, "Disconnected successfully"
        except Exception as e:
            print(f"❌ Error disconnecting: {e}")
            trading_system.add_log(f"Disconnect error: {e}", "ERROR")
            return False, str(e)
    
    return True, "Already disconnected"

# ===========================
# ML Functions
# ===========================

def load_ml_model(model_path):
    """تحميل ML Model"""
    try:
        trading_system.ml_model = joblib.load(model_path)
        trading_system.model_name = os.path.basename(model_path)
        print(f"✅ ML Model loaded: {trading_system.model_name}")
        trading_system.add_log(f"Model loaded: {trading_system.model_name}", "INFO")
        broadcast_status()
        return True, f"Model loaded: {trading_system.model_name}"
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        trading_system.add_log(f"Model load error: {e}", "ERROR")
        broadcast_status()
        return False, str(e)

def evaluate_setup_with_ml(setup):
    """تقييم Setup باستخدام ML - بنفس طريقة الباكتست تماماً"""
    if not trading_system.ml_model:
        return None, "No ML model loaded"
    
    try:
        trading_system.reset_daily_counters()
        
        # استخراج Features - بنفس طريقة الباكتست تماماً
        # 1. Risk, Target, RR - بنفس الحساب
        risk = abs(setup['entry'] - setup['sl'])
        target = abs(setup['tp'] - setup['entry'])
        rr = target / risk if risk > 0 else 0
        
        # 2. Type - بنفس الطريقة
        type_num = 1 if setup['type'].upper() == 'BUY' else 0
        
        # 3. Strength - استخدام 'strength' أولاً (من الباكتست)، ثم 'ob_strength' كبديل
        # في الباكتست: يستخدم s['strength'] مباشرة
        strength = setup.get('strength', setup.get('ob_strength', 10))
        
        # 4. Session - بنفس الخريطة
        session_map = {'London': 2, 'NY_AM': 1, 'NY_PM': 0}
        session_num = session_map.get(setup.get('session', 'London'), 0)
        
        # 5. Time features - بنفس الطريقة
        ts = pd.to_datetime(setup['timestamp'])
        hour = ts.hour
        day_of_week = ts.weekday()
        
        # 6. Priority - بنفس الطريقة
        priority = setup.get('priority', 10)
        
        # Features بنفس الترتيب تماماً كما في الباكتست
        features = [
            type_num,      # 0: type_num (BUY=1, SELL=0)
            strength,      # 1: strength (Order Block strength)
            risk,          # 2: risk (points)
            target,        # 3: target (points)
            rr,            # 4: rr (risk/reward ratio)
            priority,      # 5: priority (killzone priority)
            session_num,   # 6: session_num (London=2, NY_AM=1, NY_PM=0)
            hour,          # 7: hour (0-23)
            day_of_week    # 8: day_of_week (0=Monday, 6=Sunday)
        ]
        
        # Logging للتحقق من التطابق
        if trading_system.debug_setup_detection:
            trading_system.add_log(
                f"🔍 ML Features: type={type_num}, strength={strength:.1f}, risk={risk:.2f}, "
                f"target={target:.2f}, rr={rr:.2f}, priority={priority}, "
                f"session={session_num}, hour={hour}, dow={day_of_week}", "DEBUG"
            )
        
        # التنبؤ
        probability = trading_system.ml_model.predict_proba([features])[0][1]
        
        decision = 'TAKE' if probability >= 0.70 else 'SKIP'
        reason = f"{'High' if probability >= 0.70 else 'Low'} confidence ({probability*100:.1f}%)"
        
        # فحوصات إضافية
        if trading_system.daily_trades >= trading_system.max_trades_per_day:
            decision = 'SKIP'
            reason = f"Max daily trades reached ({trading_system.max_trades_per_day})"
        
        if trading_system.stats['daily_pnl'] <= -trading_system.max_daily_loss:
            decision = 'SKIP'
            reason = f"Max daily loss reached (${trading_system.max_daily_loss})"
        
        return {
            'probability': float(probability),
            'decision': decision,
            'reason': reason
        }, None
    
    except Exception as e:
        print(f"❌ Error evaluating setup: {e}")
        return None, str(e)

# ===========================
# Flask Routes
# ===========================

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return send_from_directory('templates', 'real_platform.html')

@app.route('/api/status')
def get_status():
    """حالة النظام"""
    return jsonify(status_payload())

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """قراءة أو تحديث إعدادات التداول"""
    if request.method == 'GET':
        return jsonify({'success': True, 'settings': trading_system.settings_payload()})
    
    data = request.json or {}
    min_prob = data.get('min_probability_percent')
    max_trades = data.get('max_trades_per_day')
    max_loss = data.get('max_daily_loss')
    threshold_mult = data.get('threshold_multiplier')
    min_candles = data.get('min_candles_for_setup')
    min_momentum = data.get('min_momentum_ticks')
    debug_setup = data.get('debug_setup_detection')
    
    try:
        updates = {}
        if min_prob is not None:
            updates['min_probability'] = float(min_prob) / 100.0
        if max_trades is not None:
            updates['max_trades'] = int(max_trades)
        if max_loss is not None:
            updates['max_loss'] = float(max_loss)
        if threshold_mult is not None:
            updates['threshold_multiplier'] = float(threshold_mult)
        if min_candles is not None:
            updates['min_candles'] = int(min_candles)
        if min_momentum is not None:
            updates['min_momentum'] = int(min_momentum)
        if debug_setup is not None:
            updates['debug_setup'] = bool(debug_setup)
        
        if not updates:
            return jsonify({'success': False, 'message': 'No valid settings provided'})
        
        trading_system.update_settings(**updates)
        broadcast_settings()
        return jsonify({'success': True, 'settings': trading_system.settings_payload()})
    except Exception as e:
        trading_system.add_log(f"Settings update error: {e}", "ERROR")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """الاتصال بـ ProjectX"""
    data = request.json
    username = data.get('username')
    api_key = data.get('api_key')
    environment = data.get('environment', 'TOPSTEP_X')
    
    if not username or not api_key:
        return jsonify({'success': False, 'message': 'Username and API Key required'})
    
    try:
        trading_system.start_event_loop()
        success, message = trading_system.run_async(
            connect_to_projectx(username, api_key, environment)
        )
        
        return jsonify({
            'success': True,
            'message': message,
            'account_info': trading_system.account_info
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """قطع الاتصال"""
    try:
        success, message = trading_system.run_async(disconnect_from_projectx())
        trading_system.is_trading = False
        broadcast_status()
        
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/accounts')
def get_accounts():
    """قائمة الحسابات المتاحة"""
    if not trading_system.is_connected:
        return jsonify({'error': 'Not connected', 'accounts': []})
    
    try:
        return jsonify({
            'accounts': trading_system.available_accounts,
            'selected': trading_system.selected_account_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'accounts': []})

@app.route('/api/select_account', methods=['POST'])
def api_select_account():
    """اختيار حساب للتداول"""
    if not trading_system.is_connected:
        return jsonify({'success': False, 'message': 'Not connected'})
    
    data = request.json
    account_id = data.get('account_id')
    
    if not account_id:
        return jsonify({'success': False, 'message': 'Account ID required'})
    
    try:
        result = trading_system.run_async(select_account(account_id))
        
        if result:
            # تحديث معلومات الحساب
            for acc in trading_system.available_accounts:
                if acc.get('id') == account_id:
                    trading_system.account_info = {
                        'account_id': acc.get('id'),
                        'account_name': acc.get('name'),
                        'balance': acc.get('balance', 0.0),
                        'can_trade': acc.get('canTrade', False),
                        'status': 'Active' if acc.get('canTrade') else 'Inactive',
                        'last_update': datetime.now().isoformat()
                    }
                    break
            
            broadcast_account_info(trading_system.account_info)
            trading_system.start_market_stream()
            broadcast_status()
            
            return jsonify({
                'success': True,
                'message': 'Account selected',
                'account': trading_system.account_info
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to select account'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/account')
def get_account():
    """معلومات الحساب الحالي"""
    if not trading_system.is_connected:
        return jsonify({'error': 'Not connected'})
    
    return jsonify(trading_system.account_info or {})

@app.route('/api/contracts')
def get_contracts():
    """قائمة العقود المتاحة"""
    if not trading_system.is_connected:
        return jsonify({'error': 'Not connected', 'contracts': []})
    
    try:
        import requests
        
        # جلب العقود المتاحة مباشرة من API
        url = f"{trading_system.api_endpoint}/api/Contract/available"
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        
        # جرب Live أولاً
        response = requests.post(url, headers=headers, json={"live": True}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                contracts = result.get('contracts', [])
                # Filter ES contracts only
                es_contracts = [c for c in contracts if 'EP' in c.get('symbolId', '')]
                # Auto-select active contract
                active_contract = next((c for c in es_contracts if c.get('activeContract')), None)
                if active_contract:
                    trading_system.selected_contract = active_contract['id']
                    print(f"✅ Auto-selected active contract: {active_contract['id']}")
                    trading_system.refresh_contract_details(active_contract['id'])
                    trading_system.add_log(f"Live contract selected: {active_contract['id']}", "INFO")
                return jsonify({'success': True, 'contracts': es_contracts, 'active_contract': active_contract['id'] if active_contract else None})
        
        # Fallback to Sim
        response = requests.post(url, headers=headers, json={"live": False}, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                contracts = result.get('contracts', [])
                es_contracts = [c for c in contracts if 'EP' in c.get('symbolId', '')]
                active_contract = next((c for c in es_contracts if c.get('activeContract')), None)
                if active_contract:
                    trading_system.selected_contract = active_contract['id']
                    trading_system.refresh_contract_details(active_contract['id'])
                    trading_system.add_log(f"SIM contract selected: {active_contract['id']}", "INFO")
                return jsonify({'success': True, 'contracts': es_contracts, 'active_contract': active_contract['id'] if active_contract else None})
        
        return jsonify({'error': 'Failed to fetch contracts', 'contracts': []})
    
    except Exception as e:
        print(f"❌ Error fetching contracts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'contracts': []})

@app.route('/api/select_contract', methods=['POST'])
def select_contract():
    """اختيار Contract للتداول"""
    if not trading_system.trading_engine:
        return jsonify({'success': False, 'message': 'Trading engine not initialized'})
    
    data = request.json
    contract_id = data.get('contract_id')
    
    if not contract_id:
        return jsonify({'success': False, 'message': 'Contract ID required'})
    
    try:
        # حفظ Contract المختار
        trading_system.selected_contract = contract_id
        
        # جلب تفاصيل العقد
        contract = trading_system.trading_engine.get_contract_by_id(contract_id)
        trading_system.refresh_contract_details(contract_id)
        trading_system.add_log(f"Contract selected manually: {contract_id}", "INFO")
        
        return jsonify({
            'success': True,
            'message': 'Contract selected',
            'contract': contract
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/positions')
def get_positions():
    """المراكز المفتوحة الحالية"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return jsonify({'positions': []})
    
    try:
        positions = retrieve_open_positions()
        broadcast_positions(positions)
        return jsonify({'positions': positions})
    except Exception as e:
        trading_system.add_log(f"Positions route error: {e}", "ERROR")
        return jsonify({'error': str(e), 'positions': []})

@app.route('/api/orders')
def get_orders():
    """الأوامر المفتوحة الحالية"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return jsonify({'orders': []})
    
    try:
        orders = retrieve_open_orders()
        broadcast_orders(orders)
        return jsonify({'orders': orders})
    except Exception as e:
        trading_system.add_log(f"Orders route error: {e}", "ERROR")
        return jsonify({'error': str(e), 'orders': []})

@app.route('/api/executed_trades')
def get_executed_trades():
    """إرجاع الصفقات المنفذة مؤخراً"""
    try:
        return jsonify({'success': True, 'trades': list(trading_system.executed_trades)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'trades': []})

@app.route('/api/skipped_trades')
def get_skipped_trades():
    """إرجاع الصفقات التي تم تجاهلها"""
    try:
        return jsonify({'success': True, 'trades': list(trading_system.skipped_trades)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'trades': []})

@app.route('/api/place_trade', methods=['POST'])
def place_trade():
    """تنفيذ صفقة حقيقية بناءً على ML"""
    if not trading_system.trading_engine:
        return jsonify({'success': False, 'message': 'Trading engine not initialized'})
    
    if not trading_system.ml_model:
        return jsonify({'success': False, 'message': 'ML model not loaded'})
    
    data = request.json
    setup = data.get('setup')  # {type, entry, sl, tp}
    
    if not setup:
        return jsonify({'success': False, 'message': 'Setup required'})
    
    # Fill missing metadata for ML
    setup.setdefault('timestamp', datetime.utcnow().isoformat())
    setup.setdefault('session', 'NY_AM')
    setup.setdefault('priority', 10)
    if 'strength' not in setup and 'ob_strength' in setup:
        setup['strength'] = setup['ob_strength']
    if 'ob_strength' not in setup and 'strength' in setup:
        setup['ob_strength'] = setup['strength']
    
    evaluation, error = evaluate_setup_with_ml(setup)
    if error:
        return jsonify({'success': False, 'message': error})
    
    if evaluation['decision'] != 'TAKE':
        trading_system.record_skipped_trade({
            'type': setup.get('type'),
            'entry': setup.get('entry'),
            'sl': setup.get('sl'),
            'tp': setup.get('tp'),
            'probability': evaluation['probability'],
            'reason': evaluation.get('reason'),
            'source': 'MANUAL',
            'contract': trading_system.selected_contract,
            'session': setup.get('session'),
            'priority': setup.get('priority')
        })
        return jsonify({
            'success': True,
            'decision': evaluation['decision'],
            'probability': evaluation['probability'],
            'message': evaluation['reason']
        })
    
    result = execute_projectx_trade(
        setup,
        source='MANUAL',
        probability=evaluation['probability'],
        reason=evaluation.get('reason')
    )
    result['decision'] = evaluation['decision']
    return jsonify(result)

@app.route('/api/place_trade_direct', methods=['POST'])
def place_trade_direct():
    """تنفيذ صفقة مباشرة بدون ML"""
    if not trading_system.is_connected or not trading_system.selected_account_id:
        return jsonify({'success': False, 'message': 'Not connected or account not selected'})
    
    data = request.json
    setup = data.get('setup')
    
    if not setup:
        return jsonify({'success': False, 'message': 'Setup required'})
    
    setup.setdefault('timestamp', datetime.utcnow().isoformat())
    setup.setdefault('session', 'NY_AM')
    setup.setdefault('priority', 9)
    setup.setdefault('strength', setup.get('strength', 10))
    setup.setdefault('ob_strength', setup.get('ob_strength', setup['strength']))
    
    result = execute_projectx_trade(setup, source='DIRECT', probability=None, reason='Manual direct execution')
    return jsonify(result)

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """إغلاق مركز"""
    if not trading_system.trading_engine:
        return jsonify({'success': False, 'message': 'Trading engine not initialized'})
    
    data = request.json
    contract_id = data.get('contract_id')
    
    if not contract_id:
        return jsonify({'success': False, 'message': 'Contract ID required'})
    
    try:
        result = trading_system.trading_engine.close_position(contract_id)
        return jsonify({'success': result})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/chart_data')
def get_chart_data():
    """الحصول على بيانات الشموع للرسم البياني - LIVE DATA"""
    import json  # Import json at the start of the function
    
    # Check connection first
    if not trading_system.is_connected:
        return jsonify({'success': False, 'message': 'Not connected to ProjectX'})
    
    if not trading_system.session_token:
        return jsonify({'success': False, 'message': 'No session token'})
    
    # Parameters
    contract_id = request.args.get('contract', trading_system.selected_contract)
    
    if not contract_id:
        return jsonify({'success': False, 'message': 'No contract selected'})
    
    timeframe = int(request.args.get('timeframe', 5))  # 5 minutes default
    # زيادة limit ليشمل اليوم السابق + اليوم الحالي
    # 5 دقائق × 24 ساعة × 2 يوم = 288 شمعة على الأقل، نستخدم 500 للتأكد
    limit = int(request.args.get('limit', 500))  # 500 bars default (ليومين كاملين)
    use_live = request.args.get('live', 'true').lower() == 'true'  # Use live data by default
    
    try:
        from datetime import datetime, timedelta
        import requests
        
        print(f"📊 [CHART] Fetching {'LIVE' if use_live else 'SIM'} data for {contract_id}")
        print(f"   Timeframe: {timeframe}m, Limit: {limit} bars")
        
        # Calculate time range - اليوم السابق + اليوم الحالي
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=2)  # آخر يومين (اليوم السابق + اليوم الحالي)
        
        print(f"📅 [CHART] Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        
        # Make direct API call for live data
        url = f"{trading_system.api_endpoint}/api/History/retrieveBars"
        
        payload = {
            "contractId": contract_id,
            "live": use_live,  # TRUE for live market data
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "unit": 2,  # Minute
            "unitNumber": timeframe,
            "limit": limit,
            "includePartialBar": True  # Include current incomplete bar
        }
        
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        
        print(f"📤 [CHART] API Request:")
        print(f"   URL: {url}")
        print(f"   Contract: {contract_id}")
        print(f"   Live: {use_live}")
        print(f"   Period: {start_time.date()} to {end_time.date()}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 [CHART] API Response: Status {response.status_code}")
        
        result = response.json()
        
        print(f"📥 [CHART] Result: success={result.get('success')}, errorCode={result.get('errorCode')}")
        try:
            print(f"📥 [CHART] Full response: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            print(f"📥 [CHART] Full response (error serializing): {result}")
        
        if not result.get('success'):
            error_code = result.get('errorCode', 'Unknown')
            error_msg = result.get('errorMessage') or result.get('message') or f"API Error Code: {error_code}"
            
            print(f"❌ [CHART] API Error: {error_msg}")
            print(f"❌ [CHART] Error Code: {error_code}")
            print(f"❌ [CHART] Contract: {contract_id}")
            print(f"❌ [CHART] Live: {use_live}")
            
            # إذا كان Live data فشل، جرب Sim data تلقائياً
            if use_live:
                print(f"🔄 [CHART] Live data failed, trying Sim data...")
                payload['live'] = False
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                result = response.json()
                
                if result.get('success') and result.get('bars'):
                    print(f"✅ [CHART] Sim data successful!")
                    # Continue processing with Sim data
                    use_live = False  # Update flag for response
                else:
                    # إذا فشل Sim أيضاً، جرب عقود أخرى
                    print(f"🔄 [CHART] Sim data also failed, trying alternative contracts...")
                    alternative_contracts = [
                        "CON.F.US.EP.Z25",  # December 2025
                        "CON.F.US.EP.H25",  # March 2025
                        "CON.F.US.EP.M25",  # June 2025
                    ]
                    
                    for alt_contract in alternative_contracts:
                        print(f"   Trying {alt_contract}...")
                        payload['contractId'] = alt_contract
                        response = requests.post(url, headers=headers, json=payload, timeout=30)
                        result = response.json()
                        
                        if result.get('success') and result.get('bars'):
                            bars = result.get('bars', [])
                            contract_id = alt_contract
                            trading_system.selected_contract = alt_contract
                            print(f"✅ [CHART] Found data with {alt_contract}!")
                            use_live = False
                            break
                    
                    if not result.get('success') or not result.get('bars'):
                        return jsonify({
                            'success': False, 
                            'message': f"Failed to get data. Live error: {error_msg}, Sim also failed. Tried contracts: {contract_id}, {', '.join(alternative_contracts)}"
                        })
            else:
                return jsonify({'success': False, 'message': error_msg})
        
        # Get bars (either from original Live request or Sim fallback)
        bars = result.get('bars', [])
        
        print(f"📊 [CHART] Received {len(bars)} bars")
        
        # إذا لم نحصل على bars، جرب عقود أخرى
        if not bars:
            print(f"⚠️  [CHART] No bars returned from API for {contract_id}")
            print(f"🔄 [CHART] Trying alternative contracts...")
            
            alternative_contracts = [
                "CON.F.US.EP.Z25",  # December 2025
                "CON.F.US.EP.H25",  # March 2025
                "CON.F.US.EP.M25",  # June 2025
            ]
            
            # إزالة العقد الحالي من القائمة
            if contract_id in alternative_contracts:
                alternative_contracts.remove(contract_id)
            
            # محاولة كل عقد
            for alt_contract in alternative_contracts:
                print(f"   Trying {alt_contract}...")
                payload['contractId'] = alt_contract
                payload['live'] = False  # جرب Sim data
                
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    result = response.json()
                    
                    if result.get('success') and result.get('bars'):
                        bars = result.get('bars', [])
                        contract_id = alt_contract  # تحديث العقد المستخدم
                        trading_system.selected_contract = alt_contract  # حفظ العقد الجديد
                        print(f"✅ [CHART] Found data with {alt_contract}!")
                        use_live = False
                        break
                except Exception as e:
                    print(f"   ❌ Error trying {alt_contract}: {e}")
                    continue
            
            # إذا لم نجد بيانات في أي عقد
            if not bars:
                print(f"❌ [CHART] No data available in any contract")
                return jsonify({
                    'success': False, 
                    'message': f'No data available for ES contracts. Tried: {contract_id}, {", ".join(alternative_contracts)}'
                })
        
        # Convert to pandas DataFrame
        import pandas as pd
        bars_df = pd.DataFrame(bars)
        
        if bars_df.empty:
            print(f"⚠️  [CHART] Empty DataFrame returned from API")
            return jsonify({'success': False, 'message': 'No data available'})
        
        print(f"📊 [CHART] DataFrame shape: {bars_df.shape}")
        print(f"📊 [CHART] Columns: {bars_df.columns.tolist()}")
        
        bars_df['t'] = pd.to_datetime(bars_df['t'], errors='coerce')
        bars_df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
        
        if 'volume' not in bars_df.columns:
            bars_df['volume'] = 0.0
        
        bars_df['timestamp'] = bars_df['timestamp'].dt.tz_localize(None)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        bars_df[numeric_cols] = bars_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        bars_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        bars_df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close'], inplace=True)
        bars_df['volume'] = bars_df['volume'].fillna(0)
        bars_df = bars_df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        
        if bars_df.empty:
            print(f"⚠️  [CHART] No usable data after cleaning")
            return jsonify({'success': False, 'message': 'No clean data available'})
        
        print(f"✅ [CHART] Data ready: {len(bars_df)} cleaned bars from {bars_df['timestamp'].iloc[0]} to {bars_df['timestamp'].iloc[-1]}")
        print(f"✅ [CHART] Latest price: ${bars_df['close'].iloc[-1]:.2f}")
        print(f"📅 [CHART] Date range: {bars_df['timestamp'].iloc[0].date()} to {bars_df['timestamp'].iloc[-1].date()}")
        
        # استخراج Order Blocks من البيانات - استخدام lookback أكبر
        # استخدام كل البيانات المتاحة (أو 200 شمعة على الأقل)
        lookback_size = min(max(200, len(bars_df) // 2), len(bars_df))
        obs = find_order_blocks_from_bars(bars_df, lookback=lookback_size, min_strength=8)  # تقليل min_strength لاكتشاف أكثر
        print(f"📊 [CHART] Found {len(obs)} Order Blocks from {len(bars_df)} candles (lookback={lookback_size})")
        
        # استخراج FVGs من البيانات
        fvgs = find_fvgs_from_bars(bars_df, min_gap=2.0)
        print(f"📊 [CHART] Found {len(fvgs)} FVGs from {len(bars_df)} candles")
        
        # Log أول 5 Order Blocks للتحقق
        for i, ob in enumerate(obs[:5]):
            print(f"   OB {i+1}: {ob['type']} @ ${ob['low']:.2f}-${ob['high']:.2f}, strength={ob['strength']:.1f}, time={ob.get('time', 'N/A')}")
        
        # تحويل Order Blocks إلى تنسيق للشارت
        order_blocks_chart = []
        for ob in obs:
            try:
                # إيجاد timestamp للـ Order Block - استخدام index إن أمكن
                ob_time = None
                ob_index = ob.get('index')
                
                # إذا كان هناك index محفوظ، استخدمه مباشرة
                if ob_index is not None and ob_index < len(bars_df):
                    ob_time = bars_df.iloc[ob_index]['timestamp']
                    print(f"📊 [CHART] OB using index {ob_index}: {ob['type']} @ ${ob['low']:.2f}-${ob['high']:.2f}")
                else:
                    # fallback: استخدام timestamp المباشر
                    ob_time = ob.get('time')
                    
                    # إذا كان timestamp غير صالح، ابحث عن أقرب candle
                    if ob_time is None or (not isinstance(ob_time, pd.Timestamp) and not isinstance(ob_time, str)):
                        # البحث عن أقرب candle بناءً على high/low (مع tolerance أكبر)
                        matching_candles = bars_df[
                            (bars_df['high'] >= ob['high'] - 0.5) & 
                            (bars_df['high'] <= ob['high'] + 0.5) &
                            (bars_df['low'] >= ob['low'] - 0.5) & 
                            (bars_df['low'] <= ob['low'] + 0.5)
                        ]
                        if len(matching_candles) > 0:
                            ob_time = matching_candles.iloc[0]['timestamp']
                            print(f"📊 [CHART] OB found by matching: {ob['type']} @ ${ob['low']:.2f}-${ob['high']:.2f}")
                        else:
                            # استخدام candle في منتصف البيانات كبديل
                            mid_idx = len(bars_df) // 2
                            ob_time = bars_df.iloc[mid_idx]['timestamp']
                            print(f"⚠️  [CHART] OB using fallback timestamp: {ob['type']} @ ${ob['low']:.2f}-${ob['high']:.2f}")
                
                # تحويل timestamp إلى Unix timestamp
                if isinstance(ob_time, str):
                    ob_time = pd.to_datetime(ob_time)
                elif not isinstance(ob_time, pd.Timestamp):
                    ob_time = pd.to_datetime(ob_time)
                
                if isinstance(ob_time, pd.Timestamp):
                    ob_timestamp = int(ob_time.timestamp())
                else:
                    ob_timestamp = int(pd.to_datetime(ob_time).timestamp())
                
                order_blocks_chart.append({
                    'time': ob_timestamp,
                    'type': ob['type'],  # 'Bullish' or 'Bearish'
                    'high': float(ob['high']),
                    'low': float(ob['low']),
                    'strength': float(ob['strength'])
                })
                print(f"✅ [CHART] OB added: {ob['type']} @ ${ob['low']:.2f}-${ob['high']:.2f}, time={ob_timestamp} ({pd.Timestamp(ob_timestamp, unit='s')})")
            except Exception as e:
                print(f"⚠️  [CHART] Error processing OB: {e}")
                print(f"   OB data: {ob}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"📊 [CHART] Processed {len(order_blocks_chart)} Order Blocks for chart (from {len(obs)} raw OBs)")
        
        # Log sample للتحقق
        if len(order_blocks_chart) > 0:
            print(f"   Sample OB: {order_blocks_chart[0]}")
        
        # تحويل FVGs إلى تنسيق للشارت
        fvgs_chart = []
        for fvg in fvgs:
            try:
                fvg_time = fvg.get('time')
                if fvg_time is None:
                    continue
                
                # تحويل timestamp
                if isinstance(fvg_time, str):
                    fvg_time = pd.to_datetime(fvg_time)
                elif not isinstance(fvg_time, pd.Timestamp):
                    fvg_time = pd.to_datetime(fvg_time)
                
                if isinstance(fvg_time, pd.Timestamp):
                    fvg_timestamp = int(fvg_time.timestamp())
                else:
                    fvg_timestamp = int(pd.to_datetime(fvg_time).timestamp())
                
                fvgs_chart.append({
                    'time': fvg_timestamp,
                    'type': fvg['type'],  # 'Bullish' or 'Bearish'
                    'top': float(fvg['top']),
                    'bottom': float(fvg['bottom']),
                    'size': float(fvg['size'])
                })
            except Exception as e:
                print(f"⚠️  [CHART] Error processing FVG: {e}")
                continue
        
        print(f"📊 [CHART] Processed {len(fvgs_chart)} FVGs for chart (from {len(fvgs)} raw FVGs)")
        
        candles = []
        for _, row in bars_df.iterrows():
            try:
                candles.append({
                    'time': int(row['timestamp'].timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
            except (TypeError, ValueError) as candle_error:
                print(f"⚠️  [CHART] Skipping invalid candle: {candle_error}")
                continue
        
        if not candles:
            print(f"⚠️  [CHART] No candles after processing rows")
            return jsonify({'success': False, 'message': 'No data available'})
        
        # جلب الصفقات المنفذة لعرضها على الشارت
        executed_trades = list(trading_system.executed_trades)[:50]  # آخر 50 صفقة
        
        # Convert to JSON format for chart
        # التأكد من أن order_blocks_chart و fvgs_chart موجودة حتى لو كانت فارغة
        if 'order_blocks_chart' not in locals():
            order_blocks_chart = []
        if 'fvgs_chart' not in locals():
            fvgs_chart = []
        if 'executed_trades' not in locals():
            executed_trades = []
        
        # التأكد من أن القوائم ليست None
        if order_blocks_chart is None:
            order_blocks_chart = []
        if fvgs_chart is None:
            fvgs_chart = []
        if executed_trades is None:
            executed_trades = []
        
        print(f"🔍 [CHART] BEFORE chart_data creation:")
        print(f"   order_blocks_chart: {order_blocks_chart} (type: {type(order_blocks_chart)}, length: {len(order_blocks_chart)})")
        print(f"   fvgs_chart: {fvgs_chart} (type: {type(fvgs_chart)}, length: {len(fvgs_chart)})")
        print(f"   executed_trades: {executed_trades} (type: {type(executed_trades)}, length: {len(executed_trades)})")
        
        # Create chart_data dictionary step by step
        chart_data = {
            'success': True,
            'labels': bars_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'data': {
                'open': bars_df['open'].tolist(),
                'high': bars_df['high'].tolist(),
                'low': bars_df['low'].tolist(),
                'close': bars_df['close'].tolist(),
                'volume': bars_df['volume'].tolist()
            },
            'candles': candles,
            'current_price': float(bars_df['close'].iloc[-1]),
            'contract': contract_id,
            'timeframe': f"{timeframe}m",
            'data_source': 'LIVE' if use_live else 'SIM'
        }
        
        # Add fields explicitly after creation to ensure they're included
        chart_data['order_blocks'] = list(order_blocks_chart) if order_blocks_chart else []
        chart_data['fvgs'] = list(fvgs_chart) if fvgs_chart else []
        chart_data['executed_trades'] = list(executed_trades) if executed_trades else []
        
        # Double-check that fields are present
        if 'order_blocks' not in chart_data:
            chart_data['order_blocks'] = []
        if 'fvgs' not in chart_data:
            chart_data['fvgs'] = []
        if 'executed_trades' not in chart_data:
            chart_data['executed_trades'] = []
        
        print(f"🔍 [CHART] AFTER chart_data creation:")
        print(f"   chart_data keys: {list(chart_data.keys())}")
        print(f"   'order_blocks' in chart_data: {'order_blocks' in chart_data}")
        print(f"   'fvgs' in chart_data: {'fvgs' in chart_data}")
        print(f"   'executed_trades' in chart_data: {'executed_trades' in chart_data}")
        print(f"   chart_data['order_blocks']: {chart_data.get('order_blocks')} (length: {len(chart_data.get('order_blocks', []))})")
        print(f"   chart_data['fvgs']: {chart_data.get('fvgs')} (length: {len(chart_data.get('fvgs', []))})")
        print(f"   chart_data['executed_trades']: {chart_data.get('executed_trades')} (length: {len(chart_data.get('executed_trades', []))})")
        
        # التحقق من أن البيانات موجودة قبل الإرسال
        assert 'order_blocks' in chart_data, "order_blocks missing from chart_data!"
        assert 'fvgs' in chart_data, "fvgs missing from chart_data!"
        assert 'executed_trades' in chart_data, "executed_trades missing from chart_data!"
        print(f"✅ [CHART] Response prepared: {len(candles)} candles, {len(order_blocks_chart)} Order Blocks, {len(fvgs_chart)} FVGs, {len(executed_trades)} trades")
        print(f"📊 [CHART] Order Blocks in response: {len(order_blocks_chart)}")
        print(f"📊 [CHART] FVGs in response: {len(fvgs_chart)}")
        print(f"📊 [CHART] Executed trades in response: {len(executed_trades)}")
        if len(order_blocks_chart) > 0:
            print(f"   Sample OB in response: {order_blocks_chart[0]}")
        if len(fvgs_chart) > 0:
            print(f"   Sample FVG in response: {fvgs_chart[0]}")
        
        # التأكد من إرسال البيانات حتى لو كانت فارغة
        print(f"🔍 [CHART] Checking response keys: {list(chart_data.keys())}")
        print(f"🔍 [CHART] order_blocks type: {type(chart_data.get('order_blocks'))}, length: {len(chart_data.get('order_blocks', []))}")
        print(f"🔍 [CHART] fvgs type: {type(chart_data.get('fvgs'))}, length: {len(chart_data.get('fvgs', []))}")
        print(f"🔍 [CHART] executed_trades type: {type(chart_data.get('executed_trades'))}, length: {len(chart_data.get('executed_trades', []))}")
        
        # التأكد من أن البيانات موجودة قبل الإرسال
        print(f"🔍 [CHART] Final chart_data keys before jsonify: {list(chart_data.keys())}")
        print(f"🔍 [CHART] order_blocks exists: {'order_blocks' in chart_data}")
        print(f"🔍 [CHART] fvgs exists: {'fvgs' in chart_data}")
        print(f"🔍 [CHART] executed_trades exists: {'executed_trades' in chart_data}")
        
        # إرسال Order Blocks و FVGs عبر Socket.IO أيضاً
        try:
            print(f"📡 [CHART] Broadcasting via Socket.IO:")
            print(f"   Order Blocks: {len(order_blocks_chart)}")
            print(f"   FVGs: {len(fvgs_chart)}")
            print(f"   Executed Trades: {len(executed_trades)}")
            broadcast_chart_data(bars_df, contract_id, timeframe=f"{timeframe}m", data_source='LIVE' if use_live else 'SIM', 
                               order_blocks=order_blocks_chart, fvgs=fvgs_chart, executed_trades=executed_trades)
            print(f"✅ [CHART] Broadcast completed")
        except Exception as broadcast_error:
            print(f"⚠️  [CHART] Error broadcasting via Socket.IO: {broadcast_error}")
            import traceback
            traceback.print_exc()
        
        # التحقق من أن البيانات موجودة قبل jsonify
        print(f"🔍 [CHART] PRE-JSONIFY CHECK:")
        print(f"   chart_data type: {type(chart_data)}")
        print(f"   chart_data keys: {list(chart_data.keys())}")
        print(f"   'order_blocks' in chart_data: {'order_blocks' in chart_data}")
        print(f"   'fvgs' in chart_data: {'fvgs' in chart_data}")
        print(f"   'executed_trades' in chart_data: {'executed_trades' in chart_data}")
        if 'order_blocks' in chart_data:
            print(f"   order_blocks value: {chart_data['order_blocks']}")
            print(f"   order_blocks length: {len(chart_data['order_blocks'])}")
        if 'fvgs' in chart_data:
            print(f"   fvgs value: {chart_data['fvgs']}")
            print(f"   fvgs length: {len(chart_data['fvgs'])}")
        if 'executed_trades' in chart_data:
            print(f"   executed_trades value: {chart_data['executed_trades']}")
            print(f"   executed_trades length: {len(chart_data['executed_trades'])}")
        
        # تحويل إلى JSON للتأكد (json already imported at start of function)
        json_str = json.dumps(chart_data, default=str, indent=2)
        print(f"🔍 [CHART] JSON string length: {len(json_str)}")
        print(f"🔍 [CHART] JSON contains 'order_blocks': {'order_blocks' in json_str}")
        print(f"🔍 [CHART] JSON contains 'fvgs': {'fvgs' in json_str}")
        print(f"🔍 [CHART] JSON contains 'executed_trades': {'executed_trades' in json_str}")
        
        # التحقق من chart_data مرة أخيرة قبل jsonify
        print(f"🔍 [CHART] FINAL CHECK before jsonify:")
        print(f"   chart_data keys count: {len(chart_data.keys())}")
        print(f"   chart_data has 'order_blocks': {'order_blocks' in chart_data}")
        print(f"   chart_data has 'fvgs': {'fvgs' in chart_data}")
        print(f"   chart_data has 'executed_trades': {'executed_trades' in chart_data}")
        
        # إرسال response
        try:
            # Force convert to dict to ensure all fields are included
            final_chart_data = dict(chart_data)
            # Explicitly ensure all fields are present
            final_chart_data['order_blocks'] = list(final_chart_data.get('order_blocks', []))
            final_chart_data['fvgs'] = list(final_chart_data.get('fvgs', []))
            final_chart_data['executed_trades'] = list(final_chart_data.get('executed_trades', []))
            
            print(f"🔍 [CHART] FINAL chart_data before jsonify:")
            print(f"   Final chart_data keys: {list(final_chart_data.keys())}")
            print(f"   'order_blocks' in final_chart_data: {'order_blocks' in final_chart_data}")
            print(f"   'fvgs' in final_chart_data: {'fvgs' in final_chart_data}")
            print(f"   'executed_trades' in final_chart_data: {'executed_trades' in final_chart_data}")
            
            response = jsonify(final_chart_data)
            print(f"✅ [CHART] jsonify successful")
            
            # التحقق من response بعد jsonify
            response_data = response.get_json()
            print(f"🔍 [CHART] Response after jsonify:")
            print(f"   Response type: {type(response)}")
            print(f"   Response data keys: {list(response_data.keys()) if response_data else 'None'}")
            if response_data:
                print(f"   Response has 'order_blocks': {'order_blocks' in response_data}")
                print(f"   Response has 'fvgs': {'fvgs' in response_data}")
                print(f"   Response has 'executed_trades': {'executed_trades' in response_data}")
                if 'order_blocks' in response_data:
                    print(f"   Response order_blocks: {response_data['order_blocks']} (length: {len(response_data['order_blocks'])})")
                if 'fvgs' in response_data:
                    print(f"   Response fvgs: {response_data['fvgs']} (length: {len(response_data['fvgs'])})")
                if 'executed_trades' in response_data:
                    print(f"   Response executed_trades: {response_data['executed_trades']} (length: {len(response_data['executed_trades'])})")
            
            return response
        except Exception as jsonify_error:
            print(f"❌ [CHART] Error in jsonify: {jsonify_error}")
            import traceback
            traceback.print_exc()
            # إرجاع response بدون هذه الحقول في حالة الخطأ
            fallback_data = {
                'success': True,
                'labels': bars_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
                'data': {
                    'open': bars_df['open'].tolist(),
                    'high': bars_df['high'].tolist(),
                    'low': bars_df['low'].tolist(),
                    'close': bars_df['close'].tolist(),
                    'volume': bars_df['volume'].tolist()
                },
                'candles': candles,
                'order_blocks': [],
                'fvgs': [],
                'executed_trades': [],
                'current_price': float(bars_df['close'].iloc[-1]),
                'contract': contract_id,
                'timeframe': f"{timeframe}m",
                'data_source': 'LIVE' if use_live else 'SIM'
            }
            return jsonify(fallback_data)
    
    except Exception as e:
        import traceback
        print(f"❌ [CHART] Exception occurred:")
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Chart error: {str(e)}'})

@app.route('/api/market_data', methods=['GET'])
def get_market_data_raw():
    """جلب بيانات السوق الخام (بدون chart) - للتأكد من وصول البيانات"""
    if not trading_system.is_connected:
        return jsonify({'success': False, 'message': 'Not connected'})
    
    contract_id = request.args.get('contract', trading_system.selected_contract)
    use_live = request.args.get('live', 'false').lower() == 'true'
    
    try:
        import requests
        from datetime import datetime, timedelta
        
        url = f"{trading_system.api_endpoint}/api/History/retrieveBars"
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {trading_system.session_token}'
        }
        
        # سحب بيانات اليوم السابق + اليوم الحالي (يومين)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=2)  # آخر يومين
        
        payload = {
            "contractId": contract_id,
            "live": use_live,
            "startTime": start_time.isoformat() + "Z",
            "endTime": end_time.isoformat() + "Z",
            "unit": 2,  # Minute
            "unitNumber": 5,
            "limit": 500,  # زيادة الحد الأقصى لليومين
            "includePartialBar": True
        }
        
        print(f"📊 [MARKET_DATA] Fetching raw data for {contract_id}...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        
        if result.get('success'):
            bars = result.get('bars', [])
            print(f"✅ [MARKET_DATA] Received {len(bars)} bars")
            return jsonify({
                'success': True,
                'contract': contract_id,
                'bars_count': len(bars),
                'bars': bars[:10] if bars else [],  # أول 10 فقط للعرض
                'latest_bar': bars[-1] if bars else None,
                'data_source': 'LIVE' if use_live else 'SIM'
            })
        else:
            # جرب Sim
            payload['live'] = False
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                bars = result.get('bars', [])
                return jsonify({
                    'success': True,
                    'contract': contract_id,
                    'bars_count': len(bars),
                    'bars': bars[:10] if bars else [],
                    'latest_bar': bars[-1] if bars else None,
                    'data_source': 'SIM'
                })
            
            return jsonify({
                'success': False,
                'message': result.get('errorMessage', 'Unknown error'),
                'error_code': result.get('errorCode')
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/upload_model', methods=['POST'])
def api_upload_model():
    """رفع ML Model"""
    if 'model' not in request.files:
        return jsonify({'success': False, 'message': 'No model file'})
    
    file = request.files['model']
    
    if not file.filename.endswith('.pkl'):
        return jsonify({'success': False, 'message': 'Only .pkl files allowed'})
    
    try:
        # حفظ الملف
        models_dir = r'C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\ml_models'
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, file.filename)
        file.save(model_path)
        
        # تحميل Model
        success, message = load_ml_model(model_path)
        
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/evaluate_setup', methods=['POST'])
def api_evaluate_setup():
    """تقييم Setup"""
    data = request.json
    
    result, error = evaluate_setup_with_ml(data)
    
    if error:
        return jsonify({'success': False, 'message': error})
    
    return jsonify({'success': True, 'result': result})

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """بدء التداول الآلي"""
    if not trading_system.is_connected:
        return jsonify({'success': False, 'message': 'Not connected'})
    
    if not trading_system.ml_model:
        return jsonify({'success': False, 'message': 'No ML model loaded'})
    
    if not trading_system.selected_account_id:
        return jsonify({'success': False, 'message': 'No account selected'})
    
    trading_system.is_trading = True
    trading_system.start_auto_trading_loop()
    print("🚀 [AUTO] Auto-trading started with ML model")
    trading_system.add_log("Auto trading started", "INFO")
    broadcast_status()
    return jsonify({'success': True, 'message': 'Auto-trading started'})

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    """إيقاف التداول الآلي"""
    trading_system.is_trading = False
    trading_system.add_log("Auto trading stopped", "INFO")
    broadcast_status()
    return jsonify({'success': True, 'message': 'Auto-trading stopped'})

@app.route('/api/stats')
def get_stats():
    """إحصائيات التداول"""
    return jsonify(stats_payload())

@app.route('/api/logs')
def get_logs():
    """إرجاع آخر سجلات النظام"""
    return jsonify({
        'success': True,
        'logs': list(trading_system.auto_logs)
    })

@app.route('/api/today_signals_csv_info')
def get_today_signals_csv_info():
    """معلومات عن ملف CSV لإشارات اليوم"""
    try:
        today = datetime.utcnow().date()
        csv_file = os.path.join(LIVE_SIGNALS_DIR, f"signals_{today.strftime('%Y%m%d')}.csv")
        
        if not os.path.exists(csv_file):
            return jsonify({
                'success': False,
                'exists': False,
                'file_path': csv_file,
                'date': today.isoformat(),
                'message': f'لا يوجد ملف CSV لليوم {today.strftime("%Y-%m-%d")}'
            })
        
        # قراءة الملف لمعرفة عدد الإشارات
        try:
            df = pd.read_csv(csv_file)
            total_signals = len(df)
            executed = len(df[df['status'] == 'EXECUTED']) if 'status' in df.columns else 0
            skipped = len(df[df['status'] == 'SKIPPED']) if 'status' in df.columns else 0
            file_size = os.path.getsize(csv_file)
        except Exception as e:
            total_signals = 0
            executed = 0
            skipped = 0
            file_size = 0
        
        return jsonify({
            'success': True,
            'exists': True,
            'file_path': csv_file,
            'file_name': os.path.basename(csv_file),
            'date': today.isoformat(),
            'total_signals': total_signals,
            'executed': executed,
            'skipped': skipped,
            'file_size_bytes': file_size,
            'file_size_kb': round(file_size / 1024, 2) if file_size > 0 else 0,
            'download_url': '/api/today_signals_csv'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/today_signals_csv')
def get_today_signals_csv():
    """تحميل ملف CSV لإشارات اليوم"""
    from flask import send_file
    
    try:
        today = datetime.utcnow().date()
        csv_file = os.path.join(LIVE_SIGNALS_DIR, f"signals_{today.strftime('%Y%m%d')}.csv")
        
        if not os.path.exists(csv_file):
            return jsonify({
                'success': False,
                'message': f'لا يوجد ملف CSV لليوم {today.strftime("%Y-%m-%d")}'
            })
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'signals_{today.strftime("%Y%m%d")}.csv'
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/today_signals')
def get_today_signals():
    """عرض جميع إشارات اليوم (منفذة ومتجاهلة)"""
    from datetime import datetime, timedelta
    
    try:
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())
        
        # تصفية الصفقات المنفذة لليوم
        executed_today = []
        for trade in trading_system.executed_trades:
            try:
                # تحويل timestamp إلى datetime إذا كان string
                if isinstance(trade.get('timestamp'), str):
                    trade_time = pd.to_datetime(trade['timestamp'])
                else:
                    trade_time = trade.get('timestamp')
                
                if isinstance(trade_time, pd.Timestamp):
                    trade_time = trade_time.to_pydatetime()
                
                if trade_time and trade_time.date() == today:
                    executed_today.append(trade)
            except Exception:
                # إذا فشل التحويل، أضف الصفقة على أي حال
                executed_today.append(trade)
        
        # تصفية الصفقات المتجاهلة لليوم
        skipped_today = []
        for trade in trading_system.skipped_trades:
            try:
                if isinstance(trade.get('timestamp'), str):
                    trade_time = pd.to_datetime(trade['timestamp'])
                else:
                    trade_time = trade.get('timestamp')
                
                if isinstance(trade_time, pd.Timestamp):
                    trade_time = trade_time.to_pydatetime()
                
                if trade_time and trade_time.date() == today:
                    skipped_today.append(trade)
            except Exception:
                skipped_today.append(trade)
        
        # إحصائيات اليوم
        stats_today = {
            'total_executed': len(executed_today),
            'total_skipped': len(skipped_today),
            'total_signals': len(executed_today) + len(skipped_today),
            'win_count': sum(1 for t in executed_today if t.get('result') == 'WIN'),
            'loss_count': sum(1 for t in executed_today if t.get('result') == 'LOSS'),
            'pending_count': sum(1 for t in executed_today if t.get('result') == 'PENDING' or t.get('result') is None),
        }
        
        # حساب Win Rate
        completed = stats_today['win_count'] + stats_today['loss_count']
        if completed > 0:
            stats_today['win_rate'] = round((stats_today['win_count'] / completed) * 100, 2)
        else:
            stats_today['win_rate'] = 0.0
        
        # حساب PnL الإجمالي
        total_pnl = sum(float(t.get('pnl', 0)) for t in executed_today)
        stats_today['total_pnl'] = round(total_pnl, 2)
        
        return jsonify({
            'success': True,
            'date': today.isoformat(),
            'stats': stats_today,
            'executed': executed_today,
            'skipped': skipped_today,
            'all_signals': executed_today + skipped_today  # جميع الإشارات معاً
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'executed': [],
            'skipped': [],
            'all_signals': []
        })

# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("="*80)
    print("🚀 REAL Trading Platform")
    print("="*80)
    print()
    
    if not PROJECTX_AVAILABLE:
        print("❌ WARNING: ProjectX API not installed!")
        print("📦 Install: pip install projectx-api")
        print()
    else:
        print("✅ ProjectX API available")
        print()
    
    print("🌐 Starting server...")
    print("📍 URL: http://localhost:5000")
    print()
    print("⚙️  Features:")
    print("   ✅ REAL ProjectX API Connection")
    print("   ✅ REAL Account Data")
    print("   ✅ REAL Order Execution")
    print("   ✅ ML Model Upload")
    print("   ✅ Auto-Trading")
    print()
    print("💡 Press Ctrl+C to stop")
    print("="*80)
    print()
    
    # بدء Event Loop
    trading_system.start_event_loop()
    
    # فتح المتصفح
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    
    # بدء الخادم مع WebSocket
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)


