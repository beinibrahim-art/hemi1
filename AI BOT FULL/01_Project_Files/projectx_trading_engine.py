"""
ProjectX Trading Engine - Full Integration
==========================================
Handles all real trading operations via ProjectX API
"""

import requests
import json
from datetime import datetime, timedelta
import pandas as pd


class ProjectXTradingEngine:
    """محرك التداول الحقيقي مع ProjectX API"""
    
    def __init__(self, api_endpoint, session_token, account_id):
        self.api_endpoint = api_endpoint
        self.session_token = session_token
        self.account_id = account_id
        self.headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {session_token}'
        }
    
    # ============================================================
    # MARKET DATA
    # ============================================================
    
    def search_contracts(self, search_text="ES", live=False):
        """البحث عن Contracts"""
        url = f"{self.api_endpoint}/api/Contract/search"
        
        payload = {
            "searchText": search_text,
            "live": live
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('contracts', [])
            else:
                print(f"⚠️  Contract search failed: {result.get('errorMessage')}")
                return []
        
        except Exception as e:
            print(f"❌ Error searching contracts: {e}")
            return []
    
    def get_contract_by_id(self, contract_id):
        """الحصول على Contract محدد"""
        url = f"{self.api_endpoint}/api/Contract/searchById"
        
        payload = {"contractId": contract_id}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('contract')
            else:
                return None
        
        except Exception as e:
            print(f"❌ Error getting contract: {e}")
            return None
    
    def get_available_contracts(self, live=True):
        """قائمة جميع Contracts المتاحة"""
        url = f"{self.api_endpoint}/api/Contract/available"
        
        payload = {"live": live}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('contracts', [])
            else:
                return []
        
        except Exception as e:
            print(f"❌ Error getting available contracts: {e}")
            return []
    
    def retrieve_bars(self, contract_id, start_time, end_time, unit=2, unit_number=5, limit=1000):
        """
        سحب البيانات التاريخية (OHLC)
        
        Parameters:
        - contract_id: معرف العقد (مثل "CON.F.US.EP.U25")
        - start_time: وقت البداية (ISO format)
        - end_time: وقت النهاية (ISO format)
        - unit: 1=Second, 2=Minute, 3=Hour, 4=Day, 5=Week, 6=Month
        - unit_number: عدد الوحدات (مثلاً 5 = 5 دقائق)
        - limit: الحد الأقصى (20000 max)
        """
        url = f"{self.api_endpoint}/api/History/retrieveBars"
        
        payload = {
            "contractId": contract_id,
            "live": False,  # استخدام Sim data للباكتست
            "startTime": start_time,
            "endTime": end_time,
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": False
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            result = response.json()
            
            if result.get('success'):
                bars = result.get('bars', [])
                
                # تحويل إلى DataFrame
                if bars:
                    df = pd.DataFrame(bars)
                    df['t'] = pd.to_datetime(df['t'])
                    df.rename(columns={
                        't': 'timestamp',
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    }, inplace=True)
                    return df
                else:
                    return pd.DataFrame()
            else:
                print(f"⚠️  Retrieve bars failed: {result.get('errorMessage')}")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"❌ Error retrieving bars: {e}")
            return pd.DataFrame()
    
    # ============================================================
    # ORDER MANAGEMENT
    # ============================================================
    
    def place_order(self, contract_id, order_type, side, size, 
                    limit_price=None, stop_price=None, trail_price=None,
                    sl_ticks=None, tp_ticks=None, custom_tag=None):
        """
        تنفيذ أمر حقيقي
        
        Parameters:
        - contract_id: معرف العقد
        - order_type: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
        - side: 0=Buy, 1=Sell
        - size: عدد العقود
        - sl_ticks: عدد التكات لـ Stop Loss
        - tp_ticks: عدد التكات لـ Take Profit
        """
        url = f"{self.api_endpoint}/api/Order/place"
        
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": side,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
            "customTag": custom_tag
        }
        
        # إضافة Stop Loss Bracket
        if sl_ticks:
            payload["stopLossBracket"] = {
                "ticks": sl_ticks,
                "type": 4  # Stop order
            }
        
        # إضافة Take Profit Bracket
        if tp_ticks:
            payload["takeProfitBracket"] = {
                "ticks": tp_ticks,
                "type": 1  # Limit order
            }
        
        try:
            # طباعة وحفظ التفاصيل
            log_msg = f"""
{'='*80}
📤 SENDING ORDER REQUEST
{'='*80}
URL: {url}
Payload:
{json.dumps(payload, indent=2)}
{'='*80}
"""
            print(log_msg)
            
            # حفظ في ملف أيضاً
            try:
                with open('order_debug.log', 'a', encoding='utf-8') as f:
                    from datetime import datetime
                    f.write(f"\n[{datetime.now()}]\n{log_msg}\n")
            except:
                pass
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            response_log = f"""
{'='*80}
📥 RESPONSE RECEIVED
{'='*80}
Status Code: {response.status_code}
Response Body:
{response.text[:1000]}
{'='*80}
"""
            print(response_log)
            
            # حفظ في ملف أيضاً
            try:
                with open('order_debug.log', 'a', encoding='utf-8') as f:
                    f.write(response_log + "\n")
            except:
                pass
            
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                order_id = result.get('orderId')
                print(f"✅ Order placed: ID {order_id}")
                return order_id
            else:
                error_msg = result.get('errorMessage') or result.get('error') or 'Unknown error'
                error_code = result.get('errorCode')
                print(f"❌ Order failed:")
                print(f"   Error Code: {error_code}")
                print(f"   Error Message: {error_msg}")
                print(f"   Full response: {json.dumps(result, indent=2)}")
                return None
        
        except Exception as e:
            print(f"❌ Error placing order: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cancel_order(self, order_id):
        """إلغاء أمر"""
        url = f"{self.api_endpoint}/api/Order/cancel"
        
        payload = {
            "accountId": self.account_id,
            "orderId": order_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            return result.get('success', False)
        
        except Exception as e:
            print(f"❌ Error canceling order: {e}")
            return False
    
    def modify_order(self, order_id, size=None, limit_price=None, stop_price=None):
        """تعديل أمر"""
        url = f"{self.api_endpoint}/api/Order/modify"
        
        payload = {
            "accountId": self.account_id,
            "orderId": order_id,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            return result.get('success', False)
        
        except Exception as e:
            print(f"❌ Error modifying order: {e}")
            return False
    
    def get_open_orders(self):
        """الحصول على الأوامر المفتوحة"""
        url = f"{self.api_endpoint}/api/Order/searchOpen"
        
        payload = {"accountId": self.account_id}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('orders', [])
            else:
                return []
        
        except Exception as e:
            print(f"❌ Error getting open orders: {e}")
            return []
    
    def search_orders(self, start_time, end_time=None):
        """البحث عن الأوامر في فترة زمنية"""
        url = f"{self.api_endpoint}/api/Order/search"
        
        payload = {
            "accountId": self.account_id,
            "startTimestamp": start_time,
            "endTimestamp": end_time
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('orders', [])
            else:
                return []
        
        except Exception as e:
            print(f"❌ Error searching orders: {e}")
            return []
    
    # ============================================================
    # POSITION MANAGEMENT
    # ============================================================
    
    def get_open_positions(self):
        """الحصول على المراكز المفتوحة"""
        url = f"{self.api_endpoint}/api/Position/searchOpen"
        
        payload = {"accountId": self.account_id}
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('positions', [])
            else:
                return []
        
        except Exception as e:
            print(f"❌ Error getting open positions: {e}")
            return []
    
    def close_position(self, contract_id):
        """إغلاق مركز كاملاً"""
        url = f"{self.api_endpoint}/api/Position/closeContract"
        
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Position closed: {contract_id}")
                return True
            else:
                print(f"❌ Close failed: {result.get('errorMessage')}")
                return False
        
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False
    
    def partial_close_position(self, contract_id, size):
        """إغلاق جزء من مركز"""
        url = f"{self.api_endpoint}/api/Position/partialCloseContract"
        
        payload = {
            "accountId": self.account_id,
            "contractId": contract_id,
            "size": size
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            return result.get('success', False)
        
        except Exception as e:
            print(f"❌ Error partial closing position: {e}")
            return False
    
    # ============================================================
    # TRADE HISTORY
    # ============================================================
    
    def search_trades(self, start_time, end_time=None):
        """البحث عن الصفقات المنفذة"""
        url = f"{self.api_endpoint}/api/Trade/search"
        
        payload = {
            "accountId": self.account_id,
            "startTimestamp": start_time,
            "endTimestamp": end_time
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success'):
                return result.get('trades', [])
            else:
                return []
        
        except Exception as e:
            print(f"❌ Error searching trades: {e}")
            return []
    
    # ============================================================
    # SMART TRADING FUNCTIONS
    # ============================================================
    
    def execute_ml_trade(self, setup, ml_decision, contract_id="CON.F.US.EP.U25"):
        """
        تنفيذ صفقة بناءً على قرار ML
        
        Parameters:
        - setup: dict مع (type, entry, sl, tp)
        - ml_decision: TAKE أو SKIP
        - contract_id: معرف العقد
        """
        
        if ml_decision != "TAKE":
            print(f"⏭️  ML Decision: SKIP")
            return None
        
        # التحقق من صحة Contract
        print(f"🔍 Verifying contract: {contract_id}")
        contract = self.get_contract_by_id(contract_id)
        if not contract:
            print(f"❌ Contract not found: {contract_id}")
            print(f"   Try using search_contracts() to find available contracts")
            return None
        else:
            print(f"✅ Contract verified: {contract.get('name')} - {contract.get('description')}")
            print(f"   Tick Size: {contract.get('tickSize')}, Tick Value: {contract.get('tickValue')}")
        
        # استخراج التفاصيل
        trade_type = setup.get('type')  # BUY or SELL
        entry = setup.get('entry')
        sl = setup.get('sl')
        tp = setup.get('tp')
        
        # تحديد side
        side = 0 if trade_type == 'BUY' else 1  # 0=Buy, 1=Sell
        
        # حساب SL/TP بالتكات (Ticks)
        # لـ ES: 1 tick = 0.25
        tick_size = 0.25
        
        if trade_type == 'BUY':
            sl_ticks = int((entry - sl) / tick_size)
            tp_ticks = int((tp - entry) / tick_size)
        else:  # SELL
            sl_ticks = int((sl - entry) / tick_size)
            tp_ticks = int((entry - tp) / tick_size)
        
        print(f"📊 Executing {trade_type} order:")
        print(f"   Account ID: {self.account_id}")
        print(f"   Contract: {contract_id}")
        print(f"   Entry: {entry}")
        print(f"   SL: {sl} ({sl_ticks} ticks)")
        print(f"   TP: {tp} ({tp_ticks} ticks)")
        
        # التحقق من القيم
        if sl_ticks <= 0 or tp_ticks <= 0:
            print(f"❌ Invalid ticks: SL={sl_ticks}, TP={tp_ticks}")
            return None
        
        # تنفيذ Market Order مع SL/TP
        order_id = self.place_order(
            contract_id=contract_id,
            order_type=2,  # Market
            side=side,
            size=1,  # عقد واحد
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            custom_tag=f"ML_ICT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if order_id:
            print(f"✅ Order successfully placed: {order_id}")
        else:
            print(f"❌ Order placement failed - check logs above")
        
        return order_id
    
    def get_current_price(self, contract_id="CON.F.US.EP.U25"):
        """الحصول على السعر الحالي"""
        # استخدام retrieve_bars للحصول على آخر شمعة
        end_time = datetime.utcnow().isoformat() + "Z"
        start_time = (datetime.utcnow() - timedelta(minutes=10)).isoformat() + "Z"
        
        bars = self.retrieve_bars(
            contract_id=contract_id,
            start_time=start_time,
            end_time=end_time,
            unit=2,  # Minute
            unit_number=1,
            limit=1
        )
        
        if not bars.empty:
            return bars.iloc[-1]['close']
        else:
            return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_order_status(status_code):
    """تحويل Order Status Code إلى نص"""
    statuses = {
        0: "None",
        1: "Open",
        2: "Filled",
        3: "Cancelled",
        4: "Expired",
        5: "Rejected",
        6: "Pending"
    }
    return statuses.get(status_code, "Unknown")

def format_order_type(type_code):
    """تحويل Order Type Code إلى نص"""
    types = {
        0: "Unknown",
        1: "Limit",
        2: "Market",
        3: "StopLimit",
        4: "Stop",
        5: "TrailingStop",
        6: "JoinBid",
        7: "JoinAsk"
    }
    return types.get(type_code, "Unknown")

def format_position_type(type_code):
    """تحويل Position Type Code إلى نص"""
    types = {
        0: "Undefined",
        1: "Long",
        2: "Short"
    }
    return types.get(type_code, "Unknown")

