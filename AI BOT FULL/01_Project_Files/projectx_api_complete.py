"""
ProjectX API - Complete Implementation
========================================
Based on official ProjectX Gateway API documentation
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any


class ProjectXAPI:
    """
    Complete ProjectX API implementation following official documentation
    https://gateway.docs.projectx.com/
    """
    
    def __init__(self, platform: str = "topstepx"):
        """
        Initialize ProjectX API client
        
        Args:
            platform: Platform name (topstepx, e8x, etc.)
        """
        self.base_url = f"https://api.{platform}.com"
        self.session_token = None
        self.headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json'
        }
    
    # =========================================================================
    # AUTHENTICATION
    # =========================================================================
    
    def login_with_api_key(self, username: str, api_key: str) -> Dict[str, Any]:
        """
        Authenticate with API Key
        
        API: POST /api/Auth/loginKey
        
        Args:
            username: User's email/username
            api_key: API Key from platform
        
        Returns:
            dict with 'success', 'token', 'errorCode', 'errorMessage'
        """
        url = f"{self.base_url}/api/Auth/loginKey"
        
        payload = {
            "userName": username,
            "apiKey": api_key
        }
        
        print(f"🔐 Authenticating...")
        print(f"   URL: {url}")
        print(f"   Username: {username}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            print(f"   Status: {response.status_code}")
            
            if result.get('success') or result.get('errorCode') == 0:
                self.session_token = result.get('token')
                self.headers['Authorization'] = f'Bearer {self.session_token}'
                print(f"✅ Authentication successful")
                return result
            else:
                print(f"❌ Authentication failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    # =========================================================================
    # ACCOUNT
    # =========================================================================
    
    def search_accounts(self, only_active: bool = True) -> Dict[str, Any]:
        """
        Search for accounts
        
        API: POST /api/Account/search
        
        Args:
            only_active: Whether to filter only active accounts
        
        Returns:
            dict with 'accounts', 'success', 'errorCode', 'errorMessage'
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Account/search"
        
        payload = {
            "onlyActiveAccounts": only_active
        }
        
        print(f"🔍 Searching accounts...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                accounts = result.get('accounts', [])
                print(f"✅ Found {len(accounts)} account(s)")
                for acc in accounts:
                    print(f"   - ID: {acc.get('id')} | Name: {acc.get('name')} | Balance: ${acc.get('balance'):,.2f} | Can Trade: {acc.get('canTrade')}")
                return result
            else:
                print(f"❌ Search failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    # =========================================================================
    # CONTRACTS
    # =========================================================================
    
    def list_available_contracts(self, live: bool = False) -> Dict[str, Any]:
        """
        List all available contracts
        
        API: POST /api/Contract/available
        
        Args:
            live: Whether to retrieve live contracts (True) or sim contracts (False)
        
        Returns:
            dict with 'contracts', 'success', 'errorCode', 'errorMessage'
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Contract/available"
        
        payload = {
            "live": live
        }
        
        print(f"📊 Listing available contracts (live={live})...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                contracts = result.get('contracts', [])
                print(f"✅ Found {len(contracts)} contract(s)")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    def search_contracts(self, search_text: str, live: bool = False) -> Dict[str, Any]:
        """
        Search for contracts by name
        
        API: POST /api/Contract/search
        
        Args:
            search_text: Name to search for (e.g., "ES", "NQ")
            live: Whether to search live or sim contracts
        
        Returns:
            dict with 'contracts', 'success', 'errorCode', 'errorMessage'
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Contract/search"
        
        payload = {
            "searchText": search_text,
            "live": live
        }
        
        print(f"🔍 Searching contracts for '{search_text}'...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                contracts = result.get('contracts', [])
                print(f"✅ Found {len(contracts)} contract(s)")
                for contract in contracts:
                    print(f"   - ID: {contract.get('id')} | Name: {contract.get('name')} | {contract.get('description')}")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    def get_contract_by_id(self, contract_id: str) -> Dict[str, Any]:
        """
        Get contract details by ID
        
        API: POST /api/Contract/searchById
        
        Args:
            contract_id: Contract ID (e.g., "CON.F.US.EP.U25")
        
        Returns:
            dict with 'contract', 'success', 'errorCode', 'errorMessage'
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Contract/searchById"
        
        payload = {
            "contractId": contract_id
        }
        
        print(f"🔍 Getting contract details for '{contract_id}'...")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                contract = result.get('contract')
                if contract:
                    print(f"✅ Contract found:")
                    print(f"   - Name: {contract.get('name')}")
                    print(f"   - Description: {contract.get('description')}")
                    print(f"   - Tick Size: {contract.get('tickSize')}")
                    print(f"   - Tick Value: {contract.get('tickValue')}")
                    print(f"   - Active: {contract.get('activeContract')}")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    # =========================================================================
    # ORDERS
    # =========================================================================
    
    def place_order(self,
                    account_id: int,
                    contract_id: str,
                    order_type: int,
                    side: int,
                    size: int,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_price: Optional[float] = None,
                    custom_tag: Optional[str] = None,
                    stop_loss_ticks: Optional[int] = None,
                    stop_loss_type: int = 4,
                    take_profit_ticks: Optional[int] = None,
                    take_profit_type: int = 1) -> Dict[str, Any]:
        """
        Place an order
        
        API: POST /api/Order/place
        
        Args:
            account_id: Account ID
            contract_id: Contract ID (e.g., "CON.F.US.EP.U25")
            order_type: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop, 6=JoinBid, 7=JoinAsk
            side: 0=Bid (buy), 1=Ask (sell)
            size: Number of contracts
            limit_price: Limit price (optional)
            stop_price: Stop price (optional)
            trail_price: Trail price (optional)
            custom_tag: Custom tag (optional)
            stop_loss_ticks: Number of ticks for stop loss (optional)
            stop_loss_type: Stop loss bracket type (default: 4=Stop)
            take_profit_ticks: Number of ticks for take profit (optional)
            take_profit_type: Take profit bracket type (default: 1=Limit)
        
        Returns:
            dict with 'orderId', 'success', 'errorCode', 'errorMessage'
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Order/place"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": order_type,
            "side": side,
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "trailPrice": trail_price,
            "customTag": custom_tag
        }
        
        # Add stop loss bracket if specified
        if stop_loss_ticks is not None and stop_loss_ticks > 0:
            payload["stopLossBracket"] = {
                "ticks": stop_loss_ticks,
                "type": stop_loss_type
            }
        
        # Add take profit bracket if specified
        if take_profit_ticks is not None and take_profit_ticks > 0:
            payload["takeProfitBracket"] = {
                "ticks": take_profit_ticks,
                "type": take_profit_type
            }
        
        print("\n" + "="*80)
        print("📤 PLACING ORDER")
        print("="*80)
        print(f"URL: {url}")
        print(f"Account ID: {account_id}")
        print(f"Contract: {contract_id}")
        print(f"Type: {order_type} (1=Limit, 2=Market, 4=Stop)")
        print(f"Side: {side} (0=Buy, 1=Sell)")
        print(f"Size: {size}")
        if stop_loss_ticks:
            print(f"Stop Loss: {stop_loss_ticks} ticks (type={stop_loss_type})")
        if take_profit_ticks:
            print(f"Take Profit: {take_profit_ticks} ticks (type={take_profit_type})")
        print(f"\nPayload:")
        print(json.dumps(payload, indent=2))
        print("="*80)
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            print(f"\n📥 RESPONSE:")
            print(f"Status Code: {response.status_code}")
            print(f"Body: {response.text[:1000]}")
            print("="*80 + "\n")
            
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                order_id = result.get('orderId')
                print(f"✅ Order placed successfully!")
                print(f"   Order ID: {order_id}")
                return result
            else:
                error_code = result.get('errorCode')
                error_msg = result.get('errorMessage') or 'Unknown error'
                print(f"❌ Order failed!")
                print(f"   Error Code: {error_code}")
                print(f"   Error Message: {error_msg}")
                return result
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'errorMessage': str(e)}
    
    def search_open_orders(self, account_id: int) -> Dict[str, Any]:
        """
        Get all open orders for an account
        
        API: POST /api/Order/searchOpen
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Order/searchOpen"
        
        payload = {
            "accountId": account_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                orders = result.get('orders', [])
                print(f"✅ Found {len(orders)} open order(s)")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    # =========================================================================
    # POSITIONS
    # =========================================================================
    
    def search_open_positions(self, account_id: int) -> Dict[str, Any]:
        """
        Get all open positions for an account
        
        API: POST /api/Position/searchOpen
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Position/searchOpen"
        
        payload = {
            "accountId": account_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                positions = result.get('positions', [])
                print(f"✅ Found {len(positions)} open position(s)")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    def place_stop_loss_order(self, account_id: int, contract_id: str, stop_price: float, size: int = 1, custom_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Place a Stop Loss order for existing position
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            stop_price: Stop price
            size: Number of contracts
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Order/place"
        
        # For long position, SL is a sell stop
        if not custom_tag:
            from datetime import datetime
            custom_tag = f"SL_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": 4,  # Stop
            "side": 1,  # Sell (to close long)
            "size": size,
            "limitPrice": None,
            "stopPrice": stop_price,
            "trailPrice": None,
            "customTag": custom_tag
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                print(f"✅ Stop Loss order placed at {stop_price}")
                return result
            else:
                print(f"❌ SL order failed: {result.get('errorMessage')}")
                return result
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    def place_take_profit_order(self, account_id: int, contract_id: str, limit_price: float, size: int = 1, custom_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Place a Take Profit order for existing position
        
        Args:
            account_id: Account ID
            contract_id: Contract ID
            limit_price: Limit price for take profit
            size: Number of contracts
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Order/place"
        
        # For long position, TP is a sell limit
        if not custom_tag:
            from datetime import datetime
            custom_tag = f"TP_{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "type": 1,  # Limit
            "side": 1,  # Sell (to close long)
            "size": size,
            "limitPrice": limit_price,
            "stopPrice": None,
            "trailPrice": None,
            "customTag": custom_tag
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                print(f"✅ Take Profit order placed at {limit_price}")
                return result
            else:
                print(f"❌ TP order failed: {result.get('errorMessage')}")
                return result
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}
    
    def close_position(self, account_id: int, contract_id: str) -> Dict[str, Any]:
        """
        Close entire position for a contract
        
        API: POST /api/Position/closeContract
        """
        if not self.session_token:
            return {'success': False, 'errorMessage': 'Not authenticated'}
        
        url = f"{self.base_url}/api/Position/closeContract"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id
        }
        
        print(f"🔴 Closing position: {contract_id}")
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            result = response.json()
            
            if result.get('success') or result.get('errorCode') == 0:
                print(f"✅ Position closed")
                return result
            else:
                print(f"❌ Failed: {result.get('errorMessage')}")
                return result
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'errorMessage': str(e)}

