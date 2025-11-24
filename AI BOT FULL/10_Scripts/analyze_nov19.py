import databento as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🎯 تحليل ICT - 19 نوفمبر 2025")
print("="*80)

# تحميل البيانات
print("\n📂 تحميل البيانات...")
store = db.DBNStore.from_file('/mnt/user-data/uploads/glbx-mdp3-20251119_trades_dbn.zst')
df = store.to_df()
df = df[df['symbol'] == 'ESZ5'].copy()

print(f"✅ {len(df):,} صفقة")
print(f"   السعر: {df['price'].min():.2f} - {df['price'].max():.2f}")
print(f"   المدى: {df['price'].max() - df['price'].min():.2f} نقاط")

# إنشاء OHLC
print("\n📊 إنشاء OHLC...")
df['hour'] = df.index.hour
df['delta'] = np.where(df['side'] == 'A', df['size'], -df['size'])

ohlc_5m = df.groupby(pd.Grouper(freq='5T')).agg({
    'price': ['first', 'max', 'min', 'last'],
    'size': 'sum',
    'delta': 'sum'
})
ohlc_5m.columns = ['open', 'high', 'low', 'close', 'volume', 'delta']
ohlc_5m = ohlc_5m.dropna()

print(f"✅ {len(ohlc_5m)} شمعة (5 دقائق)")

# Order Blocks
print("\n🔍 البحث عن Order Blocks...")
obs = []
lookback = 20

for i in range(lookback, len(ohlc_5m)):
    # Bullish OB
    if ohlc_5m['close'].iloc[i] > ohlc_5m['close'].iloc[i-1]:
        for j in range(i-1, max(0, i-lookback), -1):
            if ohlc_5m['close'].iloc[j] < ohlc_5m['open'].iloc[j]:
                move = ohlc_5m['high'].iloc[i] - ohlc_5m['low'].iloc[j]
                if move >= 8:
                    obs.append({
                        'time': ohlc_5m.index[j],
                        'type': 'Bullish',
                        'high': ohlc_5m['high'].iloc[j],
                        'low': ohlc_5m['low'].iloc[j],
                        'strength': move
                    })
                break
    
    # Bearish OB
    elif ohlc_5m['close'].iloc[i] < ohlc_5m['close'].iloc[i-1]:
        for j in range(i-1, max(0, i-lookback), -1):
            if ohlc_5m['close'].iloc[j] > ohlc_5m['open'].iloc[j]:
                move = ohlc_5m['high'].iloc[j] - ohlc_5m['low'].iloc[i]
                if move >= 8:
                    obs.append({
                        'time': ohlc_5m.index[j],
                        'type': 'Bearish',
                        'high': ohlc_5m['high'].iloc[j],
                        'low': ohlc_5m['low'].iloc[j],
                        'strength': move
                    })
                break

obs_df = pd.DataFrame(obs) if obs else pd.DataFrame()
print(f"✅ {len(obs_df)} Order Block")
if len(obs_df) > 0:
    print(f"   Bullish: {len(obs_df[obs_df['type']=='Bullish'])}")
    print(f"   Bearish: {len(obs_df[obs_df['type']=='Bearish'])}")

# Fair Value Gaps
print("\n🔍 البحث عن FVGs...")
fvgs = []

for i in range(2, len(ohlc_5m)):
    # Bullish FVG
    if ohlc_5m['low'].iloc[i] > ohlc_5m['high'].iloc[i-2]:
        gap = ohlc_5m['low'].iloc[i] - ohlc_5m['high'].iloc[i-2]
        if gap >= 2.0:
            fvgs.append({
                'time': ohlc_5m.index[i],
                'type': 'Bullish',
                'top': ohlc_5m['low'].iloc[i],
                'bottom': ohlc_5m['high'].iloc[i-2],
                'size': gap
            })
    
    # Bearish FVG
    elif ohlc_5m['high'].iloc[i] < ohlc_5m['low'].iloc[i-2]:
        gap = ohlc_5m['low'].iloc[i-2] - ohlc_5m['high'].iloc[i]
        if gap >= 2.0:
            fvgs.append({
                'time': ohlc_5m.index[i],
                'type': 'Bearish',
                'top': ohlc_5m['low'].iloc[i-2],
                'bottom': ohlc_5m['high'].iloc[i],
                'size': gap
            })

fvgs_df = pd.DataFrame(fvgs) if fvgs else pd.DataFrame()
print(f"✅ {len(fvgs_df)} Fair Value Gap")

# إيجاد الصفقات
print("\n💡 البحث عن Setups...")

def get_killzone(hour):
    if 2 <= hour < 5: return 'London', 10
    elif 7 <= hour < 10: return 'NY_AM', 10
    elif 13 <= hour < 17: return 'NY_PM', 9
    return None, 0

setups = []

for _, ob in obs_df.iterrows():
    session, priority = get_killzone(ob['time'].hour)
    if priority < 9:
        continue
    
    # ابحث عن Retest
    future = ohlc_5m[(ohlc_5m.index > ob['time']) & 
                     (ohlc_5m.index <= ob['time'] + pd.Timedelta(hours=1))]
    
    for t, c in future.iterrows():
        entry, sl, tp, direction = None, None, None, None
        
        # Bullish Setup
        if ob['type'] == 'Bullish' and c['low'] <= ob['high'] and c['close'] > ob['low']:
            entry = (ob['high'] + ob['low']) / 2
            sl = ob['low'] - 2.0
            tp = entry + 8.0
            direction = 'BUY'
        
        # Bearish Setup
        elif ob['type'] == 'Bearish' and c['high'] >= ob['low'] and c['close'] < ob['high']:
            entry = (ob['high'] + ob['low']) / 2
            sl = ob['high'] + 2.0
            tp = entry - 8.0
            direction = 'SELL'
        
        if entry:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            
            if risk <= 4.0 and reward >= 7.0 and reward/risk >= 2.0:
                setups.append({
                    'time': t,
                    'session': session,
                    'type': direction,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'risk': risk,
                    'target': reward,
                    'rr': reward/risk,
                    'strength': ob['strength']
                })

print(f"✅ {len(setups)} Setup")

# اختيار أفضل 3
if len(setups) > 0:
    setups_df = pd.DataFrame(setups)
    setups_df = setups_df.sort_values(['strength', 'rr'], ascending=[False, False])
    
    # اختر حسب الجلسات
    top_3 = []
    sessions_used = set()
    
    for _, s in setups_df.iterrows():
        if len(top_3) < 3:
            if s['session'] not in sessions_used or len(top_3) == 0:
                top_3.append(s)
                sessions_used.add(s['session'])
    
    # أكمل لـ 3
    if len(top_3) < 3:
        for _, s in setups_df.iterrows():
            if len(top_3) >= 3:
                break
            if not any(x['time'] == s['time'] for x in top_3):
                top_3.append(s)
    
    print("\n" + "="*80)
    print("🏆 أفضل 3 صفقات - 19 نوفمبر 2025")
    print("="*80)
    
    total_target = 0
    total_risk = 0
    
    for i, trade in enumerate(top_3, 1):
        print(f"\n{'='*80}")
        print(f"📍 صفقة #{i} - {trade['type']}")
        print(f"{'='*80}")
        print(f"   ⏰ الوقت:         {trade['time']}")
        print(f"   🎪 الجلسة:        {trade['session']}")
        print(f"   🎯 Entry:         {trade['entry']:.2f}")
        print(f"   🛑 Stop Loss:     {trade['sl']:.2f} (Risk: {trade['risk']:.2f} نقاط)")
        print(f"   ✅ Take Profit:   {trade['tp']:.2f} (Target: {trade['target']:.2f} نقاط)")
        print(f"   💰 R:R:           1:{trade['rr']:.2f}")
        print(f"   ⚡ القوة:         {trade['strength']:.1f} نقاط")
        print(f"   💵 الربح المحتمل: ${trade['target'] * 50:.0f} لكل عقد")
        
        total_target += trade['target']
        total_risk += trade['risk']
    
    print(f"\n{'='*80}")
    print("📊 الملخص:")
    print(f"{'='*80}")
    print(f"   إجمالي الهدف:     {total_target:.1f} نقاط = ${total_target * 50:.0f}")
    print(f"   إجمالي المخاطرة:  {total_risk:.1f} نقاط = ${total_risk * 50:.0f}")
    print(f"   متوسط الهدف:      {total_target/3:.1f} نقاط")
    print(f"   متوسط R:R:        1:{sum(t['rr'] for t in top_3)/3:.2f}")
    
    # حفظ
    pd.DataFrame(top_3).to_csv('/mnt/user-data/outputs/nov19_trades.csv', index=False)
    print(f"\n✅ تم الحفظ: nov19_trades.csv")
    
    # رسم بياني
    print("\n📊 إنشاء رسم بياني...")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # رسم الشموع
    ax.plot(ohlc_5m.index, ohlc_5m['close'], 'k-', linewidth=0.8, alpha=0.5, label='Price')
    
    # Order Blocks
    for _, ob in obs_df.iterrows():
        color = 'green' if ob['type'] == 'Bullish' else 'red'
        alpha = 0.1
        ax.axhspan(ob['low'], ob['high'], 
                   xmin=(ob['time'] - ohlc_5m.index[0]).total_seconds() / (ohlc_5m.index[-1] - ohlc_5m.index[0]).total_seconds(),
                   xmax=min(1.0, (ob['time'] - ohlc_5m.index[0] + pd.Timedelta(hours=2)).total_seconds() / (ohlc_5m.index[-1] - ohlc_5m.index[0]).total_seconds()),
                   color=color, alpha=alpha)
    
    # الصفقات
    for i, trade in enumerate(top_3, 1):
        color = 'blue' if trade['type'] == 'BUY' else 'orange'
        ax.scatter(trade['time'], trade['entry'], s=200, color=color, marker='^' if trade['type'] == 'BUY' else 'v', 
                   edgecolors='black', linewidth=2, zorder=5, label=f"#{i} {trade['type']}")
        ax.hlines(trade['sl'], trade['time'], trade['time'] + pd.Timedelta(hours=2), colors='red', linestyles='--', linewidth=2)
        ax.hlines(trade['tp'], trade['time'], trade['time'] + pd.Timedelta(hours=2), colors='green', linestyles='--', linewidth=2)
    
    ax.set_title('ICT Analysis - 19 نوفمبر 2025 (ESZ5)', fontsize=16, fontweight='bold')
    ax.set_xlabel('الوقت', fontsize=12)
    ax.set_ylabel('السعر', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/nov19_chart.png', dpi=150, bbox_inches='tight')
    print("✅ تم الحفظ: nov19_chart.png")

else:
    print("\n❌ لم يتم العثور على Setups")

print("\n" + "="*80)
print("✅ التحليل مكتمل!")
print("="*80)
