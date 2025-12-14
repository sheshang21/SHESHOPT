import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

st.set_page_config(page_title="Options Pricing Engine", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);}
    h1, h2, h3 {color: #60a5fa;}
    .stAlert {background-color: rgba(30, 41, 59, 0.8);}
</style>
""", unsafe_allow_html=True)

def black_scholes(S, K, T, r, sigma, opt_type='call'):
    """Black-Scholes option pricing"""
    if T <= 0:
        return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if opt_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, opt_type='call'):
    """Calculate all option Greeks"""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    
    # Theta
    term1 = -S*norm.pdf(d1)*sigma / (2*np.sqrt(T))
    if opt_type == 'call':
        theta = (term1 - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        theta = (term1 + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    
    # Vega
    vega = S*norm.pdf(d1)*np.sqrt(T) / 100
    
    # Rho
    if opt_type == 'call':
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def generate_options_chain(symbol, spot, r, sigma, strikes_config, expiries_config):
    """Generate complete options chain"""
    
    # Generate strikes
    strikes = []
    if strikes_config['mode'] == 'percentage':
        for pct in strikes_config['percentages']:
            strikes.append(round(spot * (1 + pct/100)))
    else:  # interval mode
        atm = round(spot / strikes_config['interval']) * strikes_config['interval']
        for i in range(-strikes_config['count']//2, strikes_config['count']//2 + 1):
            strike = atm + i*strikes_config['interval']
            if strike > 0:
                strikes.append(strike)
    
    # Generate expiries
    expiries = []
    for days in expiries_config['days']:
        exp_date = datetime.now() + timedelta(days=days)
        expiries.append({
            'date': exp_date,
            'label': exp_date.strftime("%d %b %Y"),
            'dte': days
        })
    
    # Generate all options
    options = []
    
    for exp in expiries:
        T = exp['dte'] / 365.0
        
        for strike in strikes:
            for opt_type in ['call', 'put']:
                # Calculate option price and Greeks
                price = black_scholes(spot, strike, T, r, sigma, opt_type)
                intrinsic = max(spot-strike, 0) if opt_type=='call' else max(strike-spot, 0)
                time_value = max(price - intrinsic, 0)
                greeks = calculate_greeks(spot, strike, T, r, sigma, opt_type)
                
                # Simulate market data
                spread_pct = 0.02 + (0.03 * abs(spot-strike)/spot)  # Wider spread for OTM
                spread = max(0.05, price * spread_pct)
                bid = max(0.01, price - spread/2)
                ask = price + spread/2
                
                # Simulate volume and OI based on moneyness
                moneyness = abs(spot - strike) / spot
                volume_factor = max(0.1, 1 - moneyness*5)
                base_volume = np.random.randint(100, 5000)
                volume = int(base_volume * volume_factor)
                oi = int(volume * np.random.uniform(2, 10))
                
                itm = (opt_type=='call' and spot>strike) or (opt_type=='put' and spot<strike)
                
                options.append({
                    'Option Code': f"{symbol} {exp['label']} {strike} {'CE' if opt_type=='call' else 'PE'}",
                    'Symbol': symbol,
                    'Spot': spot,
                    'Strike': strike,
                    'Type': 'CALL' if opt_type=='call' else 'PUT',
                    'Expiry': exp['label'],
                    'DTE': exp['dte'],
                    'Theoretical Price': price,
                    'LTP': price,
                    'Bid': bid,
                    'Ask': ask,
                    'Spread': spread,
                    'Volume': volume,
                    'Open Interest': oi,
                    'IV': sigma*100,
                    'Intrinsic Value': intrinsic,
                    'Time Value': time_value,
                    'Delta': greeks['delta'],
                    'Gamma': greeks['gamma'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Rho': greeks['rho'],
                    'ITM': itm,
                    'Moneyness': round((spot/strike - 1)*100, 2) if opt_type=='call' else round((strike/spot - 1)*100, 2)
                })
    
    return pd.DataFrame(options)

st.title("📊 Professional Options Pricing Engine")
st.markdown("**Black-Scholes Model with Complete Greeks & Market Simulation**")

st.info("🎯 **No API Required!** Enter stock details below to generate a complete, professional options chain with theoretical pricing, Greeks, and simulated market data.")

# Input Section
st.header("📝 Stock & Market Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input("Stock Symbol", "AAPL", help="Enter any stock symbol").upper()
    spot_price = st.number_input("Current Stock Price ($)", min_value=0.01, value=180.00, step=5.0, help="Enter the current market price")

with col2:
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 20.0, 5.0, 0.5, help="Treasury yield / risk-free rate") / 100
    iv = st.slider("Implied Volatility (%)", 5.0, 150.0, 30.0, 5.0, help="Expected volatility") / 100

with col3:
    currency = st.selectbox("Currency", ["USD ($)", "INR (₹)", "EUR (€)", "GBP (£)"])
    currency_symbol = currency.split('(')[1].strip(')')

# Strike Configuration
st.header("🎯 Strike Prices Configuration")

strike_mode = st.radio("Strike Generation Mode", ["Percentage Based", "Interval Based"], horizontal=True)

if strike_mode == "Percentage Based":
    st.info("Generate strikes at specific percentages from current price")
    
    preset = st.selectbox("Preset", ["Narrow (±10%)", "Medium (±20%)", "Wide (±30%)", "Custom"])
    
    if preset == "Narrow (±10%)":
        percentages = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
    elif preset == "Medium (±20%)":
        percentages = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    elif preset == "Wide (±30%)":
        percentages = [-30, -20, -10, 0, 10, 20, 30]
    else:
        percentages_str = st.text_input("Enter percentages (comma-separated)", "-20,-10,-5,0,5,10,20")
        percentages = [float(x.strip()) for x in percentages_str.split(',')]
    
    strikes_config = {'mode': 'percentage', 'percentages': percentages}
    st.caption(f"Will generate {len(percentages)} strikes: {', '.join([f'{p:+.1f}%' for p in percentages])}")

else:  # Interval Based
    st.info("Generate strikes at fixed price intervals around ATM")
    
    col1, col2 = st.columns(2)
    with col1:
        strike_count = st.slider("Number of Strikes", 5, 51, 15, 2)
    with col2:
        strike_interval = st.number_input("Strike Interval ($)", min_value=1, value=5, step=1)
    
    strikes_config = {'mode': 'interval', 'count': strike_count, 'interval': strike_interval}
    st.caption(f"Will generate {strike_count} strikes with {currency_symbol}{strike_interval} intervals")

# Expiry Configuration
st.header("📅 Expiration Dates Configuration")

expiry_preset = st.selectbox("Expiry Preset", ["Weekly (4 weeks)", "Monthly (6 months)", "Quarterly (1 year)", "Custom"])

if expiry_preset == "Weekly (4 weeks)":
    expiry_days = [7, 14, 21, 28]
elif expiry_preset == "Monthly (6 months)":
    expiry_days = [30, 60, 90, 120, 150, 180]
elif expiry_preset == "Quarterly (1 year)":
    expiry_days = [90, 180, 270, 365]
else:
    expiry_days_str = st.text_input("Days to expiry (comma-separated)", "7,30,60,90,180,365")
    expiry_days = [int(x.strip()) for x in expiry_days_str.split(',')]

expiries_config = {'days': expiry_days}
st.caption(f"Will generate options for {len(expiry_days)} expiry dates")

# Generate Button
st.markdown("---")
generate = st.button("🚀 Generate Complete Options Chain", type="primary", use_container_width=True)

if generate:
    with st.spinner("Generating options chain..."):
        df = generate_options_chain(symbol, spot_price, risk_free, iv, strikes_config, expiries_config)
        
        st.session_state['df'] = df
        st.session_state['spot'] = spot_price
        st.session_state['symbol'] = symbol
        st.session_state['currency'] = currency_symbol
        
        st.success(f"✅ Generated {len(df):,} options contracts!")
        st.balloons()

# Display Options Chain
if 'df' in st.session_state and st.session_state['df'] is not None:
    st.markdown("---")
    st.header(f"📈 Options Chain: {st.session_state['symbol']}")
    
    df = st.session_state['df']
    curr = st.session_state['currency']
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Strike, code, expiry...")
    with col2:
        type_filter = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        money_filter = st.selectbox("Moneyness", ["All", "ITM", "ATM (±5%)", "OTM"])
    with col4:
        expiry_filter = st.selectbox("Expiry", ["All"] + sorted(df['Expiry'].unique().tolist()))
    
    # Apply filters
    filtered = df.copy()
    
    if search:
        mask = (
            filtered['Option Code'].str.contains(search, case=False, na=False) |
            filtered['Strike'].astype(str).str.contains(search, na=False) |
            filtered['Expiry'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    
    if type_filter != "All":
        filtered = filtered[filtered['Type'] == type_filter]
    
    if money_filter == "ITM":
        filtered = filtered[filtered['ITM']]
    elif money_filter == "OTM":
        filtered = filtered[~filtered['ITM']]
    elif money_filter == "ATM (±5%)":
        filtered = filtered[abs(filtered['Moneyness']) <= 5]
    
    if expiry_filter != "All":
        filtered = filtered[filtered['Expiry'] == expiry_filter]
    
    # Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Options", f"{len(filtered):,}")
    with col2:
        st.metric("Call Options", f"{len(filtered[filtered['Type']=='CALL']):,}")
    with col3:
        st.metric("Put Options", f"{len(filtered[filtered['Type']=='PUT']):,}")
    with col4:
        st.metric("ITM Options", f"{len(filtered[filtered['ITM']]):,}")
    with col5:
        st.metric("Spot Price", f"{curr}{st.session_state['spot']:.2f}")
    with col6:
        total_vol = filtered['Volume'].sum()
        st.metric("Total Volume", f"{total_vol:,}")
    
    # Display DataFrame
    display_cols = [
        'Option Code', 'Strike', 'Type', 'Expiry', 'DTE',
        'Theoretical Price', 'LTP', 'Bid', 'Ask', 'Spread',
        'Volume', 'Open Interest', 'IV',
        'Intrinsic Value', 'Time Value',
        'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'
    ]
    
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        height=600,
        column_config={
            "Strike": st.column_config.NumberColumn(format=f"{curr}%.2f"),
            "Theoretical Price": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "LTP": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "Bid": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "Ask": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "Spread": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "Intrinsic Value": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "Time Value": st.column_config.NumberColumn(format=f"{curr}%.3f"),
            "IV": st.column_config.NumberColumn(format="%.2f%%"),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Theta": st.column_config.NumberColumn(format="%.4f"),
            "Vega": st.column_config.NumberColumn(format="%.4f"),
            "Rho": st.column_config.NumberColumn(format="%.4f"),
        }
    )
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Complete Data as CSV",
        csv,
        f"{st.session_state['symbol']}_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        use_container_width=True
    )
    
    # Analytics
    st.markdown("---")
    st.header("📊 Options Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Volume Analysis", "Open Interest", "Greeks Heatmap", "Volatility Smile"])
    
    with tab1:
        st.subheader("Top 10 by Volume")
        top_vol = filtered.nlargest(10, 'Volume')[['Option Code', 'Type', 'Strike', 'LTP', 'Volume', 'Open Interest']]
        st.dataframe(top_vol, use_container_width=True)
    
    with tab2:
        st.subheader("Top 10 by Open Interest")
        top_oi = filtered.nlargest(10, 'Open Interest')[['Option Code', 'Type', 'Strike', 'LTP', 'Volume', 'Open Interest']]
        st.dataframe(top_oi, use_container_width=True)
    
    with tab3:
        st.subheader("Greeks Summary by Expiry")
        greeks_summary = filtered.groupby('Expiry').agg({
            'Delta': 'mean',
            'Gamma': 'mean',
            'Theta': 'mean',
            'Vega': 'mean'
        }).round(4)
        st.dataframe(greeks_summary, use_container_width=True)
    
    with tab4:
        st.subheader("IV by Strike (Volatility Smile)")
        iv_by_strike = filtered.groupby(['Strike', 'Type'])['IV'].mean().reset_index()
        st.dataframe(iv_by_strike, use_container_width=True)

else:
    st.info("👆 Configure parameters above and click 'Generate Complete Options Chain' to start!")
    
    st.markdown("### 💡 What You Get:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Pricing Models**
        - Black-Scholes theoretical prices
        - Intrinsic value calculation
        - Time value decomposition
        - Bid/ask spread simulation
        """)
    
    with col2:
        st.markdown("""
        **Complete Greeks**
        - Delta (price sensitivity)
        - Gamma (delta sensitivity)
        - Theta (time decay)
        - Vega (volatility sensitivity)
        - Rho (rate sensitivity)
        """)
    
    with col3:
        st.markdown("""
        **Market Data**
        - Simulated volume
        - Open interest estimates
        - ITM/OTM classification
        - Moneyness calculation
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 About This Tool")
st.sidebar.markdown("""
**Options Pricing Engine**

A professional-grade options analysis tool using the Black-Scholes model.

**Features:**
- ✅ No API required
- ✅ Complete options chains
- ✅ All Greeks calculated
- ✅ Customizable parameters
- ✅ Market data simulation
- ✅ CSV export

**Perfect for:**
- Options strategy planning
- Risk analysis
- Educational purposes
- Backtesting strategies
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust IV to match market conditions for more accurate pricing!")
