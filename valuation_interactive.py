import streamlit as st
import numpy as np
from valuation import compute_npv_offerA, expected_npv_offerB
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import copy

# Set page config
st.set_page_config(
    page_title="Startup vs Corporate Compensation Calculator",
    page_icon="💰",
    layout="wide"
)

# Page title and description
st.title("Job Offer Value Comparison: Corporate vs Startup")
st.markdown("""
This tool helps you compare the expected value of two job offers:
- **Offer A**: Established company (e.g., Natera) with stable compensation
- **Offer B**: Startup with lower base but equity upside and associated risks

Adjust the parameters using the sliders and see how the comparison changes dynamically.
""")

# Initialize parameters with defaults
default_params = {
    # Offer A (Natera) compensation assumptions
    "offerA_base": 230000.0,       # initial base salary ($)
    "offerA_raise": 0.03,         # annual raise (%)
    "offerA_bonus_pct": 0.15,     # annual bonus (% of base)
    "offerA_rsu_initial": 175000.0,   # initial RSU grant value ($)
    "offerA_rsu_refresh": 50000.0,    # refresher RSU grant value ($)
    "offerA_rsu_vest_years": 4,   # vesting period for RSUs (years)
    "offerA_rsu_growth": 0.05,    # assumed annual stock growth for RSUs
    "offerA_401k_match": 0.06,    # 401k match (% of salary)
    "offerA_401k_vest_years": 4,  # vesting period for 401k match (years)

    # Offer B (Startup) compensation assumptions
    "offerB_base": 180000.0,      # initial base salary ($)
    "offerB_raise": 0.03,        # annual raise (%)
    "offerB_signing_bonus": 45000.0, # signing bonus at t=0 ($)
    "offerB_equity_shares": 88000.0, # shares granted to employee
    "offerB_total_shares": 16000000.0, # total shares outstanding initially
    "offerB_vest_years": 4.0,      # equity vesting period (years)
    "offerB_cliff_years": 1.0,     # equity cliff (years before any vest)

    # Startup funding timeline and valuations
    "seriesA_year": 1.0,   # time of Series A round (years from start)
    "seriesB_year": 3.0,   # time of Series B round
    "seriesC_year": 5.0,   # time of Series C round
    "IPO_year":   6.5,     # time of IPO (liquidity event)

    "seriesA_valuation": 110e6,  # pre-money valuation for Series A ($120M)
    "seriesA_raise":     30e6,   # amount raised in Series A ($30M)
    "seriesB_valuation": 280e6,  # pre-money valuation for Series B ($300M)
    "seriesB_raise":     90e6,   # amount raised in Series B ($80M)
    "seriesC_valuation": 700e6,  # pre-money valuation for Series C ($750M)
    "seriesC_raise":     150e6,  # amount raised in Series C ($110M)
    "exit_valuation":    1.5e9,    # IPO exit valuation ($1B)

    # Failure probabilities at each round (tunable risk parameters)
    "P_fail_A": 0.2,  # Probability of failing at Series A
    "P_fail_B": 0.4,  # Probability of failing at Series B
    "P_fail_C": 0.6,  # Probability of failing at Series C

    # Discount rate for NPV calculations
    "discount_rate": 0.045,  # 4.5% per year
    "time_step": 0.5,        # simulation time step (years)
    
    # Failure valuation multiplier
    "valuation_multiplier_if_fail": 0.3,  # If company fails, value equity at 30% of current valuation
}

# Create a placeholder for active parameters
params = copy.deepcopy(default_params)

# Debug helper to check parameter types
def ensure_float(val):
    """Convert integer values to float for slider compatibility"""
    if isinstance(val, int):
        return float(val)
    return val

# Convert integer parameters to float for slider compatibility
for key in default_params:
    if isinstance(default_params[key], int):
        default_params[key] = float(default_params[key])

# Function to format values as currency
def format_currency(value):
    if value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"

# Function to format percentage values
def format_pct(value):
    return f"{value*100:.1f}%"

# Create sidebar for parameters
st.sidebar.title("Simulation Parameters")

# Create tabs for parameter categories
tab1, tab2, tab3, tab4 = st.sidebar.tabs(["Offer A", "Offer B", "Startup Timeline", "Risk Factors"])

# Tab 1: Offer A parameters
with tab1:
    st.header("Offer A (Corporate)")
    params["offerA_base"] = st.slider("Base Salary ($)", 150000.0, 350000.0, float(default_params["offerA_base"]), 5000.0, format="$%.0f", key="offerA_base")
    params["offerA_raise"] = st.slider("Annual Raise (decimal)", 0.0, 0.10, float(default_params["offerA_raise"]), 0.005, format="%.3f", key="offerA_raise")
    params["offerA_bonus_pct"] = st.slider("Annual Bonus (decimal of base)", 0.0, 0.30, float(default_params["offerA_bonus_pct"]), 0.01, format="%.2f", key="offerA_bonus_pct")
    params["offerA_rsu_initial"] = st.slider("Initial RSU Grant ($)", 0.0, 500000.0, float(default_params["offerA_rsu_initial"]), 10000.0, format="$%.0f", key="offerA_rsu_initial")
    params["offerA_rsu_refresh"] = st.slider("Annual RSU Refresh ($)", 0.0, 200000.0, float(default_params["offerA_rsu_refresh"]), 5000.0, format="$%.0f", key="offerA_rsu_refresh")
    params["offerA_rsu_growth"] = st.slider("Expected Stock Growth (decimal)", -0.10, 0.20, float(default_params["offerA_rsu_growth"]), 0.01, format="%.2f", key="offerA_rsu_growth")
    params["offerA_401k_match"] = st.slider("401k Match (decimal)", 0.0, 0.10, float(default_params["offerA_401k_match"]), 0.005, format="%.3f", key="offerA_401k_match")

# Tab 2: Offer B parameters
with tab2:
    st.header("Offer B (Startup)")
    params["offerB_base"] = st.slider("Base Salary ($)", 100000.0, 300000.0, float(default_params["offerB_base"]), 5000.0, format="$%.0f", key="offerB_base")
    params["offerB_raise"] = st.slider("Annual Raise (decimal)", 0.0, 0.10, float(default_params["offerB_raise"]), 0.005, format="%.3f", key="offerB_raise")
    params["offerB_signing_bonus"] = st.slider("Signing Bonus ($)", 0.0, 100000.0, float(default_params["offerB_signing_bonus"]), 5000.0, format="$%.0f", key="offerB_signing_bonus")
    
    equity_percentage = float(default_params["offerB_equity_shares"]) / float(default_params["offerB_total_shares"]) * 100
    equity_pct = st.slider("Equity (decimal of company)", 0.0001, 0.02, float(equity_percentage)/100, 0.0005, format="%.4f", key="offerB_equity_pct")
    # Update both values based on percentage
    params["offerB_equity_shares"] = equity_pct * 100 * params["offerB_total_shares"] / 100
    
    params["offerB_cliff_years"] = st.slider("Equity Cliff (years)", 0.0, 2.0, float(default_params["offerB_cliff_years"]), 0.25, format="%.2f", key="offerB_cliff_years")

# Tab 3: Startup Timeline
with tab3:
    st.header("Startup Funding Timeline")
    
    # Series A
    st.subheader("Series A")
    params["seriesA_year"] = st.slider("Series A Timing (years)", 0.5, 2.0, float(default_params["seriesA_year"]), 0.25, format="%.2f", key="seriesA_year")
    params["seriesA_valuation"] = st.slider("Series A Pre-money ($M)", 50.0, 200.0, float(default_params["seriesA_valuation"])/1e6, 5.0, format="%.0f", key="seriesA_valuation") * 1e6
    params["seriesA_raise"] = st.slider("Series A Raise ($M)", 10.0, 50.0, float(default_params["seriesA_raise"])/1e6, 5.0, format="%.0f", key="seriesA_raise") * 1e6
    
    # Series B
    st.subheader("Series B")
    params["seriesB_year"] = st.slider("Series B Timing (years)", params["seriesA_year"] + 0.5, 4.0, float(default_params["seriesB_year"]), 0.25, format="%.2f", key="seriesB_year")
    params["seriesB_valuation"] = st.slider("Series B Pre-money ($M)", 150.0, 400.0, float(default_params["seriesB_valuation"])/1e6, 10.0, format="%.0f", key="seriesB_valuation") * 1e6
    params["seriesB_raise"] = st.slider("Series B Raise ($M)", 30.0, 150.0, float(default_params["seriesB_raise"])/1e6, 10.0, format="%.0f", key="seriesB_raise") * 1e6
    
    # Series C
    st.subheader("Series C")
    params["seriesC_year"] = st.slider("Series C Timing (years)", params["seriesB_year"] + 0.5, 6.0, float(default_params["seriesC_year"]), 0.25, format="%.2f", key="seriesC_year")
    params["seriesC_valuation"] = st.slider("Series C Pre-money ($M)", 300.0, 1000.0, float(default_params["seriesC_valuation"])/1e6, 25.0, format="%.0f", key="seriesC_valuation") * 1e6
    params["seriesC_raise"] = st.slider("Series C Raise ($M)", 50.0, 300.0, float(default_params["seriesC_raise"])/1e6, 25.0, format="%.0f", key="seriesC_raise") * 1e6
    
    # IPO
    st.subheader("Exit (IPO)")
    params["IPO_year"] = st.slider("IPO Timing (years)", params["seriesC_year"] + 0.5, 8.0, float(default_params["IPO_year"]), 0.25, format="%.2f", key="IPO_year")
    params["exit_valuation"] = st.slider("Exit Valuation ($B)", 0.5, 5.0, float(default_params["exit_valuation"])/1e9, 0.1, format="%.1f", key="exit_valuation") * 1e9

# Tab 4: Risk factors
with tab4:
    st.header("Risk Parameters")
    params["P_fail_A"] = st.slider("P(Fail at Series A) (decimal)", 0.0, 0.5, float(default_params["P_fail_A"]), 0.05, format="%.2f", key="P_fail_A")
    params["P_fail_B"] = st.slider("P(Fail at Series B) (decimal)", 0.0, 0.7, float(default_params["P_fail_B"]), 0.05, format="%.2f", key="P_fail_B")
    params["P_fail_C"] = st.slider("P(Fail at Series C) (decimal)", 0.0, 0.8, float(default_params["P_fail_C"]), 0.05, format="%.2f", key="P_fail_C")
    params["valuation_multiplier_if_fail"] = st.slider("Valuation if Fail (decimal of pre-money)", 0.0, 0.7, float(default_params["valuation_multiplier_if_fail"]), 0.05, format="%.2f", key="valuation_multiplier_if_fail")
    params["discount_rate"] = st.slider("Discount Rate (decimal)", 0.01, 0.10, float(default_params["discount_rate"]), 0.005, format="%.3f", key="discount_rate")

# Additional options in main area
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Options")
sim_trials = st.sidebar.slider("Number of Monte Carlo Trials", 1000, 50000, 10000, 1000, format="%d", key="sim_trials")
max_horizon = st.sidebar.slider("Maximum Time Horizon (years)", 5.0, 15.0, 10.0, 0.5, format="%.1f", key="max_horizon")

# Reset button
if st.sidebar.button("Reset to Defaults"):
    # Force a full page refresh (this is hacky but works)
    st.experimental_rerun()

# Create main visualization area
st.subheader("Net Present Value Comparison")

# Calculate horizons
horizons = [i/2 for i in range(int(2.5*2), int(max_horizon*2)+1)]  # [2.5, 3.0, ..., max_horizon]

# Calculate NPV results for each horizon
with st.spinner("Running simulations..."):
    progress_bar = st.progress(0)
    results = []
    
    for i, h in enumerate(horizons):
        npv_A = compute_npv_offerA(h, params)
        npv_B, q025, q975, q10, q90, q40, q60, q70, q80 = expected_npv_offerB(h, params, trials=sim_trials)
        results.append((h, npv_A, npv_B, q025, q975, q10, q90, q40, q60, q70, q80))
        progress_bar.progress((i + 1) / len(horizons))

# Extract data for plotting
horizons_plot = [r[0] for r in results]
npv_a_values = [r[1] for r in results]
npv_b_values = [r[2] for r in results]
npv_b_q025 = [r[3] for r in results]  # 2.5th percentile
npv_b_q975 = [r[4] for r in results]  # 97.5th percentile
npv_b_q10 = [r[5] for r in results]   # 10th percentile
npv_b_q90 = [r[6] for r in results]   # 90th percentile

# Extract additional quantiles
npv_b_q40 = [r[7] for r in results]   # 40th percentile
npv_b_q60 = [r[8] for r in results]   # 60th percentile
npv_b_q70 = [r[9] for r in results]   # 70th percentile
npv_b_q80 = [r[10] for r in results]  # 80th percentile

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data
ax.plot(horizons_plot, npv_a_values, 'bo-', label="Offer A (Corporate)", linewidth=2)
ax.plot(horizons_plot, npv_b_values, 'ro-', label="Offer B (Startup)", linewidth=2)

# Remove the confidence intervals plots
# Plot quantile lines with different markers (removed 2.5% and 10% quantiles)
ax.plot(horizons_plot, npv_b_q40, 'r:', marker='x', alpha=0.6, linewidth=1, label="40% Quantile")
ax.plot(horizons_plot, npv_b_q60, 'r:', marker='o', alpha=0.6, linewidth=1, label="60% Quantile")
ax.plot(horizons_plot, npv_b_q70, 'r:', marker='s', alpha=0.6, linewidth=1, label="70% Quantile")
ax.plot(horizons_plot, npv_b_q80, 'r:', marker='p', alpha=0.6, linewidth=1, label="80% Quantile")
ax.plot(horizons_plot, npv_b_q90, 'r:', marker='>', alpha=0.6, linewidth=1, label="90% Quantile")
ax.plot(horizons_plot, npv_b_q975, 'r:', marker='^', alpha=0.6, linewidth=1, label="95% Quantile")

# Format y-axis as currency
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# Add grid, legend, and labels
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12)
ax.set_xlabel('Time Horizon (Years)', fontsize=12)
ax.set_ylabel('Net Present Value (NPV)', fontsize=12)
ax.set_title("NPV Comparison: Corporate vs Startup Offer", fontsize=14)

# Find crossover point if it exists
crossover_found = False
crossover_x = None
crossover_y = None

for i in range(1, len(horizons_plot)):
    if (npv_a_values[i-1] > npv_b_values[i-1] and npv_a_values[i] <= npv_b_values[i]) or \
       (npv_a_values[i-1] < npv_b_values[i-1] and npv_a_values[i] >= npv_b_values[i]):
        # Linear interpolation to find crossover point
        h1, h2 = horizons_plot[i-1], horizons_plot[i]
        a1, a2 = npv_a_values[i-1], npv_a_values[i]
        b1, b2 = npv_b_values[i-1], npv_b_values[i]
        
        # Calculate intersection point
        x_intersect = h1 + (h2 - h1) * (b1 - a1) / ((a2 - a1) - (b2 - b1))
        y_intersect = a1 + (a2 - a1) * (x_intersect - h1) / (h2 - h1)
        
        # Mark the crossover point
        ax.plot(x_intersect, y_intersect, 'gx', markersize=10, markeredgewidth=2)
        ax.annotate(f'Crossover: {x_intersect:.2f} years\n${y_intersect:,.0f}',
                    xy=(x_intersect, y_intersect), xytext=(x_intersect+0.5, y_intersect),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, color='green')
        crossover_found = True
        crossover_x = x_intersect
        crossover_y = y_intersect
        break

# Display the plot
st.pyplot(fig)

# Display key metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Value at 5 Years", 
              format_currency(npv_a_values[horizons_plot.index(5.0)]), 
              format_currency(npv_a_values[horizons_plot.index(5.0)] - npv_b_values[horizons_plot.index(5.0)]))

with col2:
    st.metric("Value at 7 Years (if available)", 
              format_currency(npv_a_values[horizons_plot.index(7.0)] if 7.0 in horizons_plot else npv_a_values[-1]),
              format_currency((npv_a_values[horizons_plot.index(7.0)] if 7.0 in horizons_plot else npv_a_values[-1]) - 
                            (npv_b_values[horizons_plot.index(7.0)] if 7.0 in horizons_plot else npv_b_values[-1])))

with col3:
    st.metric("Value at 10 Years (if available)", 
              format_currency(npv_a_values[horizons_plot.index(10.0)] if 10.0 in horizons_plot else npv_a_values[-1]),
              format_currency((npv_a_values[horizons_plot.index(10.0)] if 10.0 in horizons_plot else npv_a_values[-1]) - 
                             (npv_b_values[horizons_plot.index(10.0)] if 10.0 in horizons_plot else npv_b_values[-1])))

# Display crossover point
if crossover_found:
    st.success(f"The offers have equal value at the **{crossover_x:.2f} year** mark, with an NPV of **{format_currency(crossover_y)}**.")
    
    if crossover_x <= 5:
        st.info("Offer B (Startup) becomes more valuable relatively quickly. If you plan to stay more than a few years, the startup may be the better choice, assuming the risk profile is acceptable.")
    elif crossover_x <= 7:
        st.info("Offer B (Startup) becomes more valuable in the medium term. The corporate offer provides better value in the first several years.")
    else:
        st.info("Offer B (Startup) only becomes more valuable in the longer term. If you're not planning to stay long-term, the corporate offer likely provides better value.")
else:
    if npv_a_values[-1] > npv_b_values[-1]:
        st.warning(f"Offer A (Corporate) remains more valuable throughout the entire time horizon analyzed (up to {max_horizon} years).")
    else:
        st.warning(f"Offer B (Startup) remains more valuable throughout the entire time horizon analyzed (up to {max_horizon} years).")

# Display information on how to run the app
st.markdown("---")
st.subheader("How to Run This App")
st.markdown("""
1. Install the required packages:
   ```
   pip install streamlit numpy matplotlib
   ```
2. Run the application:
   ```
   streamlit run valuation_interactive.py
   ```
3. A browser window will open automatically with the interactive app.
""")
