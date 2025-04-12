import random
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick
# Define global parameters for the simulation
params = {
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
    "offerB_base": 190000.0,      # initial base salary ($)
    "offerB_raise": 0.03,        # annual raise (%)
    "offerB_signing_bonus": 45000.0, # signing bonus at t=0 ($)
    "offerB_equity_shares": 129000.0, # shares granted to employee
    "offerB_total_shares": 16000000.0, # total shares outstanding initially
    "offerB_vest_years": 4,      # equity vesting period (years)
    "offerB_cliff_years": 1,     # equity cliff (years before any vest)

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
    "valuation_multiplier_if_fail": 0.3,  # If company fails, value equity at 20% of current valuation
}


def compute_npv_offerA(horizon, params):
    """Compute the NPV of Offer A (Natera) compensation over a given time horizon."""
    r = params["discount_rate"]
    t_step = params["time_step"]
    t = 0.0
    year = 0
    base = params["offerA_base"]
    base_growth = params["offerA_raise"]
    bonus_pct = params["offerA_bonus_pct"]
    npv = 0.0

    # Track total 401k match contributions (for vesting calculations)
    total_match_contrib = 0.0

    # Simulate salary + bonus in half-year increments
    while t < horizon - 1e-9:
        # Apply annual raise at year boundaries
        if year > 0 and abs(t - year) < 1e-9:
            base *= (1 + base_growth)
        # Determine length of this step (might be shorter if horizon is not integer multiple of 0.5)
        step = min(t_step, horizon - t)
        # Cash flows in this period
        salary_flow = base * step    # base salary for this half-year (step=0.5 years gives half-year salary)
        bonus_flow  = base * bonus_pct * step  # assume bonus accrues evenly (half-year portion)
        # Discount these flows to present
        t_end = t + step  # time at end of this period
        npv += salary_flow / ((1 + r) ** t_end)
        npv += bonus_flow  / ((1 + r) ** t_end)
        # 401k match contribution (vested fully by year4 for our horizons)
        match_flow = params["offerA_401k_match"] * salary_flow
        # We will add match flows later (after accounting for vesting)
        total_match_contrib += match_flow
        # Increment time
        t += step
        if abs(t - (year + 1)) < 1e-9:
            year += 1

    # Add RSU vesting events (at end of each vest year)
    # Initial RSUs vest in years 1..4
    for vest_year in range(1, params["offerA_rsu_vest_years"] + 1):
        if vest_year <= horizon:
            portion = params["offerA_rsu_initial"] / params["offerA_rsu_vest_years"]
            value_at_vest = portion * ((1 + params["offerA_rsu_growth"]) ** vest_year)
            npv += value_at_vest / ((1 + r) ** vest_year)
    # Refresh RSUs vest in years 2..5 (if grant at year1)
    for vest_year in range(2, params["offerA_rsu_vest_years"] + 2):
        if vest_year <= horizon:
            portion = params["offerA_rsu_refresh"] / params["offerA_rsu_vest_years"]
            # These RSUs granted at year1, so growth = (vest_year - 1) years
            years_since_grant = vest_year - 1
            value_at_vest = portion * ((1 + params["offerA_rsu_growth"]) ** years_since_grant)
            npv += value_at_vest / ((1 + r) ** vest_year)

    # 401k match vesting:
    if horizon >= params["offerA_401k_vest_years"]:
        # If stayed ≥4 years, all match contributions vest. Discount each contribution at its time.
        # (Approximate by distributing match flows over each half-year as if vested then)
        t = 0.0
        year = 0
        base = params["offerA_base"]
        while t < horizon - 1e-9:
            if year > 0 and abs(t - year) < 1e-9:
                base *= (1 + base_growth)
            step = min(t_step, horizon - t)
            salary_flow = base * step
            match_flow = params["offerA_401k_match"] * salary_flow
            t_end = t + step
            npv += match_flow / ((1 + r) ** t_end)
            t += step
            if abs(t - (year + 1)) < 1e-9:
                year += 1
    else:
        # If left before 4 years (not in our scenario), vest only a proportional amount of match
        vested_fraction = horizon / params["offerA_401k_vest_years"]
        vested_match_value = total_match_contrib * vested_fraction
        npv += vested_match_value / ((1 + r) ** horizon)
    return npv

def simulate_offerB_once(params, horizon):
    """Simulate one realization of Offer B (Startup) compensation up to the given horizon.
       Returns the discounted NPV of that trial."""
    r = params["discount_rate"]
    t_step = params["time_step"]
    base = params["offerB_base"]
    base_growth = params["offerB_raise"]
    # Failure probabilities
    P_fail_A = params["P_fail_A"]
    P_fail_B = params["P_fail_B"]
    P_fail_C = params["P_fail_C"]
    # Failure valuation multiplier
    valuation_multiplier = params["valuation_multiplier_if_fail"]

    t = 0.0
    year = 0
    npv = 0.0
    alive = True
    # Pay signing bonus at t=0
    npv += params["offerB_signing_bonus"]  # (no discount, time 0)

    # Flags to track if each round succeeded (for equity payout)
    success_A = success_B = success_C = False

    # Calculate initial share count
    total_shares = params["offerB_total_shares"]
    employee_shares = params["offerB_equity_shares"]
    
    # Vested shares over time (to calculate payout if failure)
    vested_fraction = 0.0
    
    # Simulate cash flows in half-year increments until horizon or failure
    while t < horizon - 1e-9:
        # Update vested share fraction (simplified linear vesting after cliff)
        cliff_time = params["offerB_cliff_years"]
        vest_time = params["offerB_vest_years"]
        if t >= cliff_time:
            # Start vesting after cliff
            vested_fraction = min(1.0, (t - cliff_time) / (vest_time - cliff_time))
            
        # Check for funding events at this time point
        if abs(t - params["seriesA_year"]) < 1e-9 and success_A is False:
            # Series A happens now
            if random.random() < P_fail_A:
                # Company fails at Series A
                alive = False
                
                # Calculate current valuation and equity payout
                pre_money_val = params["seriesA_valuation"]
                failure_val = pre_money_val * valuation_multiplier
                equity_value = vested_fraction * employee_shares * (failure_val / total_shares)
                
                # Add discounted equity payout to NPV
                npv += equity_value / ((1 + r) ** t)
                break
            else:
                success_A = True
                # Update share count due to dilution
                total_shares *= (1 + params["seriesA_raise"] / params["seriesA_valuation"])
                
        if abs(t - params["seriesB_year"]) < 1e-9 and alive and success_B is False:
            # Series B round
            if random.random() < P_fail_B:
                # Company fails at Series B
                alive = False
                
                # Calculate current valuation and equity payout
                pre_money_val = params["seriesB_valuation"]
                failure_val = pre_money_val * valuation_multiplier
                equity_value = vested_fraction * employee_shares * (failure_val / total_shares)
                
                # Add discounted equity payout to NPV
                npv += equity_value / ((1 + r) ** t)
                break
            else:
                success_B = True
                # Update share count due to dilution
                total_shares *= (1 + params["seriesB_raise"] / params["seriesB_valuation"])
                
        if abs(t - params["seriesC_year"]) < 1e-9 and alive and success_C is False:
            # Series C round
            if random.random() < P_fail_C:
                # Company fails at Series C
                alive = False
                
                # Calculate current valuation and equity payout
                pre_money_val = params["seriesC_valuation"]
                failure_val = pre_money_val * valuation_multiplier
                equity_value = vested_fraction * employee_shares * (failure_val / total_shares)
                
                # Add discounted equity payout to NPV
                npv += equity_value / ((1 + r) ** t)
                break
            else:
                success_C = True
                # Update share count due to dilution
                total_shares *= (1 + params["seriesC_raise"] / params["seriesC_valuation"])
        
        # Check if company is alive; if not, break out
        if not alive:
            break
            
        # Apply raise at year boundaries
        if year > 0 and abs(t - year) < 1e-9:
            base *= (1 + base_growth)
        # Determine step length
        step = min(t_step, horizon - t)
        t_end = t + step
        # Salary (if alive during this period)
        salary_flow = base * step if alive else 0.0
        npv += salary_flow / ((1 + r) ** t_end)
        # (No startup annual bonus or 401k match to add)
        # Advance time
        t = t_end
        
        if abs(t - (year + 1)) < 1e-9:
            year += 1

    # If the company survived through Series C and horizon extends to IPO time, include equity payout
    if horizon >= params["IPO_year"] and success_A and success_B and success_C:
        equity_value = employee_shares * (params["exit_valuation"] / total_shares)
        npv += equity_value / ((1 + r) ** params["IPO_year"])
    return npv

def expected_npv_offerB(horizon, params, trials=10000):
    """
    Simulate Offer B for a number of trials and return the average (expected) NPV
    and empirical quantiles for confidence intervals.
    """
    npv_values = []
    for _ in range(trials):
        npv_values.append(simulate_offerB_once(params, horizon))
    
    mean_npv = np.mean(npv_values)
    
    # Calculate empirical quantiles for 95% CI and 80% CI
    q025 = np.percentile(npv_values, 2.5)   # 2.5th percentile
    q975 = np.percentile(npv_values, 97.5)  # 97.5th percentile
    q10 = np.percentile(npv_values, 10)     # 10th percentile
    q90 = np.percentile(npv_values, 90)     # 90th percentile
    
    # Calculate additional percentiles
    q40 = np.percentile(npv_values, 40)     # 40th percentile
    q60 = np.percentile(npv_values, 60)     # 60th percentile
    q70 = np.percentile(npv_values, 70)     # 70th percentile
    q80 = np.percentile(npv_values, 80)     # 80th percentile
    
    return mean_npv, q025, q975, q10, q90, q40, q60, q70, q80

def plot_npv_comparison(horizons, results, params, output_dir="valuation_plots"):
    """
    Create and save a plot comparing NPV of Offer A and Offer B.
    
    Args:
        horizons: List of time horizons
        results: List of tuples (horizon, npv_A, npv_B, q025, q975, q10, q90, q40, q60, q70, q80)
        params: Parameter dictionary used for simulation
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine next scenario number
    existing_plots = [f for f in os.listdir(output_dir) 
                     if f.startswith("valuation_scenario_") and f.endswith(".png")]
    scenario_nums = [int(f.split("_")[2].split(".")[0]) for f in existing_plots if f.split("_")[2].split(".")[0].isdigit()]
    scenario_num = 1 if not scenario_nums else max(scenario_nums) + 1
    
    # Extract data for plotting
    horizons = [r[0] for r in results]
    npv_a_values = [r[1] for r in results]
    npv_b_values = [r[2] for r in results]
    
    # Extract empirical quantiles for confidence intervals
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
    ax.plot(horizons, npv_a_values, 'bo-', label="Offer A (Natera)", linewidth=2)
    ax.plot(horizons, npv_b_values, 'ro-', label="Offer B (Startup)", linewidth=2)
    
    # Remove the 95% and 80% confidence interval plots
    # Plot quantile lines with different markers (removed 2.5% and 10% quantiles)
    ax.plot(horizons, npv_b_q40, 'r:', marker='x', alpha=0.6, linewidth=1, label="40% Quantile")
    ax.plot(horizons, npv_b_q60, 'r:', marker='o', alpha=0.6, linewidth=1, label="60% Quantile")
    ax.plot(horizons, npv_b_q70, 'r:', marker='s', alpha=0.6, linewidth=1, label="70% Quantile")
    ax.plot(horizons, npv_b_q80, 'r:', marker='p', alpha=0.6, linewidth=1, label="80% Quantile")
    ax.plot(horizons, npv_b_q90, 'r:', marker='>', alpha=0.6, linewidth=1, label="90% Quantile") 
    ax.plot(horizons, npv_b_q975, 'r:', marker='^', alpha=0.6, linewidth=1, label="95% Quantile")
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    
    # Add grid, legend, and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_xlabel('Time Horizon (Years)', fontsize=12)
    ax.set_ylabel('Net Present Value (NPV)', fontsize=12)
    ax.set_title('NPV Comparison: Natera vs Startup Offer', fontsize=14)
    
    # Add parameter information as text
    param_text = "Simulation Parameters:\n"
    # Offer A params
    param_text += "\nOffer A (Natera):\n"
    param_text += f"Base Salary: ${params['offerA_base']:,.0f}\n"
    param_text += f"Annual Raise: {params['offerA_raise']*100:.1f}%\n"
    param_text += f"Bonus: {params['offerA_bonus_pct']*100:.1f}% of base\n"
    param_text += f"Initial RSU: ${params['offerA_rsu_initial']:,.0f}\n"
    param_text += f"RSU Refresh: ${params['offerA_rsu_refresh']:,.0f}\n"
    param_text += f"RSU Growth: {params['offerA_rsu_growth']*100:.1f}%\n"
    param_text += f"401k Match: {params['offerA_401k_match']*100:.1f}%\n"
    
    # Offer B params
    param_text += "\nOffer B (Startup):\n"
    param_text += f"Base Salary: ${params['offerB_base']:,.0f}\n"
    param_text += f"Annual Raise: {params['offerB_raise']*100:.1f}%\n"
    param_text += f"Signing Bonus: ${params['offerB_signing_bonus']:,.0f}\n"
    param_text += f"Equity: {params['offerB_equity_shares']:,.0f} shares\n" 
    param_text += f"Total Shares: {params['offerB_total_shares']:,.0f}\n"
    
    # Exit timeline
    param_text += "\nStartup Timeline:\n"
    param_text += f"Series A: Year {params['seriesA_year']:.1f}, ${params['seriesA_valuation']/1e6:.0f}M pre-money, ${params['seriesA_raise']/1e6:.0f}M raised\n"
    param_text += f"Series B: Year {params['seriesB_year']:.1f}, ${params['seriesB_valuation']/1e6:.0f}M pre-money, ${params['seriesB_raise']/1e6:.0f}M raised\n"
    param_text += f"Series C: Year {params['seriesC_year']:.1f}, ${params['seriesC_valuation']/1e6:.0f}M pre-money, ${params['seriesC_raise']/1e6:.0f}M raised\n"
    param_text += f"IPO: Year {params['IPO_year']:.1f}, ${params['exit_valuation']/1e9:.1f}B valuation\n"
    
    # Risk factors
    param_text += "\nRisk Factors:\n"
    param_text += f"P(Fail at A): {params['P_fail_A']*100:.0f}%\n"
    param_text += f"P(Fail at B): {params['P_fail_B']*100:.0f}%\n"
    param_text += f"P(Fail at C): {params['P_fail_C']*100:.0f}%\n"
    param_text += f"Valuation if Fail: {params['valuation_multiplier_if_fail']*100:.0f}% of pre-money\n"
    param_text += f"Discount Rate: {params['discount_rate']*100:.1f}%"
    
    # Add parameter text to plot
    plt.figtext(1.02, 0.5, param_text, ha='left', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    # Adjust layout to make room for parameters
    plt.subplots_adjust(right=0.7)
    
    # Add crossover point
    crossover_found = False
    for i in range(1, len(horizons)):
        if (npv_a_values[i-1] > npv_b_values[i-1] and npv_a_values[i] <= npv_b_values[i]) or \
           (npv_a_values[i-1] < npv_b_values[i-1] and npv_a_values[i] >= npv_b_values[i]):
            # Linear interpolation to find crossover point
            h1, h2 = horizons[i-1], horizons[i]
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
            break
    
    # Save the plot
    output_path = os.path.join(output_dir, f"valuation_scenario_{scenario_num}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")
    return output_path


# Example: compute expected NPV for Offer B at a 10-year horizon
print(expected_npv_offerB(10.0, params, trials=10000))

horizons = [i/2 for i in range(5, 21)]  # [5.0, 5.5, ..., 10.0]
results = []
for h in horizons:
    npv_A = compute_npv_offerA(h, params)
    npv_B, q025, q975, q10, q90, q40, q60, q70, q80 = expected_npv_offerB(h, params, trials=20000)  # using 20k simulations for stability
    results.append((h, npv_A, npv_B, q025, q975, q10, q90, q40, q60, q70, q80))
    print(f"Horizon {h:.1f} years -> NPV Offer A = ${npv_A:,.0f}, NPV Offer B = ${npv_B:,.0f} ± ${q975 - q025:,.0f}")

# Generate and save the NPV comparison plot
plot_npv_comparison(horizons, results, params)

def create_scenario(scenario_name, param_adjustments):
    """
    Create and evaluate a new scenario with modified parameters.
    
    Args:
        scenario_name: String description of the scenario
        param_adjustments: Dictionary of parameter adjustments
    
    Returns:
        Path to the saved plot
    """
    # Create a deep copy of the original parameters
    import copy
    new_params = copy.deepcopy(params)
    
    # Apply parameter adjustments
    for key, value in param_adjustments.items():
        if key in new_params:
            new_params[key] = value
        else:
            print(f"Warning: Parameter {key} not found in params dictionary")
    
    # Calculate results for new scenario
    print(f"\nEvaluating scenario: {scenario_name}")
    results = []
    for h in horizons:
        npv_A = compute_npv_offerA(h, new_params)
        npv_B, q025, q975, q10, q90, q40, q60, q70, q80 = expected_npv_offerB(h, new_params, trials=20000)
        results.append((h, npv_A, npv_B, q025, q975, q10, q90, q40, q60, q70, q80))
        print(f"Horizon {h:.1f} years -> NPV Offer A = ${npv_A:,.0f}, NPV Offer B = ${npv_B:,.0f} ± ${q975 - q025:,.0f}")
    
    # Generate and save plot with the new parameters
    plot_path = plot_npv_comparison(horizons, results, new_params)
    return plot_path

# Create a higher risk scenario
higher_risk_scenario = {
    "P_fail_A": 0.2,    # Increase failure probability at Series A (from 40% to 60%)
    "P_fail_B": 0.35,    # Increase failure probability at Series B (from 30% to 45%)
    "P_fail_C": 0.5,    # Increase failure probability at Series C (from 20% to 30%)
    "exit_valuation": 2.0e9  # Increased exit valuation to compensate for higher risk
}

create_scenario("Scenario 1 - 2B valuation ", higher_risk_scenario)

# Create a lower equity, higher salary scenario for Offer B
competitive_salary_scenario = {
    "offerB_base": 210000.0,      # Higher base salary
    "offerB_equity_shares": 60000.0,  # Lower equity
    "seriesA_valuation": 150e6,  # Higher Series A valuation
    "seriesB_valuation": 350e6,  # Higher Series B valuation
    "seriesC_valuation": 800e6,  # Higher Series C valuation
    "exit_valuation": 1e9     # Higher exit valuation
}

#create_scenario("Competitive Salary, Lower Equity", competitive_salary_scenario)

# Create a higher salvage value scenario
higher_salvage_scenario = {
    "valuation_multiplier_if_fail": 0.2,  # Increase salvage value to 50% of valuation
    "P_fail_A": 0.20,    # Keep the same failure probabilities
    "P_fail_B": 0.30,
    "P_fail_C": 0.40,
}

#create_scenario("Higher Salvage Value on Failure", higher_salvage_scenario)

# Create a higher risk but also higher salvage scenario
high_risk_high_salvage_scenario = {
    "P_fail_A": 0.60,    # Increase failure probability at Series A
    "P_fail_B": 0.45,    # Increase failure probability at Series B
    "P_fail_C": 0.30,    # Increase failure probability at Series C
    "valuation_multiplier_if_fail": 0.5,  # Higher salvage value (50%)
    "exit_valuation": 2.0e9  # Higher exit valuation to compensate for higher risk
}

#create_scenario("High Risk, High Salvage Value", high_risk_high_salvage_scenario)

