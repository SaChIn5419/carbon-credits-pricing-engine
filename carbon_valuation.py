import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import jarque_bera, kstest, norm, t
import scipy.stats as stats
import real_data_calibration as rdc

# ==========================================
# MODULE 1: THE CORE MATH ENGINE (Merton Model)
# ==========================================
class CarbonCreditValuator:
    def __init__(self, s0, mu, sigma, risk_free_rate=0.07):
        """
        Initialize the Valuation Engine.
        :param s0: Current Spot Price (Proxy or Shadow Price in INR)
        :param mu: Annual Drift (Expected inflation in abatement cost)
        :param sigma: Volatility (Standard Deviation of returns)
        :param risk_free_rate: Risk-free rate (India 10Y Bond Yield approx 7%)
        """
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.r = risk_free_rate

    def merton_jump_diffusion_simulation(self, T, dt, lambda_j, jump_mean, jump_std, n_sims=10000):
        """
        Simulates price paths using Geometric Brownian Motion with Jumps (GBMPJ).
        Validated by Daskalakis et al. (2009) as superior to Black-Scholes for Carbon.
        """
        n_steps = int(T / dt)
        prices = np.zeros((n_steps, n_sims))
        prices[0] = self.s0
        
        for t in range(1, n_steps):
            z = np.random.normal(0, 1, n_sims)
            
            # Poisson Jump Process
            n_jumps = np.random.poisson(lambda_j * dt, n_sims)
            
            # Helper to handle nan params if calibration fails gracefully
            j_mean = jump_mean if not np.isnan(jump_mean) else 0.0
            j_std = jump_std if not np.isnan(jump_std) else 0.0
            
            jump_magnitude = np.random.normal(j_mean, j_std, n_sims) * n_jumps
            
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z
            
            prices[t] = prices[t-1] * np.exp(drift + diffusion + jump_magnitude)
            
        return prices

    def calculate_fair_value(self, T=1.0, lambda_j=0.33, jump_mean=-0.05, jump_std=0.15):
        """
        Calculates Fair Value and returns simulations for Risk Cone.
        """
        dt = 1/252 
        simulations = self.merton_jump_diffusion_simulation(T, dt, lambda_j, jump_mean, jump_std)
        fair_value = np.mean(simulations[-1]) * np.exp(-self.r * T)
        return fair_value, simulations

# ==========================================
# MODULE 2: STRATEGIC & MACRO LAYER
# ==========================================
class StrategicAdvisor(CarbonCreditValuator):
    def __init__(self, s0, mu=0.05, sigma=0.42, risk_free_rate=0.07):
        # Initialize with dynamic params
        super().__init__(s0, mu=mu, sigma=sigma, risk_free_rate=risk_free_rate)
        
        self.geopolitics = {
            "EU": 1.3,      # Strict (CBAM Tax threat)
            "USA": 1.1,     # Transactional
            "GlobalSouth": 0.8 # Friendly
        }

    def enforcement_risk_filter(self, fair_value, sector_type):
        """Splits value based on domestic vs export enforcement reality."""
        if sector_type == "Export_Heavy":
            return fair_value * 1.15, "STRONG BUY (Export Compliance)"
        elif sector_type == "Domestic_Only":
            return fair_value * 0.70, "CAUTION (Enforcement Risk)"
        return fair_value, "HOLD"

# ==========================================
# MODULE 3: CREDIT QUALITY & CONVERGENCE (The New Part)
# ==========================================
class AdvancedCarbonAdvisor(StrategicAdvisor): 
    def __init__(self, s0, mu=0.05, sigma=0.42):
        # Pass params up
        super().__init__(s0, mu=mu, sigma=sigma)
        
        # Quality Multipliers (Current Tech)
        self.quality_matrix = {
            "DAC": 1.5,          # High Tech Removal
            "Biochar": 1.2,      # High Durability Removal
            "Reforestation": 1.0,# Nature Based Removal
            "Cookstoves": 0.6,   # Avoidance (High Fraud Risk)
            "Renewables": 0.2    # Junk (Old Tech Avoidance)
        }

    def assess_asset_specific_risk(self, credit_type, target_market, target_year, lambda_j=0.33, jump_mean=-0.05, jump_std=0.15):
        """
        The 'Time Bomb' Logic:
        If Year >= 2026 AND Market is EU AND Credit is Avoidance -> BANNED.
        """
        # 1. Base Quant Value using specific parameters
        base_fv, simulations = self.calculate_fair_value(T=2.0, lambda_j=lambda_j, jump_mean=jump_mean, jump_std=jump_std)
        
        # 2. Geopolitical Adj
        friction = self.geopolitics.get(target_market, 1.0)
        macro_price = base_fv * friction
        
        # 3. Quality Adj
        quality_factor = self.quality_matrix.get(credit_type, 0.5)
        
        # 4. The "Convergence Cliff" (New Logic)
        # EU bans 'Avoidance' credits starting 2026 (CBAM phase in)
        is_avoidance = credit_type in ["Cookstoves", "Renewables"]
        is_strict_market = target_market in ["EU", "USA"]
        
        if is_avoidance and is_strict_market and target_year >= 2026:
            print(f"   [CRITICAL ALERT] {credit_type} credits are INVALID in {target_market} post-2026.")
            final_price = 0.0 # Asset becomes worthless
            signal = "SELL IMMEDIATELY (Regulatory Ban)"
        else:
            final_price = macro_price * quality_factor
            signal = "BUY / HOLD" if final_price > base_fv else "SELL / AVOID"
            
        return final_price, signal, base_fv, simulations

# ==========================================
# MODULE 4: STATISTICAL VALIDATION
# ==========================================
class StatisticalValidator:
    def __init__(self, real_returns, simulated_prices):
        self.real_returns = real_returns
        self.sim_prices = simulated_prices
        
    def run_normality_tests(self):
        print("\n--- STATISTICAL SIGNIFICANCE TESTS (Jumps vs. Noise) ---")
        jb_stat, jb_p_value = jarque_bera(self.real_returns)
        print(f"1. Jarque-Bera Test:")
        print(f"   - Statistic: {jb_stat:.2f}")
        print(f"   - P-Value:   {jb_p_value:.5f}")
        
        if jb_p_value < 0.05:
            print("   -> RESULT: REJECT H0. Data is NON-NORMAL (Jumps are statistically significant).")
        else:
            print("   -> RESULT: FAIL TO REJECT H0. Data looks Normal.")

    def run_goodness_of_fit(self):
        std_returns = (self.real_returns - np.mean(self.real_returns)) / np.std(self.real_returns)
        ks_stat, ks_p_value = kstest(std_returns, 'norm')
        print(f"\n2. Kolmogorov-Smirnov Test:")
        print(f"   - P-Value:   {ks_p_value:.5f}")
        if ks_p_value < 0.05:
            print("   -> RESULT: REJECT Normality. Confirms non-linear behavior.")
            
    def calculate_valuation_confidence(self, fair_value, confidence=0.95):
        terminal_prices = self.sim_prices[-1] 
        std_err = np.std(terminal_prices) / np.sqrt(len(terminal_prices))
        df = len(terminal_prices) - 1
        t_score = t.ppf((1 + confidence) / 2, df)
        margin_of_error = t_score * std_err
        lower_bound = fair_value - margin_of_error
        upper_bound = fair_value + margin_of_error
        
        print(f"\n--- VALUATION CONFIDENCE INTERVAL ({confidence:.0%}) ---")
        print(f"   Model Fair Value:  INR {fair_value:.2f}")
        print(f"   Confidence Range: [INR {lower_bound:.2f} - INR {upper_bound:.2f}]")
        print(f"   Margin of Error:   INR {margin_of_error:.2f}")
        return lower_bound, upper_bound

class RigorousValidator(StatisticalValidator):
    def run_qq_plot(self):
        print("\n--- VISUAL VALIDATION (Q-Q Plot) ---")
        plt.figure(figsize=(8, 6))
        stats.probplot(self.real_returns, dist="norm", plot=plt)
        plt.title("Q-Q Plot: Real Carbon Data vs. Normal Distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig('qq_plot.png')
        print("   -> Q-Q Plot saved to qq_plot.png")

    def run_3_sigma_test(self):
        print("\n--- THE '3-SIGMA' REALITY CHECK ---")
        std_dev = np.std(self.real_returns)
        mean = np.mean(self.real_returns)
        upper_limit = mean + 3 * std_dev
        lower_limit = mean - 3 * std_dev
        outliers = self.real_returns[(self.real_returns > upper_limit) | (self.real_returns < lower_limit)]
        num_outliers = len(outliers)
        total_days = len(self.real_returns)
        expected_normal = total_days * 0.0027 
        
        print(f"   Total Trading Days: {total_days}")
        print(f"   Actual Outliers:    {num_outliers} days (Expected: {expected_normal:.1f})")
        if num_outliers > expected_normal * 2:
            print("   -> RESULT: VALIDATED. Market creates 2x+ more shocks than predicted.")

# ==========================================
# EXECUTION & VALIDATION
# ==========================================
if __name__ == "__main__":
    print("--- C-RISK ENGINE: INITIALIZING PIPELINE ---")
    
    # 1. PIPELINE STEP 1: Fetch/Load Real Market Data
    df = rdc.fetch_real_market_data()
    if df.empty:
        print("CRITICAL ERROR: No market data found.")
        exit(1)
        
    # 2. PIPELINE STEP 2: Calibrate Parameters
    sigma, lam, j_mean, j_std, mu_proxy = rdc.calibrate_model_params(df['Carbon_EU'])
    print(f"\n--- CALIBRATION COMPLETE ---")
    print(f"   Real Volatility: {sigma:.2%}")
    print(f"   Shock Intensity: {lam:.2f} / year")
    
    # 3. Initialize Advanced Advisor with REAL params
    Indian_Spot_Price = 1500
    advisor = AdvancedCarbonAdvisor(s0=Indian_Spot_Price, mu=0.05, sigma=sigma)
    
    print("\n--- C-RISK ENGINE: FINAL ASSET VALUATION ---")
    
    # SCENARIO 1: The "Junk" Asset (Wind Power in 2027)
    print("\n1. ASSET: Wind Power Credits (Renewables)")
    print("   Target: Export to EU | Year: 2027")
    price_wind, sig_wind, _, _ = advisor.assess_asset_specific_risk("Renewables", "EU", 2027, lambda_j=lam, jump_mean=j_mean, jump_std=j_std)
    print(f"   Valuation: INR {price_wind:.2f}")
    print(f"   Signal:    {sig_wind}")
    
    # SCENARIO 2: The "Gold" Asset (Biochar in 2027)
    print("\n2. ASSET: Biochar Removal Credits")
    print("   Target: Export to EU | Year: 2027")
    price_bio, sig_bio, _, _ = advisor.assess_asset_specific_risk("Biochar", "EU", 2027, lambda_j=lam, jump_mean=j_mean, jump_std=j_std)
    print(f"   Valuation: INR {price_bio:.2f}")
    print(f"   Signal:    {sig_bio}")

    # SCENARIO 3: The "Domestic" Asset (Cookstoves in India)
    # Using this scenario to drive the statistical validation (Risk Cone source)
    print("\n3. ASSET: Cookstoves")
    print("   Target: Domestic (GlobalSouth) | Year: 2027")
    price_cook, sig_cook, base_fv, sims = advisor.assess_asset_specific_risk("Cookstoves", "GlobalSouth", 2027, lambda_j=lam, jump_mean=j_mean, jump_std=j_std)
    print(f"   Valuation: INR {price_cook:.2f}")
    print(f"   Signal:    {sig_cook}")
    
    # SCENARIO 4: VISUAL PROOF (Risk Cone)
    print("\n--- GENERATING RISK CONE ---")
    plt.figure(figsize=(10,6))
    plt.plot(sims[:, :50], color='green', alpha=0.1) # Plot 50 paths
    plt.plot(np.mean(sims, axis=1), color='black', linewidth=2, label="Mean Price Path")
    plt.title(f"C-RISK: Indian Carbon Price Forecast (2026-2028)\nModel: Jump-Diffusion (Sigma={sigma:.2f}, Lambda={lam:.2f})")
    plt.xlabel("Trading Days")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.savefig('risk_cone_real.png')
    print("Visualization saved to risk_cone_real.png")

    # SCENARIO 5: STATISTICAL VALIDATION
    print("\n--- RUNNING RIGOROUS VALIDATION ---")
    real_returns = np.log(df['Carbon_EU'] / df['Carbon_EU'].shift(1)).dropna()
    validator = RigorousValidator(real_returns, sims)
    validator.run_normality_tests()
    validator.run_3_sigma_test()
    validator.run_qq_plot()
    validator.calculate_valuation_confidence(price_bio) # Validate our "Gold" asset
