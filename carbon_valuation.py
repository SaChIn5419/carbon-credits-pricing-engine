import numpy as np
import real_data_calibration as rdc
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kstest, norm, t
import scipy.stats as stats

class StatisticalValidator:
    def __init__(self, real_returns, simulated_prices):
        self.real_returns = real_returns
        self.sim_prices = simulated_prices
        
    def run_normality_tests(self):
        """
        Hypothesis: Carbon returns are NOT Normal (Gaussian).
        H0 (Null Hypothesis): Returns follow a Normal Distribution.
        H1 (Alternative): Returns contain Jumps/Fat Tails.
        """
        print("\n--- STATISTICAL SIGNIFICANCE TESTS (Jumps vs. Noise) ---")
        
        # 1. Jarque-Bera Test (The standard for checking Fat Tails)
        # It measures Skewness and Kurtosis. 
        jb_stat, jb_p_value = jarque_bera(self.real_returns)
        
        print(f"1. Jarque-Bera Test:")
        print(f"   - Statistic: {jb_stat:.2f}")
        print(f"   - P-Value:   {jb_p_value:.5f}")
        
        if jb_p_value < 0.05:
            print("   -> RESULT: REJECT H0. Data is NON-NORMAL (Jumps are statistically significant).")
            print("   -> Implication: Black-Scholes would fail here. Jump-Diffusion is required.")
        else:
            print("   -> RESULT: FAIL TO REJECT H0. Data looks Normal.")

    def run_goodness_of_fit(self):
        """
        2. Kolmogorov-Smirnov Test (KS Test)
        Checks if real data fits a Normal Distribution.
        """
        # Standardize returns
        std_returns = (self.real_returns - np.mean(self.real_returns)) / np.std(self.real_returns)
        ks_stat, ks_p_value = kstest(std_returns, 'norm')
        
        print(f"\n2. Kolmogorov-Smirnov Test:")
        print(f"   - P-Value:   {ks_p_value:.5f}")
        if ks_p_value < 0.05:
            print("   -> RESULT: REJECT Normality. Confirms non-linear behavior.")
            
    def calculate_valuation_confidence(self, fair_value, confidence=0.95):
        """
        3. Monte Carlo Confidence Intervals
        How sure are we about the valuation?
        """
        # Get terminal prices from the simulation matrix
        terminal_prices = self.sim_prices[-1] 
        
        # Standard Error
        std_err = np.std(terminal_prices) / np.sqrt(len(terminal_prices))
        
        # T-Score for 95% Confidence
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

class RigorousValidator(StatisticalValidator): # Extends your previous class
    def run_qq_plot(self):
        """
        Visual Test: The Q-Q Plot.
        If the Blue Dots deviate from the Red Line at the corners, 
        it proves Fat Tails (Extreme Events) exist.
        """
        print("\n--- VISUAL VALIDATION (Q-Q Plot) ---")
        plt.figure(figsize=(8, 6))
        
        # Compare Real Carbon Returns vs. Theoretical Normal Distribution
        stats.probplot(self.real_returns, dist="norm", plot=plt)
        
        plt.title("Q-Q Plot: Real Carbon Data vs. Normal Distribution")
        plt.grid(True, alpha=0.3)
        plt.savefig('qq_plot.png')
        print("   -> Q-Q Plot saved to qq_plot.png")
        print("   -> INTERPRETATION: Look at the tails (top right, bottom left).")
        print("      If dots curl away from the red line, the '0.0 P-Value' is real.")

    def run_3_sigma_test(self):
        """
        The 'Trader Sanity Check'.
        Count the number of extreme days (> 3 Std Devs).
        """
        print("\n--- THE '3-SIGMA' REALITY CHECK ---")
        
        std_dev = np.std(self.real_returns)
        mean = np.mean(self.real_returns)
        
        # Define thresholds
        upper_limit = mean + 3 * std_dev
        lower_limit = mean - 3 * std_dev
        
        # Count outliers
        outliers = self.real_returns[(self.real_returns > upper_limit) | (self.real_returns < lower_limit)]
        num_outliers = len(outliers)
        total_days = len(self.real_returns)
        
        # Theoretical expectation under Normal Distribution
        expected_normal = total_days * 0.0027 # 0.27% probability
        
        print(f"   Total Trading Days: {total_days}")
        print(f"   Threshold (3x Sigma): +/- {3*std_dev:.2%}")
        print(f"   Expected Outliers (if Normal): {expected_normal:.1f} days")
        print(f"   ACTUAL Outliers Observed:      {num_outliers} days")
        
        if num_outliers > expected_normal * 2:
            print("   -> RESULT: VALIDATED. Market creates 2x+ more shocks than Black-Scholes predicts.")
            print("   -> This confirms that the '0.00000' P-Value is not a bug.")
        else:
            print("   -> RESULT: Suspicious. Data looks surprisingly normal.")

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
        
        :param lambda_j: Jump Intensity (Expected shocks per year, e.g., 0.33 for 3-year cycle)
        :param jump_mean: Average size of the jump (e.g., -0.1 for supply glut, +0.2 for tightening)
        :param jump_std: Volatility of the jump size
        """
        n_steps = int(T / dt)
        prices = np.zeros((n_steps, n_sims))
        prices[0] = self.s0
        
        for t in range(1, n_steps):
            # 1. Standard Diffusion (Continuous Volatility)
            z = np.random.normal(0, 1, n_sims)
            
            # 2. Jump Component (Poisson Process) - The "Information Shock"
            # Logic: BEE releases new targets or EU CBAM rules change
            n_jumps = np.random.poisson(lambda_j * dt, n_sims)
            # Handle potential scalar/nan issues
            try:
                if np.isnan(jump_mean): jump_mean = 0
                if np.isnan(jump_std): jump_std = 0
            except:
                pass
            jump_magnitude = np.random.normal(jump_mean, jump_std, n_sims) * n_jumps
            
            # Combine: Previous Price * exp(Drift + Diffusion + Jumps)
            # Drift adjustment for jumps: (mu - 0.5*sigma^2 - lambda*(exp(jump_mean+0.5*jump_std^2)-1))
            # Simplified drift for clarity:
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * z
            
            prices[t] = prices[t-1] * np.exp(drift + diffusion + jump_magnitude)
            
        return prices

    def calculate_fair_value(self, T=1.0, lambda_j=0.33, jump_mean=0.10, jump_std=0.15):
        """
        Calculates the Risk-Neutral Fair Value (Discounted Expectation).
        """
        dt = 1/252 # Daily steps
        simulations = self.merton_jump_diffusion_simulation(T, dt, lambda_j, jump_mean, jump_std)
        terminal_values = simulations[-1]
        
        # Discount back to present
        fair_value = np.mean(terminal_values) * np.exp(-self.r * T)
        return fair_value

class StrategicAdvisor(CarbonCreditValuator):
    def __init__(self, s0, mu=0.05, sigma=0.60):
        # Initialize with passed params (Real Data Calibration or Default)
        super().__init__(s0, mu=mu, sigma=sigma, risk_free_rate=0.07)
        
        # Affinity Matrix: Geopolitical Friction Multipliers
        # > 1.0 means High Risk (Hostile/Strict), < 1.0 means Low Risk (Friendly)
        self.geopolitics = {
            "EU": 1.3,      # Strict (CBAM Tax threat)
            "USA": 1.1,     # Transactional (IRA competition)
            "GlobalSouth": 0.8 # Friendly (Article 6.2 partners)
        }

    def enforcement_risk_filter(self, fair_value, sector_type):
        """
        The 'Realpolitik' Module: Adjusts price based on sector compliance reality.
        
        Logic:
        - Export Sectors (Steel) CANNOT default because EU CBAM is the 'Shadow Regulator'.
        - Domestic Sectors (Cement) might default if Indian enforcement is weak ('Paper Tiger').
        """
        if sector_type == "Export_Heavy":
            # Price Premium: They MUST buy credits, driving demand up.
            return fair_value * 1.15, "STRONG BUY (CBAM Forced Compliance)"
            
        elif sector_type == "Domestic_Only":
            # Regulatory Discount: Risk of weak enforcement / corruption.
            # 30% discount for 'Paper Tiger' risk
            return fair_value * 0.70, "CAUTION (Enforcement Risk High)"
            
        return fair_value, "HOLD"

    def get_strategic_price(self, target_market, sector_type, lambda_j=0.33):
        # 1. Base Quant Valuation (Merton Model)
        base_fv = self.calculate_fair_value(T=2.0, lambda_j=lambda_j) 
        
        # 2. Apply Geopolitical Friction
        friction = self.geopolitics.get(target_market, 1.0)
        macro_adj_price = base_fv * friction
        
        # 3. Apply Enforcement Filter
        final_price, signal = self.enforcement_risk_filter(macro_adj_price, sector_type)
        
        return {
            "Model_Fair_Value": round(base_fv, 2),
            "Geopolitics_Adj": round(macro_adj_price, 2),
            "Final_Strategic_Price": round(final_price, 2),
            "Signal": signal
        }

if __name__ == "__main__":
    print("--- STARTING CARBON VALUATION PIPELINE ---")
    
    # 1. PIPELINE STEP 1: Fetch/Load Real Market Data
    df = rdc.fetch_real_market_data()
    
    if df.empty:
        print("CRITICAL ERROR: No market data found. Cannot calibrate.")
        exit(1)
        
    # 2. PIPELINE STEP 2: Calibrate Parameters from History
    # We use the EU data (proxy) to find the 'Physics' of the market (Sigma, Lambda)
    sigma, lam, j_mean, j_std, mu_proxy = rdc.calibrate_model_params(df['Carbon_EU'])
    
    print(f"\n--- PIPELINE CALIBRATION COMPLETE ---")
    print(f"Using Real-World Volatility: {sigma:.2%}")
    print(f"Using Real-World Shock Intensity: {lam:.2f} / year")
    
    # 3. PIPELINE STEP 3: Initialize Strategic Advisor with Real Params
    # Scenario: Current Shadow Price is 1500 INR
    # We use a conservative Indian growth drift (5%) but REAL volatility
    Indian_Spot_Price = 1500
    Indian_Mu = 0.05
    
    advisor = StrategicAdvisor(s0=Indian_Spot_Price, mu=Indian_Mu, sigma=sigma)

    # 3b. PIPELINE Output: Save Calibration Evidence
    print("\nGenerating Calibration Check Chart...")
    # Run a quick 1-year sim just for the calibration comparison chart
    calib_sims = advisor.merton_jump_diffusion_simulation(T=1.0, dt=1/252, lambda_j=lam, jump_mean=j_mean, jump_std=j_std, n_sims=100)
    rdc.visualize_calibration(df, calib_sims, sigma, lam)

    print("\n--- STRATEGIC CARBON VALUATION REPORT (INDIA CCTS) ---")

    # Case A: Tata Steel (Exports to Europe)
    steel_report = advisor.get_strategic_price(target_market="EU", sector_type="Export_Heavy", lambda_j=lam)
    print(f"\nSector: STEEL (Export Heavy to EU)")
    print(f"Base Quant Value:     INR {steel_report['Model_Fair_Value']} (Jump-Diffusion)")
    print(f"Strategic Value:      INR {steel_report['Final_Strategic_Price']}")
    print(f"Action:               {steel_report['Signal']}")
    print(f"Reasoning:            EU CBAM acts as a 'Shadow Regulator', forcing compliance despite local friction.")

    # Case B: UltraTech Cement (Domestic Consumption)
    cement_report = advisor.get_strategic_price(target_market="GlobalSouth", sector_type="Domestic_Only", lambda_j=lam)
    print(f"\nSector: CEMENT (Domestic / Global South)")
    print(f"Base Quant Value:     INR {cement_report['Model_Fair_Value']} (Jump-Diffusion)")
    print(f"Strategic Value:      INR {cement_report['Final_Strategic_Price']}")
    print(f"Action:               {cement_report['Signal']}")
    print(f"Reasoning:            Market pricing in 30% probability of weak BEE enforcement ('Paper Tiger' scenario).")

    # --- VISUALIZATION: THE "RISK CONE" WITH REAL DATA ---
    print("\nGenerating Real-Data Risk Cone...")
    sims = advisor.merton_jump_diffusion_simulation(T=2, dt=1/252, lambda_j=lam, jump_mean=j_mean, jump_std=j_std, n_sims=1000) # Increased N for robust stats

    plt.figure(figsize=(10, 6))
    plt.plot(sims[:, :50], color='green', alpha=0.1) # Plot only 50 paths to avoid clutter
    plt.plot(np.mean(sims, axis=1), color='black', linewidth=2, label=f'Expected Price Path (Real $\lambda$={lam:.2f})')
    plt.title("Indian Carbon Price Pathways (Calibrated on Real EU Data)\nCapturing Actual Market Shocks")
    plt.xlabel("Trading Days (2 Years)")
    plt.ylabel("Price (INR/tCO2e)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('risk_cone_real.png')
    print("Visualization saved to risk_cone_real.png")
    
    # --- STATISTICAL VALIDATION ---
    print("\n--- RUNNING STATISTICAL VALIDATION MODULE ---")
    
    # 1. Calculate Real Returns for Testing
    real_returns = np.log(df['Carbon_EU'] / df['Carbon_EU'].shift(1)).dropna()
    
    # 2. Init Validator 
    # Note: We use the large 'sims' matrix for robust confidence intervals
    # validator = StatisticalValidator(real_returns, sims)
    validator = RigorousValidator(real_returns, sims)
    
    # 3. Run Hypothesis Tests
    validator.run_normality_tests()
    validator.run_goodness_of_fit()
    
    # 3b. Run Visual & Reality Checks (New Layer)
    validator.run_qq_plot()
    validator.run_3_sigma_test()
    
    # 4. Check Confidence on Valuation
    # We validate the STEEL valuation (our primary export case)
    steel_fv = steel_report['Final_Strategic_Price']
    validator.calculate_valuation_confidence(steel_fv)
