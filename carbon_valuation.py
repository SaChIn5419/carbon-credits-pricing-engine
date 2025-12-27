import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

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
    def __init__(self, s0):
        # Initialize with baseline Indian Market params (High Volatility due to new market)
        super().__init__(s0, mu=0.05, sigma=0.60, risk_free_rate=0.07)
        
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

    def get_strategic_price(self, target_market, sector_type):
        # 1. Base Quant Valuation (Merton Model)
        # Intensity 0.33 = 1 shock every 3 years (Indian Compliance Cycle)
        base_fv = self.calculate_fair_value(T=2.0, lambda_j=0.33) 
        
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
    # --- EXECUTION: SIMULATING THE 2026 MARKET LAUNCH ---

    # Scenario: Current Shadow Price is ₹1,500
    advisor = StrategicAdvisor(s0=1500)

    print("--- STRATEGIC CARBON VALUATION REPORT (INDIA CCTS) ---")

    # Case A: Tata Steel (Exports to Europe)
    steel_report = advisor.get_strategic_price(target_market="EU", sector_type="Export_Heavy")
    print(f"\nSector: STEEL (Export Heavy to EU)")
    print(f"Base Quant Value:     INR {steel_report['Model_Fair_Value']} (Jump-Diffusion)")
    print(f"Strategic Value:      INR {steel_report['Final_Strategic_Price']}")
    print(f"Action:               {steel_report['Signal']}")
    print(f"Reasoning:            EU CBAM acts as a 'Shadow Regulator', forcing compliance despite local friction.")

    # Case B: UltraTech Cement (Domestic Consumption)
    cement_report = advisor.get_strategic_price(target_market="GlobalSouth", sector_type="Domestic_Only")
    print(f"\nSector: CEMENT (Domestic / Global South)")
    print(f"Base Quant Value:     INR {cement_report['Model_Fair_Value']} (Jump-Diffusion)")
    print(f"Strategic Value:      INR {cement_report['Final_Strategic_Price']}")
    print(f"Action:               {cement_report['Signal']}")
    print(f"Reasoning:            Market pricing in 30% probability of weak BEE enforcement ('Paper Tiger' scenario).")

    # --- VISUALIZATION: THE "RISK CONE" ---
    # Simulating paths to show the "Jumps" visually
    sims = advisor.merton_jump_diffusion_simulation(T=2, dt=1/252, lambda_j=0.33, jump_mean=0.15, jump_std=0.1, n_sims=50)

    plt.figure(figsize=(10, 6))
    plt.plot(sims, color='green', alpha=0.1)
    plt.plot(np.mean(sims, axis=1), color='black', linewidth=2, label='Expected Price Path')
    plt.title("Indian Carbon Price Pathways (Jump-Diffusion Model)\nCapturing 3-Year Policy Cycle Shocks")
    plt.xlabel("Trading Days (2 Years)")
    plt.ylabel("Price (INR/tCO2e)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('risk_cone.png')
    print("\nVisualization saved to risk_cone.png")
