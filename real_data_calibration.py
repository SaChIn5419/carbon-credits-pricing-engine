import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- PART 1: FETCH REAL DATA (The "Input" Layer) ---
def fetch_real_market_data(cache_file='market_data_cache.csv'):
    # 1. Try loading from Cache
    try:
        print(f"Checking for cached data ({cache_file})...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print("Loaded data from cache.")
        return df
    except FileNotFoundError:
        print("Cache not found. Fetching Real-World Data via yfinance...")
    
    # 2. PROXY ASSET: KraneShares European Carbon Allowance ETF (KEUA)
    def get_carbon_data(ticker):
        print(f"Attempting to fetch {ticker}...")
        try:
            d = yf.download(ticker, start="2022-01-01", end="2025-01-01", progress=False)
            if not d.empty and 'Close' in d.columns:
                 return d['Close']
            elif not d.empty: # Maybe it's a series or different format
                 return d
        except Exception as e:
            print(f"Exception fetching {ticker}: {e}")
        return pd.Series()

    carbon_data = get_carbon_data("KEUA")
    if carbon_data.empty:
        print("KEUA data empty. Trying KRBN...")
        carbon_data = get_carbon_data("KRBN")
    
    if carbon_data.empty:
        print("Error: Could not fetch Carbon Proxy data (KEUA or KRBN).")
        return pd.DataFrame()

    # 3. FUNDAMENTAL DRIVER: Coal India Ltd (COALINDIA.NS)
    # Used to check correlation (High Coal Price often = High Carbon Demand)
    try:
        print("Fetching COALINDIA.NS...")
        coal_data = yf.download("COALINDIA.NS", start="2022-01-01", end="2025-01-01", progress=False)['Close']
    except Exception as e:
        print(f"Error fetching Coal India data: {e}")
        coal_data = pd.Series()

    print(f"Data Sizes - Carbon: {len(carbon_data)}, Coal: {len(coal_data)}")
    
    # Align Data
    # Use concat which is more robust for aligning Series
    frames = [carbon_data]
    column_names = ['Carbon_EU']
    
    if not coal_data.empty:
        frames.append(coal_data)
        column_names.append('Coal_India')
        
    try:
        df = pd.concat(frames, axis=1)
        df.columns = column_names
    except Exception as e:
        print(f"Error validating data for DataFrame: {e}")
        return pd.DataFrame()
        
    df.ffill(inplace=True)
    df.dropna(inplace=True) # Drop initial NaNs
    
    # Save to cache
    df.to_csv(cache_file)
    print(f"Data saved to {cache_file}")
    
    return df

# --- PART 2: CALIBRATE PARAMETERS (The "Quant" Layer) ---
def calibrate_model_params(price_series):
    """
    Instead of guessing, we calculate Sigma and Lambda from history.
    """
    if price_series.empty:
        return 0.5, 0.5, 0, 0, 0 # Default values if empty

    # Calculate Daily Log Returns: ln(Pt / Pt-1)
    returns = np.log(price_series / price_series.shift(1)).dropna()
    
    if returns.empty:
         return 0.5, 0.5, 0, 0, 0
         
    # 1. Volatility (Sigma): Standard Deviation of returns * sqrt(252)
    sigma_annual = returns.std() * np.sqrt(252)
    
    # 2. Jump Detection (The "Shock" Logic)
    # We define a "Jump" as any day where return > 3 Standard Deviations
    threshold = 3 * returns.std()
    jumps = returns[abs(returns) > threshold]
    
    # Lambda (Jump Intensity): How many jumps per year?
    # (Total Jumps / Total Years)
    years = len(returns) / 252
    if years < 0.1: years = 1 # Safety check
    lambda_annual = len(jumps) / years
    
    # Jump Size Statistics
    jump_mean = jumps.mean()
    jump_std = jumps.std()
    
    print(f"\n--- CALIBRATED PARAMETERS FROM REAL DATA ---")
    print(f"Annual Volatility (Sigma): {sigma_annual:.2%}")
    print(f"Detected Jumps (Total):    {len(jumps)}")
    print(f"Jump Intensity (Lambda):   {lambda_annual:.2f} jumps/year")
    print(f"Avg Jump Magnitude:        {jump_mean:.2%}")
    
    return sigma_annual, lambda_annual, jump_mean, jump_std, returns.mean()*252

def visualize_calibration(df, simulated_paths, sigma, lam):
    plt.figure(figsize=(12, 6))

    # Subplot 1: Real Data (The "Training Set")
    plt.subplot(1, 2, 1)
    plt.plot(df['Carbon_EU'], color='blue')
    plt.title("REAL DATA: EU Carbon Price (Proxy)\n(Source: yfinance 'KEUA')")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha=0.3)

    # Subplot 2: Your Model's Prediction for India
    plt.subplot(1, 2, 2)
    plt.plot(simulated_paths, color='green', alpha=0.1)
    # Calculate mean manually to avoid axis issues if numpy versions mismatch
    mean_path = np.mean(simulated_paths, axis=1)
    plt.plot(mean_path, color='black', linewidth=2, label='Mean Forecast')
    plt.title(r"YOUR MODEL: Indian CCTS Forecast" + "\n" + rf"(Calibrated: $\sigma$={sigma:.2f}, $\lambda$={lam:.2f})")
    plt.xlabel("Trading Days (Next 1 Year)")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('calibration_chart.png')
    print("Chart saved to calibration_chart.png")

# --- PART 3: THE JUMP-DIFFUSION ENGINE (Your Original Code) ---
def run_simulation(S0, mu, sigma, lambda_j, jump_mean, jump_std, T=1.0):
    dt = 1/252
    steps = int(T * 252)
    paths = 100  # Number of scenarios
    
    price_paths = np.zeros((steps, paths))
    price_paths[0] = S0
    
    for t in range(1, steps):
        z = np.random.normal(0, 1, paths)
        
        # Poisson Jump Process
        n_jumps = np.random.poisson(lambda_j * dt, paths)
        try:
             # handle case where jumps are detected but size is nan if param is nan
             if np.isnan(jump_mean): jump_mean = 0
             if np.isnan(jump_std): jump_std = 0
             jump_impact = n_jumps * np.random.normal(jump_mean, jump_std, paths)
        except:
             jump_impact = 0

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        
        price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion + jump_impact)
        
    return price_paths

if __name__ == "__main__":
    # --- MAIN EXECUTION ---

    # 1. Get Data
    df = fetch_real_market_data()
    
    if df.empty:
        print("Error: Could not fetch data. Please check your internet connection or tickers.")
    else:
        # 2. Calibrate based on EU Carbon History
        # Note: We use the EU data to get the 'Sigma' and 'Lambda' structure, 
        # but we will apply it to the Indian Starting Price (S0).
        sigma, lam, j_mean, j_std, mu = calibrate_model_params(df['Carbon_EU'])

        # 3. Set Inputs for Indian Market (Hypothetical Start)
        Indian_Spot_Price = 1500  # Current Shadow Price in INR
        Indian_Mu = 0.05          # We assume 5% growth trend for India

        # 4. Run Model
        simulated_paths = run_simulation(Indian_Spot_Price, Indian_Mu, sigma, lam, j_mean, j_std, T=1)

        # 5. Visualize: Real History vs. Future Simulation
        plt.figure(figsize=(12, 6))

        # Subplot 1: Real Data (The "Training Set")
        plt.subplot(1, 2, 1)
        plt.plot(df['Carbon_EU'], color='blue')
        plt.title("REAL DATA: EU Carbon Price (Proxy)\n(Source: yfinance 'KEUA')")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)

        # Subplot 2: Your Model's Prediction for India
        plt.subplot(1, 2, 2)
        plt.plot(simulated_paths, color='green', alpha=0.1)
        # Calculate mean manually to avoid axis issues if numpy versions mismatch
        mean_path = np.mean(simulated_paths, axis=1)
        plt.plot(mean_path, color='black', linewidth=2, label='Mean Forecast')
        plt.title(r"YOUR MODEL: Indian CCTS Forecast" + "\n" + rf"(Calibrated: $\sigma$={sigma:.2f}, $\lambda$={lam:.2f})")
        plt.xlabel("Trading Days (Next 1 Year)")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('calibration_chart.png')
        print("Chart saved to calibration_chart.png")
        
        # 6. Strategic Output
        print("\n--- STRATEGIC INSIGHT ---")
        print(f"The Real Data shows that Carbon Markets are highly volatile ({sigma:.0%}).")
        print(f"Your model detected {lam:.1f} major price shocks per year in the real data.")
        print(f"Applying this to India: If we start at INR {Indian_Spot_Price}, the Fair Value Risk Premium suggests holding until INR {mean_path[-1]:.0f}.")
