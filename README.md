# Indian Carbon Credit Valuation System (CCTS 2026)

This project implements a **Carbon Credit Valuation Engine** specifically tailored for the upcoming **Indian Carbon Market (CCTS 2026)**.

## Core Features

1.  **Real-Data Pipeline**: Automatically fetches EU ETS (Proxy) and Coal India data to "calibrate" the model's physics (Volatility & Jump Intensity).
2.  **Merton's Jump-Diffusion Model**: Simulates carbon price paths accounting for sudden policy shocks (e.g., BEE target changes).
3.  **Data Caching**: Saves market data to `market_data_cache.csv` for faster subsequent runs and offline reliability.
4.  **Strategic Advisor**: Adjusts fair value based on Geopolitics (EU/USA/Global South) and Enforcement Risk (Paper Tiger vs. CBAM).

## Project Structure

-   `carbon_valuation.py`: **Main Pipeline.** Fetches data, calibrates model, runs valuation, and generates reports/charts.
-   `real_data_calibration.py`: Helper module for fetching YFinance data and performing statistical calibration.
-   `market_data_cache.csv`: Automatically created file storing historical market data.
-   `risk_cone_real.png`: Visualization of projected Indian Carbon Price pathways.
-   `calibration_chart.png`: Visualization comparing Real Data history vs. Model Forecast.
-   `requirements.txt`: Python dependencies.
-   `run_valuation.bat`: One-click runner for Windows.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Pipeline**:
    Double-click `run_valuation.bat` 
    OR run:
    ```bash
    python carbon_valuation.py
    ```

## Logic Flow

1.  **Fetch**: Checks `market_data_cache.csv`. If missing, downloads `KEUA` (EU Carbon ETF) and `COALINDIA.NS` from Yahoo Finance.
2.  **Calibrate**: Calculates Annual Volatility ($\sigma$) and Jump Intensity ($\lambda$) from the downloaded history.
3.  **Simulate**: Uses these real-world parameters to simulate thousands of scenarios for the Indian market starting at ₹1500.
4.  **Evaluate**: Applies "Realpolitik" filters (Sector Type + Target Market) to give a final Buy/Sell signal.

## Interpretation of Outputs

-   **Base Quant Value**: The theoretical price based on pure math (Merton Model).
-   **Strategic Value**: The real-world price adjusted for political friction and compliance mandates.
-   **Action**: `STRONG BUY` (Mandatory compliance sectors like Steel) vs `CAUTION` (Domestic sectors with weak enforcement).
