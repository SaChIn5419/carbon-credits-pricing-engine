# Indian Carbon Credit Valuation System (CCTS 2026)

This project implements a **Carbon Credit Valuation Engine** specifically tailored for the upcoming **Indian Carbon Market (CCTS 2026)**.

## Core Features

1.  **Merton's Jump-Diffusion Model**: Simulates carbon price paths using Geometric Brownian Motion with Jumps (GBMPJ) to account for sudden policy shocks (e.g., EU CBAM announcements, BEE target changes).
2.  **Geopolitical Friction Matrix**: Adjusts the base fair value based on the target market's affinity/hostility (EU, USA, Global South).
3.  **Enforcement Risk Filter**: Accounts for the "Paper Tiger" risk – differentiating between export sectors (forced compliance via CBAM) and domestic sectors (potential for weak enforcement).

## Project Structure

-   `carbon_valuation.py`: The main Python script containing the `CarbonCreditValuator` and `StrategicAdvisor` classes.
-   `risk_cone.png`: Generated visualization of Monte Carlo price simulations.
-   `requirements.txt`: List of Python dependencies.
-   `run_valuation.bat`: Helper script to run the model on Windows using Anaconda.

## How to Run

1.  Ensure you have **Python** (or Anaconda) installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the valuation script:
    ```bash
    python carbon_valuation.py
    ```
    Or simply double-click `run_valuation.bat`.

## Model Parameters

-   **Spot Price ($S_0$)**: Current proxy price (e.g., ₹1500).
-   **Jump Intensity ($\lambda$)**: Set to 0.33 (one shock every 3 years) to match the ICAP compliance cycle.
-   **Drift ($\mu$)**: 5% (Abatement cost inflation).
-   **Volatility ($\sigma$)**: 60% (High volatility for new markets).
