# Deep-Learning-for-Conditional-Asset-Pricing-Models
Implementation of Siamese LSTM for conditional asset pricing. Processes firm characteristics and market factors in parallel branches for stock return prediction. Paper: https://arxiv.org/abs/2509.04812

Overview
This repository contains a Siamese LSTM architecture implementation for stock return prediction. The model uses two parallel LSTM branches to process:

Individual firm characteristics
Common market factors (macroeconomic variables and average firm characteristics)

Requirements
See requirements.txt for dependencies.
Data Requirements
The model expects two CSV files:

full.csv: Firm-level data with columns:

permno: Firm identifier (CRSP permanent number)
DATE: Date column
eret: Excess returns (target variable)
94 firm characteristics (see Data Sources section)


common_factors.csv: Market-level data with columns:

date: Date column
Macroeconomic variables
Cross-sectional averages of firm characteristics



Data Sources and Preparation
Obtaining the Data
The datasets combine multiple financial databases:

CRSP (Center for Research in Security Prices)

Stock returns and trading data
Access via WRDS (Wharton Research Data Services) subscription
Monthly frequency data


Compustat

Firm fundamentals and accounting variables
Also available through WRDS
Quarterly/annual data converted to monthly frequency


FRED (Federal Reserve Economic Data)

Macroeconomic variables (freely available)
https://fred.stlouisfed.org/
Examples: inflation rates, term spreads, default spreads



Feature Engineering
Based on the paper's methodology:
Firm Characteristics (94 features for full.csv)
The characteristics follow those documented in:

Green, Hand, and Zhang (2017): "The characteristics that provide independent information about average U.S. monthly stock returns"
Gu, Kelly, and Xiu (2020): "Empirical asset pricing via machine learning"

Common categories include:

Momentum: Past returns (1-month, 3-month, 6-month, 12-month)
Value: Book-to-market, earnings-to-price, cash flow-to-price
Profitability: ROE, ROA, gross profitability
Investment: Asset growth, inventory growth, NOA
Trading frictions: Size, dollar volume, bid-ask spread
Risk measures: Beta, idiosyncratic volatility, downside beta

Calculating Excess Returns (eret)
python# Excess return = Stock return - Risk-free rate
eret = monthly_return - rf_rate
Where rf_rate is typically the 1-month T-bill rate from CRSP or FRED.
Common Factors (common_factors.csv)

Cross-sectional averages: Calculate mean of each firm characteristic across all firms for each month
Macroeconomic variables from FRED:

Default spread (BAA - AAA corporate bond yields)
Term spread (10-year - 3-month Treasury yields)
T-bill rates
Inflation rates
VIX (when available)



Data Processing Example
pythonimport pandas as pd
import numpy as np

# Example structure for creating full.csv
# Assume you have CRSP and Compustat data merged
def prepare_firm_data(crsp_data, compustat_data):
    # Merge CRSP and Compustat on permno and date
    merged = pd.merge(crsp_data, compustat_data, on=['permno', 'date'])
    
    # Calculate characteristics
    # Momentum
    merged['mom1m'] = merged['ret'].shift(1)
    merged['mom3m'] = merged['ret'].rolling(3).mean().shift(1)
    merged['mom6m'] = merged['ret'].rolling(6).mean().shift(1)
    merged['mom12m'] = merged['ret'].rolling(12).mean().shift(1)
    
    # Size
    merged['size'] = np.log(merged['mktcap'])
    
    # Book-to-market
    merged['bm'] = merged['book_equity'] / merged['mktcap']
    
    # ... additional characteristics
    
    # Calculate excess returns
    merged['eret'] = merged['ret'] - merged['rf']
    
    return merged

# Example for creating common_factors.csv
def prepare_common_factors(full_data, macro_data):
    # Calculate cross-sectional averages
    avg_chars = full_data.groupby('date').mean()
    
    # Merge with macro data
    common_factors = pd.merge(avg_chars, macro_data, on='date')
    
    return common_factors
Alternative Data Sources
If you don't have access to WRDS:

Yahoo Finance / yfinance: Basic price and volume data (limited characteristics)
FRED API: All macroeconomic variables (free)
OpenBB: Open-source financial data aggregator
Academic datasets: Some universities provide processed datasets for research

Data Quality Notes

Remove penny stocks (price < $5) and micro-caps if desired
Handle missing values appropriately (forward fill for prices, interpolation for fundamentals)
Winsorize extreme values at 1% and 99% percentiles
Ensure alignment of reporting dates to avoid look-ahead bias

Usage
bashpython pseudo_Siamese.py
The script uses the following fixed hyperparameters:

n_epoch: 50 (training epochs)
batch_size: 16
ln_hidden: 50 (hidden units in left LSTM for firm characteristics)
rn_hidden: 200 (hidden units in right LSTM for common factors)
dropout_rate: 0.3
learning_rate: 0.0001

Model Architecture
The Siamese LSTM consists of:

Left Branch: LSTM processing firm-specific characteristics

LSTM layer with 50 hidden units, ReLU activation
Dropout layer (0.3 rate)
Dense layer (output dimension: 1)


Right Branch: LSTM processing common factors

LSTM layer with 200 hidden units, ReLU activation
Dropout layer (0.3 rate)
Dense layer (output dimension: 1)


Fusion: Element-wise multiplication of both branches using Lambda layer
Output: Dense layer for final stock return prediction

Data Processing Pipeline

Normalization: All features are normalized to (0, 1) range using MinMaxScaler
Temporal Split:

Training: Before 2006-01-01
Validation: 2006-01-01 to 2011-01-01
Test: After 2011-01-01


Reshaping: Data is reshaped to 3D format (samples, timesteps=1, features) for LSTM input

Output
The trained model is saved as:
SLSTM_50_16_50_200_0.3_0.0001.h5
Format: SLSTM_[n_epoch]_[batch_size]_[ln_hidden]_[rn_hidden]_[dropout_rate]_[learning_rate].h5
Model Evaluation
The script includes commented code for model evaluation that calculates:

R² score for training set
Predictions for train/validation/test sets
Mispricing errors (residuals)

To enable evaluation, uncomment the relevant sections at the end of the script.
Notes

Data is processed firm by firm, then concatenated for batch training
Common factors are matched to firm dates using inner join
The model uses MSE loss and Adam optimizer
Features are scaled independently for firm characteristics and common factors

Citation

@misc{liu2025deeplearningconditionalasset,
      title={Deep Learning for Conditional Asset Pricing Models}, 
      author={Hongyi Liu},
      year={2025},
      eprint={2509.04812},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP},
      url={https://arxiv.org/abs/2509.04812}, 
}
