# [...Work in progress..]💰Financial Fraud Detection
**Note:** This repository is in work progress.

## Introduction

This repository focuses on developing and implementing machine learning models to detect financial fraud efficiently. 

## Dataset

You can download the dataset from here:  
[📥 Download Dataset](https://docs.google.com/spreadsheets/d/1ANOYFKEC7GJtriPFpXc3PNvHanHP3oXJ/edit?usp=share_link&ouid=111569509993326914459&rtpof=true&sd=true)

This dataset contains financial data from companies traded on the NYSE, NASDAQ, NYSE American, and other markets. Here, the variable "misstate" is the fraud label (1 denotes fraud, and 0 denotes non-fraud).

### The description of the 28 raw accounting variables are as follows:

- **act**: Current Assets, Total  
- **ap**: Account Payable, Trade  
- **at**: Assets, Total  
- **ceq**: Common/Ordinary Equity, Total  
- **che**: Cash and Short-Term Investments  
- **cogs**: Cost of Goods Sold  
- **csho**: Common Shares Outstanding  
- **dlc**: Debt in Current Liabilities, Total  
- **dltis**: Long-Term Debt Issuance  
- **dltt**: Long-Term Debt, Total  
- **dp**: Depreciation and Amortization  
- **ib**: Income Before Extraordinary Items  
- **invt**: Inventories, Total  
- **ivao**: Investment and Advances, Other  
- **ivst**: Short-Term Investments, Total  
- **lct**: Current Liabilities, Total  
- **lt**: Liabilities, Total  
- **ni**: Net Income (Loss)  
- **ppegt**: Property, Plant and Equipment, Total  
- **pstk**: Preferred/Preference Stock (Capital), Total  
- **re**: Retained Earnings  
- **rect**: Receivables, Total  
- **sale**: Sales/Turnover (Net)  
- **sstk**: Sale of Common and Preferred Stock  
- **txp**: Income Taxes Payable  
- **txt**: Income Taxes, Total  
- **xint**: Interest and Related Expense, Total  
- **prcc_f**: Price Close, Annual, Fiscal  

### The description of the 14 financial ratio variables are as follows:

- **dch_wc**: WC accruals  
- **ch_rsst**: RSST accruals  
- **dch_rec**: Change in receivables  
- **dch_inv**: Change in inventory  
- **soft_assset**: % Soft assets  
- **dpi**: Depreciation index  
- **ch_cs**: Change in cash sales  
- **ch_cm**: Change in cash margin  
- **ch_roa**: Change in return on assets  
- **ch_fcf**: Change in free cash flows  
- **reoa**: Retained earnings over total assets  
- **EBIT**: Earnings before interest and taxes over total assets  
- **issue**: Actual issuance  
- **bm**: Book-to-market

### 1.Exploratory data analysis:

![Histogram](Plots/Fraudes.jpg)
![Table_I](Plots/Table.jpg)

### 2.Model:
Implementation of logistic regression for fraud detection. First, we will train a Logistic Regression model with data from 1990-2011 and evaluate it with data from 2011-2014.

![PLot_I](Plots/Regression_logistic_1.jpg)

An AUC of 0.5408 indicates that the model has poor performance, as an AUC value close to 0.5 suggests that the model is no better than a random model at classifying the observations.

![PLot_II](Plots/Regression_logistic_2.jpg)

Using logistic regression, we did not achieve optimal results. However, there are other models for fraud prediction, such as RUSBoost and LightGBM, which tend to perform better in handling imbalanced datasets and capturing more complex patterns in the data.

![PLot_III](Plots/Comparison_ROC_LGBM.jpg)


