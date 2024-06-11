# Grid-Scale Battery Energy Trader

Project aims to optimize charge and discharge of grid-scale battery storge leveraging data science techniques. Aka charge/buy when electricity prices are low (i.e. high proportion of renewables on grid and low demand) and discharge/sell electricity when prices are high (i.e. low proportion of renewables on grid and high demand).
The goal is to explore strategies for maximizing profits through optimal charging and discharging of the battery, based on historical and forecasted electricity prices. The project also involves data exploration and statistical comparison of different electricity market data sets.

## Optimization Strategies

The project examines two scenarios for optimizing the use of a battery storage system in the electricity market. 

### Scenario 1: Known Future Prices

- **Battery Specifications:** 1MW power, 1MWh capacity.
- **Objective:** Determine the optimal times to charge and discharge the battery each day to maximize revenue.
- **Data Used:** Hourly wholesale electricity prices for the first 6 months of 2022.
- **Methodology:** Profit maximization by analyzing price patterns and identifying the best times to buy (charge) and sell (discharge) electricity. This involves using heuristics and optimization techniques.
- **Extension:** Analysis is extended to a 1MW, 2MWh battery to understand how increased capacity affects the charging/discharging strategy.

### Scenario 2: Unknown Future Prices

- **Objective:** Develop a strategy for charging and discharging the battery without prior knowledge of future prices.
- **Data Used:** Historical price data.
- **Methodology:** Create a predictive model using past price data to forecast future prices and inform trading decisions. The strategy is trained and tested on historical data to evaluate its effectiveness compared to the scenario with known future prices.

## Data Exploration

The project includes a thorough exploration and comparison of two data sets: hourly and 15-minute electricity prices for the first 6 months of 2022. 

- **Data Sets:** 
  - 60-minute data: Prices for hourly contracts.
  - 15-minute data: Prices for 15-minute contracts.
- **Objective:** Summarize and compare the statistical properties of both data sets to identify patterns and differences.
- **Analysis:** Identify which type of contract (hourly or 15-minute) might be more beneficial for trading, considering patterns and market behavior.

## Key Findings

- **Optimal Trading Times:** Identification of daily patterns for charging and discharging the battery to maximize revenue, including the impact of battery capacity on these strategies.
- **Predictive Modeling:** Development of a forecasting model to inform trading decisions without future price knowledge, highlighting the challenges and potential strategies for real-time trading.
- **Data Insights:** Statistical comparison of different contract durations, providing recommendations on which contract type might be more advantageous for trading in various market conditions.

## Setup

To install the conda environment first run:

```conda env create -f environment.yml```


Then activate the environment with:

```conda activate battery_energy_trader```
