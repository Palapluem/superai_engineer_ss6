import json
import re

old_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-lv2-nb-expresson-hackathon (4).ipynb"
split_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-MASTER-SPLIT.ipynb"
pure_nb_path = r"Level 2\Hackathon 5_Demand Forecasting Coffee Chain Hackathon\hackathon-5-EDA-PURE-MASTER.ipynb"

# 1. Detailed Descriptions for Original Charts
descriptions = {
    "Target Distribution": {
        "desc": "A set of histograms displaying the frequency distribution of the primary target variables, particularly the daily units sold.",
        "insight": "Understanding the skewness of the target variable is critical. If 'units_sold' is heavily right-skewed with a long tail of high-volume outliers, traditional mean-squared-error (MSE) models will over-predict low-volume days. A log-transformation objective may be required."
    },
    "Full Time Series Trend": {
        "desc": "A continuous line plot charting the aggregate daily units sold across the entire historical data window.",
        "insight": "Exposes the macro trend of the business. If the overall trajectory is flat but features high volatility spikes, the model must rely heavily on external covariates (holidays, promos) rather than autoregressive baseline drift."
    },
    "Store-level Heatmap": {
        "desc": "A matrix heatmap visualizing the density of sales across different store locations and product categories.",
        "insight": "Highlights structural discrepancies between branches. Stores that show zero volume for specific categories might have physical constraints (e.g., no bakery oven). This dictates that global models must allow for zero-prediction bounds on specific Store-Category combinations."
    },
    "Calendar Effects": {
        "desc": "Boxplots and bar charts aggregating sales by day of the week, month, and quarters to isolate calendar-driven seasonality.",
        "insight": "Human consumption of coffee is strictly habitual. Identifying which day of the week represents the global peak allows the model to assign high-weight autoregressive lags to capture weekly repeating patterns."
    },
    "Payday Effect by Category": {
        "desc": "A comparative visualization of sales volume specifically on designated Paydays versus regular days, segmented by product category.",
        "insight": "Payday injects discretionary income into the consumer base. If 'Merchandise' (high ticket items) spikes exclusively on paydays while 'Coffee' remains stable, the forecasting model must trigger strong multiplier effects for non-essential categories precisely on these dates."
    },
    "Rainy Season × Category Interaction": {
        "desc": "An interaction plot demonstrating how the Rainy Season alters the volume of sales across different product categories.",
        "insight": "Weather acts as a substitute driver. While cold/rainy weather might depress total foot traffic, it dramatically shifts the product mix towards 'Hot Beverages'. The model needs this interaction term to avoid under-forecasting hot drinks during monsoons."
    },
    "Operating Hours Effect": {
        "desc": "An analysis correlating a store's total daily sales capacity against its designated operating hours (e.g., 24-hour stores vs. standard 8-to-5).",
        "insight": "Stores with extended operating hours inherently possess a higher throughput ceiling. If operating hours change over time, the historical baseline must be normalized to prevent the model from misinterpreting a revenue jump as structural demand growth."
    },
    "Promotion Analysis": {
        "desc": "An evaluation of baseline sales versus sales during active promotional campaigns, detailing the aggregate lift generated.",
        "insight": "Quantifies the elasticity of demand. If promotions generate massive spikes but are followed by immediate 'post-promo dips' (pantry-loading behavior), the model must include a 'days_since_promo' feature to correctly forecast the subsequent sales depression."
    },
    "Promo Cannibalization": {
        "desc": "A cross-category analysis examining whether discounting one product line negatively impacts the sales of an adjacent, non-discounted product line.",
        "insight": "Crucial for multi-variate forecasting. If discounting Coffee causes a drop in normal-priced Tea sales, the model must map these cross-elastic dependencies to avoid overestimating total store revenue during heavy promo periods."
    },
    "Local Events Analysis": {
        "desc": "A statistical review of sales volume on days when localized events (e.g., concerts, festivals) are taking place near the store.",
        "insight": "Local events act as powerful, isolated demand shocks. Since these do not follow a standard cyclical pattern, they must be fed into the model as explicit binary flags to prevent the algorithm from confusing an event spike with permanent baseline growth."
    },
    "Event Type × Category Cross": {
        "desc": "A granular breakdown showing which specific types of events boost which specific product categories.",
        "insight": "Different crowds exhibit different purchasing behaviors. A sporting event might drive extreme volumes of iced beverages, while an art exhibition might drive premium bakery items. Category-level forecasting requires interacting the event type with the specific SKU."
    },
    "Customer Analysis": {
        "desc": "An overview of customer demographics, comparing the purchasing frequency and basket sizes of registered members against anonymous walk-ins.",
        "insight": "Members provide the predictable 'floor' of daily revenue. If the percentage of sales from members is rising, the overall volatility of the time-series decreases, allowing for tighter prediction intervals in future forecasting horizons."
    },
    "Weekend Effect by Neighborhood": {
        "desc": "A geographic interaction plot contrasting weekend sales performance across different neighborhood types (e.g., Office vs. Tourist).",
        "insight": "Prevents catastrophic generalization. Applying a universal 'Weekend' multiplier destroys accuracy because Office locations die on weekends while Tourist locations thrive. The model absolutely requires the 'is_weekend * neighborhood_type' interaction feature."
    },
    "Autocorrelation": {
        "desc": "A correlogram mapping the statistical correlation between today's sales and sales from previous days (lags).",
        "insight": "Mathematically proves the presence of cyclical patterns. Strong correlations at Lag 7 and Lag 14 dictate that the most powerful predictive features for any tree-based model will be the exact sales volume from one and two weeks prior."
    },
    "Seasonal & Limited Edition Products": {
        "desc": "A performance lifecycle chart for products designated as seasonal or limited-edition.",
        "insight": "Seasonal items defy standard autoregressive patterns; they launch with a massive spike and exhibit exponential decay. Forecasting these items requires unique features such as 'days_since_launch' rather than relying on historical moving averages."
    },
    "School Break Effect": {
        "desc": "An assessment of demand shifts during official school holiday periods.",
        "insight": "School breaks alter the geographic distribution of demand. Stores near universities will see structural crashes, while residential or mall stores will see sustained elevated baselines. This requires a 'School_Break * Store_Location' interaction."
    },
    "Feature Correlation Matrix": {
        "desc": "A numerical heatmap displaying the Pearson correlation coefficients between all engineered features and the target variable.",
        "insight": "Identifies multicollinearity and feature importance. Features with the highest absolute correlation to 'units_sold' should be prioritized for further polynomial expansion."
    }
}

def format_md(title, desc, insight):
    return {"cell_type": "markdown", "metadata": {}, "source": [f"### Original Chart: {title}\n\n", f"**Description:**\n{desc}\n\n", f"**Actionable Insight:**\n{insight}"]}

def clean_markdown(cell):
    src = cell.get('source', [])
    new_src = [re.sub(r'[\U00010000-\U0010ffff]', '', line) for line in src]
    cell['source'] = new_src
    return cell

print("Loading Original Notebook...")
with open(old_nb_path, 'r', encoding='utf-8') as f:
    old_nb = json.load(f)

pure_cells = []

# Title
pure_cells.append({
    "cell_type": "markdown", "metadata": {}, 
    "source": ["# 100% PURE EDA NOTEBOOK\n", "This notebook contains exclusively Exploratory Data Analysis. All modeling, feature engineering, and inference code has been completely removed to provide a clean analytical environment. Explanations follow a strict, professional standard."]
})

print("Extracting and standardizing Original Charts...")
for cell in old_nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = "".join(cell.get('source', []))
        if "AutoGluon TimeSeries Forecasting Pipeline" in src:
            print("Reached Modeling Section. Halting extraction.")
            break
        pure_cells.append(clean_markdown(cell))
    elif cell['cell_type'] == 'code':
        src_lines = cell.get('source', [])
        if len(src_lines) > 0 and src_lines[0].startswith("# CELL ") and "[CHART" in src_lines[0] or "[NEW CHART" in src_lines[0]:
            # Try to match the chart title
            title_line = src_lines[0]
            for key in descriptions.keys():
                if key in title_line:
                    pure_cells.append(format_md(key, descriptions[key]["desc"], descriptions[key]["insight"]))
                    break
        pure_cells.append(cell)

print("Loading New Advanced Charts...")
with open(split_nb_path, 'r', encoding='utf-8') as f:
    split_nb = json.load(f)

# Find where the advanced charts start
advanced_cells = []
capture = False
for cell in split_nb['cells']:
    if cell['cell_type'] == 'markdown':
        src = "".join(cell.get('source', []))
        if "## 1. Member vs. Non-Member" in src:
            capture = True
    if capture:
        advanced_cells.append(cell)

pure_cells.extend(advanced_cells)

final_nb = {
    "cells": pure_cells,
    "metadata": old_nb.get("metadata", {}),
    "nbformat": old_nb.get("nbformat", 4),
    "nbformat_minor": old_nb.get("nbformat_minor", 5)
}

with open(pure_nb_path, 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Pure EDA Notebook created at: {pure_nb_path}")
