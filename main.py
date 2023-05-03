import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def plot_hourly_dependency(df,save_plots=False):
    hours = range(24)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), sharex=True, sharey=True)

    for i, hour in enumerate(hours):
        hourly_data = df[df['hour_of_day'] == hour]
        row = i // 6
        col = i % 6
        axes[row, col].scatter(hourly_data['SpotPriceEUR'], hourly_data['HourlySettledConsumption'], s=5, alpha=0.5)
        axes[row, col].set_title(f"Hour {hour}")

    plt.tight_layout()
    if save_plots:
        plt.savefig(f'hour_{hour}_demand_vs_price.pdf', format='pdf', bbox_inches='tight')
    else:
        plt.show()
        

#calculate the elasticity of the demand
df = df_joint_prices_and_demand
df_filtered = df[(df['SpotPriceEUR'] > 0) & (df['HourlySettledConsumption'] > 0)]

df_filtered['ln_price'] = np.log(df_filtered['SpotPriceEUR'])
df_filtered['ln_demand'] = np.log(df_filtered['HourlySettledConsumption'])

model = smf.ols('ln_demand ~ ln_price', data=df_filtered).fit()

elasticity = model.params['ln_price']

def hourly_price_elasticity(df_filtered):
    hourly_elasticities = []

    for hour in range(24):
        hourly_data = df_filtered[df_filtered['hour_of_day'] == hour]
        model = smf.ols('ln_demand ~ ln_price', data=hourly_data).fit()
        elasticity = model.params['ln_price']
        hourly_elasticities.append(elasticity)

    return hourly_elasticities

hourly_elasticities = hourly_price_elasticity(df_filtered)        

fig = plot_hourly_dependency(df_joint_prices_and_demand,save_plots='true')


df_joint_prices_and_demand = pd.read_csv('prices_and_demand_2022_gridcompany_31',delimiter=';',decimal=',')
