import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# I didn't manually calualte much I based it all off the model.summary 
anova_regression= {
    "df":[],	
    "SS":[],
        "MS":[],	
        "F":[],	
        "Significance F":[]

}
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 13\Practice Portfolio 13 Data-3.xlsx",sheet_name="Regression")

df_subcolums = df[['Number of M&Ms in a bag, x_i', 'Number of Blues, y_i']]
x_axis = df['Number of M&Ms in a bag, x_i']
y_aix = df[ 'Number of Blues, y_i']

X = sm.add_constant(df['Number of M&Ms in a bag, x_i'])
model = sm.OLS(df['Number of Blues, y_i'], X).fit()

n = len(df)

conf_intervals = model.conf_int()

# Create a custom summary with 95% confidence intervals
custom_summary = pd.DataFrame({
    'Coefficient': model.params,
    'Lower 95.0%': conf_intervals[0],
    'Upper 95.0%': conf_intervals[1]
})

summary = model.summary()
# Extract both R and R^2
multiple_r = model.rsquared ** 0.5
multiple_r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj
residuals = model.resid
rse = np.sqrt(np.sum(residuals**2) / (len(df) - 2))  #residual standard error (RSE)
num_observations = model.nobs   #number of observations

Regression_Summary = {
    "Multiple R":[multiple_r],
"R Square":[multiple_r_squared],
"Adjusted R Square":[adjusted_r_squared],
"Standard Error":[rse],
"Number of Observations":[num_observations]
}
Regression_Summary_df = pd.DataFrame(Regression_Summary).transpose()
label_regression =  ["Regression Summary Table"]
Regression_Summary_df.columns = label_regression
print("Regression Stastisic Summary")
print(Regression_Summary_df)
print("\n\n")
y_mean = df['Number of Blues, y_i'].mean()
#Regression section

predicted_values = model.predict(X)
ssr = np.sum((predicted_values - y_mean)**2)
df_regression = len(model.params) - 1
msr = ssr / df_regression
f_statistic = model.fvalue
f_signficant = model.f_pvalue
anova_regression['df'].append(df_regression)
anova_regression['SS'].append(ssr)
anova_regression['MS'].append(msr)
anova_regression['F'].append(f_statistic)
anova_regression['Significance F'].append(f_signficant)
#Resduual section
sse = np.sum(residuals**2)
mse = sse / (len(df) - len(model.params))
df_residual = len(df) - len(model.params)
anova_regression['df'].append(df_residual)
anova_regression['SS'].append(sse)
anova_regression['MS'].append(mse)
anova_regression['F'].append(np.nan)
anova_regression['Significance F'].append(np.nan)


#Total sectiion
total_obs = df_regression+df_residual
sst = np.sum((df['Number of Blues, y_i'] - y_mean)**2)
anova_regression['df'].append(total_obs)
anova_regression['SS'].append(sst)
anova_regression['MS'].append(np.nan)
anova_regression['F'].append(np.nan)
anova_regression['Significance F'].append(np.nan)
label_anova = ["Regression","Residual","Total"]
anova_regression_df = pd.DataFrame(anova_regression,index=label_anova)
print("Anova Summary")
print(anova_regression_df)

intercept_table = model.summary2().tables[1]

intercept_table_df =pd.DataFrame(intercept_table)
coef_table = model.summary2().tables[1]

ci_table = model.conf_int().loc['const'].to_frame().rename_axis('95% Confidence Interval', axis=1)

coef_table = model.conf_int(alpha=0.05)
# Add the coefficient values to the table
coef_table['Coefficient'] = model.params
print("\n\n")
intercept_table['Lower 95.0%'] = [ci_table.loc[0][0],coef_table.iloc[1][0]]
intercept_table['Upper 95.0%'] = [ci_table.loc[1][0],coef_table.iloc[1][1]]
print("\n\n")
intercept_table.columns = ["Coefficient","Standard Error",	"t Stat",	"P-value",	"Lower 95%",	"Upper 95%",	"Lower 95.0%",	"Upper 95.0%"]
intercept_table.index = ["Intercept","Number of M&Ms in a bag, x_i"]
print("Table of Coefficient")
print(intercept_table)

print("\n\n")
plt.scatter(x_axis, y_aix, label='Data Points')

# Plot the regression line
plt.plot(x_axis, model.predict(X), color='red', label='Regression Line')
equation = f'Y = {model.params[0]:.2f} + {model.params[1]:.4f}X\nR-squared = {multiple_r_squared:.4f}'
plt.annotate(equation, xy=(0.75, .5), xycoords='axes fraction', ha='center', fontsize=10)

# Set labels and title
plt.xlabel('Number of M&Ms in a bag, x_i')
plt.ylabel( 'Number of Blues, y_i')
plt.show()