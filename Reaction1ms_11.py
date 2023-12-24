import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.stats import levene
from scipy.stats import f 
from matplotlib.patches import Patch

pd.set_option('display.float_format', '{:.7f}'.format)
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",sheet_name="Practice Portfolio 10 data")

df['Reaction time_1 (ms)']=  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=["Reaction time_1 (ms)"])
df['Reaction time_1_Submitted Before (ms)'] =  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Reaction time_1_Submitted Before (ms)'])
df['Reaction time_1_Submitted After (ms)'] = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Reaction time_1_Submitted After (ms)'])


x_time_1_before = df['Reaction time_1_Submitted Before (ms)'].dropna()
y_time_1_after = df['Reaction time_1_Submitted After (ms)'].dropna()


description_reaction_1 = df['Reaction time_1 (ms)'].describe().dropna()
description_reaction_2= df["Reaction time_1_Submitted After (ms)"].describe().dropna()


print("\n")
#Right and Left tail represents the  Confidence Interval
std_deviation = description_reaction_1['std']   
after_mean = round(np.mean(y_time_1_after),6)
before_mean = round(np.mean(x_time_1_before),6)

left_chi_value = chi2.ppf(0.975, 24)
converserion_left_value = (24* pow(std_deviation,2)) / left_chi_value
left_tail = round(np.sqrt(converserion_left_value),8)


right_chi_value = chi2.ppf(0.025, 24)
converserion_right_value = (24* pow(std_deviation,2)) / right_chi_value
right_tail = round(np.sqrt(converserion_right_value),8)

print(right_tail)

print(left_tail)

statistic, p_value = levene(x_time_1_before , y_time_1_after)

alpha = 0.1
if p_value < alpha / 2:
    print("The variances are significantly different (lower tail).")
else:
    print("There is no significant difference in variances (lower tail).")


print("H_0 The variance in  reaction time 2 for those who submitted before is less than or equal to  those who submiited after")
print("H_a The variance in reaction time 1 for those who submitted before is more than those who submitted after")

variance1 = round(np.var(x_time_1_before, ddof=1),6)  # ddof=1 for sample variance
variance2 = round(np.var(y_time_1_after, ddof=1),6)

#the F-statistic
f_statistic = variance1 / variance2

# Degrees of freedom for the two samples
df1 = len(x_time_1_before) - 1
df2 = len(y_time_1_after) - 1
left_p_value = 1 - p_value

p_value = 2 * (1 - f.cdf(f_statistic, df1, df2))

# significance level
alpha =  0.10 


# Print results
print("F-Statistic:", f_statistic)
print("P-Value:", p_value)
print("Degrees of Freedom:", df1, df2)
p_value_right = f.cdf(f_statistic, df1, df2)
# Print the result
print("Right-tail p-value:", p_value_right)
f_critical = f.ppf(alpha, df1, df2)
print("Critical Value for F (one-tail):", f_critical)


p_value_right = f.cdf(f_statistic, df1, df2)
f_critical = f.ppf(alpha, df1, df2)
#creating table 
Table_Before = {"Mean":[],
                "Variance":[],
                "Observations":[],
                "df":[],
                "F":[],
                "P(F<=f) one-tail":[],
                "F Critical one-tail":[]}

Table_After = {"Mean":[],
               "Variance":[],
               "Observations":[],
               "df":[]}


Table_Before['Mean'].append(before_mean)
Table_Before['Variance'].append(variance1)
Table_Before["Observations"].append(df1+1)
Table_Before['df'].append(df1)
Table_Before["F"].append(f_statistic)
Table_Before['P(F<=f) one-tail'].append(p_value_right)
Table_Before['F Critical one-tail'].append(f_critical)
#moding table after
Table_After["Mean"].append(after_mean)
Table_After['Variance'].append(variance2)
Table_After['df'].append(df2)
Table_After['Observations'].append(df2+1)
Two_sample_var_1 = pd.DataFrame(Table_Before).transpose()
Two_sample_var_2 = pd.DataFrame(Table_After).transpose()
Two_Sam_Var =  pd.concat([Two_sample_var_1, Two_sample_var_2], ignore_index=True, axis=1)

print("F-Test Two-Sample for Variances\n")

Labels = ["Reaction time_1_Submitted Before (ms)",'Reaction time_1_Submitted After (ms)']
Two_Sam_Var.columns = Labels
print(Two_Sam_Var)

#Box plotting 
out=plt.boxplot([x_time_1_before, y_time_1_after],meanline=True,showfliers=True, flierprops=dict(markerfacecolor='pink', marker='o'),patch_artist=True)
target_box_color='orange'
out['boxes'][0].set_facecolor(target_box_color)
out['boxes'][1].set_facecolor('blue')

#Plots mean 
plt.plot( 1,before_mean, 'rx', markersize=10)
plt.plot(2, after_mean,'rx',markersize=10)

# Add vertical lines at each y-axis tick
y_ticks = plt.yticks()[0]
for y_tick in y_ticks:
    if y_tick < 0:
        continue
    else:
      plt.axhline(y=y_tick, color='black', linestyle='-', alpha=0.5)
legend_handles = [
    Patch(facecolor='orange', edgecolor='black',linewidth=1),
    Patch(facecolor='blue', edgecolor='black', linewidth=1),
]
 #Rejecting Null Hypothesis or Not
if p_value_right < alpha :
    print(f"Rejct null because P {round(p_value_right,4)} < alpha {alpha} is true ")
else:
    print(f"Don't rejct null because P {round(p_value_right,4)} < alpha {alpha} is false ")
plt.xticks([])
plt.legend(Labels)
plt.legend(legend_handles, Labels, handlelength=2, handleheight=2,bbox_to_anchor=(0.23, 1))
plt.ylabel('Reaction time_1 (ms)')
plt.show()
