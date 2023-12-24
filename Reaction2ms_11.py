import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.stats import levene
from scipy.stats import f 
from matplotlib.patches import Patch
pd.set_option('display.float_format', '{:.7f}'.format)
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",sheet_name="Practice Portfolio 10 data")
'''The variance Reaction Time 2'''

df['Reaction time_2 (ms)']=  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=["Typing Speed (wpm)"])
df['Reaction time_2_Submitted Before (ms)'] =  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Reaction time_2_Submitted Before (ms)'])
df['Reaction time_2_Submitted After (ms)'] = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Reaction time_2_Submitted After (ms)'])

time_2_before = df['Reaction time_2_Submitted Before (ms)'].dropna()
time_2_after = df['Reaction time_2_Submitted After (ms)'].dropna()

description_reaction_1 = df['Reaction time_2 (ms)'].describe().dropna()
description_reaction_2= df["Reaction time_2_Submitted After (ms)"].describe().dropna()
std_deviation = description_reaction_1['std']  
after_mean = round(np.mean(time_2_after),6)
before_mean = round(np.mean(time_2_before),6)
df1 = len(time_2_before) - 1
df2 = len(time_2_after) - 1
print("H_0 The variance in reaction time 2 for those who submitted before is less than or equal to those who sumbited after")
print("H_a The variance in reaction time 2 for those who submitted before is more than those who sumitted after")
statistic, p_value = levene(time_2_before , time_2_after)
variance1 = round(np.var(time_2_before, ddof=1),6)  # ddof=1 for sample variance
variance2 = round(np.var(time_2_after, ddof=1),6)
f_statistic = variance1 / variance2

alpha = 0.1
p_value_right = f.cdf(f_statistic, df1, df2)
f_critical = f.ppf(alpha, df1, df2)

Table_Before = {"Mean":[],"Variance":[],"Observations":[],"df":[],"F":[],"P(F<=f) one-tail":[],"F Critical one-tail":[]}
Table_After = {"Mean":[],"Variance":[],"Observations":[],"df":[]}
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
#Two_sample_var['Reaction time_1_Submitted After (ms)'] = Table_After

print("F-Test Two-Sample for Variances\n")
Two_Sam_Var =  pd.concat([Two_sample_var_1, Two_sample_var_2], ignore_index=True, axis=1)
Labels = ['Reaction time_2_Submitted Before (ms)','Reaction time_2_Submitted After (ms)']
Two_Sam_Var.columns = Labels
print(Two_Sam_Var)
out=plt.boxplot([time_2_before, time_2_after],meanline=True,showfliers=True, flierprops=dict(markerfacecolor='pink', marker='o'),patch_artist=True)
target_box_color='orange'
out['boxes'][0].set_facecolor(target_box_color)
out['boxes'][1].set_facecolor('blue')
plt.plot( 1,before_mean, 'rx', markersize=10)
plt.plot(2, after_mean,'rx',markersize=10)
y_ticks = plt.yticks()[0]
# Add vertical lines at each y-axis tick
for y_tick in y_ticks:
    if y_tick < 0:
        continue
    else:
      plt.axhline(y=y_tick, color='black', linestyle='-', alpha=0.5)
legend_handles = [
    Patch(facecolor='orange', edgecolor='black',linewidth=1),
    Patch(facecolor='blue', edgecolor='black', linewidth=1),
]
print("\n")
if p_value_right < alpha :
   print(f"Rejct null because P {round(p_value_right,4)} < alpha {alpha} is true ")
else:
    print(f"Don't rejct null because P {round(p_value_right,4)} < alpha {alpha} is false ")
plt.xticks([])
plt.legend(Labels)
plt.legend(legend_handles, Labels, handlelength=2, handleheight=2,bbox_to_anchor=(0.23, 1))
plt.ylabel('Reaction time_2 (ms)')
plt.show()