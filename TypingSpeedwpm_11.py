import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levene
from scipy.stats import f 
from matplotlib.patches import Patch

pd.set_option('display.float_format', '{:.7f}'.format)
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",sheet_name="Practice Portfolio 10 data")


df['Typing Speed (wpm)']=  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=["Typing Speed (wpm)"])
df['Typing Speed_High Quiz Scores (wpm)'] =  pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Typing Speed_High Quiz Scores (wpm)'])
df['Typing Speed_Low Quiz Scores (wpm)'] = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 11\Practice Portfolio 11 data.xlsx",usecols=['Typing Speed_Low Quiz Scores (wpm)'])

Wpm_Low = df['Typing Speed_Low Quiz Scores (wpm)'].dropna()
Wpm_High = df['Typing Speed_High Quiz Scores (wpm)'].dropna()

description_low = df['Typing Speed_Low Quiz Scores (wpm)'].describe().dropna()
description_high= df['Typing Speed_High Quiz Scores (wpm)'].describe().dropna()

std_deviation = description_low['std'] 
low_mean = round(np.mean(Wpm_Low),6)
high_mean = round(np.mean(Wpm_High),6)

# Degrees of freedom for the two samples
df1 = len(Wpm_Low) - 1
df2 = len(Wpm_High) - 1

print("H_0 The variance in typing speed is the same for those who have high vs low quiz scores")
print("H_a The variance in typing speed is different for those who have high vs low quiz scores")

statistic, p_value = levene(Wpm_Low , Wpm_High)

variance1 = round(np.var(Wpm_Low , ddof=1),6)  # ddof=1 for sample variance
variance2 = round(np.var(Wpm_High, ddof=1),6)
#the F-statistic
f_statistic = variance1 / variance2

alpha = 0.1
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
Table_Before['Mean'].append(low_mean)
Table_Before['Variance'].append(variance1)
Table_Before["Observations"].append(df1+1)
Table_Before['df'].append(df1)
Table_Before["F"].append(f_statistic)
Table_Before['P(F<=f) one-tail'].append(p_value_right)
Table_Before['F Critical one-tail'].append(f_critical)
#moding table after
Table_After["Mean"].append(high_mean)
Table_After['Variance'].append(variance2)
Table_After['df'].append(df2)
Table_After['Observations'].append(df2+1)
Two_sample_var_1 = pd.DataFrame(Table_Before).transpose()
Two_sample_var_2 = pd.DataFrame(Table_After).transpose()

Two_Sam_Var =  pd.concat([Two_sample_var_1, Two_sample_var_2], ignore_index=True, axis=1)
Labels = ['Typing Speed_Low Quiz Scores (wpm)','Typing Speed_High Quiz Scores (wpm)']
Two_Sam_Var.columns = Labels
print(Two_Sam_Var)

#Box Plotting 
out=plt.boxplot([Wpm_Low, Wpm_High],meanline=True,showfliers=True, flierprops=dict(markerfacecolor='pink', marker='o'),patch_artist=True)
target_box_color='orange'
out['boxes'][0].set_facecolor(target_box_color)
out['boxes'][1].set_facecolor('blue')
#Plots mean 
plt.plot( 1,low_mean, 'rx', markersize=10)
plt.plot(2, high_mean,'rx',markersize=10)

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
print("\n")
#Rejecting Null Hypothesis or Not
if p_value_right < alpha :
    print(f"Rejct null because P {round(p_value_right,4)} < alpha {alpha} is true ")
else:
    print(f"Don't reject null because P {round(p_value_right,4)} < alpha {alpha} is false ")
plt.xticks([])
plt.legend(Labels)
plt.legend(legend_handles, Labels, handlelength=2, handleheight=2,bbox_to_anchor=(0.23, 1))
plt.ylabel('Typing Speed (wpm)')
plt.show()
