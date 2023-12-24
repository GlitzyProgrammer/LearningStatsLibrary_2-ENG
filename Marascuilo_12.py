import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import chisquare
from scipy.stats import chi2
pd.set_option('display.float_format', lambda x: '%.6f' % x)
print("\n")
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 12\Practice Portfolio 12_data.xlsx",sheet_name="Practice Portfolio 12_data")
df["Christmas Season"] = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 12\Practice Portfolio 12_data.xlsx",usecols=["Christmas Season"])
df["Winter attitudes"]= pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 12\Practice Portfolio 12_data.xlsx",usecols=["Winter attitudes"])


cross_tab = pd.crosstab(df['Winter attitudes'], df['Christmas Season'], margins=True, margins_name='Grand Total')


print("Cross tab of Winter attitudes and Chrismas Season takes")
print(cross_tab)

observed_contingency = pd.crosstab(df['Winter attitudes'], df['Christmas Season'], margins=True, margins_name='Grand Total')
observed_chisquare_use = pd.crosstab(df['Winter attitudes'], df['Christmas Season'])

# Calculate the expected contingency table
row_totals = observed_contingency.iloc[:-1, -1]
col_totals = observed_contingency.iloc[-1, :-1]
grand_total = observed_contingency.iloc[-1, -1]

expected_contingency = np.outer(row_totals, col_totals) / grand_total
# Create a DataFrame for the expected contingency table
expected_contingency_df = pd.DataFrame(expected_contingency, index=row_totals.index, columns=col_totals.index)

# Display the observed and expected contingency tables

print("\nExpected Contingency Table:")
print(expected_contingency_df)

L1 = ['Hate','Love','Meh']
L2 = ['After','Before']
manual_chisquare = {
    "Winter Attitudes":[],
    "Christmas Season Op":[],
     "Observed Frequency, f_ij":[],
     "Expected Frequencey, e_ij":[],
      "f_ij-e_ij":[],
      "(f_ij-e_ij)^2":[],
      "(f_ij-e_ij)^2/e_ij":[]

}

for i in L1:
    for j in L2:
        manual_chisquare['Winter Attitudes'].append(i)
        manual_chisquare['Christmas Season Op'].append(j)
        cell_value_eij = expected_contingency_df.loc[i,j]
        cell_value_fij = observed_contingency.loc[i,j]
        manual_chisquare["Expected Frequencey, e_ij"].append(cell_value_eij) 
        manual_chisquare["Observed Frequency, f_ij"].append(cell_value_fij) 
        fijsubeij = cell_value_fij- cell_value_eij
        manual_chisquare['f_ij-e_ij'].append(fijsubeij)
        fsubesqr = fijsubeij ** 2 
        manual_chisquare['(f_ij-e_ij)^2'].append(fsubesqr)
        fsubesqr_eij = fsubesqr/ cell_value_eij
        manual_chisquare['(f_ij-e_ij)^2/e_ij'].append(fsubesqr_eij) 


man_chi = pd.DataFrame(manual_chisquare)
print("This is the table used for Manual Chi Caluations\n")
print(man_chi)

chi_square_value = 0 
total_expected = 0
total_observed = 0 
for i in manual_chisquare["(f_ij-e_ij)^2/e_ij"]:
    chi_square_value+= i

for i in manual_chisquare["Expected Frequencey, e_ij"]:
    total_expected += i
print("Total Expeceted", math.ceil(total_expected))
for i in manual_chisquare['Observed Frequency, f_ij']:
    total_observed += i
print("Total Observed", total_observed)

print("\n")

chi_square_value = round(chi_square_value,7 )
print("This is the manually calulated chi_square for 7 signifcant digits",chi_square_value,"\n")
chi2_stat, p_value = chisquare(f_obs=observed_chisquare_use, f_exp=expected_contingency_df)
print("Chi squre value using python and p value")
p_value = 1 - chi2.cdf(chi_square_value, 2)  #2 represents the number of categories - 1
auto_chi = chi2_stat[0]+chi2_stat[1]
auto_chi = round(auto_chi,7)
print(f"This is what I got using the keeping 7 signifcant digits scipy.stats package {auto_chi}\nThis is what I got form manually calulating the chi-value {chi_square_value}")
print("This is the p value I got from using the scipy stats package", p_value,"\n")    


        


print("Performing the Marascuilo Multiple Comparisons Procedure\n")

p1_bar = observed_contingency.loc['Love','After']/ observed_contingency.loc['Love','Grand Total']

print(f"This is the p-bar for people that love winter and think that christmas season should start after Thanksgiving:\n{p1_bar}")

p1_bar = observed_contingency.loc['Love','After']/ observed_contingency.loc['Love','Grand Total']
p2_bar = observed_contingency.loc['Meh','After']/ observed_contingency.loc['Meh','Grand Total']

print(f"This is the p-bar for people that are meh about winter and think that christmas season should start after Thanksgiving:\n{p2_bar}")

p3_bar = observed_contingency.loc['Hate','After']/ observed_contingency.loc['Hate','Grand Total']

print(f"This is the p-bar for people that hate winter and think that christmas season should start after Thanksgiving:\n{p3_bar}")

print("\n")
pair_wise= {
    "ABS |pi_bar-pj_bar|": [],
     "CV_ij":[],
     "Null Hypothesis":[np.nan,np.nan,np.nan]
}
critical_value = round(chi2.ppf(0.95, 2),9)   #use for CV_ij
print(f"Using 0.95 as the probability and setting degrees of freedom to 2 to calulate the critical value we find it to be\n{critical_value} with in 7 signficant digits")
print("\n")
print("Pairwise comparision table")
p1_p2 =  abs(p1_bar-p2_bar)
p1_p3 =  abs(p1_bar-p3_bar)
p2_p3 =  abs(p2_bar-p3_bar)
pair_wise["ABS |pi_bar-pj_bar|"].append(abs(p1_p2))
pair_wise["ABS |pi_bar-pj_bar|"].append(abs(p1_p3))
pair_wise["ABS |pi_bar-pj_bar|"].append(abs(p2_p3))
 
# this takes care of the  p_ibar(1-p_ibar)/n_i for each of the  p_bars found before
p1 = round((p1_bar*(1-p1_bar))/ observed_contingency.loc["Love","Grand Total"],7)
p2 = round((p2_bar*(1-p2_bar))/ observed_contingency.loc["Meh","Grand Total"],7)
p3 = round((p3_bar*(1-p3_bar))/ observed_contingency.loc["Hate","Grand Total"],7)


# this combines values gathered from the previous step in a pairwise fashion
p1vp2 = round(p1+p2,7)
p1vp3 = round(p1+p3,7)
p2vp3 = round(p2+p3,7)


# The Critial Values for each of the pair-wise comparisions 
Cv_1 =  round(math.sqrt(critical_value) * math.sqrt(p1vp2),9)
Cv_2 =  round(math.sqrt(critical_value) * math.sqrt(p1vp3),9)
Cv_3 = round(math.sqrt(critical_value) * math.sqrt(p2vp3),9)
pair_wise["CV_ij"].append(Cv_1)
pair_wise["CV_ij"].append(Cv_2)
pair_wise["CV_ij"].append(Cv_3)



list_label =["p1_bar vs p2_bar","p1_bar vs p3_bar","p2_bar vs p2_bar"]
p_wise = pd.DataFrame(pair_wise)
p_wise = p_wise.set_index(pd.Index(list_label))
p_wise.index.name = 'Pairwise Comparision'
print(p_wise)



#This algorithm is testing wither to reject the Null Hypothesis       
count = 0 
for index, row in p_wise.iterrows():
    if row["ABS |pi_bar-pj_bar|"] > row["CV_ij"]:
        p_wise.at[index, "Null Hypothesis"] = 'Reject Null'
    else:
        p_wise.at[index, "Null Hypothesis"] = 'Do not Reject Null'
print("\n\n") 
print(p_wise)
print("A chi-square test of independence was performed to examine the relation between students' belief that Christmas starts after Thanksgiving and their attitudes toward the Winter season.")
print(f"The relation between these variables was significant,  (2,N = {total_observed}) = {round(chi_square_value,2)} p = {round(p_value,3)}")
print("The Marascuilo Pairwise Comparison Procedure showed that students who are 'meh' about winter are more likely to think Christmas starts after Thanksgiving compared to students who are love winter.")