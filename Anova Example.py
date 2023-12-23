
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmap styling
import numpy as np
from scipy.stats import t
import numpy as np
from scipy.stats import f_oneway
from sklearn.metrics import mean_squared_error
import itertools
from scipy.stats import f



def pairwise_caluation_num(Num_Reg,Num_Peanut,Num_PB,desired_alpha,
RegularvsPeanut,RegularvsPB,PeanutvsPB):
     Number_array = [Num_Reg, Num_Peanut,Num_PB]
     g1_mean = np.mean(Num_Reg)
     sse_group1 = np.sum((Num_Reg - g1_mean)**2)
     g2_mean = np.mean(Num_Peanut)
     sse_group2 = np.sum((Num_Peanut - g2_mean)**2)
     g3_mean = np.mean(Num_PB)
     sse_group3 = np.sum((Num_PB - g3_mean)**2)
     sse = sse_group1+sse_group2+sse_group3
     mse = sse/dfw
     Bonferroni_Correction = desired_alpha/3  
     count = 0 
     for pair in itertools.combinations(Number_array, 2): 
             element1, element2 = pair 
             # num does the caluation for the mean difference between arrays 
             num  = round(element1.mean()- element2.mean(),2)  #rounding for the sake of getting consistant results takes 
             #Takes the square root of the MSE* 1/n + 1/n1 which  n represents the observed sample size 
             denom = round(np.sqrt(mse*((1/25)+(1/25))),9)#rounding to get consistant results of excel sheet 
             t_value = round(num/denom,9)     
             p_value = 2 * (1 - t.cdf(abs(t_value), dfw)) #two-tailed Student's t-distributio
             if count == 0:
              count+=1
              RegularvsPeanut["numerator"].append(num)
              RegularvsPeanut["denominator"].append(denom)
              RegularvsPeanut["t_value"].append(t_value)
              RegularvsPeanut["p-value"].append(p_value)
              if p_value < Bonferroni_Correction:           #check p value against the bonferroni_correction  to see signifances
                  RegularvsPeanut["Yes/No"].append("Yes")
                  continue
              else:
                  RegularvsPeanut["Yes/No"].append("No")
                  continue

             if count ==1: 
                count+=1
                RegularvsPB["numerator"].append(num)
                RegularvsPB["denominator"].append(denom)
                RegularvsPB["t_value"].append(t_value)
                RegularvsPB["p-value"].append(p_value)
                if p_value < Bonferroni_Correction:   #check p value against the bonferroni_correction  to see signifances
                  RegularvsPB["Yes/No"].append("Yes")  
                  continue
                else:
                  RegularvsPB["Yes/No"].append("No")
                  continue
             if count == 2:
                count+=1
                PeanutvsPB["numerator"].append(num)
                PeanutvsPB["denominator"].append(denom)
                PeanutvsPB["t_value"].append(t_value)
                PeanutvsPB["p-value"].append(p_value)
                if p_value < Bonferroni_Correction:     #check p value against the bonferroni_correction  to see signifances
                  PeanutvsPB["Yes/No"].append("Yes")
                  continue
                else:
                  PeanutvsPB["Yes/No"].append("No")
                  continue
    
    
def pairwise_caluation_blue(Blue_Reg,Blue_Peanuts,Blue_PB,desired_alpha,
BlueRegularvsPeanut,BlueRegularvsPB,BluePeanutvsPB):
     Number_array = [Blue_Reg, Blue_Peanuts,Blue_PB]
     g1_mean = np.mean(Blue_Reg)
     sse_group1 = np.sum((Blue_Reg - g1_mean)**2)
     g2_mean = np.mean(Blue_Peanuts)
     sse_group2 = np.sum((Blue_Peanuts- g2_mean)**2)
     g3_mean = np.mean(Blue_PB)
     sse_group3 = np.sum((Blue_PB - g3_mean)**2)
     sse = sse_group1+sse_group2+sse_group3
     mse = sse/dfw
     Bonferroni_Correction = desired_alpha/3  
     count = 0 
     for pair in itertools.combinations(Number_array, 2): 
             element1, element2 = pair 
             # num does the caluation for the mean difference between arrays 
             num  = round(element1.mean()- element2.mean(),2)  #rounding for the sake of getting consistant results takes 
             #Takes the square root of the MSE* 1/n + 1/n1 which  n represents the observed sample size 
             denom = round(np.sqrt(mse*((1/25)+(1/25))),9)#rounding to get consistant results of excel sheet 
             t_value = round(num/denom,9)     
             p_value = 2 * (1 - t.cdf(abs(t_value), dfw)) #two-tailed Student's t-distributio
             if count == 0:
              count+=1
              BlueRegularvsPeanut["numerator"].append(num)
              BlueRegularvsPeanut["denominator"].append(denom)
              BlueRegularvsPeanut["t_value"].append(t_value)
              BlueRegularvsPeanut["p-value"].append(p_value)
              if p_value < Bonferroni_Correction:           #check p value against the bonferroni_correction  to see signifances
                  BlueRegularvsPeanut["Yes/No"].append("Yes")
                  continue
              else:
                  BlueRegularvsPeanut["Yes/No"].append("No")
                  continue

             if count ==1: 
                count+=1
                BlueRegularvsPB["numerator"].append(num)
                BlueRegularvsPB["denominator"].append(denom)
                BlueRegularvsPB["t_value"].append(t_value)
                BlueRegularvsPB["p-value"].append(p_value)
                if p_value < Bonferroni_Correction:   #check p value against the bonferroni_correction  to see signifances
                  BlueRegularvsPB["Yes/No"].append("Yes")  
                  continue
                else:
                  BlueRegularvsPB["Yes/No"].append("No")
                  continue
             if count == 2:
                count+=1
                BluePeanutvsPB["numerator"].append(num)
                BluePeanutvsPB["denominator"].append(denom)
                BluePeanutvsPB["t_value"].append(t_value)
                BluePeanutvsPB["p-value"].append(p_value)
                if p_value < Bonferroni_Correction:     #check p value against the bonferroni_correction  to see signifances
                  BluePeanutvsPB["Yes/No"].append("Yes")
                  continue
                else:
                  BluePeanutvsPB["Yes/No"].append("No")
                  continue
#print("Orginal Data Frame \n")
df = pd.read_excel(r"C:\Users\ljwil\Desktop\Intro STATS\Project Stats 2\Chapter 13\Practice Portfolio 13 Data-3.xlsx",sheet_name="ANOVA",header=1)
# Setting header to 1 as keeping it as it was made itterating impossible
# Also seeting appart the two tables in the dataframe into 2 other data frames will make programing this easier
#print(df)
#print("\n")
#print("Data Fame 'Split' into two different ones\n")
Number_of_MnMs = {
    "Regular": [],
    "Peanut": [],
    "PB": []
}
Blue_MnMs = {
    "Regular": [],
    "Peanut": [],
    "PB": []
}
BlueRegularvsPeanut = {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []

}
BlueRegularvsPB = {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []

}
BluePeanutvsPB = {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []
}
RegularvsPeanut = {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []

}
RegularvsPB =  {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []

}
PeanutvsPB =  {
     "numerator":[],
     "denominator":[],
     "t_value": [],
     "p-value": [],
     "Yes/No": []

}


for index, row in df.iterrows():
    Number_of_MnMs["Regular"].append(row["Regular"])
    Number_of_MnMs["Peanut"].append(row['Peanut'])
    Number_of_MnMs["PB"].append(row['PB'])
    Blue_MnMs["Regular"].append(row["Regular.1"])
    Blue_MnMs["Peanut"].append(row["Peanut.1"])
    Blue_MnMs["PB"].append(row['PB.1'])


Number_Anova = pd.DataFrame(Number_of_MnMs)
Blue_Anova =  pd.DataFrame(Blue_MnMs)

#print("Number of MnMs\n")
#print(Number_Anova)
#print("\n")
#print("Blue MnMs Data\n")
#print(Blue_Anova)

Num_Reg = Number_Anova.loc[:, 'Regular']
Num_Reg_dev = np.std(Num_Reg)           #caluating standard deviation 
Num_Reg_var = np.var(Num_Reg)           #caluating variance
Num_Reg_mean = np.mean(Num_Reg)
Num_Reg_sum = np.sum(Num_Reg)
Num_Reg_count = len(Num_Reg)

Num_Peanut = Number_Anova.loc[:,'Peanut']
Num_Peanut_dev = np.std(Num_Peanut)    #caluating standard deviation 
Num_Peanut_var = np.var(Num_Peanut)           #caluating variance
Num_Peanut_mean = np.mean(Num_Peanut)
Num_Peanut_sum = np.sum(Num_Peanut)
Num_Peanut_count = len(Num_Peanut)

Num_PB =  Number_Anova.loc[:,'PB']
Num_PB_dev = np.std(Num_PB)           #caluating standard deviation 
Num_PB_var = np.var(Num_PB)           #caluating variance
Num_PB_mean = np.mean(Num_PB)
Num_PB_sum = np.sum(Num_PB)
Num_PB_count = len(Num_PB)

Blue_Reg = Blue_Anova.loc[:,'Regular']
Blue_Reg_dev = np.std(Blue_Reg)         #caluating standard deviation 
Blue_Reg_var = np.var(Blue_Reg)           #caluating variance
Blue_Reg_mean = np.mean(Blue_Reg)
Blue_Reg_sum = np.sum(Blue_Reg)
Blue_Reg_count= len(Blue_Reg)

Blue_Peanuts = Blue_Anova.loc[:,'Peanut']
Blue_Peanuts_dev = np.std(Blue_Peanuts)    #caluating standard deviation 
Blue_Peanuts_var = np.var(Blue_Peanuts)           #caluating variance
Blue_Peanuts_mean = np.mean(Blue_Peanuts)
Blue_Peanuts_sum = np.sum(Blue_Peanuts)
Blue_Peanuts_count =len(Blue_Peanuts)

Blue_PB = Blue_Anova.loc[:,'PB'] 
Blue_PB_dev = np.std(Blue_PB)            #caluating standard deviation 
Blue_PB_var = np.var(Blue_PB)           #caluating variance
Blue_PB_mean = np.mean(Blue_PB)
Blue_PB_sum = np.sum(Blue_PB)
Blue_PB_count =len(Blue_PB)


desired_alpha = 0.05 
Bonferroni_Correction =desired_alpha/3 #3 representing the number of categories 
dfw = len(Num_Reg) + len(Num_Peanut) + len(Num_PB) - 3  #df equals 72


pairwise_caluation_num(Num_Reg,Num_Peanut,Num_PB,desired_alpha,RegularvsPeanut,RegularvsPB,PeanutvsPB)
pairwise_caluation_blue(Blue_Reg,Blue_Peanuts,Blue_PB,desired_alpha,BlueRegularvsPeanut,BlueRegularvsPB,BluePeanutvsPB)

data_nummnms = {"Regular vs Peanuts":RegularvsPeanut,
             "Regular vs PB":RegularvsPB,
             "Peanut vs PB":PeanutvsPB}
data_bluemnms = {"Regular vs Peanuts":BlueRegularvsPeanut,
             "Regular vs PB":BlueRegularvsPB,
             "Peanut vs PB":BluePeanutvsPB}

Numb_Anova_table =  pd.DataFrame(data_nummnms).transpose()  # contains the first table for Number of Mnms for anova caluations
Blue_Anova_table =  pd.DataFrame(data_bluemnms).transpose()  # contains the first table for Blue Mnms for anova caluations

Number_Summary = {
   "Regular": [Num_Reg_count,Num_Reg_sum,Num_Reg_mean,Num_Reg_var,Num_Reg_dev],
   "Peanut":  [Num_Peanut_count,Num_Peanut_sum,Num_Peanut_mean,Num_Peanut_var,Num_Peanut_dev],
   "PB": [Num_PB_count,Num_PB_sum,Num_PB_mean,Num_PB_var,Num_PB_dev]
}
Anova_Number_Summary = {
   "Between Groups":[],
   "Within Groups": [],
   "Total":[]

}
Blue_Summary = {
   "Regular": [Blue_Reg_count,Blue_Reg_sum,Blue_Reg_mean,Blue_Reg_var,Blue_Reg_dev],
     "Peanut":  [Blue_Peanuts_count,Blue_Peanuts_sum,Blue_Peanuts_mean,Blue_Peanuts_var,Blue_Peanuts_dev],
     "PB": [Blue_PB_count,Blue_PB_sum,Blue_PB_mean,Blue_PB_var,Blue_PB_dev]
}
Anova_Blue_Summary = {
   "Between Groups":[],
   "Within Groups":[],
   "Total":[]
   
}
labels = ['Count', 'Sum', 'Average','Variance',"Standard Devivation"]
Number_Summary_df = pd.DataFrame(Number_Summary).transpose()
Number_Summary_df.columns = labels
Blue_Summary_df = pd.DataFrame(Blue_Summary).transpose()
Blue_Summary_df.columns = labels


# Combine all data into one array
all_data = np.concatenate([Num_Reg, Num_Peanut, Num_PB])

# Calculate overall mean
overall_mean = np.mean(all_data)

# Calculate group means
group1_mean = np.mean(Num_Reg)
group2_mean = np.mean(Num_Peanut)
group3_mean = np.mean(Num_PB)

# Calculate SS between groups
ssb_group1 = len(Num_Reg) * (group1_mean - overall_mean)**2
ssb_group2 = len(Num_Peanut) * (group2_mean - overall_mean)**2
ssb_group3 = len(Num_PB) * (group3_mean - overall_mean)**2

ssb_total = round(ssb_group1 + ssb_group2 + ssb_group3,7) #rounding to get consistent result
df = len(['Regular','Peanut','PB']) -1 #because this df is based off the number of catecoriges 
#print(df)
#print(ssb_total)
#print(ssb_total/df)
msb = ssb_total/df

g1_mean = np.mean(Num_Reg)
sse_group1 = np.sum((Num_Reg - g1_mean)**2)
g2_mean = np.mean(Num_Peanut)
sse_group2 = np.sum((Num_Peanut - g2_mean)**2)
g3_mean = np.mean(Num_PB)
sse_group3 = np.sum((Num_PB - g3_mean)**2)
sse = sse_group1+sse_group2+sse_group3

dfw = len(Num_Reg) + len(Num_Peanut) + len(Num_PB) - 3  #df equals 72
mse = sse/dfw
#print(mse)
f_table = msb/mse
p_value = 1 - f.cdf(f_table, df, dfw)
alpha = 0.05
f_critical = f.ppf(1 - alpha, df, dfw)

Anova_Number_Summary['Between Groups'].append(ssb_total)
Anova_Number_Summary["Between Groups"].append(df)
Anova_Number_Summary['Between Groups'].append(msb)
Anova_Number_Summary["Between Groups"].append(f_table)
Anova_Number_Summary['Between Groups'].append(p_value)
Anova_Number_Summary['Between Groups'].append(f_critical)

#print(Anova_Number_Summary["Between Groups"])
Anova_Number_Summary["Within Groups"].append(sse)
Anova_Number_Summary['Within Groups'].append(dfw)
Anova_Number_Summary['Within Groups'].append(mse)
Anova_Number_Summary["Within Groups"].append(np.nan)
Anova_Number_Summary["Within Groups"].append(np.nan)
Anova_Number_Summary["Within Groups"].append(np.nan)
#print(Anova_Number_Summary["Within Groups"])
Anova_Number_Summary["Total"].append(sse+ssb_total)
Anova_Number_Summary["Total"].append(df+dfw)
Anova_Number_Summary["Total"].append(np.nan)
Anova_Number_Summary["Total"].append(np.nan)
Anova_Number_Summary["Total"].append(np.nan)
Anova_Number_Summary["Total"].append(np.nan)
#print(Anova_Number_Summary["Total"])
Anova_num_sum_df = pd.DataFrame(Anova_Number_Summary).transpose()
labels_Anovas = ['SS','df','MS','F','p-value','F Critical']
Anova_num_sum_df.columns = labels_Anovas

# Combine all data into one array
all_data = np.concatenate([Blue_Reg, Blue_Peanuts, Blue_PB])

# Calculate overall mean
overall_mean = np.mean(all_data)

# Calculate group means
group1_mean = np.mean(Blue_Reg)
group2_mean = np.mean(Blue_Peanuts)
group3_mean = np.mean(Blue_PB)

# Calculate SS between groups
ssb_group1 = len(Blue_Reg) * (group1_mean - overall_mean)**2
ssb_group2 = len(Blue_Peanuts) * (group2_mean - overall_mean)**2
ssb_group3 = len(Blue_PB) * (group3_mean - overall_mean)**2

ssb_total = round(ssb_group1 + ssb_group2 + ssb_group3,7) #rounding to get consistent result
df = len(['Regular','Peanut','PB']) -1 #because this df is based off the number of catecoriges 
#print(df)
#print(ssb_total)
#print(ssb_total/df)
msb = ssb_total/df

g1_mean = np.mean(Blue_Reg)
sse_group1 = np.sum((Blue_Reg - g1_mean)**2)
g2_mean = np.mean(Blue_Peanuts)
sse_group2 = np.sum((Blue_Peanuts - g2_mean)**2)
g3_mean = np.mean(Blue_PB)
sse_group3 = np.sum((Blue_PB - g3_mean)**2)
sse = sse_group1+sse_group2+sse_group3

dfw = len(Blue_Reg) + len(Blue_Peanuts) + len(Blue_PB) - 3  #df equals 72
mse = sse/dfw
#print(mse)
f_table = msb/mse
p_value = 1 - f.cdf(f_table, df, dfw)
alpha = 0.05
f_critical = f.ppf(1 - alpha, df, dfw)


Anova_Blue_Summary['Between Groups'].append(ssb_total)
Anova_Blue_Summary["Between Groups"].append(df)
Anova_Blue_Summary['Between Groups'].append(msb)
Anova_Blue_Summary["Between Groups"].append(f_table)
Anova_Blue_Summary['Between Groups'].append(p_value)
Anova_Blue_Summary['Between Groups'].append(f_critical)


Anova_Blue_Summary["Within Groups"].append(sse)
Anova_Blue_Summary['Within Groups'].append(dfw)
Anova_Blue_Summary['Within Groups'].append(mse)
Anova_Blue_Summary["Within Groups"].append(np.nan)
Anova_Blue_Summary["Within Groups"].append(np.nan)
Anova_Blue_Summary["Within Groups"].append(np.nan)

Anova_Blue_Summary["Total"].append(sse+ssb_total)
Anova_Blue_Summary["Total"].append(df+dfw)
Anova_Blue_Summary["Total"].append(np.nan)
Anova_Blue_Summary["Total"].append(np.nan)
Anova_Blue_Summary["Total"].append(np.nan)
Anova_Blue_Summary["Total"].append(np.nan)

Anova_blue_sum_df = pd.DataFrame(Anova_Blue_Summary).transpose()
Anova_blue_sum_df.columns = labels_Anovas
print("SECTION NUMBER OF MNM\n\n")
print("Summary of Number of MnMs\n")
print(Number_Summary_df)
print("\n")
print("Anova Table for Number of MnMs\n")
print(Anova_num_sum_df)
print("\n")
print("Fisher's LSD multiple comparison test on the number of Mnms\n")
print(Numb_Anova_table)
print("\n")
print(f"An analysis of variance showed that the effect of type of MnM was significant,F(2,75) = {round(Anova_num_sum_df.iat[0,3],2)} and p < 0.001. Post hoc analyses using Fisher's LSD indicated that the number of peanut MnMs was significantly lower (M = {Number_Summary_df.iat[1,2]}, SD = {round(Number_Summary_df.iat[1,4],3)} and penut butter (M = {Number_Summary_df.iat[2,2]}), SD = {round(Number_Summary_df.iat[2,4],3)})\n") 
f_statistic, p_value = f_oneway(Num_Reg, Num_Peanut, Num_PB)
#print(f"I got the p values two different wais")
#print(p_value)
#print("\n")
#print(Anova_Number_Summary['Between Groups'][4])
#print("\n")

#print(f"The orginal F statstic I got was {f_statistic} but for the sake of matching data from class\nI will round up to 7 decmial places as done here {round(f_statistic,7)}\n")
f_stat_7d = round(f_statistic,7)



print("SECTION BLUE MNM\n\n")
print("Summary of Blue of MnMs\n")
print(Blue_Summary_df)
print("\n")
print("Anova Table for Blue MnMs\n")
print(Anova_blue_sum_df)
print("\n")
print("Fisher's LSD multiple comparison test on the Blue Mnms\n")
print(Blue_Anova_table)
print("\n")
print(f"An analysis of variance showed that the effect of the number of blue mnms was significant,F(2,75) = {round(Anova_blue_sum_df.iat[0,3],3)} and p < 0.001. Post hoc analyses using Fisher's LSD indicated that the amount of Blue MnMs in peanut MnMs were significantly higher (M = {Blue_Summary_df.iat[1,2]}, SD =  {round(Blue_Summary_df.iat[1,4],3)})  compared to  Regular MnMs (M = {Blue_Summary_df.iat[0,2]}, SD = {round(Blue_Summary_df.iat[0,4],3)}) and  Peanut Butter(M = {Blue_Summary_df.iat[2,2]}, SD ={round(Blue_Summary_df.iat[2,4],3)} ")
#print(Number_Summary)
#print("\n")
#print(Blue_Summary)






#sstr = len(Num_Reg) * (Num_Reg.mean() - mean_within_group)**2 +  len(Num_Peanut) * (Num_Peanut.mean() - mean_within_group)**2 + len(Num_PB) * (Num_PB.mean() -mean_within_group)**2
#print(sstr)

#ssw =  

#print(sse_group1+sse_group2+sse_group3)




#print(denom_1)
#print(t_value)
#print(p_value)   

#mse_within_groups = ssw / dfw
#print(mse_within_groups 
#4.376666666666667

    

#for index, row in test.iterrows():
 #   print(f"Index: {index}, num: {row['numerator']}, denom: {row['denominator']}, t_value: {row['t_value']}, Significange {row['Yes/No']}")

categories = ['Regular',"Peanut","PB"]
num_means = [Num_Reg_mean,Num_Peanut_mean,Num_PB_mean]
num_dev =  [Num_Reg_dev/2,Num_Peanut_dev/2,Num_PB_dev/2]
plt.ylabel('Number of Mnms')
plt.bar(categories, num_means, yerr=num_dev, capsize=5, color='gold', alpha=0.5)
plt.grid(axis='y', linestyle='-', alpha=0.5)
plt.show()
blue_means =  [Blue_Reg_mean,Blue_Peanuts_mean,Blue_PB_mean]
blue_dev = [Blue_Reg_dev/2,Blue_Peanuts_dev/2,Blue_PB_dev/2]
plt.ylabel("Number of Blue MnMs")
plt.bar(categories, blue_means, yerr=blue_dev, capsize=5, color='red', alpha=0.5)
plt.grid(axis='y', linestyle='-', alpha=0.5)
plt.show()