#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import all python packages I'll be using
import pandas as pd
import researchpy as rp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy.stats import norm

# In[2]:
# import the Assignment data as pandas dataframe to facilitate further analysis
df = pd.read_excel(
    'C:\\Users\\vasco\\Documents\\Faculdade\\1st Semester\\Statistical Analysis\\Assignment\\Assignment data_10.xlsx',
    'Series10', index_col=None)
data_df = pd.DataFrame(df)
print(data_df)
# In[3]:
# for the first step, let's get some summary statistics
Summary_Statistics = rp.summary_cont(data_df)
print(Summary_Statistics)

# In[4]:
# Since it doesn't give us the Sample Variance, we need to calculate it separately
Sample_Var = data_df.var()
print(Sample_Var)

# In[5]:
# Now to generate the 5 number Summary, along with some other additional data already analysed
Five_Number_Summary = data_df.describe()
print(Five_Number_Summary)

# In[6]:
# now for measures of Kurtosis and Skewness
Skewness = df.skew()
Kurtosis = df.kurt()
print(Skewness)
print(Kurtosis)

# In[20]:
# now to translate this into visuals
Boxplot = sns.boxplot(data=df).set(xlabel="Group Types", ylabel="Sales")
# plt.savefig('C:\\Users\\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\boxplot.png')
plt.show()

# In[8]:
# Now to generate Histograms for each Group, including an overlay of their distribution vs the Normal Distribution
Ad1 = df['Advertisement1']
Ad1_Hist = sns.distplot(Ad1, hist=True, kde=True, fit=norm, color='green', axlabel='Sales').set_title(
    "Histogram Group 1")
# plt.savefig('C:\\Users\\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\Ad1.png')
plt.show()

# In[9]:
Ad2 = df['Advertisement2']
Ad2_Hist = sns.distplot(Ad2, hist=True, kde=True, fit=norm, color='blue', axlabel='Sales').set_title(
    "Histogram Group 2")
# plt.savefig('C:\\Users\\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\Ad2.png')
plt.show()

# In[10]:
Ad3 = df['Advertisement3']
Ad3_Hist = sns.distplot(Ad3, hist=True, kde=True, fit=norm, color='red', axlabel='Sales').set_title("Histogram Group 3")
# plt.savefig('C:\\Users\\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\Ad3.png')
plt.show()

# In[11]:
Ad4 = df['Advertisement4']
Ad4_Hist = sns.distplot(Ad4, hist=True, kde=True, fit=norm, color='orange', axlabel='Sales').set_title(
    "Histogram Group 4")
# plt.savefig('C:\\Users\\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\Ad4.png')
plt.show()

# In[12]:
# Before performing an ANOVA, it's necessary to check it's assumptions
# 1st, the Assumption of Independence cannot be tested since it relies on good experimental design
# 2nd, the Assumption of Normality requires to perform the Shapiro-Wilk test
SW_Test = stats.shapiro(data_df)
print(SW_Test)

# In[21]:
# Since the W-statistic is higher than the p-value, we cannot reject the null hypothesis,
# so the data seems to be normally distributed
# Now for the last assumption: Homoscedasticy, i.e., equal variances, for which I'll use Levene's test, using the mean
Levene_Test = stats.levene(Ad1, Ad2, Ad3, Ad4, center='mean')
print(Levene_Test)

# In[14]:
# Since the statistic is higher than the p-value, we do not reject the null hypothesis,
# suggesting that the samples have equal variances
# Having fulfilled all of ANOVA's assumptions, we can safely perform one,
# but in order to run ANOVA with more detail,
# we need to convert our original data into a format liable for that analysis to be done
df_transform = pd.melt(data_df)
# make sure that the columns have the right datatype
pd.to_numeric(df_transform['value'])
print(df_transform)

# In[15]:
ANOVA = pg.anova(data=df_transform, dv='value', between='variable', detailed=True, effsize='np2')
print(ANOVA)

# In[16]:
# Since p-value is lower than our significance level (alpha = 0.05)
# we have enough evidence to reject the null hypothesis
# So, there is at least one pair of means that differs,
# therefore we need to check which one(s). For that we'll run a Tuskey HSD
TukeyHSD = pg.pairwise_tukey(data=df_transform, dv='value', between='variable', effsize='cohen')
print(TukeyHSD)

# In[17]:
# As previously seen, we can safely deduce that except for the pair Advertisement1 and Advertisement2,
# all other pairs have differing means.
# Now to export all our data to an Excel file
# writer = pd.ExcelWriter(
#    "C:\\Users\VascoRod\\OneDrive\\Documents\\Faculdade\\Statistical Analysis\\Assignment\\Python_Outputs.xlsx",
#    engine='xlsxwriter')

# Summary_Statistics.to_excel(writer, sheet_name='Summary Stats', index=False)
# Sample_Var.to_excel(writer, sheet_name='Sample_Var', index=True)
# Five_Number_Summary.to_excel(writer, sheet_name='5 nº Summ')
# Skewness.to_excel(writer, sheet_name='Skewnews')
# Kurtosis.to_excel(writer, sheet_name='Kurtosis')
# The next two need to first be converted to Pandas Dataframes before export
# pd.DataFrame(SW_Test).to_excel(writer, sheet_name='Shapiro-Wilk')
# pd.DataFrame(Levene_Test).to_excel(writer, sheet_name='Levene')
# ANOVA.to_excel(writer, sheet_name='ANOVA', index=False)
# TukeyHSD.to_excel(writer, sheet_name='Tukey HSD', index=False)

# writer.save()

# Now we have all the info we need to build our report!
