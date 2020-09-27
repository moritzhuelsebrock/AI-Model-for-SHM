import pandas as pd
import os
fpath = 'C:/Users/XW0823/Documents/Master/Arbeit/Hiwi/ARP_LBF-master/ARP_LBF-master/Data/version_5/daten1P_gerundet.csv'
fpath2 ="..\daten1P.csv"
# with open(fpath,'r') as f:
#     a=0
#     if(a \
#     >1):
#         print(f.read())
df=pd.read_csv(fpath2)
path=os.getcwd()
df=pd.read_csv("daten1P.csv")
# with open(path,'r') as f:
#     print(os.getcwd())
# df=pd.read_csv(fpath2)
# print(df['Tem'])
# print(df.min())
print(os.listdir('.'))


