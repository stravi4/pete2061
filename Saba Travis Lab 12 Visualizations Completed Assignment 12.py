
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

conn = sqlite3.connect("DCA.db")
titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13
for wellID in range (1, 18):
    
############### Question 1
    prodDF = pd.read_sql_query(f"SELECT time,rate,Cum FROM Rates WHERE wellID={wellID};", conn)
    dcaDF = pd.read_sql_query("SELECT * FROM DCAparams;", conn) 
    
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    ax1.plot(prodDF['time'], prodDF['rate'], color="green", ls='None', marker='o', markersize=4,)
    ax2.plot(prodDF['time'], prodDF['Cum']/1000, 'b-' )
    
    ax1.set_xlabel('Time, Months')
    ax1.set_ylabel('Production Rate, bopm' , color='g')
    ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')

    plt.show()
##############  Question 2
prodDF.drop(["rate","Cum"],axis=1, inplace = True)
dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='gas';", conn)
for i in dcaDF['wellID']:
    prodDF['Well' + str(i)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID={i};", conn)
    
production = prodDF.iloc[:,1:].values
time = prodDF['time'].values

labels = prodDF.columns
labels = list(labels[1:])
print (labels)
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(production),labels=labels)
ax.legend(loc='upper right')
plt.title('Stacked Field Gas Production')
plt.show()
############## Question 3
oilRatesDF = pd.DataFrame(prodDF['time'])
dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='oil';", conn)
for i in dcaDF['wellID']:
    oilRatesDF['Well' + str(i)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID={1};", conn)
production = oilRatesDF.iloc[:,1:].values
time = oilRatesDF['time'].values
labels = oilRatesDF.columns
labels = list(labels[1:])
fig, ax = plt.subplots()
ax.stackplot(time, np.transpose(production),labels=labels)
ax.legend(loc='upper right')
plt.title('Stacked Field Oil Production')
plt.show()
##################### Question 4
N=6
ind = np.arange(1,N+1)
months = ['Jan','Feb','Mar','Apr','May','Jun']
result = np.zeros(len(months))
labels = []
loc_plts = []
width = 0.5

cumDF = pd.DataFrame(prodDF['time'])
dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='gas';", conn)
for i in dcaDF['wellID']:
    cumDF['Well' + str(i)] = pd.read_sql_query(f"SELECT Cum FROM Rates WHERE wellID={i};", conn)
    
j = 1
for i in dcaDF['wellID']:

    p1 = plt.bar(cumDF['time'][0:N], cumDF['Well' + str(i)][0:N]/1000,width, bottom = result)
    labels.append('Well' + str(i))
    loc_plts.append(p1)
    plt.ylabel('Gas Production, Mbbls')
    plt.title('Cumulative Gas Field Production')
    plt.xticks(ind, months, fontweight='bold')
    j +=1
    split = cumDF.iloc[0:6,1:j].values
    result = np.sum(split,axis=1)/1000

plt.legend(loc_plts,labels)
plt.show(loc_plts)
######################### Question 5
N = 6
ind = np.arange(1, N + 1)
months = ['Jan','Feb','Mar','Apr','May','Jun']
result = np.zeros(len(months))
labels=[]
loc_plts = []
width = 0.5

cumDF = pd.DataFrame(prodDF['time'])
dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='oil';", conn)
for i in dcaDF['wellID']:
    cumDF['Well' + str(i)] = pd.read_sql_query(f"SELECT Cum FROM Rates WHERE wellID={i};", conn)

j = 1
for i in dcaDF['wellID']:   
    
    p1 = plt.bar(cumDF['time'][0:N], cumDF['Well' + str(i)][0:N]/1000,width,bottom = result)
    labels.append('Well' + str(i))
    loc_plts.append(p1)
    plt.ylabel('Oil Production, Mbbls')
    plt.title('Cumulative Oil Field Production')
    plt.xticks(ind, months, fontweight='bold')
    j +=1
    split = cumDF.iloc[0:6,1:j].values
    result = np.sum(split,axis=1)/1000
    
plt.legend(loc_plts,labels)
loc_plts = plt.figure(figsize=(36,20),dpi=100)

############QUESTION 6###################
data1 = np.loadtxt("volve_logs/15_9-F-1B_INPUT.LAS",skiprows=69)
DZ1,rho1=data1[:,0], data1[:,16]
DZ1=DZ1[np.where(rho1>0)]
rho1=rho1[np.where(rho1>0)]

titleFontSize = 22
fontSize = 20

fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho1,DZ1, color='red')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,DT1=data1[:,0], data1[:,2]
DZ1=DZ1[np.where(DT1>0)]
DT1=DT1[np.where(DT1>0)]

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(DT1,DZ1, color='green')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,DTS1=data1[:,0], data1[:,3]
DZ1=DZ1[np.where(DTS1>0)]
DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(DTS1,DZ1, color='blue')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,GR1=data1[:,0], data1[:,4]
DZ1=DZ1[np.where(GR1>0)]
GR1=GR1[np.where(GR1>0)]

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(GR1,DZ1, color='black')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,NPHI1=data1[:,0], data1[:,5]
DZ1=DZ1[np.where(NPHI1>0)]
NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(NPHI1,DZ1, color='brown')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1,CALI1=data1[:,0], data1[:,6]
DZ1=DZ1[np.where(CALI1>0)]
CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(CALI1,DZ1, color='grey')
plt.title('Caliper vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, m', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('well_1_log.png', dpi=600) 

########

data2 = np.loadtxt("volve_logs/15_9-F-4_INPUT.LAS",skiprows=65)
DZ2,rho2=data2[:,0], data2[:,7]
DZ2=DZ2[np.where(rho2>0)]
rho2=rho2[np.where(rho2>0)]

titleFontSize = 22
fontSize = 20

fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho2,DZ2, color='red')
plt.title('Density vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,DT2=data2[:,0], data2[:,2]
DZ2=DZ2[np.where(DT2>0)]
DT2=DT2[np.where(DT2>0)]

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(DT2,DZ2, color='green')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,DTS2=data2[:,0], data2[:,3]
DZ2=DZ2[np.where(DTS2>0)]
DTS2=DTS2[np.where(DTS2>0)]

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(DTS2,DZ2, color='blue')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,GR2=data2[:,0], data2[:,4]
DZ2=DZ2[np.where(DTS2>0)]
GR2=GR2[np.where(DTS2>0)]

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(GR2,DZ2, color='black')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,NPHI2=data2[:,0], data2[:,5]
DZ2=DZ2[np.where(NPHI2>0)]
NPHI2=NPHI2[np.where(NPHI2>0)]

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(NPHI2,DZ2, color='brown')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ2,CALI2=data1[:,0], data2[:,6]
DZ2=DZ2[np.where(CALI2>0)]
CALI2=CALI2[np.where(CALI2>0)]

plt.subplot(1, 6, 6)
plt.grid(axis='both')
plt.plot(CALI2,DZ2, color='grey')
plt.title('Caliper vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, m', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('well_2_log.png', dpi=600)

###############################
