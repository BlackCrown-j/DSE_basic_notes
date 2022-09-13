#!/usr/bin/env python
# coding: utf-8

# # Lecture1

# In[2]:


import numpy as np


# In[3]:


#manual


# In[4]:


l1=[1,2,3,4,5,6,7,8,9,10]


# In[5]:


np.array(l1)


# In[ ]:





# In[6]:


l2=[1,2,3,4,5,6,7,8,9,10.2]


# In[7]:


np.array(l2)


# In[8]:


l3=[1,2,3,4,5,6,7,8,9,10.2,11+5j]


# In[9]:


np.array(l3)


# In[10]:


#complex>float>integer ---- priority sequence in decending order.


# In[11]:


l4=[1,2,3,4,5,6,7,8]


# In[12]:


np.array(l4)


# In[ ]:





# In[ ]:





# In[13]:


#dimensions


# In[14]:


a=np.array([1,2,3,4,5])


# In[15]:


a


# In[16]:


a.ndim #number of dimensions


# In[17]:


abc=[[1,2,3],[4,5,6]]


# In[18]:


np.array(abc)


# In[19]:


abc1=[[1,2,3],[4,5,6],[7,8,9]]


# In[20]:


x=np.array(abc1)


# In[21]:


x.ndim


# In[ ]:





# In[ ]:





# latest,Automatic methods

# In[22]:


list(range(1,10)) #in numpy the last no. (here 10) is exclusive


# In[23]:


y=np.arange(1,10)
y


# In[24]:


np.arange(10)


# In[ ]:





# In[25]:


#create 1 dimensional array of even numbersnbetween 0 and 100


# In[26]:


np.arange(0,100,2)#(start,end,step)


# In[27]:


np.arange(1,100,2)#odd numbers


# In[28]:


np.arange(-20,100,2)


# In[29]:


list(range(1.2,10.2))


# In[ ]:


np.arange(1.2,10.2,2.6)


# In[ ]:


np.arange(3+10j,10+23j)


# In[ ]:


#but the sbove thing can't be done in basic python.


# In[ ]:





# In[ ]:





# the below method will not make changes in original data

# In[ ]:


np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float64)#64 here is bits


# In[ ]:


np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.complex128)#128 i 64 bits for real and 64 for imaginary part.


# In[ ]:





# In[ ]:





# In[ ]:


np.zeros(3)#==> 1D dimensional zero array.


# In[ ]:


np.zeros((2,2))


# In[ ]:


np.zeros((4,2))


# In[ ]:


np.zeros((4,5))


# In[ ]:


np.ones(3)#generating unity array


# In[ ]:


np.ones((4,5))


# Random method

# In[ ]:


np.random.randint(1,50)


# In[ ]:


np.random.randint(1,50,10)


# In[ ]:


q=np.random.randint(1,50,11)
q


# In[ ]:


np.random.randint(1,25,20)#here repitions will be there 


# In[ ]:


set(np.random.randint(1,25,20))#unique elements


# In[ ]:


np.random.randint(1,25,size=(4,2))


# In[ ]:


np.random.randint(1,25,size=4)


# In[ ]:


np.random.randn(5)  #here n in randn stands for normal numbers. Normalizatoin of number means generating no as close as possible to zero


# In[ ]:


np.random.randn(4,2)


# In[ ]:


np.random.randn(3,4)


# In[ ]:


np.random.randn(3,4,2)


# In[ ]:


np.random.randn(3,4,2,3)


# In[ ]:


np.random.rand(6,5)#uniform distribution, number of range [0,1) 


# In[ ]:


#numpy only allows to see 8 bits after decimal,if blank before 8 bits then that is zero.


# In[ ]:





# In[ ]:





# # Lecture 2

# indexing of arrays,
#  slicing,
#  broadcasting,
#  arithmetic ops,
#  tri,
#  log,
#  exp,
#  stack and split

# In[ ]:


import numpy as np


# In[ ]:


abc=np.arange(10)


# In[ ]:


abc


# In[ ]:


abc[-1]


# In[ ]:


abc[-9]


# In[ ]:


abc[0:4] #index 4 is exclusive


# In[ ]:


abc[-4:-1]


# In[ ]:


abc[-1:-4]#here step is 1 and we get an empty error


# In[ ]:


abc[-1:-4:-1]#here step is -1 as we mentioned it separately


# In[ ]:


abc[0:-1] # here python see's zero and proceed for left to right reading but since -1 will be exclusive it will ignore it and will proceed in the direction till 2nd last element


# In[ ]:


abc[-1:0]


# In[ ]:


abc[::-1]#the simplest way to reverse your array. also remember the mechanism.


# In[ ]:


#in the above method it takes more time as it goes from 0 to last index and the from last index to 0. Hence it has to travel 2 times.


# In[ ]:


copy=abc[0:4]#this is slicing
copy


# In[ ]:


abc[0:4:-1]#it won't stop in the middle and start reversing. though there are some exception like above.


# In[ ]:


a=abc[0:4]


# In[ ]:


a[::-1]


# In[ ]:


abc[-2::-1]


# In[ ]:


abc[-2::]


# In[ ]:





# In[ ]:





# In[ ]:


abc[:4:-1]


# In[ ]:


abc[0:4:-1]


# In[ ]:


abc[4::-1]#here by seeing 4 it will start reading in forward direction, and will go to last index and then will see -1 n will print from 4 to 0th index


# In[ ]:





# In[ ]:


a=np.random.randint(10,50,size=(4,3))


# In[ ]:


a


# In[ ]:


a[0]#row 0


# ###### extract a row with reverse indexing

# In[ ]:


a


# In[ ]:


a[-1]


# In[ ]:


a[-2]


# In[ ]:


#a[rows,columns]


# In[ ]:


a[:,0]#data is extracted


# In[ ]:


a[:,0:1]#data and shape both are extracted. The shape here is retained.    


# In[ ]:


a


# In[ ]:


a[0][0]


# In[ ]:


a[0,0]


# In[ ]:


a[3,0]


# In[ ]:


a[3][0]


# In[ ]:


a[0:1,0]#faster and in array form


# In[ ]:


a[1:2,0:1]#better


# In[ ]:


a[0:2,1:]


# In[ ]:





# ##### Arithmetic operation

# In[ ]:


a1=np.arange(11,23)


# In[ ]:


a1


# In[ ]:


a1+5


# In[ ]:


a1-5


# In[ ]:


a1*5


# In[ ]:


a1/5


# In[ ]:


a1//5


# In[ ]:


a1%5


# In[ ]:


np.sum(a1)


# In[ ]:


np.mean(a1)


# In[ ]:


np.median(a1)


# In[ ]:


np.sqrt(a1)


# In[ ]:


np.square(a1)


# In[ ]:


np.cumsum(a1)#cumulative sum


# In[ ]:


a1


# In[ ]:


np.cumprod(a1)


# In[ ]:


np.sin(a1)#here calculations are in radians


# In[ ]:


np.rad2deg(np.sin(a1))


# In[ ]:


np.cos(a1)


# In[ ]:


x=np.sin(a1)
y=np.cos(a1)
z=x/y


# In[ ]:


z


# In[ ]:


a1**3


# In[ ]:


np.power(a1,3)#faster than above command


# In[ ]:


c=np.arange(22,33)


# In[ ]:


c


# In[ ]:


d=np.arange(11,22)


# In[ ]:


d


# In[ ]:


c+d


# In[ ]:


np.add(c,d)


# In[ ]:


np.log(a1)#natural log.(base e)


# In[ ]:


np.log2(a1)


# In[ ]:


np.log10(a1)


# In[ ]:


a2=np.arange(10)


# In[ ]:


a2


# In[ ]:


a3=np.arange(11,22)


# In[ ]:


a3


# In[ ]:


a2+a3


# In[ ]:


a3=np.arange(11,21)


# In[ ]:


a3


# In[ ]:


a2+a3#to add two array their size and dimensions should be same.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


p=np.random.randint(1,15,size=(3,3))
q=np.random.randint(2,16,size=(3,3))


# In[ ]:


p


# In[ ]:


q


# In[ ]:


p+q


# In[ ]:


p%q


# In[ ]:


qwe=np.arange(11,20)#1-Dimensional ordered static array.


# In[ ]:


qwe


# In[ ]:


n=np.random.randint(1,10,9)#1-Dimensional unordered dynamic array.
n


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


a=np.array([1,2,3])
b=np.array([4,5,6])


# In[ ]:


np.hstack((a,b))


# In[ ]:


np.vstack((a,b))


# In[ ]:


np.concatenate((a,b))


# In[ ]:


np.concatenate((p,q))


# In[ ]:


a


# In[ ]:


np.split(a,3)#the array has to be divided into equal parts


# In[ ]:





# In[ ]:





# In[ ]:





# # Lecture 3

# In[ ]:


import numpy as np


# hsplit, 
# vsplit, 
# attributes of an array, 
# reshape

# pandas and data manipulation

# In[ ]:


d=np.arange(12)


# In[ ]:


d


# In[ ]:


d.reshape(2,6)


# In[ ]:


d.reshape(3,4)


# In[ ]:


d1=np.arange(19)


# In[ ]:


d1


# In[ ]:


d1.reshape(19,1)


# In[ ]:


d1.shape


# In[ ]:


d.shape


# In[ ]:


d1.ndim


# In[ ]:


d1.size#total no of data points


# In[ ]:


d2=d.reshape(3,4)


# In[ ]:


d2


# In[ ]:


d2.shape


# In[ ]:


d2.size


# In[ ]:


d2.itemsize# every data point occupies 4 bytes of data (12*4=48 bytes)


# In[ ]:


d3=np.arange(10).reshape(5,2)


# In[ ]:


d3


# In[ ]:


np.hsplit(d3,2)# splits your array colum wise. spliting is done in equal parts only.

#note it gives you a list and not an array. Hence other operations of array can't be performed on this after spliting. 


# In[ ]:


q=np.hsplit(d3,2)
q


# In[ ]:


type(q)


# In[ ]:


np.array(q)


# In[ ]:


np.array(q).shape#((2,3,1)===>3-Dimensional,2=2 arrays,5=5 rows, 1=1 coloumn)


# In[ ]:


np.vsplit(d3,2)


# In[ ]:


d4=d3.reshape(2,5)


# In[ ]:


d4


# In[ ]:


np.vsplit(d4,2)#rowwise. Spliting is alwaays done in equal parts.Note : return type =list


# In[ ]:


np.array(np.vsplit(d4,2)).shape


# In[ ]:


a=np.vsplit(d4,2)##we will show conversion of array to list, list to df,df array in pandas.


# In[ ]:


d4


# In[ ]:


d4.sum()#total sum


# In[ ]:


d4.sum(axis=0)# for column wise addition 


# In[ ]:


d4.sum(axis=1)# row wise sum


# In[ ]:


##################################################################################################################


# In[ ]:





# # Pandas

# 1. python module which help in data manipulation==> convertinh unstructed to structured data.

# 2. create a table/structure

# 3. extraction, indexing, slicing,

# 4. mathematical

# 5. business queries

# 6. format

# 7. concat, join, merge

# 8. diagrams==> visualization

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#list python ==>method 1


# In[3]:


list_of_names=['rahul','shambhavi','akriti','john']
list_of_countries=['india','U.S','russia','japan']


# In[4]:


pd.DataFrame(list_of_names)


# In[5]:


pd.DataFrame(list_of_names,columns=['name of students'],index=['a','b','c','d'])#to change default index


# In[6]:


pd.DataFrame(list_of_names,columns=['name of students'],index=range(1,5))#re-numbering the index


# In[7]:


pd.DataFrame(list_of_names,columns=['name of students'])#naming the column pass a list of names for columns.


# In[8]:


pd.DataFrame(list_of_names,columns=['name of students'] ,index='w x y z'.split())#string passed as sequence for index


# pd.DataFrame(data,index,columns)

# In[9]:


pd.DataFrame([list_of_names,list_of_countries],columns=['A','B','C','D'],index=['po','lo'])


# In[10]:


data=pd.DataFrame([[list_of_names],[list_of_countries]])
data

creating data frame using python dictionary
# In[11]:


corona_cases= {'india':1000,'U.S':1200,'japan':300}


# In[12]:


pd.DataFrame(corona_cases,index=[0])#while passing scalar values we have to give index otherwise it will give error.


# while converting the dictionary to data frame, key are converted to columns.

# In[13]:


pd.DataFrame(corona_cases,index=['a'])


# In[14]:


corona_cases= {'india':[1000,200],'U.S':[1200,345],'japan':[300,9898]}


# In[15]:


pd.DataFrame(corona_cases,index=['a','b'])


# In[16]:


corona_cases= {'india':[1000,200],'U.S':[1200,345],'japan':[300]}#not giving equal number of values


# In[17]:


pd.DataFrame(corona_cases,index=['a','b'])#in such case the value appearing most no. of times will appear in the empty place.(repition mode)


# In[18]:


pd.DataFrame(corona_cases)#error will be given if index or no. of rows is not mentioned.


# In[ ]:


pd.DataFrame(corona_cases,index=['a','b'])


# method-3.Creating data frames using numpy arrays

# In[19]:


pd.DataFrame(np.arange(10,21),columns=['numbers'])


# In[20]:


pd.DataFrame(np.arange(10,21),columns=['numbers'],index=range(20,31))


# In[21]:


pd.DataFrame(np.random.randint(23,67,15))


# In[22]:


pd.DataFrame(np.random.randint(23,67,size=(4,2)))


# In[23]:


pd.DataFrame(np.random.randint(23,67,size=(4,2)),columns=['A','B'])


# In[24]:


pd.DataFrame(np.random.randn(3,4))


# In[25]:


pd.DataFrame(np.random.randn(3,4),columns=['A','B','C','D'])


# In[ ]:





# In[26]:


# dataframe


# In[27]:


#read data from microsoft excel

#read data from microsoft excel--> csv


# In[28]:


pd.read_excel('C:\\Users\\pranj\\Desktop\\test.xlsx')


# In[29]:


data=pd.read_excel('C:/Users/pranj/Desktop/test.xlsx')
data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Lecture 4

# 1 real data
# 2 count
# 3 unique
# 4 nunique
# 5 sum
# 6 text
# 7 shape
# 8 head
# 9 tail
# 10 sample
# 11 value counts
# 12 pivot
# 13 groupby
# 14 map
# 15 replace
# 16 drop
# 17 drop duplicates

# In[ ]:


import numpy as np#mathematical computation
import pandas as pd# data manipulation
import seaborn as sns# data visualization lib
import matplotlib.pyplot as plt # oldest and versatile data viz librarry
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


data=sns.load_dataset('tips')
data


# In[49]:


#how to read 1st 5 rows of your data


# In[50]:


data.head()#1st 5 rows by default


# In[51]:


data.head(3)#for 1st three rows


# In[52]:


data.tail()#for last 5 rows by default.


# In[53]:


data.tail(10)


# In[54]:


# # picking random data samples


# In[55]:


data.sample(9)


# In[56]:


data.sample()


# In[57]:


# What is the size of your data ? or What is the shape od your data?


# In[58]:


data.shape


# In[59]:


#there are 244 rows and 7 columns in my dataset


# In[ ]:





# In[60]:


#What are the columns in your data set


# In[61]:


data.columns


# In[62]:


#These are the columns in my data frame


# In[ ]:





# In[63]:


#What is the index range of your data


# In[64]:


data.index


# In[ ]:





# In[65]:


#what is the maximum bill (USD) that was generated by a customer in your cafe?


# In[66]:


data['total_bill'].max()#it is from NUMPY library. 

#here half comand is from pandas and half from numpy. This is advanced numpy and pandas command.

#most preferable 


# In[67]:


#The max bill generated by a customer in my cafe is 50 usd 81 cents.


# In[68]:


max(data['total_bill'])#note= this is a basic python comand and is slower that numpy comand . Avoid this.


# In[ ]:


np.max(data['total_bill'])#completely numpy command without any pandas involved.


# In[ ]:





# In[ ]:


#What is the average bill in usd that was generated in your cafe?


# In[ ]:


data['total_bill'].mean()


# In[ ]:


np.mean(data['total_bill'])


# In[ ]:


#the avg bill is 19.7 usd


# In[ ]:





# In[69]:


data.head()


# In[70]:


#what is the total tip given to waiters in your cafe?


# In[71]:


data['tip'].sum()# golbally accepted and most preferable.


# In[72]:


data['tip'].sum().round(2)


# In[73]:


data.tip.sum()# . is accesion operator 


# In[ ]:





# In[74]:


data.head()


# In[75]:


#Calculate how many male and female customers come to your cafe?


# In[76]:


data['sex'].value_counts()# value_counts() is used only and only for categorical columns


# In[77]:


data.sex.value_counts()


# In[ ]:





# In[78]:


# What are the unique "time" when people visit your cafe?


# In[81]:


l=data['time'].unique()
l


# In[82]:


data['total_bill'].unique()


# In[83]:


#.unique() works for both categorigal and numerical columns but its application should be understood.


# In[84]:


data['time'].nunique()# .nunique() returns the number of unique elements in a column


# In[85]:


data['total_bill'].nunique()


# In[ ]:





# In[86]:


data.head()


# In[87]:


#What is the command that will tell more info about your datasaet


# In[88]:


data.info()


# In[89]:


#What is the five point summary of your dataframe? what is the statistical descod your df? how will you describe?


# In[90]:


data.describe()#it works only with numerical columns. it doesn't work for categorical columns.


# In[91]:


data.head()


# In[ ]:





# In[ ]:





# In[92]:


#What was the total bill generated on Sunday?


# In[93]:


data[data['day']=='Sun']#pulls our data here day is Sunday.


# In[94]:


data[data['day']=='Sun']['total_bill']#pulls out only total bill column for day=Sunday


# In[95]:


data[data['day']=='Sun']['total_bill'].sum()#returns the sum of tatal_bill column for day=Sunday.


# In[96]:


#on sunday,a total bill generation was 1627.16 USD


# In[ ]:





# In[ ]:





# In[97]:


data.head()


# In[98]:


#how many male members came on saturday to your cafe who were smoking?


# In[99]:


data.head()


# In[100]:


#multiple conditions


# In[103]:


(data['sex']=='Male') (data['day']=='Sat') (data['smoker']=='Yes')


# In[104]:


(data['sex']=='Male') & (data['day']=='Sat') & (data['smoker']=='Yes').count()


# In[105]:


data[(data['sex']=='Male') & (data['day']=='Sat') & (data['smoker']=='Yes')].count()


# In[106]:


#How many female visiting your cafe have paid a tip of more than 5 USD?


# In[107]:


data[(data['sex']=='Female') & (data['tip']>5)].count()


# In[108]:


data[data['sex']=='Female']['tip']>5


# In[110]:


data[data[data['sex']=='Female']['tip']>5] # if we write like this we get indexing error therefore we go by the 1st method


# In[111]:


data[(data['sex']=='Female') & (data['tip']>5)].count()


# In[112]:


#there are 4 females who visited the cafe and paid a ti of more than 5 USD.


# In[ ]:





# In[ ]:





# In[113]:


data


# In[114]:


#Replace the word 'Sun' in your data frame to 'Sunday' and store it permanently.


# In[115]:


data.head()


# In[116]:


data['day'].replace('Sun',"Sunday")#this only temperory changes the replacement.


# In[117]:


data.head()


# In[119]:


#method 1 to make permanent changes

data['day'].replace('Sun',"Sunday",inplace=True)#inplace=True will make the changes permanent.


# In[120]:


data.head()


# In[121]:


#method 2 to make permanent changes

data['day']=data['day'].replace('Sun','Sunday')


# In[122]:


data.head()


# In[ ]:





# In[ ]:





# In[123]:


# delete or drop a column from my data set


# In[124]:


data.head()


# In[125]:


data.drop('size',axis=1)#axis=1 means column, axis=0 means row. Since size is a column we took axis is 1.
#this is temperory , for permanent make inplace='True'


# In[ ]:





# In[126]:


data.head()


# In[127]:


copy_of_my_data=data.copy()#to create a copy of my data


# In[128]:


copy_of_my_data


# In[129]:


data.drop(['smoker','size'],axis=1)#to delete multiple columns temperory . For permanent do inplace=True.


# In[130]:


data.drop(0,axis=0)# to delete 0th row


# In[131]:


data.drop([0,5,239],axis=0)#to delete multiple rows temperory. for permanent deletion make inplace=True


# In[ ]:





# In[ ]:





# In[ ]:





# ###### Lecture 5

# In[132]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings(action='ignore')# to ignore warnings due to version difference, outdated methods, OS differences etc...


# In[133]:


data={'employee id':[111,112,113], 'names': ['Ashley','Robert','Shruti'],'gender':['male','male','female']}


# In[134]:


data


# In[137]:


df=pd.DataFrame(data)
df


# loc, iloc, pivot, duplicated, drop duplicates, groupby, map, concat, merge, join

# In[139]:


df1=pd.DataFrame(data,index=['emp1','emp2','emp3'])
df1


# In[140]:


df1.loc['emp1']#pulls out the details about the row emp1

#loc stands for location. Go to the location of row passed in the square brackets


# In[142]:


df.loc[0:1]#remember numbers work only if there is no name for the rows. You always have to pass the name of the rows.


# In[143]:


#indexing 

#we use iloc for this


# In[144]:


df1.iloc[0,1]#pulling out single data point


# In[146]:


df1.iloc[1]#priority given to rows. Pulling one row


# In[148]:


df1.iloc[:,2]#pulling out all the data of column 2. here we can see the out put is not looking lika a data frame for that we have to use : after 2


# In[149]:


df1.iloc[:,2:]#pulling out a single column


# In[154]:


df.iloc[0:2]#it works on indexing hence 2 is exclusive. using iloc.


# In[152]:


df.loc[0:2]#here 2 is inclusive because we are using loc. #it works on name.


# In[157]:


df.iloc[:,2:]


# In[158]:


df.loc[:,'names']


# #concat, append

# In[163]:


data2={'corona_cases':[1200,3000,4000],'countries':['usa','ussr','india']}
data3={'corona_cases':[2200,3200,4200],'countries':['japan','germany','singapore']}


# In[ ]:





# In[168]:


a1=pd.DataFrame(data2)


# In[169]:


a2=pd.DataFrame(data3)
a2


# In[170]:


# same index
#same column names
# different data


# In[171]:


pd.concat([a1,a2],axis=0) #here axis =0 is row wise. which is opposite to that in numpy


# In[172]:


#problem 1 ==> indecies are repeating . this will create problem while extracting values


# In[173]:


pd.concat([a1,a2],axis=1) #axis=1 means column wise


# In[174]:


#problem 2 ==> here we can find more that one column with same names


# In[175]:


pd.concat([a1,a2],axis=0,ignore_index=True) #solution to problem 1


# In[178]:


pd.concat([a1,a2],axis=1).T #Transpose as a solution to problem 2. Not a good solution .


# # 

# # 

# In[179]:


a1


# In[180]:


a2


# In[181]:


a2=pd.DataFrame(data3,index=[4,5,6])
a2


# In[185]:


a2.rename(columns={'corona_cases':'cc','countries':'nations'},inplace=True)#rename your columns. for permanent us e inplace='True'


# In[186]:


a2


# In[187]:


pd.concat([a1,a2],axis=0)#bad solution


# In[188]:


pd.concat([a1,a2],axis=1)


# In[189]:


#problem3 ==> columns have different names therefore NaN values are shown. and also the rows have different indecies.

#Therefore never change the name of the orginal columns


# In[191]:


a2.rename(columns={'cc':'corona_cases','nations':'countries'},inplace=True)


# In[192]:


a2


# In[193]:


pd.concat([a1,a2],axis=0)


# In[195]:


new_a2=pd.DataFrame(data3)


# In[199]:


new_a2.rename(columns={'corona_cases':'cc','countries':'nations'},inplace=True)


# In[200]:


pd.concat([a1,new_a2],axis=1)


# In[ ]:





# In[ ]:





# In[201]:


data2={'corona_cases':[1200,3000,4000],'countries':['usa','ussr','india']}
data3={'corona_cases':[2200,3200,4200],'countries':['japan','germany','singapore']}


# In[202]:


d1=pd.DataFrame(data2)
d1


# In[203]:


d2=pd.DataFrame(data3,index=[4,5,6])
d2


# In[204]:


d1.append(d2)#when ever you need to add two datasets.


# In[ ]:





# In[ ]:





# In[ ]:





# In[205]:


#merge

#merge is only about the columns.


# In[210]:


data3={'season':['winter','summer','spring'],'sales':[3000,4500,6700],'media_channel':['amazon','flipkart','bestbuy']}
data4={'season':['winter','summer','spring'],'country':['usa','ussr','japan'],'advertising':['posters','instagram','influencer']}


# In[215]:


df1=pd.DataFrame(data3)
df1


# In[216]:


df2=pd.DataFrame(data4)
df2


# In[217]:


#we apply merge when at least colum is exactly same.


# In[218]:


pd.merge(df1,df2,on=['season'])#traditional method. preferable


# In[220]:


pd.merge(df1,df2)#don't use this marks will be deducted in exams..


# Merge Sub categories
# 
# inner join, outer join,
#  left join, right join

# In[224]:


data3={'season':['winter','summer','fall'],'sales':[3000,4500,6700],'country':['usa','ussr','india']}
data4={'season':['winter','summer','spring'],'country':['usa','ussr','japan'],'advertising':['posters','instagram','influencer']}


# In[227]:


a1=pd.DataFrame(data3)
a1


# In[228]:


a2=pd.DataFrame(data4)
a2


# In[230]:


pd.merge(a1,a2,on=['season','country'],how='inner') #inner jin ==> intersection ==> common data


# In[232]:


pd.merge(a1,a2) #by default common columns, common data automatically. Don't use this , marks will be cut.


# In[233]:


pd.merge(a1,a2,on=['season','country'],how='outer')#outer join ==> union==> all data


# In[239]:


data3={'season':['winter','summer','fall'],'sales':[3000,4500,6700],'media_channel':['amazon','flipkart','bestbuy']}
data4={'season':['winter','summer','spring'],'country':['usa','ussr','japan'],'advertising':['posters','instagram','influencer']}


# In[240]:


a1=pd.DataFrame(data3)
a1


# In[241]:


a2=pd.DataFrame(data4)
a2


# In[242]:


#pd.merge(first priority, second priority)


# In[243]:


pd.merge(a1,a2,on=['season'],how ='left')#here left means a1 is kept constant (hence spring is there instead of fall) 


# In[245]:


pd.merge(a1,a2,on=['season'],how ='right')#here right means a2 is kept constant(hence fall is there instead of right)


#Addn Notes= play priority is in the sequence you write, how= 'left' /'right ' decide the data priority.


# In[ ]:





# In[246]:


#pivot tables, groupby, duplicated, drop_duplicates


# In[247]:


sales={'Months':['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec'],'Sales':[22000,27000,25000,29000,35000,67000,78000,67000,56000,56000,89000,60000],
      'Seasons': ['Winter','Winter','Spring','Spring','Spring','Summer','Summer','Summer','Fall','Fall','Fall','Winter']}

df_sales=pd.DataFrame(sales, columns=['Months','Sales','Seasons'])


# In[248]:


df_sales


# In[252]:


#What is the average sale of walmart in different seasons?
pd.pivot_table(df_sales,index=['Seasons'],values=['Sales'],aggfunc=np.mean)


# In[253]:


#What is the min sale of walmart in different seasons?
pd.pivot_table(df_sales,index=['Seasons'],values=['Sales'],aggfunc=np.min)


# In[254]:


df_sales


# In[255]:


pd.pivot_table(df_sales,index=['Seasons'])

#pivot tables work only with numerical columns. Here there is only one numerical column here  'Sales'. Therefore it will take that in sonsideration
#by default the aggfunc is mean.


# In[257]:


pd.pivot_table(df_sales,index=['Seasons','Months'],values=['Sales'],aggfunc=np.mean)


# In[258]:


pd.pivot_table(df_sales,index=['Seasons','Months'],values=['Sales'],aggfunc=np.mean).T


# In[259]:


df_sales.sort_values('Sales',ascending=False)


# In[260]:


df_sales.sort_values('Sales',ascending=False,ignore_index=True)


# In[ ]:





# In[261]:


df_sales.groupby(['Seasons'])['Sales'].mean()


# In[263]:


df_sales.groupby(['Seasons'])['Sales'].sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Lecture 6

# In[315]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline #generates the code along code without opening a new tab')
import warnings
warnings.filterwarnings(action='ignore')


# In[293]:


data=sns.load_dataset('tips')


# In[294]:


data


# In[268]:


data.head()


# In[269]:


#scatterplot
#Scatter plot works only with numerical data.


# Q. Whether total_bill and tip given to waiters is in positive correlation?
# 
# Q. total bill and tip are linearly proportional

# In[271]:


sns.scatterplot(x='total_bill',y='tip',data=data,color='g')


# In[274]:


sns.scatterplot(x='total_bill',y='tip',data=data,color='black')


# In[275]:


sns.scatterplot(x='total_bill',y='tip',data=data,color='g',hue='time')
#hue means give more importance to. Hue works only with categorical column

# less lucch dots maybe because of bad lunch menu


# In[276]:


sns.scatterplot(x='total_bill',y='tip',data=data,color='g',hue='day')

#Here we can see that we don't have dta of Mon,Tue and Wed. Therefore incomplete data.


# In[278]:


sns.scatterplot(x='total_bill',y='tip',data=data,color='g',hue='sex')


# In[ ]:


#Scatter plot using matplotlib ==>


# In[280]:


plt.scatter('total_bill','tip',data=data)


# In[281]:


plt.scatter(data['total_bill'],data['tip'])


# In[282]:


plt.scatter('total_bill','tip')#idk what is this..


# In[285]:


plt.figure(figsize=(14,5))# mention figure line at the starting
plt.scatter('total_bill','tip',data=data)
plt.xlabel('total_bill')#naming x-axis
plt.ylabel('tip')#naming y axis
plt.title('Scatter plot')

#Seaborn is better than matplotlib as in that we don't have to lable the axis separately. they are automatically labeled.
#to make changes in labels in seaborn we have to make changes in the data frame.


# Addn note---#in seaborn its hue and in matplotlib its label

# In[288]:


#barplot


# In[291]:


data.head()


# In[296]:


sns.barplot(x='day',y='total_bill',data=data)


# In[297]:


sns.barplot(y='day',x='total_bill',data=data)#interchanging axis


# In[299]:


sns.barplot(x='time',y='tip',data=data)

#one should be numerical and other should be numerical colum for bar plot


# In[301]:


sns.barplot(x='size',y='tip',data=data,hue='sex')


# In[303]:


sns.barplot(x='size',y='total_bill',data=data)


# In[309]:


plt.bar(x='total_bill',data=data,height=400)

#we can use matplotlib for barplot of single variables but that doesn't make sense therefore we use seaborn library.


# In[310]:


data.head()


# In[316]:


#How many male and females are visiting the cafe
sns.countplot(x='sex',data=data)


# In[317]:


#through pandas.
data['sex'].value_counts()


# In[318]:


sns.countplot(x='day',data=data)


# In[319]:


data['day'].value_counts()


# In[320]:


sns.countplot(x='size',data=data)


# In[321]:


data.head()


# In[322]:


data1=data['tip']
data2=np.square(data1)


# In[323]:


data2


# In[328]:


plt.figure(figsize=(12,5))
plt.plot(data1)#line plot


# In[329]:


plt.plot(data1,data2)


# In[ ]:





# In[330]:


data1=np.arange(10)
data1


# In[331]:


data2=np.square(data1)


# In[336]:


plt.figure(figsize=(12,5))
plt.plot(data2,color='gold',marker='*')
plt.title("diagram")


# In[ ]:





# In[ ]:





# In[ ]:





# In[337]:


#histogram
#Gaussian curve 
#(bell shaped curve) 


# In[340]:


#total_bill
sns.histplot(data['total_bill'],kde=True)
#kde=kernal density , will be discussed in stats.


# In[342]:


sns.histplot(data['total_bill'])
#these rectangles are called bins.


# In[344]:


sns.histplot(data['total_bill'],bins=40,color='red')


# In[ ]:





# In[347]:


plt.hist(x='total_bill',bins=30,data=data)


# In[ ]:





# In[ ]:





# In[348]:


#pie plot


# In[349]:


data.head()


# In[350]:


data['day'].value_counts()


# In[357]:


plt.pie(data['day'].value_counts(),labels=['Sat','Sun','Thu','Fri'],radius=2,autopct='%0.2f%%')

#remember the names in the list of labels are to be passed in decending order always.


# In[360]:


plt.pie(data['day'].value_counts(),labels=['Sat','Sun','Thu','Fri'],radius=2,autopct='%0.2f%%',explode=[0.0,0.0,0.0,0.5])


# In[ ]:





# In[ ]:





# # BOX PLOT

# In[361]:


data.head()


# In[362]:


sns.boxplot(x='time',y='total_bill',data=data)


# In[ ]:





# In[364]:


data1={'state':['Karnataka','Maharastra','Goa','Assam'],'capital':['banglore','mumbai','panaji','guwahati']}


# In[365]:


a=pd.DataFrame(data1)


# In[366]:


a


# In[367]:


language={'banglore':'kannada','mumbai':'marathi','panaji':'goanese','ghuwati':'assamese'}


# In[368]:


a


# In[369]:


a['language']=a['capital'].map(language)


# In[370]:


a


# In[ ]:





# In[ ]:


data={'Name':['Anne','Bobby','James','Lewis','Ross','Cathrine','Anne','Bobby','Jack','Alisa']}

