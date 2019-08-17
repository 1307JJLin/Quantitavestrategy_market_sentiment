# 例：5交易日滚动值求和，使用DataFrame.rolling(window)计算时间窗口数据
import numpy as np
import pandas as pd
data=pd.read_excel('D:\\量化实战\\Day10\\情绪代理变量.xlsx',sheet_name=0)
data['滚动值求和']=data['成交量'].rolling(5).sum()
data_volprice=pd.read_excel('D:\\量化实战\\Day10\\情绪代理变量.xlsx',sheet_name=0,index_col=0).iloc[:,:2]
data_volprice['滚动值求和']=data_volprice['成交量'].rolling(5).sum()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
x=data_volprice.index
y=data_volprice['沪深300A']
y1=data_volprice['滚动值求和']

fig=plt.figure(figsize=(10,7))
ax1=fig.add_subplot(111)
ax1.plot(x,y1,color='#6699CC',linewidth=2,label='滚动值求和')
ax2=ax1.twinx()
ax2.plot(x,y,color='r',linestyle='--',linewidth=2,label='沪深300A')

ax1.set_title('成交量滚动值求和和沪深300A指数',fontsize=18)
ax1.set_ylabel('滚动值求和',fontsize=15)
ax2.set_ylabel('沪深300A',fontsize=15)
#ax1.set_xticklabels(list(range(2009,2018)),rotation=45)
fig.legend(loc=1)
plt.savefig("D:/成交量滚动值求和和沪深300A指数.png")


data_hand=pd.read_excel('D:\\量化实战\\Day10\\情绪代理变量.xlsx',sheet_name=1,index_col=0).iloc[:,:2]
data_hand['滚动值求和']=data_hand['换手率'].rolling(5).sum()
y2=data_hand['滚动值求和']

fig=plt.figure(figsize=(10,7))
ax1=fig.add_subplot(111)
ax1.plot(x,y2,color='#6699CC',linewidth=2,label='滚动值求和')
ax2=ax1.twinx()
ax2.plot(x,y,color='r',linestyle='--',linewidth=2,label='沪深300A')

ax1.set_title('换手率滚动值求和和沪深300A指数',fontsize=18)
ax1.set_ylabel('滚动值求和',fontsize=15)
ax2.set_ylabel('沪深300A',fontsize=15)
#ax1.set_xticklabels(list(range(2009,2018)),rotation=
fig.legend(loc=1)
plt.savefig("D:/换手率滚动值求和和沪深300A指数.png")

data_pe=pd.read_excel('D:\\量化实战\\Day10\\情绪代理变量.xlsx',sheet_name=2,index_col=0).iloc[:,:2]
data_pe['滚动值求和']=data_pe['PE'].rolling(5).sum()
y3=data_pe['滚动值求和']

fig=plt.figure(figsize=(10,7))
ax1=fig.add_subplot(111)
ax1.plot(x,y3,color='#6699CC',linewidth=2,label='PE')
ax2=ax1.twinx()
ax2.plot(x,y,color='r',linestyle='--',linewidth=2,label='沪深300A')

ax1.set_title('PE和沪深300A指数',fontsize=18)
ax1.set_ylabel('PE',fontsize=15)
ax2.set_ylabel('沪深300A',fontsize=15)
fig.legend(loc=1)
#ax1.set_xticklabels(list(range(2009,2018)),rotation=45)
plt.savefig("D:/PE和沪深300A指数.png")
#plt.show()


data_pb=pd.read_excel('D:\\量化实战\\Day10\\情绪代理变量.xlsx',sheet_name=3,index_col=0).iloc[:,:2]
data_pb['滚动值求和']=data_pb['PB'].rolling(5).sum()
y4=data_pb['PB']

fig=plt.figure(figsize=(10,7))
ax1=fig.add_subplot(111)
ax1.plot(x,y4,color='#6699CC',linewidth=2,label='滚动值求和')
ax2=ax1.twinx()
ax2.plot(x,y,color='r',linestyle='--',linewidth=2,label='沪深300A')

ax1.set_title('PB和沪深300A指数',fontsize=18)
ax1.set_ylabel('PB',fontsize=15)
ax2.set_ylabel('沪深300A',fontsize=15)
fig.legend(loc=1)
#ax1.set_xticklabels(list(range(2009,2018)),rotation=45)
#plt.show()
plt.savefig("D:/PB和沪深300A指数.png")


data=pd.DataFrame(index=data_pe.index)
data['成交金额滚动值求和']=data_volprice['滚动值求和']
data['换手率滚动值求和']=data_hand['滚动值求和']
data['PE']=data_pe['滚动值求和']
data['PB']=data_pb['滚动值求和']
data=data.dropna()
data.corr()

#PCA
def f(s):
    miu=s.mean()
    std=s.std()
    s=(s-miu)/std
    return s

for i in data.columns:
    data[i]=f(data[i])
data[:5]
# 主成分分析
from sklearn.decomposition import PCA
X = np.array(data)
pca=PCA(1) #保留1个主成分；如要保留所有成分，括号中数字省略
pca.fit(X)
print(pca.transform(X))

print('贡献率：',pca.explained_variance_ratio_) #返回各个成分各自的方差百分比(也称贡献率）

proxy=pd.read_excel('D:\\WeChat\\WeChat Files\\JRBTGR1307\\FileStorage\\File\\2019-08\\data_part1.xlsx')
proxy['成交金额滚动']=proxy['成交金额（合计）'].rolling(5).sum()
proxy['换手率滚动']=proxy['换手率（整体法）'].rolling(5).sum()

#实战
comp=pd.read_excel('D:\\WeChat\\WeChat Files\\JRBTGR1307\\FileStorage\\File\\2019-08\\data_part2.xlsx')
stock=pd.read_excel('D:\\WeChat\\WeChat Files\\JRBTGR1307\\FileStorage\\File\\2019-08\\data_part2.xlsx',sheet_name=1)
stock=stock[:-2]
index=pd.read_excel('D:\\WeChat\\WeChat Files\\JRBTGR1307\\FileStorage\\File\\2019-08\\data_part2.xlsx',sheet_name=2)
index=index[:-2]
comp['是否停牌'].value_counts()

#剔除在每个月底交易日，停牌，或者涨跌停的股票数据
comp= comp[((comp['是否停牌']=='正常交易') | (comp['是否停牌']=='复牌')) & (comp['是否涨停'] =='否') & (comp['是否跌停']=='否')]

#个股收益率数据处理
stock.reset_index(inplace=True)
stock=pd.melt(stock,id_vars='index')
stock.columns=['date','code','ret']

#指数收益率处理
index.reset_index(inplace=True)
index.columns=['date','mktret']
index[:5]

#合并生成超额收益
df=pd.merge(stock,index,on='date')
df['excess']=df['ret']-df['mktret']
df[:5]

#每月末进行PCA，找到月末日期
month=df[['date']].drop_duplicates()
month['year']=[i.year for i in month['date']]
month['month']=[i.month for i in month['date']]
month=month.groupby(['year','month'])[['date']].max().reset_index()

import statsmodels.api as sm
#利用过去500个交易日数据，计算滚动情绪指数
sentindex=pd.DataFrame()
for i in set(month['date']):
    temp=proxy[proxy.index<=i][-500:]
    X=temp[['成交金额滚动', '换手率滚动', '市盈率PE（整体法）', '市净率PB（整体法）']]
    x=(X-X.mean())/X.std()
    pca=PCA(n_components=3)
    pca.fit(x)
    x_new=pd.DataFrame(pca.transform(x))[[0]]

    x.reset_index(inplace=True)
    x_new=pd.concat([x,x_new],axis=1)
    x_new['年份']=[i.year for i in x_new['index']]
    x_new['月份'] = [i.month for i in x_new['index']]
    x_new['senti']=x_new[0]
    x_new=x_new[(x_new['年份']==i.year) & (x_new['月份']==i.month)][['index','senti']]
    x_new.columns=['date','senti']
    sentindex=sentindex.append(x_new,ignore_index=True)

df=pd.merge(df,sentindex,on='date')
#合并，准备好回归所需要的数据
res=[]
from tqdm import tqdm
for i in tqdm(range(len(comp))):
    reg=comp.iloc[i]
    temp=df[df.code==reg['成份代码']]
    temp=temp[temp.date<=reg['截止日期']][-250:].dropna()
    Y=temp['ret']
    X=temp[['mktret','senti']]
    model=sm.OLS(Y,sm.add_constant(X))
    results=model.fit()
    res.append(results.params['senti'])

comp['sentifactor']=res

comp_s=pd.DataFrame()
for i in set(comp['截止日期']):
    temp=comp[comp['截止日期']==i]
    temp['p']=pd.qcut(temp['sentifactor'],q=5,labels=False)
    comp_s=comp_s.append(temp)

#准备月收益率
df['year']=[i.year for i in df['date']]
df['month']=[i.month for i in df['date']]
df['day']=[i.day for i in df['date']]
df['culret']=df.groupby(['code','year','month'])['excess'].apply(lambda x:((1+x/100).cumprod()-1)*100)
mon=df.groupby(['year','month','code']).apply(lambda x:x.iloc[-1])[['date','culret']].reset_index()[['date','code','culret']]

retstat=pd.merge(comp_s,mon,left_on=['截止日期','成份代码'],right_on=['date','code'])
retstat=retstat.groupby(['date','p'])[['culret']].mean().reset_index()

res=[]
for i in range(5):
    temp=retstat[retstat['p']==i].sort_values(by='date')
    res.append(((1+temp['culret']/100).cumprod()-1)*100)

plt.figure(figsize=(10,7))
plt.plot(res[0],color='r',label='情绪化指数I级')
plt.plot(res[1],color='b',label='情绪化指数II级')
plt.plot(res[2],color='k',label='情绪化指数III级')
plt.plot(res[3],color='g',label='情绪化指数IV级')
plt.plot(res[4],color='y',label='情绪化指数V级')
plt.xlabel('时间点')
plt.ylabel('收益率(%)')
plt.title('各组股票的收益率')
plt.legend(loc=1)
plt.savefig("D:/策略收益.png")
#plt.show()