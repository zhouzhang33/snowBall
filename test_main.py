import snowball_para as sp
import pandas as pd
import autocall
from time import *
import numpy as np
from sympy import *
import math
from prettytable import PrettyTable


#读入交易文件
trade = pd.read_excel\
    (r'C:\Users\RL882PV\PycharmProjects\SnowBall\tradeData_2.xlsx', sheet_name='Sheet1',header=0)

para_dict=sp.read_trade(trade)
mc_para=para_dict['mc_para'] #蒙特卡洛参数
basic_para=para_dict['basic_para'] #基础交易参数

t1 = time()
atc = autocall.AutoCall(basic_para, mc_para) #构建雪球期权对象
t2 = time()
atc.monte_carlo()
t3 = time()
s0 = atc.basic_parameter['s0'] #期初价格
t4 = time()
atc.get_stock_path(s0) #生成标的运动路径
t5 = time()
cum_rtn_1 = atc.cum_rtn_1
cum_rtn_2 = atc.cum_rtn_2
t6 = time()
spath_1 = atc.stock_path_1
spath_2 = atc.stock_path_2
t7 = time()
present_value = atc.present_value() #计算期权价值
t8 = time()
x = symbols('x') #最优化求解票息率使得期权价值为0
eq=atc.present_value_coupon_rate(x)['present_value']
target=0
a = nsolve(eq - target, 0)
t9 = time()
#110个交易日后，计算delta和gamma时，需判断110天时期权的状态，已经敲出/已经敲入/未发生，分别对应码值为ever out/ever in/noi
price_list=[k for k in range(90,111)]
delta_gamma_list=[]
hes='ever in'
delta_gamma_table=PrettyTable(['第110日期初价格\n 期权状态为'+hes,'delta','gamma','spot price up','pv up','spot price down','pv down'])
for x in price_list:
    atc=autocall.AutoCall(basic_para, mc_para)
    delta_gamma=atc.get_SN_delta_gamma(x, 0.01*x, 110,hes)
    delta_gamma_list.append(delta_gamma)
    delta_gamma_table.add_row([x, delta_gamma['delta'], delta_gamma['gamma'],
                               delta_gamma['su'],delta_gamma['pup'],
                               delta_gamma['sd'],delta_gamma['pdown']])
t10 = time()
vol_list=[k*0.01 for k in range(10,21)]
vega_list=[]
vega_table=PrettyTable(['波动率','vega','vol up','pv up','vol down','pv down'])
for x in vol_list:
    atc=autocall.AutoCall(basic_para, mc_para)
    vega=atc.get_vega(x, 0.001)
    vega_list.append(vega)
    vega_table.add_row([x, vega['vega'],
                               vega['vu'],vega['pup'],
                               vega['vd'],vega['pdown']])
t11 = time()

print('构建对象耗时：', t2 - t1)
print('蒙特卡洛耗时：', t3 - t2)
print('取单个参数耗时：', t4 - t3)
print('获取股票路径耗时：', t5 - t4)
print('获取收益率矩阵耗时：', t6 - t5)
print('获取股票矩阵耗时：', t7 - t6)
print('获取客户收益耗时：', t8 - t7)
print('计算delta&gamma耗时：', t10 - t9)
print(delta_gamma_table)
print(vega_table)
print('蒙特卡洛模拟次数:', mc_para['simulations']*2)
print('计算PV耗时:', t8-t1)
print('期权价值:', present_value)
print('PV=0的票息为 ',a)

print('正在输出路径......')
path=pd.concat([spath_1,spath_2],axis=1)
path.to_csv(r'C:\Users\RL882PV\PycharmProjects\SnowBall_Deliver\path.csv')		# 路径写入Excel文件
print('输出完毕')


