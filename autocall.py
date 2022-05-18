import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import *
from joblib import Parallel, delayed
from sympy import *
import math
import copy
from scipy.stats import norm

class AutoCall():
    def __init__(self, basic_para, mc_para):
        self.mc_parameter = copy.deepcopy(mc_para)  # 蒙特卡洛参数
        self.basic_parameter = copy.deepcopy(basic_para)  # 基础参数
        self.stock_path_1 = pd.DataFrame()  # 用于保存股票价格路径的dataframe
        self.stock_path_2 = pd.DataFrame()  # 用于保存股票价格路径的dataframe
        self.cum_rtn_1 = pd.DataFrame()  # 用于保存累计收益率路径的dataframe
        self.cum_rtn_2 = pd.DataFrame()  # 用于保存累计收益率路径的dataframe
        self.cum_rtn_1_copy = pd.DataFrame()  # 用于保存累计收益率路径的dataframe
        self.cum_rtn_2_copy = pd.DataFrame()  # 用于保存累计收益率路径的dataframe
        self.random_r_cum_1 = np.zeros([self.mc_parameter['N'], self.mc_parameter['simulations']])
        self.random_r_cum_2 = np.zeros([self.mc_parameter['N'], self.mc_parameter['simulations']])
        self.return_lst = []
        self.ob_copy = []
        self.ob_copy.extend(self.basic_parameter['observe_point'])
        np.random.seed(10)

    def monte_carlo(self):
        N = self.mc_parameter['N']
        simulations = self.mc_parameter['simulations']
        vol = self.mc_parameter['vol']
        dt = self.mc_parameter['dt']
        mu = self.mc_parameter['mu']
        # random_u=np.random.rand(N, simulations)
        # random_r=norm.ppf(random_u,loc=0,scale=1)
        random_r = np.random.normal(0, 1, (N, simulations))
        panduan_up = (0.1 - mu * dt) / (vol * np.sqrt(dt))  # 将收益率序列用涨跌幅规范
        panduan_low = (-0.1 - mu * dt) / (vol * np.sqrt(dt))
        random_r[np.where(random_r > panduan_up)] = panduan_up
        random_r[np.where(random_r < panduan_low)] = panduan_low
        random_r_cum_1 = np.cumprod(np.exp(mu * dt + vol * np.sqrt(dt) * random_r), axis=0)
        random_r_cum_2 = np.cumprod(np.exp(mu * dt + vol * np.sqrt(dt) * -random_r), axis=0)
        # 加上第一日收益率
        head = np.array([[1] * simulations])
        random_r_cum_1 = np.vstack((head, random_r_cum_1))
        random_r_cum_2 = np.vstack((head, random_r_cum_2))
        self.random_r_cum_1 = random_r_cum_1
        self.random_r_cum_2 = random_r_cum_2
        df1 = pd.DataFrame(random_r_cum_1)
        df2 = pd.DataFrame(random_r_cum_2)
        self.cum_rtn_1 = df1.copy(deep=True)
        self.cum_rtn_2 = df2.copy(deep=True)
        self.cum_rtn_1_copy = df1.copy(deep=True)
        self.cum_rtn_2_copy = df2.copy(deep=True)

    def get_stock_path(self, s0):
        self.stock_path_1 = s0 * self.cum_rtn_1
        self.stock_path_2 = s0 * self.cum_rtn_2

    def spath_expect_payoff(self, spath):
        '''
        参数准备
        '''
        s0 = self.basic_parameter['s0']
        r = self.mc_parameter['r']
        N = self.mc_parameter['N']
        LN=self.mc_parameter['LN']
        dt = self.mc_parameter['dt']
        dividend = self.basic_parameter['dividend']
        coupon_rate = self.basic_parameter['coupon_rate']
        T = self.mc_parameter['T']
        F = self.basic_parameter['F']
        discount_factor = np.exp(-r * N * dt)
        simulations = self.mc_parameter['simulations']
        ki = self.basic_parameter['ki']
        ko = self.basic_parameter['ko']
        k=self.basic_parameter['k']
        #######################################################################
        ob = self.basic_parameter['observe_point']
        spath_value = spath.values
        path_ind = spath.index.tolist()
        #        print(type(path_value))
        path_col = [i for i in range(simulations)]
        #        ob_point_price = spath_value[ob]
        ob_point_price = spath.loc[ob].values
        #        print(ob)
        path_number_calculated=0 #计算payoff时将路径进行分类，该变量统计是否路径情况考虑全
        if len(ob) != 0:

            #避险事件发生与否
            ever_nlizard=set(path_col) #没有避险条件时，路径为全集
            payoff_lizard=0
            if 'lizard_number' in self.basic_parameter: #有避险事件
                lizard_i=list(set(path_col)) #全路径
                #spath=spath.loc[:,lizard_i]
                payoff_ever_out_lizard_i=[]#记录每个lizard下的路径的payoff
                for i in range(self.basic_parameter['lizard_number']):
                    ob_lizard=self.basic_parameter['observe_point_lizard'][i]
                    ob_point_price_lizard = spath.loc[ob_lizard].values
                    ever_out_lizard = list(np.where((
                        np.min(ob_point_price_lizard, axis=0) >=self.basic_parameter['lizard_value'][i])
                                                    &(np.max(ob_point_price_lizard, axis=0) <ko[0][0]))[0])
                    #path_ob_point_lizard = spath.loc[ob_lizard, ever_out_lizard]
                    lizard_i_path=list(set(lizard_i) & set(ever_out_lizard)) #发生避险事件的路径
                    out_date_lizard = self.basic_parameter['lizard_time'][i]
                    out_discount_factor = np.exp(-r * out_date_lizard * dt)
                    #计算第i个lizard的payoff
                    payoff_ever_out_lizard = (1 + out_date_lizard / 240 * coupon_rate) * F * out_discount_factor #敲出直接计算利息
                    payoff_ever_out_lizard = payoff_ever_out_lizard*len(lizard_i_path) #路径数量乘以敲出收益
                    payoff_ever_out_lizard_i.append(payoff_ever_out_lizard)
                    path_number_calculated += len(lizard_i_path)
                    lizard_i=list(set(lizard_i) ^ set(lizard_i_path)) #去掉发生前一次避险事件的路径,最后循环完得到的路径为除去敲出和避险事件发生的所有路径
                ever_nlizard=lizard_i
                payoff_lizard=sum(payoff_ever_out_lizard_i)


            if type(ko) is np.ndarray: #处理可变敲出价格
                ever_out=list()
                first_out_date = list()
                ever_out_path = (ob_point_price >=np.transpose(ko))
                n_lizard_path=list(set(range(simulations))&set(ever_nlizard))#去掉避险事件发生的path
                for i in n_lizard_path:
                    a=np.where(ever_out_path[:, i] == True)[0]
                    if len(a)>0:
                        first_out_date.append(a[0]*20+LN)#找出第一个敲出点
                        ever_out.append(i)
                first_out_date=np.array(first_out_date)
                path_number_calculated+=len(first_out_date)
            else:
                ever_out = list(np.where(np.max(ob_point_price, axis=0) >= ko)[0])
                ever_out=list(set(ever_out)&set(ever_nlizard))
                path_ob_point = spath.loc[ob, ever_out]
                path_ob_point = (path_ob_point > ko) #判断每条路径上的观察点是否大于ko，True/False
                first_out_date = path_ob_point.idxmax() #True>False，取第一个True
                path_number_calculated += len(ever_out)


            # 计算payoff
            out_discount_factor = np.exp(-r * first_out_date * dt)
            payoff_ever_out = (first_out_date / 240 * coupon_rate) * F * out_discount_factor #敲出直接计算利息
            payoff_ever_out = payoff_ever_out.sum()
            ever_in = np.where(np.min(spath.loc[self.basic_parameter['observe_point']].values, axis=0) <= ki)[0] #发生敲入的路径
            ever_in= list(set(ever_in)&set(ever_nlizard)) #去掉发生避险事件的路径
            ever_io = list(set(ever_in) & set(ever_out)) #既敲入又敲出的路径
            only_in = list((set(ever_in) ^ set(ever_io))&set(ever_nlizard))  #只敲入的路径
            payoff_only_in = list(map(lambda x: max(x,self.basic_parameter['fl']),spath_value[-1, only_in])) #只敲入时payoff最小为保底价格
            payoff_only_in=np.exp(-r * T) * sum(list(map(lambda x: min(x,k),payoff_only_in))/k*F-F) #只敲入相当于put，payoff最大为本金
            noi = list((set(path_col) ^ set(ever_out) ^ set(only_in))&set(ever_nlizard)) #未敲入或者敲出
            payoff_noi = len(noi) * (dividend*T) * F * np.exp(-r * T)
            payoff_expect = (payoff_ever_out + payoff_noi + payoff_only_in+payoff_lizard) / simulations
            path_number_calculated += len(only_in)
            path_number_calculated += len(noi)
            print('计算的路径总数量为：', path_number_calculated)
            print('模拟的路径总数量为：', simulations)
        else:
            payoff_ever_out = 0
            ever_out = []
            ever_in = np.where(np.min(spath_value, axis=0) <= ki)[0]
            ever_io = list(set(ever_in) & set(ever_out))
            only_in = list(set(ever_in) ^ set(ever_io))
            payoff_only_in = list(map(lambda x: max(x,self.basic_parameter['fl']),spath_value[-1, only_in])) #只敲入时payoff最小为保底价格
            payoff_only_in=np.exp(-r * T) * sum(list(map(lambda x: min(x,k),payoff_only_in)))/k*F #只敲入相当于put，payoff最大为本金
            noi = list(set(path_col) ^ set(ever_out) ^ set(only_in))
            payoff_noi = len(noi) * (1 + dividend*T) * F * np.exp(-r * T)
            payoff_expect = (payoff_ever_out + payoff_noi + payoff_only_in) / simulations
        return {'expect_payoff': payoff_expect}

    def customer_expect_payoff(self):
        sp_1 = self.stock_path_1
        ep_1 = self.spath_expect_payoff(sp_1)['expect_payoff']
        sp_2 = self.stock_path_2
        ep_2 = self.spath_expect_payoff(sp_2)['expect_payoff']
        #######################################################################
        expect_payoff = (ep_1 + ep_2) / 2
        #        print('ep_1:',ep_1,'ep_2:',ep_2,'expect_payoff',expect_payoff)
        return {'expect_payoff': expect_payoff}

    def present_value(self):
        present_value=self.customer_expect_payoff()['expect_payoff']
        return {'present_value': present_value}

    def present_value_coupon_rate(self,cr):

        self.basic_parameter['coupon_rate']=cr
        present_value=self.customer_expect_payoff()['expect_payoff']
        return {'present_value': present_value}

    def get_SN_delta_gamma(self, s, ds, SN, hes): #SN=110/20, hes为目前期权所处状态

        #110交易日后需要重置日期变量,需要判断现在的期权是否已经发生敲出、敲入、或者未发生任何情形
        self.mc_parameter['T'] = self.mc_parameter['N']/240-SN/240
        self.mc_parameter['LN'] = 0  #已经过了锁定期
        self.mc_parameter['N'] =  self.mc_parameter['N'] - SN
        self.basic_parameter['observe_point']=np.array(self.basic_parameter['observe_point'])
        self.basic_parameter['observe_point_copy'] = np.array(self.basic_parameter['observe_point_copy'])
        ob_index=np.array(self.basic_parameter['observe_point'] > 110)
        self.basic_parameter['ko'] = self.basic_parameter['ko'][:,ob_index]
        self.basic_parameter['observe_point'] = self.basic_parameter['observe_point'][np.array(self.basic_parameter['observe_point'] > 110)]-110
        self.basic_parameter['observe_point_copy'] = self.basic_parameter['observe_point_copy'][
                                                    np.array(self.basic_parameter['observe_point_copy'] > 110)] - 110

        self.monte_carlo()

        if hes == 'noi':
            self.get_stock_path(s)
            p = self.customer_expect_payoff()['expect_payoff']
            su = s + ds
            self.get_stock_path(su)
            pup = self.customer_expect_payoff()['expect_payoff']
            sd = s - ds
            self.get_stock_path(sd)
            pdown = self.customer_expect_payoff()['expect_payoff']
        elif hes == 'ever out': #敲出期权价值不会发生变化
            su = s + ds
            sd = s - ds
            pup=0
            pdown=0
            p=0
        elif hes == 'ever in':
            self.basic_parameter['ki']=self.basic_parameter['s0']*100 #将ki设成较大值，保证一定敲入发生
            self.get_stock_path(s)
            p = self.customer_expect_payoff()['expect_payoff']
            su = s + ds
            self.get_stock_path(su)
            pup = self.customer_expect_payoff()['expect_payoff']
            sd = s - ds
            self.get_stock_path(sd)
            pdown = self.customer_expect_payoff()['expect_payoff']

        delta = (pup - pdown) / (2 * ds)
        gamma = (pup + pdown - 2 * p) / (ds ** 2)

        return {'delta': delta/self.basic_parameter['F']*self.basic_parameter['s0'],'gamma': gamma, 'su': su, 'sd': sd, 'pup': pup, 'pdown': pdown}

    def get_vega(self, vol,dv):

        #calculate pv vol up
        self.mc_parameter['vol'] = vol+dv
        self.mc_parameter['mu'] = self.mc_parameter['r'] - self.mc_parameter['q'] - 0.5 * self.mc_parameter['vol'] * self.mc_parameter['vol']

        self.monte_carlo()
        self.get_stock_path(self.basic_parameter['s0'])
        present_value_vol_up = self.present_value()['present_value']

        # calculate pv vol down
        self.mc_parameter['vol'] = vol-dv
        self.mc_parameter['mu'] = self.mc_parameter['r'] - self.mc_parameter['q'] - 0.5 * self.mc_parameter['vol'] * self.mc_parameter['vol']

        self.monte_carlo()
        self.get_stock_path(self.basic_parameter['s0'])
        present_value_vol_down = self.present_value()['present_value']

        vega = (present_value_vol_up- present_value_vol_down) / (2*dv)
        return {'vega': vega, 'vu': vol+dv, 'vd': vol-dv, 'pup': present_value_vol_up, 'pdown': present_value_vol_down}

