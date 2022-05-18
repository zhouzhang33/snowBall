import numpy as np


def read_trade(trade):
    # 定价参数
    mc_para = {}
    mc_para['r'] = trade['无风险利率'].values[0]
    mc_para['q'] = trade['贴水率'].values[0]
    mc_para['vol'] = trade['波动率'].values[0]
    mc_para['T'] = trade['周期（月）'].values[0] / 12
    mc_para['LN'] = 20 * trade['锁定期（月）'].values[0]  # 锁定期
    mc_para['N'] = 20 * trade['周期（月）'].values[0]+1  # 观察期
    mc_para['simulations'] = 5000
    mc_para['dt'] = 1 / 240  # dt为一个月
    mc_para['mu'] = mc_para['r'] - mc_para['q'] - 0.5 * mc_para['vol'] * mc_para['vol']

    # 交易参数
    basic_para = {}
    basic_para['s0'] = trade['期初价'].values[0]
    basic_para['ki'] = trade['敲入价'].values[0]
    basic_para['ko'] = trade['敲出价'].values[0]
    basic_para['k'] = trade['执行价'].values[0]
    basic_para['fl'] = trade['保底价格'].values[0]
    basic_para['coupon_rate'] = trade['票息'].values[0]  # 敲出票息（年化）
    basic_para['dividend'] = basic_para['coupon_rate']  # 红利票息（年化）=敲出票息
    basic_para['target'] = trade['Target'].values[0]
    basic_para['ko_rate'] = 0
    basic_para['lock_period'] = 20  # 一个月
    basic_para['period'] = 20  # 一个月
    if trade.columns.__contains__('敲出递减值'):
        basic_para['ko_rate'] = trade['敲出递减值'].values[0]
        basic_para['ko'] = np.tile([(basic_para['ko'] - (k-mc_para['LN']) * basic_para['ko_rate'] / basic_para['period']) for k in
                                    range(mc_para['LN'], mc_para['N'], basic_para['period'])],
                                   (mc_para['simulations'], 1))
    if trade.columns.__contains__('不敲入敲票息'):
        basic_para['dividend'] = trade['不敲入敲票息'].values[0]
    basic_para['observe_point'] = [k for k in range(mc_para['LN'], mc_para['N'], basic_para['period'])]
    basic_para['observe_point_copy'] = [k for k in range(mc_para['LN'], mc_para['N'], 1)]
    basic_para['calender'] = [k for k in range(mc_para['N'])]
    basic_para['F'] = 1000000  # 面值假设为1000000
    basic_para['path_count'] = 0
    if trade.columns.__contains__('避险事件个数'):
        basic_para['lizard_number'] = trade['避险事件个数'].values[0]
        basic_para['lizard_time'] = np.zeros(trade['避险事件个数'].values[0])
        basic_para['lizard_value'] = np.zeros(trade['避险事件个数'].values[0])
        basic_para['observe_point_lizard'] = [[] for k in range(trade['避险事件个数'].values[0])]
        basic_para['lizard_time'][0] = trade['避险时点' + str(1) + '（月）'].values[0] * 20  # 一个月20天
        basic_para['lizard_value'][0] = trade['避险价格' + str(1)].values[0]
        basic_para['observe_point_lizard'][0] = [k for k in
                                                 range(0, int(basic_para['lizard_time'][0]), 1)]  # 期初至该月观察日之前每一日
        for i in range(1, trade['避险事件个数'].values[0]):
            basic_para['lizard_time'][i] = trade['避险时点' + str(i + 1) + '（月）'].values[0] * 20  # 一个月20天
            basic_para['lizard_value'][i] = trade['避险价格' + str(i + 1)].values[0]
            basic_para['observe_point_lizard'][i] = [k for k in range(0, int(basic_para['lizard_time'][i]),
                                                                      1)]  # 期初至该月观察日之前每一日

    return {'mc_para':mc_para,'basic_para': basic_para}