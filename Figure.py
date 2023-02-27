# Ma Yibo
# Email:mayibo_hyh@163.com

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


########## Figure 1 ##########


# NanChang Network Capacity


def func(x, a, b, c, d):
    return a * x ** 2 + b * x + c * x ** 3 + d


Cell_Capacity_4G = np.load('data/data_old/Cell_Capacity_4G.npy', allow_pickle=True).item()
Cell_Capacity_5G = np.load('data/data_old/Cell_Capacity_5G.npy', allow_pickle=True).item()

Capacity_4G = 0
for Cell_ID in Cell_Capacity_4G:
    Capacity_4G = Capacity_4G + Cell_Capacity_4G[Cell_ID]['Capacity']  # (千字节)
Capacity_4G = Capacity_4G / 1024 / 1024 / 1024 / 1024 * 96

Capacity_5G = 0
for Cell_ID in Cell_Capacity_5G:
    Capacity_5G = Capacity_5G + Cell_Capacity_5G[Cell_ID]['Capacity']  # (千字节)
Capacity_5G = Capacity_5G / 1024 / 1024 / 1024 / 1024 * 96

Date_Before5G = np.linspace(0, 7, 8)
Date_After5G = np.linspace(7, 23, 17)
Date_Future = np.linspace(23, 42, 20)

Network_Capacity_Before5G = []
Network_Capacity_After5G = []
Network_Capacity_Future = []
for index in range(len(Date_Before5G)):
    Network_Capacity_Before5G.append(Capacity_4G)
for index in range(len(Date_After5G)):
    Network_Capacity_After5G.append(Capacity_4G + Capacity_5G)
for index in range(len(Date_Future)):
    Network_Capacity_Future.append(Capacity_4G + Capacity_5G)

# NanChang Network Traffic

Network_Traffic_Before5G = []
for index in range(len(Date_Before5G)):
    Network_Traffic_Before5G.append(1.21042719)
Network_Traffic_Before5G = np.array(Network_Traffic_Before5G)

Network_Traffic_After5G_4G = [1.15357101, 1.312790426, 1.399133649, 1.464940381, 1.513061395, 1.524572747, 1.53056248, \
                              1.533335786, 1.473622746, 1.431067853, 1.470064229, 1.498515583, 1.461390619, 1.415919433,
                              1.387327911, 1.382028557, 1.37190577]
Network_Traffic_After5G_5G = [0.056856181, 0.062154632, 0.136420432, 0.167308413, 0.195333782, 0.223651352, 0.231856969,
                              0.240518264, \
                              0.351768637, 0.382075861, 0.386963403, 0.396250525, 0.439388366, 0.488138965, 0.528031143,
                              0.540033872, 0.57771561]
Network_Traffic_After5G = np.array(Network_Traffic_After5G_4G) + np.array(Network_Traffic_After5G_5G)

popt, pcov = curve_fit(func, np.array(Date_After5G), np.array(Network_Traffic_After5G))
Network_Traffic_Future = func(np.array(Date_Future), popt[0], popt[1], popt[2], popt[3])
Network_Traffic_Future[0] = Network_Traffic_After5G[-1]

Network_Capacity_Before5G_log = [np.log(value) for value in Network_Capacity_Before5G]
Network_Capacity_After5G_log = [np.log(value) for value in Network_Capacity_After5G]
Network_Capacity_Future_log = [np.log(value) for value in Network_Capacity_Future]

Network_Traffic_After5G_4G_log = []
Network_Traffic_After5G_5G_log = []
for index in range(len(Network_Traffic_After5G_4G)):
    Network_Traffic_After5G_log = np.log(Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])
    Network_Traffic_After5G_4G_log.append(Network_Traffic_After5G_log * (Network_Traffic_After5G_4G[index] / (
            Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])))
    Network_Traffic_After5G_5G_log.append(Network_Traffic_After5G_log * (Network_Traffic_After5G_5G[index] / (
            Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])))

Network_Traffic_Before5G_log = [np.log(value) for value in Network_Traffic_Before5G]
Network_Traffic_Future_log = [np.log(value) for value in Network_Traffic_Future]

# NanChang Network Energy

K = 0.6131
K_std = 0.0061
K1 = 0.5261
K1_std = 0.053
K2 = 0.4387
K2_std = 0.0044
K3 = 0.2856
K3_std = 0.0029
gama = 0.9527
gama_std = 0.0515
Network_Energy_error_Before5G = 0.00758
Network_Energy_error_After5G = 0.00853
Network_Energy_error_Future = 0.00853
Network_Energy_Before5G = (np.array(Network_Traffic_Before5G) / np.array(
    Network_Capacity_Before5G) * 0.454 + 0.546) * 43160884 / 1_000_000 * 24
Network_Energy_After5G = (np.array(Network_Traffic_After5G) / np.array(Network_Capacity_After5G) * (1 - K) + K) * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future = (np.array(Network_Traffic_Future) / np.array(Network_Capacity_Future) * (1 - K) + K) * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Before5G_low = Network_Energy_Before5G - Network_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Network_Energy_After5G_low = Network_Energy_After5G - K_std * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future_low = Network_Energy_Future - K_std * (17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Before5G_high = Network_Energy_Before5G + Network_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Network_Energy_After5G_high = Network_Energy_After5G + K_std * (17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future_high = Network_Energy_Future + K_std * (17675608 + 43160884) / 1_000_000 * 24

# NanChang Network Carbon

Network_Carbon_error = 0.0515

Network_Carbon_Before5G = (Network_Energy_Before5G / 24 * gama) * 24
Network_Carbon_After5G = (Network_Energy_After5G / 24 * gama) * 24
Network_Carbon_Future = (Network_Energy_Future / 24 * gama) * 24
Network_Carbon_Before5G_low = (Network_Energy_Before5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_After5G_low = (Network_Energy_After5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_Future_low = (Network_Energy_Future_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_Before5G_high = (Network_Energy_Before5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Network_Carbon_After5G_high = (Network_Energy_After5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Network_Carbon_Future_high = (Network_Energy_Future_high / 24 * gama) * 24 + Network_Carbon_error * 24

# NanChang Network Energy Efficiency

Network_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G
Network_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / Network_Energy_After5G
Network_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / Network_Energy_Future
Network_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_low
Network_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / Network_Energy_After5G_low
Network_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / Network_Energy_Future_low
Network_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_high
Network_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / Network_Energy_After5G_high
Network_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / Network_Energy_Future_high

# NanChang Network Carbon Efficiency

Network_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(Network_Carbon_Before5G)
Network_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(Network_Carbon_After5G)
Network_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future)
Network_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Network_Carbon_Before5G_low)
Network_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Network_Carbon_Before5G_high)
Network_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(Network_Carbon_After5G_low)
Network_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(
    Network_Carbon_After5G_high)
Network_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future_low)
Network_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future_high)

# NanChang Additional Carbon Emissions
index = 0
Network_Wasted_Carbon = 0
Network_Wasted_Carbon_low = 0
Network_Wasted_Carbon_high = 0
while Network_Carbon_Efficiency_Before5G[-1] - Network_Carbon_Efficiency_After5G[index] > 0 and index < len(
        Network_Carbon_Efficiency_After5G) - 1:
    item = Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_After5G[index]) - \
           Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G[-1])
    item_low = Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_After5G_low[index]) - \
               Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G_low[-1])
    item_high = Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_After5G_high[index]) - \
                Network_Traffic_After5G[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G_high[-1])

    assert item > 0
    Network_Wasted_Carbon = Network_Wasted_Carbon + item
    Network_Wasted_Carbon_low = Network_Wasted_Carbon_low + item_low
    Network_Wasted_Carbon_high = Network_Wasted_Carbon_high + item_high
    index = index + 1
index = 0

while Network_Carbon_Efficiency_Before5G[-1] - Network_Carbon_Efficiency_Future[index] > 0 and index < len(
        Network_Carbon_Efficiency_Future) - 1:
    item = Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Future[index]) - \
           Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G[-1])
    item_low = Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Future_low[index]) - \
               Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G_low[-1])
    item_high = Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Future_high[index]) - \
                Network_Traffic_Future[index] * 1024 * 30 / (Network_Carbon_Efficiency_Before5G_high[-1])
    assert item > 0
    Network_Wasted_Carbon = Network_Wasted_Carbon + item
    Network_Wasted_Carbon_low = Network_Wasted_Carbon_low + item_low
    Network_Wasted_Carbon_high = Network_Wasted_Carbon_high + item_high
    index = index + 1

# Plot a

cloud = plt.figure(figsize=(12, 6))
axes = plt.subplot(111)
p1 = axes.plot(Date_Before5G, Network_Capacity_Before5G_log, color='#346fa9', linestyle='-', linewidth=3)
axes.plot([Date_Before5G[-1], Date_After5G[0]], [Network_Capacity_Before5G_log[-1], Network_Capacity_After5G_log[0]],
          color='#346fa9', linestyle='-', linewidth=3)
axes.plot(Date_After5G, Network_Capacity_After5G_log, color='#346fa9', linestyle='-', linewidth=3)
axes.plot(Date_Future, Network_Capacity_Future_log, color='#346fa9', linestyle='-', linewidth=3,
          label='Network Capacity')
axes.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')

axes.stackplot(Date_After5G, Network_Traffic_After5G_4G_log, Network_Traffic_After5G_5G_log,
               labels=['4G Traffic', '5G Traffic'], colors=["#FF4500", "#008000"], alpha=0.6)
axes.stackplot(Date_Before5G, Network_Traffic_Before5G_log, colors=["#FF4500"], alpha=0.6)
# axes.plot(Date_Before5G, Network_Traffic_Before5G, color='#FF4500',linestyle=':', linewidth = 3)
axes.stackplot(Date_Future, Network_Traffic_Future_log, colors=["darkorange"], alpha=0.6,
               labels=['Estimated Network Traffic'])
# axes.plot(Date_Future, Network_Traffic_Future, color='darkorange',linestyle=':', linewidth = 3)
axes.spines['bottom'].set_linewidth('1.0')
axes.spines['left'].set_linewidth('1.0')
axes.spines['right'].set_linewidth('1.0')
axes.spines['top'].set_linewidth('1.0')
axes.axvline(Date_After5G[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.axvline(Date_Future[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.set_ylabel('Data Volume ln(PByte) ', fontsize=24)
axes.set_xticks([7, 23, 31, 42])
axes.tick_params(axis="y", labelsize=24)
axes.set_xticklabels(['2021-01 (Launching 5G)', '2022-05', '2023-1', '2024-12'], fontsize=24)
axes.set_xlim(0, 31)
axes.set_ylim(0, 1.2 * np.max(Network_Capacity_Future_log))

axes.spines['bottom'].set_linewidth('1.0')
axes.spines['left'].set_linewidth('1.0')
axes.spines['right'].set_linewidth('1.0')
axes.spines['top'].set_linewidth('1.0')
axes.axvline(Date_After5G[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.axvline(Date_Future[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.set_ylabel('Data Volume ln(PByte) ', fontsize=24)
axes.set_xticks([7, 23, 31, 42])
axes.tick_params(axis="y", labelsize=24)
axes.set_xticklabels(['2021-01 (Launching 5G)', '2022-05', '2023-1', '2024-12'], fontsize=24)
axes.set_xlim(0, 31)
axes.set_ylim(0, 1.2 * np.max(Network_Capacity_Future_log))

cloud.legend(bbox_to_anchor=(0.8, 0.638), fontsize=22)
# plt.legend(fontsize=16)
cloud.savefig(fname=r"fig1/Network Capacity Temporal.pdf", bbox_inches='tight')

# Plot b

cloud = plt.figure(figsize=(12, 6))
axes = plt.subplot(111)

axes.plot(Date_Before5G, Network_Carbon_Before5G, color='#000000', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Before5G, Network_Carbon_Before5G_low, Network_Carbon_Before5G_high, facecolor='#346fa9',
                  alpha=0.5)
axes.plot([Date_Before5G[-1], Date_After5G[0]], [Network_Carbon_Before5G[-1], Network_Carbon_After5G[0]],
          color='#000000', linestyle='-', linewidth=2.4)
axes.plot(Date_After5G, Network_Carbon_After5G, color='#000000', linestyle='-', linewidth=2.4)
axes.fill_between(Date_After5G, Network_Carbon_After5G_low, Network_Carbon_After5G_high, facecolor='#346fa9', alpha=0.5)
P1 = axes.plot(Date_Future, Network_Carbon_Future, color='#000000', linestyle='-', linewidth=2.4,
               label='Network Carbon Emission')
P4 = axes.fill_between(Date_Future, Network_Carbon_Future_low, Network_Carbon_Future_high, facecolor='#346fa9',
                       alpha=0.5, label='Confidence interval')
axes.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')

axes2 = axes.twinx()
P2 = axes2.stackplot(Date_After5G, Network_Traffic_After5G_4G, Network_Traffic_After5G_5G,
                     colors=["#FF4500", "#008000"], labels=['4G Traffic', '5G Traffic'], alpha=0.6)
axes2.stackplot(Date_Before5G, Network_Traffic_Before5G, colors=["#FF4500"], alpha=0.6)
# axes2.plot(Date_Before5G, Network_Traffic_Before5G, color='#FF4500',linestyle=':', linewidth = 5)
P3 = axes2.stackplot(Date_Future, Network_Traffic_Future, colors=["darkorange"], alpha=0.6,
                     labels=['Estimated Network Traffic'])
# axes2.plot(Date_Future, Network_Traffic_Future, color='darkorange',linestyle=':', linewidth = 5)
axes.spines['bottom'].set_linewidth('1.0')
axes.spines['left'].set_linewidth('1.0')
axes.spines['right'].set_linewidth('1.0')
axes.spines['top'].set_linewidth('1.0')
axes.axvline(Date_After5G[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.axvline(Date_Future[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.set_ylabel('Carbon Emission (tCO$_2$)', fontsize=24)
axes2.set_ylabel('Network Traffic (PByte)', fontsize=24)
axes.set_xticks([7, 23, 31, 42])
axes.tick_params(axis="y", labelsize=24)
axes2.tick_params(axis="y", labelsize=24)
axes.set_xticklabels(['2021-01 (Launching 5G)', '2022-05', '2023-1', '2024-12'], fontsize=24)
axes.set_xlim(0, 31)
axes.set_ylim(0, 1.1 * np.max(Network_Carbon_Future))
axes2.set_ylim(0, 10)
cloud.legend((P1[0], P2[0], P2[1], P3[0], P4),
             (
                 "Network Carbon Emission", '4G Traffic', '5G Traffic', 'Estimated Network Traffic',
                 'Confidence interval'),
             bbox_to_anchor=(0.8, 0.7), fontsize=22)
cloud.savefig(r"fig1\Network Carbon Temporal.pdf", format='pdf', bbox_inches='tight')

# Plot c

cloud = plt.figure(figsize=(12, 6))
axes = plt.subplot(111)
axes.plot(Date_Before5G, Network_Carbon_Efficiency_Before5G, color='olivedrab', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Before5G, Network_Carbon_Efficiency_Before5G_low, Network_Carbon_Efficiency_Before5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_After5G[0:8],
          np.concatenate((Network_Carbon_Efficiency_Before5G, Network_Carbon_Efficiency_Before5G))[0:8],
          color='olivedrab', linestyle=':', linewidth=2)
axes.fill_between(Date_After5G[0:8],
                  np.concatenate((Network_Carbon_Efficiency_Before5G, Network_Carbon_Efficiency_Before5G))[0:8],
                  Network_Carbon_Efficiency_After5G[0:8], facecolor='dimgray', hatch='//', edgecolor="w", linewidth=0.3,
                  alpha=0.5)
axes.plot(Date_After5G, Network_Carbon_Efficiency_After5G, color='olivedrab', linestyle='-', linewidth=2.4,
          label='Network Carbon Efficiency')
axes.fill_between(Date_After5G, Network_Carbon_Efficiency_After5G_low, Network_Carbon_Efficiency_After5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_Future, Network_Carbon_Efficiency_Future, color='olivedrab', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Future, Network_Carbon_Efficiency_Future_low, Network_Carbon_Efficiency_Future_high,
                  facecolor='#346fa9', alpha=0.5)
axes.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
axes.spines['bottom'].set_linewidth('1.0')
axes.spines['left'].set_linewidth('1.0')
axes.spines['right'].set_linewidth('1.0')
axes.spines['top'].set_linewidth('1.0')
axes.axvline(Date_After5G[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.axvline(Date_Future[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.set_xticks([7, 23, 31, 42])
axes.set_xticklabels(['2021-01 (Launching 5G)', '2022-05', '2023-1', '2024-12'], fontsize=24)
axes.set_xlim(0, 31)
axes.tick_params(axis="y", labelsize=24)
axes.set_ylabel('Carbon Efficiency (TByte/tCO$_2$)', fontsize=23.5)
axes.set_ylim(0.9 * np.min(Network_Carbon_Efficiency_After5G), 0.3 * np.max(Network_Carbon_Efficiency_Future))
axes.text(Date_After5G[0] + 0.5, 1.6 * Network_Carbon_Efficiency_After5G[0], 'Carbon Efficiency Trap', fontsize=24)
cloud.savefig(r"fig1\Network Carbon Efficiency Temporal.pdf", format='pdf', bbox_inches='tight')

X = [0]
Y = [Network_Wasted_Carbon / 1000]
fig = plt.figure(figsize=(5.5, 4.5), linewidth=1)
plt.bar(X, Y, 0.2, color="steelblue")
plt.errorbar([X[0]], Y, yerr=[[np.abs(Network_Wasted_Carbon_low - Network_Wasted_Carbon) / 1000],
                              [np.abs(Network_Wasted_Carbon_high - Network_Wasted_Carbon) / 1000]], capsize=3,
             elinewidth=2, fmt=' k,')
plt.text(X[0], Y[0] + 2, '{} $\pm$ {}'.format(int(Y[0] * 100) / 100, int(np.abs(
    Network_Wasted_Carbon_low - Network_Wasted_Carbon) / 10) / 100), ha='center', va='bottom', fontsize=24)
plt.xticks([])
plt.xlim(-0.5, 0.5)
plt.ylim(0, 1.4 * Y[0])
plt.yticks(fontsize=23)
plt.xlabel("Additional Carbon Emissions", fontsize=22)
plt.ylabel("Carbon Emission (KtCO$_2$)", fontsize=21.5)
plt.savefig(r"fig1\Additional Carbon Emission.pdf", format='pdf', bbox_inches='tight')


# Province Estimation

def thr_sigma(dataset, n=1.5, method='All'):  # filter outlier
    mean = np.mean(dataset)  # 得到均值
    sigma = np.std(dataset)  # 得到标准差
    if method == 'All':
        remove_idx = np.where(abs(dataset - mean) > n * sigma)
    elif method == 'Min':
        remove_idx = np.where((mean - dataset) > n * sigma)
    elif method == 'Max':
        remove_idx = np.where((dataset - mean) > n * sigma)
    new_data_set = np.delete(dataset, remove_idx)
    return new_data_set


Cell_Capacity_4G = np.load('./data/data_old/Cell_Capacity_4G.npy', allow_pickle=True).item()
Cell_Capacity_5G = np.load('./data/data_old/Cell_Capacity_5G.npy', allow_pickle=True).item()

Capacity_4G_Nanchang = 0
for Cell_ID in Cell_Capacity_4G:
    Capacity_4G_Nanchang = Capacity_4G_Nanchang + Cell_Capacity_4G[Cell_ID]['Capacity']  # (千字节)
Capacity_4G_Nanchang = Capacity_4G_Nanchang / 1024 / 1024 / 1024 / 1024 * 96  # 统计数值是15分钟粒度的 PB 一天的

Capacity_5G_Nanchang = 0
for Cell_ID in Cell_Capacity_5G:
    Capacity_5G_Nanchang = Capacity_5G_Nanchang + Cell_Capacity_5G[Cell_ID]['Capacity']  # (千字节)
Capacity_5G_Nanchang = Capacity_5G_Nanchang / 1024 / 1024 / 1024 / 1024 * 96  # 统计数值是15分钟粒度的 PB 一天的

BS_Capacity_4G = {}
BS_Capacity_5G = {}

for Cell_ID in Cell_Capacity_4G:
    BS_ID = Cell_ID.split('-')[-2]
    if BS_ID not in BS_Capacity_4G:
        BS_Capacity_4G[BS_ID] = 0
    BS_Capacity_4G[BS_ID] = BS_Capacity_4G[BS_ID] + Cell_Capacity_4G[Cell_ID]['Capacity']

for Cell_ID in Cell_Capacity_5G:
    BS_ID = Cell_ID.split('-')[-2]
    if BS_ID not in BS_Capacity_5G:
        BS_Capacity_5G[BS_ID] = 0
    BS_Capacity_5G[BS_ID] = BS_Capacity_5G[BS_ID] + Cell_Capacity_5G[Cell_ID]['Capacity']

BS_Capacity_4G = np.array(list(BS_Capacity_4G.values())) * 96 / 1024 / 1024 / 1024  # TB
BS_Capacity_5G = np.array(list(BS_Capacity_5G.values())) * 96 / 1024 / 1024 / 1024  # TB

BS_Capacity_4G = thr_sigma(BS_Capacity_4G, 2, 'All')
# BS_Capacity_5G = thr_sigma(BS_Capacity_5G, 1, 'Min')
BS_Capacity_5G = thr_sigma(BS_Capacity_5G, 1.5, 'All')

Province_4GBS_Num = {
    'Beijing': 24305,
    'Tianjing': 15009,
    'Hebei': 107949,
    'Shanxi': 64850,
    'Inner Mongolia': 36502,
    'Liaoning': 63647,
    'Jilin': 32089,
    'Heilongjiang': 46940,
    'Shanghai': 25672,
    'Jiangsu': 93128,
    'Zhejiang': 92217,
    'Anhui': 71780,
    'Fujian': 96086,
    'Jiangxi': 71101,
    'Shandong': 117311,
    'Henan': 97658,
    'Hubei': 82828,
    'Hunan': 84584,
    'Guangdong': 124663,
    'Guangxi': 72839,
    'Hainan': 14808,
    'Chongqing': 60704,
    'Sichuan': 149584,
    'Guizhou': 79910,
    'Yunnan': 106092,
    'Tibet': 16502,
    'Shaanxi': 61301,
    'Gansu': 48530,
    'Qinghai': 12547,
    'Ningxia': 11556,
    'Xinjiang': 48386
}

Province_5GBS_Num = {
    'Beijing': 15293,
    'Tianjing': 9451,
    'Hebei': 38758,
    'Shanxi': 26122,
    'Inner Mongolia': 15993,
    'Liaoning': 19745,
    'Jilin': 9590,
    'Heilongjiang': 21158,
    'Shanghai': 24881,
    'Jiangsu': 69458,
    'Zhejiang': 58079,
    'Anhui': 35477,
    'Fujian': 29828,
    'Jiangxi': 37764,
    'Shandong': 58687,
    'Henan': 34639,
    'Hubei': 37536,
    'Hunan': 31804,
    'Guangdong': 90689,
    'Guangxi': 20929,
    'Hainan': 6690,
    'Chongqing': 23969,
    'Sichuan': 35068,
    'Guizhou': 33592,
    'Yunnan': 42795,
    'Tibet': 2880,
    'Shaanxi': 24552,
    'Gansu': 10034,
    'Qinghai': 3246,
    'Ningxia': 5588,
    'Xinjiang': 10964
}

Maximum_Energy_Before5G = {
    'Beijing': 1993544653.139595,
    'Tianjing': 1230530452.1904197,
    'Hebei': 8843046518.03212,
    'Shanxi': 5316567709.046286,
    'Inner Mongolia': 2988920675.197445,
    'Liaoning': 5213012698.141714,
    'Jilin': 2629002643.128604,
    'Heilongjiang': 3845500222.485292,
    'Shanghai': 2104920606.0905983,
    'Jiangsu': 7631332001.653536,
    'Zhejiang': 7554836973.219011,
    'Anhui': 5877636080.5023775,
    'Fujian': 7876870930.89134,
    'Jiangxi': 5823862456.179625,
    'Shandong': 9610034786.656918,
    'Henan': 7998009567.501204,
    'Hubei': 6788052358.9353485,
    'Hunan': 6930748620.535904,
    'Guangdong': 10209012998.452677,
    'Guangxi': 5970495546.0146885,
    'Hainan': 1213092271.8843272,
    'Chongqing': 4971895024.1741905,
    'Sichuan': 12262731774.680155,
    'Guizhou': 6549387072.509599,
    'Yunnan': 8691531573.419168,
    'Tibet': 1350830550.7170897,
    'Shaanxi': 5023425729.02861,
    'Gansu': 3973834621.801835,
    'Qinghai': 1029402539.8226128,
    'Ningxia': 949464627.5129029,
    'Xinjiang': 3963400879.181369
}

Maximum_Energy_std_Before5G = {
    'Beijing': 10977251.89327097,
    'Tianjing': 8603755.638326703,
    'Hebei': 24040034.293462746,
    'Shanxi': 22922108.930907544,
    'Inner Mongolia': 14486288.913480839,
    'Liaoning': 21998697.927930307,
    'Jilin': 11551889.062112242,
    'Heilongjiang': 17996403.7147921,
    'Shanghai': 12671175.23524229,
    'Jiangsu': 32249561.976358794,
    'Zhejiang': 17778992.129216537,
    'Anhui': 15337322.536070423,
    'Fujian': 16925032.410339974,
    'Jiangxi': 18793742.491810277,
    'Shandong': 27796764.45424131,
    'Henan': 18385866.697876953,
    'Hubei': 18130812.4574946,
    'Hunan': 23261119.74919323,
    'Guangdong': 28167118.003367197,
    'Guangxi': 20602534.016784918,
    'Hainan': 11546927.072526578,
    'Chongqing': 20265794.790187772,
    'Sichuan': 24715425.705309086,
    'Guizhou': 22457898.741323926,
    'Yunnan': 31494318.211030316,
    'Tibet': 9921544.959095698,
    'Shaanxi': 13183275.560417412,
    'Gansu': 13371596.382920166,
    'Qinghai': 6583252.130279812,
    'Ningxia': 7425707.297172253,
    'Xinjiang': 15846165.088056348}

Maximum_Energy_After5G = {
    'Beijing': 5062838985.873324,
    'Tianjing': 3123131501.4440494,
    'Hebei': 16622061293.703562,
    'Shanxi': 10552191915.147232,
    'Inner Mongolia': 6197591680.494571,
    'Liaoning': 9178198034.707104,
    'Jilin': 4552400787.430828,
    'Heilongjiang': 8090189680.612218,
    'Shanghai': 7092384810.185564,
    'Jiangsu': 21569017827.466908,
    'Zhejiang': 19208175447.701538,
    'Anhui': 12998578522.033813,
    'Fujian': 13860286632.836065,
    'Jiangxi': 13399182193.82479,
    'Shandong': 21386740432.776672,
    'Henan': 14941571441.697659,
    'Hubei': 14313391553.825205,
    'Hunan': 13307777518.5923,
    'Guangdong': 28416874950.972504,
    'Guangxi': 10164542350.838724,
    'Hainan': 2555489747.409697,
    'Chongqing': 9784166578.878977,
    'Sichuan': 19280982018.65919,
    'Guizhou': 13281318408.359035,
    'Yunnan': 17275925882.765244,
    'Tibet': 1931059034.0735142,
    'Shaanxi': 9945623951.569847,
    'Gansu': 5988131080.429561,
    'Qinghai': 1676420302.0909066,
    'Ningxia': 2070567607.2625527,
    'Xinjiang': 6164105181.0755825
}

Maximum_Energy_std_After5G = {
    'Beijing': 10977251.89327097,
    'Tianjing': 8603755.638326703,
    'Hebei': 24040034.293462746,
    'Shanxi': 22922108.930907544,
    'Inner Mongolia': 14486288.913480839,
    'Liaoning': 21998697.927930307,
    'Jilin': 11551889.062112242,
    'Heilongjiang': 17996403.7147921,
    'Shanghai': 12671175.23524229,
    'Jiangsu': 32249561.976358794,
    'Zhejiang': 17778992.129216537,
    'Anhui': 15337322.536070423,
    'Fujian': 16925032.410339974,
    'Jiangxi': 18793742.491810277,
    'Shandong': 27796764.45424131,
    'Henan': 18385866.697876953,
    'Hubei': 18130812.4574946,
    'Hunan': 23261119.74919323,
    'Guangdong': 28167118.003367197,
    'Guangxi': 20602534.016784918,
    'Hainan': 11546927.072526578,
    'Chongqing': 20265794.790187772,
    'Sichuan': 24715425.705309086,
    'Guizhou': 22457898.741323926,
    'Yunnan': 31494318.211030316,
    'Tibet': 9921544.959095698,
    'Shaanxi': 13183275.560417412,
    'Gansu': 13371596.382920166,
    'Qinghai': 6583252.130279812,
    'Ningxia': 7425707.297172253,
    'Xinjiang': 15846165.088056348}

for P in Maximum_Energy_std_After5G:
    Maximum_Energy_Before5G[P] = Maximum_Energy_Before5G[P] / 1000_000
    Maximum_Energy_std_Before5G[P] = Maximum_Energy_std_Before5G[P] / 1000_000
    Maximum_Energy_After5G[P] = Maximum_Energy_After5G[P] / 1000_000
    Maximum_Energy_std_After5G[P] = Maximum_Energy_std_After5G[P] / 1000_000

Maximum_Capacity_Before5G = {
    'Beijing': 24918.437712492745,
    'Tianjing': 15384.179788635864,
    'Hebei': 110747.51102425208,
    'Shanxi': 66591.06519802062,
    'Inner Mongolia': 37416.503493993674,
    'Liaoning': 65262.6151110287,
    'Jilin': 32978.13449529203,
    'Heilongjiang': 48164.857162591645,
    'Shanghai': 26391.169256045803,
    'Jiangsu': 95669.33025279762,
    'Zhejiang': 94627.45728971215,
    'Anhui': 73565.05034667291,
    'Fujian': 98615.46982084474,
    'Jiangxi': 72967.00347933048,
    'Shandong': 120337.94332397236,
    'Henan': 100370.93231025043,
    'Hubei': 85072.42648468405,
    'Hunan': 86896.78197941056,
    'Guangdong': 127889.45708489243,
    'Guangxi': 74728.12616820897,
    'Hainan': 15183.834786620278,
    'Chongqing': 62218.1280451105,
    'Sichuan': 153584.02527712708,
    'Guizhou': 82060.57820283275,
    'Yunnan': 108929.1423522208,
    'Tibet': 17005.39169283736,
    'Shaanxi': 62917.989937104496,
    'Gansu': 49832.89151889198,
    'Qinghai': 12872.07623336378,
    'Ningxia': 11867.564227700104,
    'Xinjiang': 49601.13926689687}

Maximum_Capacity_std_Before5G = {
    'Beijing': 208.04086469365953,
    'Tianjing': 140.85414895963598,
    'Hebei': 425.90159326628356,
    'Shanxi': 331.9750809676983,
    'Inner Mongolia': 254.92442513301734,
    'Liaoning': 340.3602677941905,
    'Jilin': 235.85880341736416,
    'Heilongjiang': 226.43900973937272,
    'Shanghai': 188.9567852069788,
    'Jiangsu': 352.13455157129505,
    'Zhejiang': 305.54569354938803,
    'Anhui': 365.40553766983226,
    'Fujian': 321.8918262675323,
    'Jiangxi': 292.8109077477041,
    'Shandong': 370.46188635824217,
    'Henan': 363.1488904900797,
    'Hubei': 301.9598648950953,
    'Hunan': 488.03668027366507,
    'Guangdong': 265.9529387099422,
    'Guangxi': 235.38764635967505,
    'Hainan': 165.1632660643398,
    'Chongqing': 330.7467721165372,
    'Sichuan': 519.1501469082156,
    'Guizhou': 336.53178256833803,
    'Yunnan': 401.96764640909703,
    'Tibet': 174.00313284922834,
    'Shaanxi': 276.4935103191823,
    'Gansu': 234.66423936269263,
    'Qinghai': 146.27916746146414,
    'Ningxia': 106.91772166494297,
    'Xinjiang': 223.19687244886705}

Maximum_Capacity_After5G = {
    'Beijing': 103427.05787716045,
    'Tianjing': 63900.232943401774,
    'Hebei': 309573.78596374404,
    'Shanxi': 200496.3012598362,
    'Inner Mongolia': 119465.8356700743,
    'Liaoning': 166706.30886391288,
    'Jilin': 82107.74503364437,
    'Heilongjiang': 156654.5952077875,
    'Shanghai': 153933.98473360096,
    'Jiangsu': 451957.3117036456,
    'Zhejiang': 392603.8156302228,
    'Anhui': 255614.594248283,
    'Fujian': 251762.99593427367,
    'Jiangxi': 266708.8885866975,
    'Shandong': 421375.8197617849,
    'Henan': 278002.33755988337,
    'Hubei': 277570.22155927395,
    'Hunan': 249934.33161381897,
    'Guangdong': 593152.6964999749,
    'Guangxi': 182235.84230798465,
    'Hainan': 49535.657937518394,
    'Chongqing': 185121.80860453928,
    'Sichuan': 333345.8814922295,
    'Guizhou': 254334.751769117,
    'Yunnan': 328638.11256346165,
    'Tibet': 31659.64610018045,
    'Shaanxi': 188819.1525849617,
    'Gansu': 101296.18618778053,
    'Qinghai': 29531.291164545073,
    'Ningxia': 40512.72185246395,
    'Xinjiang': 105962.2594464042}

Maximum_Capacity_std_After5G = {'Beijing': 248.28278315995985,
                                'Tianjing': 218.1325393711375,
                                'Hebei': 501.0661612404566,
                                'Shanxi': 402.3028808994284,
                                'Inner Mongolia': 294.64424810584865,
                                'Liaoning': 387.4989123611103,
                                'Jilin': 231.9950682711742,
                                'Heilongjiang': 248.08184887775536,
                                'Shanghai': 399.3509073364955,
                                'Jiangsu': 597.4855520535444,
                                'Zhejiang': 658.3438685266087,
                                'Anhui': 346.9707337881566,
                                'Fujian': 598.2777279713789,
                                'Jiangxi': 474.5977035606688,
                                'Shandong': 469.93589497095996,
                                'Henan': 487.0244595608593,
                                'Hubei': 538.4133016437805,
                                'Hunan': 658.7337196487713,
                                'Guangdong': 519.0946036320815,
                                'Guangxi': 407.94208040538956,
                                'Hainan': 228.462217148843,
                                'Chongqing': 414.5114895762192,
                                'Sichuan': 603.1609149994134,
                                'Guizhou': 464.6306825556552,
                                'Yunnan': 590.2610294335623,
                                'Tibet': 203.15733430586397,
                                'Shaanxi': 354.1452996768153,
                                'Gansu': 318.4066031958084,
                                'Qinghai': 191.20678912847322,
                                'Ningxia': 190.06542218663643,
                                'Xinjiang': 262.27375918926}

Province_Network_Capacity = {}

for Province in Province_4GBS_Num:
    Province_Network_Capacity[Province] = {}
    Province_Network_Capacity[Province]['4G'] = Province_4GBS_Num[Province] * np.mean(BS_Capacity_4G) / 1024  # PB
    Province_Network_Capacity[Province]['4GSTD'] = Province_4GBS_Num[Province] * np.std(BS_Capacity_4G) / 1024  # PB

for Province in Province_5GBS_Num:
    Province_Network_Capacity[Province]['5G'] = Province_5GBS_Num[Province] * np.mean(BS_Capacity_5G) / 1024  # PB
    Province_Network_Capacity[Province]['5GSTD'] = Province_5GBS_Num[Province] * np.std(BS_Capacity_5G) / 1024  # PB

Nanchang_User_Num = np.array(
    [724.1, 721.8, 734.4, 738.6, 743.3, 752.4, 757.5, 762.6, 779.7, 781.6, 782.7, 783.8, 798.1, 804.8, 808.4, 796.0,
     787.4])

Network_Traffic_After5G_4G = [1.15357101, 1.312790426, 1.399133649, 1.464940381, 1.513061395, 1.524572747, 1.53056248, \
                              1.533335786, 1.473622746, 1.431067853, 1.470064229, 1.498515583, 1.461390619, 1.415919433,
                              1.387327911, 1.382028557, 1.37190577]
Network_Traffic_After5G_5G = [0.056856181, 0.062154632, 0.136420432, 0.167308413, 0.195333782, 0.223651352, 0.231856969,
                              0.240518264, \
                              0.351768637, 0.382075861, 0.386963403, 0.396250525, 0.439388366, 0.488138965, 0.528031143,
                              0.540033872, 0.57771561]
Network_Traffic_After5G = np.array(Network_Traffic_After5G_4G) + np.array(Network_Traffic_After5G_5G)

PerUserTraffic = Network_Traffic_After5G / Nanchang_User_Num  # PB/万人

Province_User_Num = {
    'Beijing': [3906.4, 3870.4, 3864.3, 3851, 3846.1, 3866.1, 3883.5, 3917.8, 3945.8, 3932, 3946.4, 3972, 3950.35,
                3928.7, 3933.8, 3939.8, 3940.7, 3953.5, 3945.8],
    'Tianjing': [1711, 1706.4, 1713.6, 1718.7, 1719.4, 1736.3, 1736.1, 1724.4, 1729, 1742.1, 1744.8, 1745.1, 1743.9,
                 1742.7, 1748.4, 1753.1, 1741.5, 1761.8, 1772.6],
    'Hebei': [8336, 8347.7, 8391.2, 8403.2, 8406.1, 8430.4, 8469.9, 8506.4, 8571.8, 8601.5, 8610.6, 8643.5, 8653.45,
              8663.4, 8661.3, 8651.2, 8647.9, 8646, 8664.4],
    'Shanxi': [4022.8, 4018.2, 4045, 4040.2, 4039.4, 4067, 4068.1, 4082.2, 4114.1, 4115.4, 4113.7, 4126, 4130.8, 4135.6,
               4167.7, 4162.6, 4172.5, 4186.7, 4197.6],
    'Inner Mongolia': [2962.2, 2945.7, 2956.7, 2950.5, 2948.9, 2948.2, 2953.7, 2966.3, 2976, 2980.2, 2997.2, 3016.9,
                       3013.45, 3010, 3024.5, 3027.4, 3033.5, 3038.1, 3044.4],
    'Liaoning': [4873.8, 4857.9, 4884.8, 4886.4, 4888.4, 4896.5, 4906.1, 4919.1, 4968.1, 4971.9, 4972.0, 4975.2, 4973,
                 4970.8, 5006.8, 4994.1, 5010.2, 5021.5, 5038.6],
    'Jilin': [2870.1, 2857.1, 2876.5, 2877, 2888.4, 2901, 2910.4, 2915.3, 2932.4, 2937.6, 2956.3, 2966.5, 2967.65,
              2968.8, 2959.7, 2941.5, 2967.4, 2979.1, 2985.6],
    'Heilongjiang': [3844.4, 3725.8, 3739.7, 3731.4, 3742.3, 3745.5, 3739.5, 3743.7, 3782.7, 3800.7, 3812.9, 3759.5,
                     3773.35, 3787.2, 3792.6, 3799.5, 3807.8, 3816.6, 3825.9],
    'Shanghai': [4277.6, 4272.9, 4293.8, 4338.9, 4378, 4396.9, 4415.9, 4431.4, 4445.9, 4459.1, 4471.4, 4398.8, 4401.6,
                 4404.4, 4401.9, 4359.2, 4285.4, 4361.4, 4391.4],
    'Jiangsu': [9897.1, 9907.5, 9967.1, 9958.6, 9976.7, 10018.4, 10041.2, 10050.8, 10081.4, 10144.9, 10157.9, 10179.5,
                10225.4, 10271.3, 10335.8, 10330.2, 10362.8, 10413.2, 10438],
    'Zhejiang': [8585.2, 8510.2, 8564.3, 8647.3, 8679.6, 8701.8, 8743.2, 8797.8, 8842.5, 8862.8, 8862.5, 8859.6, 8883.3,
                 8907, 8993, 9009.8, 9045, 9079.3, 9087.3],
    'Anhui': [6025.6, 6055.8, 6076.2, 6093.2, 6131.2, 6153.2, 6164.8, 6183.5, 6202.8, 6206.9, 6202.7, 6192.6, 6227.85,
              6263.1, 6285.6, 6284.6, 6291.1, 6309.9, 6314.6],
    'Fujian': [4739.3, 4727.5, 4760.3, 4770.8, 4781.3, 4791.2, 4785.9, 4820.4, 4839.5, 4847.5, 4848.5, 4824.3, 4833.55,
               4842.8, 4863.3, 4861.4, 4860.4, 4860.2, 4873.3],
    'Jiangxi': [4249.4, 4300.7, 4332.3, 4339.9, 4361.7, 4400.7, 4418.7, 4457.4, 4497.8, 4513.9, 4476.9, 4496.8, 4562.15,
                4627.5, 4642.1, 4589.6, 4494, 4556.7, 4540.3],
    'Shandong': [10907.1, 10865.4, 10923.1, 10960.1, 10994.4, 11007.4, 11049.2, 11099.4, 11150.6, 11220.3, 11221.4,
                 11248.5, 11293.1, 11337.7, 11423.6, 11458.6, 11517.9, 11572, 11651.8],
    'Henan': [10051.4, 10116, 10181.3, 10183.3, 10194.9, 10231.1, 10250, 10245.9, 10278.3, 10342.6, 10342.1, 10352.6,
              10379.15, 10405.7, 10447.8, 10455.8, 10464.1, 10492.3, 10517.1],
    'Hubei': [5681.1, 5706.1, 5748, 5745, 5751.4, 5766.1, 5772.6, 5826.4, 5863.7, 5886.8, 5893.5, 5871.1, 5886.2,
              5901.3, 5925.4, 5938.1, 5941.1, 5970, 5980.1],
    'Hunan': [6719.4, 6761.2, 6785.9, 6800.2, 6836.7, 6853.8, 6889.4, 6895.4, 6951.9, 6952.1, 6938.7, 6942.3, 6934.7,
              6927.1, 6943, 6940.5, 6952.1, 6976.7, 6999.4],
    'Guangdong': [15536.9, 15394.3, 15471.5, 15670, 15708.8, 15758.6, 15884, 15985.7, 16122.4, 16205.8, 16254.1,
                  16267.8, 16289.55, 16311.3, 16562.1, 16634.3, 16669, 16665.3, 16691.8],
    'Guangxi': [5332.9, 5379.6, 5406.1, 5401.2, 5401.8, 5400.8, 5428.8, 5471.6, 5497.1, 5512.5, 5507.7, 5511.4, 5523.3,
                5535.2, 5606.3, 5623.1, 5661.6, 5694.2, 5714],
    'Hainan': [1135.2, 1119.5, 1137.4, 1136.6, 1136.8, 1132.4, 1129.8, 1128.5, 1150.2, 1156.6, 1157.3, 1158.9, 1156.4,
               1153.9, 1173.4, 1163.3, 1157.1, 1156.1, 1155.5],
    'Chongqing': [3640.1, 3669.1, 3692, 3684.9, 3690.6, 3692.8, 3703.1, 3728.4, 3757.1, 3766.3, 3761.2, 3751.1, 3784.35,
                  3817.6, 3835.6, 3840.8, 3850.9, 3863.6, 3878.7],
    'Sichuan': [9124.6, 9153, 9199.5, 9147.3, 9169.5, 9258.7, 9267.7, 9269.2, 9289.4, 9311.7, 9315.4, 9338.9, 9378.05,
                9417.2, 9483.5, 9508.2, 9516.8, 9550.7, 9571.3],
    'Guizhou': [4093.5, 4116.2, 4148.3, 4136.3, 4138.2, 4140.9, 4144.8, 4202.6, 4222.8, 4240.4, 4243.6, 4269.9, 4287.1,
                4304.3, 4329.2, 4327, 4331.2, 4353.8, 4368.2],
    'Yunnan': [4953.4, 4970.6, 5010.3, 4995, 4962.3, 4971.5, 5002, 5027.1, 5047.3, 5032.1, 5035.9, 5045.7, 5043.85,
               5042, 5063, 5058.3, 5047.8, 5055.3, 5079.5],
    'Tibet': [321.9, 315.5, 313.8, 319.7, 319.5, 321.2, 324.5, 325.8, 330.5, 332.2, 331.6, 333.4, 332.35, 331.3, 334.4,
              335.5, 334.9, 335.6, 341.1],
    'Shaanxi': [4589.7, 4553.4, 4540.5, 4609.5, 4622.2, 4673.5, 4706.9, 4724.2, 4749.6, 4768.1, 4772.0, 4777.8, 4772.8,
                4767.8, 4837.3, 4796.4, 4804.5, 4806.5, 4815.1],
    'Gansu': [2673.8, 2669, 2691.2, 2693.7, 2706.9, 2717.8, 2722.1, 2730.8, 2746.6, 2748.2, 2740.2, 2744.7, 2746.3,
              2747.9, 2759.7, 2758.9, 2760, 2763.1, 2762.9],
    'Qinghai': [659.4, 650.6, 652.8, 654.1, 658.8, 663, 666.1, 668.9, 675.5, 677, 678.9, 680.5, 680.4, 680.3, 684.6,
                685.7, 693, 701.2, 706.5],
    'Ningxia': [839.2, 838.3, 841.3, 841, 841.6, 842.9, 845.3, 853.6, 861.7, 864.4, 866.9, 866.1, 871.15, 876.2, 885.8,
                884.4, 886.6, 887.5, 890.4],
    'Xinjiang': [2846.6, 2855.8, 2894.5, 2906.2, 2902.4, 2906.7, 2905.3, 2912.3, 2967.9, 2967.5, 2964.4, 2965.4, 2973,
                 2980.6, 2991.1, 2996.2, 3000.2, 3007.1, 2980.5],
    'Nanchang': [724.1, 721.8, 734.4, 738.6, 743.3, 752.4, 757.5, 762.6, 779.7, 781.6, 782.7, 783.8, 798.1, 804.8,
                 808.4, 796.0, 787.4, 787.4, 787.4]
}

popt, pcov = curve_fit(func, np.array(Date_After5G), np.array(Network_Traffic_After5G))
Network_Traffic_Future = func(np.array(Date_Future), popt[0], popt[1], popt[2], popt[3])
Network_Traffic_Future[0] = Network_Traffic_After5G[-1]

Network_Traffic_After5G_Province = {}
Network_Traffic_Future_Province = {}
for Province in Province_User_Num:
    Network_Traffic_After5G_Province[Province] = np.array(Province_User_Num[Province])[0:17] * PerUserTraffic  # PB
    popt, pcov = curve_fit(func, np.array(Date_After5G), np.array(Network_Traffic_After5G_Province[Province]))
    Network_Traffic_Future_Province[Province] = func(np.array(Date_Future), popt[0], popt[1], popt[2], popt[3])
    Network_Traffic_Future_Province[Province][0] = Network_Traffic_After5G_Province[Province][-1]

Network_Energy_Province = {}
Network_Carbon_Province = {}
for Province in Province_Network_Capacity:
    Network_Carbon_Province[Province] = {}
    Network_Carbon_Province[Province]['4G'] = 48.3036 * Province_Network_Capacity[Province]['4G']
    Network_Carbon_Province[Province]['5G'] = 28.2990 * Province_Network_Capacity[Province]['5G']


def func(x, a, b, c, d):
    return a * x ** 2 + b * x + c * x ** 3 + d


Time_list_After5G = []
for year in ['2021', '2022', '2023']:
    for month in range(1, 13):
        if month < 10:
            Time_list_After5G.append(f'{year}-0{month}')
        else:
            Time_list_After5G.append(f'{year}-{month}')
Network_Traffic_Province = {}
Network_Energy_Province = {}
Network_Energy_std_Province = {}
Network_Carbon_Province = {}
Network_Carbon_std_Province = {}
Wasted_Energy_Province = {'Without Method': {}}
Wasted_Energy_std_Province = {'Without Method': {}}
Additional_Carbon_Province = {'Without Method': {}}
Additional_Carbon_std_Province = {'Without Method': {}}

for Province in Province_Network_Capacity:

    K = 0.6131
    K_std = 0.0061
    K1 = 0.5261
    K1_std = 0.053
    K2 = 0.4387
    K2_std = 0.0044
    K3 = 0.2856
    K3_std = 0.0029
    gama = 0.9527
    gama_std = 0.0515

    Date_Before5G = np.linspace(0, 7, 8)
    Date_After5G = np.linspace(7, 23, 17)
    Date_Future = np.linspace(23, 42, 20)

    Network_Capacity_Before5G = []
    Network_Capacity_After5G = []
    Network_Capacity_Future = []
    Network_Capacity_Before5G_log = []
    Network_Capacity_After5G_log = []
    Network_Capacity_Future_log = []

    Network_Capacity_Before5G_low = []
    Network_Capacity_After5G_low = []
    Network_Capacity_Future_low = []
    Network_Capacity_Before5G_log_low = []
    Network_Capacity_After5G_log_low = []
    Network_Capacity_Future_log_low = []

    Network_Capacity_Before5G_high = []
    Network_Capacity_After5G_high = []
    Network_Capacity_Future_high = []
    Network_Capacity_Before5G_log_high = []
    Network_Capacity_After5G_log_high = []
    Network_Capacity_Future_log_high = []

    for index in range(len(Date_Before5G)):
        Network_Capacity_Before5G.append(Maximum_Capacity_Before5G[Province] / 1024)
        Network_Capacity_Before5G_low.append(Maximum_Capacity_Before5G[Province] / 1024 -
                                             Maximum_Capacity_std_Before5G[Province] / 1024)
        Network_Capacity_Before5G_high.append(Maximum_Capacity_Before5G[Province] / 1024 +
                                              Maximum_Capacity_std_Before5G[Province] / 1024)
        Network_Capacity_Before5G_log.append(np.log(Maximum_Capacity_Before5G[Province] / 1024))
        Network_Capacity_Before5G_log_low.append(np.log(Maximum_Capacity_Before5G[Province] / 1024 -
                                                        Maximum_Capacity_std_Before5G[Province] / 1024))
        Network_Capacity_Before5G_log_high.append(np.log(Maximum_Capacity_Before5G[Province] / 1024 +
                                                         Maximum_Capacity_std_Before5G[Province] / 1024))
    for index in range(len(Date_After5G)):
        Network_Capacity_After5G.append(Maximum_Capacity_After5G[Province] / 1024)
        Network_Capacity_After5G_low.append(Maximum_Capacity_After5G[Province] / 1024 -
                                            Maximum_Capacity_std_After5G[Province] / 1024)
        Network_Capacity_After5G_high.append(Maximum_Capacity_After5G[Province] / 1024 +
                                             Maximum_Capacity_std_After5G[Province] / 1024)
        Network_Capacity_After5G_log.append(np.log(Maximum_Capacity_After5G[Province] / 1024))
        Network_Capacity_After5G_log_low.append(np.log(Maximum_Capacity_After5G[Province] / 1024 -
                                                       Maximum_Capacity_std_After5G[Province] / 1024))
        Network_Capacity_After5G_log_high.append(np.log(Maximum_Capacity_After5G[Province] / 1024 +
                                                        Maximum_Capacity_std_After5G[Province] / 1024))
    for index in range(len(Date_Future)):
        Network_Capacity_Future.append(Maximum_Capacity_After5G[Province] / 1024)
        Network_Capacity_Future_low.append(Maximum_Capacity_After5G[Province] / 1024 -
                                           Maximum_Capacity_std_After5G[Province] / 1024)
        Network_Capacity_Future_high.append(Maximum_Capacity_After5G[Province] / 1024 +
                                            Maximum_Capacity_std_After5G[Province] / 1024)
        Network_Capacity_Future_log.append(np.log(Maximum_Capacity_After5G[Province] / 1024))
        Network_Capacity_Future_log_low.append(np.log(Maximum_Capacity_After5G[Province] / 1024 -
                                                      Maximum_Capacity_std_After5G[Province] / 1024))
        Network_Capacity_Future_log_high.append(np.log(Maximum_Capacity_After5G[Province] / 1024 +
                                                       Maximum_Capacity_std_After5G[Province] / 1024))

    Network_Capacity_Before5G = np.array(Network_Capacity_Before5G)
    Network_Capacity_After5G = np.array(Network_Capacity_After5G)
    Network_Capacity_Future = np.array(Network_Capacity_Future)
    Network_Traffic_Before5G = []
    Network_Traffic_After5G = []
    Network_Traffic_Future = []

    Network_Traffic_Before5G_low = []
    Network_Traffic_After5G_low = []
    Network_Traffic_Future_low = []

    Network_Traffic_Before5G_high = []
    Network_Traffic_After5G_high = []
    Network_Traffic_Future_high = []
    for index in range(len(Date_Before5G)):
        Network_Traffic_Before5G.append(Network_Traffic_After5G_Province[Province][0])
        Network_Traffic_Before5G_low.append(Network_Traffic_After5G_Province[Province][0] * 0.97)
        Network_Traffic_Before5G_high.append(Network_Traffic_After5G_Province[Province][0] * 1.03)
    for index in range(len(Date_After5G)):
        Network_Traffic_After5G.append(Network_Traffic_After5G_Province[Province][index])
        Network_Traffic_After5G_low.append(Network_Traffic_After5G_Province[Province][index] * 0.97)
        Network_Traffic_After5G_high.append(Network_Traffic_After5G_Province[Province][index] * 1.03)
    for index in range(len(Date_Future)):
        Network_Traffic_Future.append(Network_Traffic_Future_Province[Province][index])
        Network_Traffic_Future_low.append(Network_Traffic_Future_Province[Province][index] * 0.97)
        Network_Traffic_Future_high.append(Network_Traffic_Future_Province[Province][index] * 1.03)

    Network_Traffic_Province[Province] = Network_Traffic_After5G + Network_Traffic_Future[1:]

    Network_Traffic_Before5G = np.array(Network_Traffic_Before5G)
    Network_Traffic_After5G = np.array(Network_Traffic_After5G)
    Network_Traffic_Future = np.array(Network_Traffic_Future)
    Network_Traffic_Before5G_log = np.log(Network_Traffic_Before5G)
    Network_Traffic_After5G_log = np.log(Network_Traffic_After5G)
    Network_Traffic_Future_log = np.log(Network_Traffic_Future)

    Network_Traffic_Before5G_low = np.array(Network_Traffic_Before5G_low)
    Network_Traffic_After5G_low = np.array(Network_Traffic_After5G_low)
    Network_Traffic_Future_low = np.array(Network_Traffic_Future_low)
    Network_Traffic_Before5G_log_low = np.log(Network_Traffic_Before5G_low)
    Network_Traffic_After5G_log_low = np.log(Network_Traffic_After5G_low)
    Network_Traffic_Future_log_low = np.log(Network_Traffic_Future_low)

    Network_Traffic_Before5G_high = np.array(Network_Traffic_Before5G_high)
    Network_Traffic_After5G_high = np.array(Network_Traffic_After5G_high)
    Network_Traffic_Future_high = np.array(Network_Traffic_Future_high)
    Network_Traffic_Before5G_log_high = np.log(Network_Traffic_Before5G_high)
    Network_Traffic_After5G_log_high = np.log(Network_Traffic_After5G_high)
    Network_Traffic_Future_log_high = np.log(Network_Traffic_Future_high)

    Network_Energy_Before5G = Maximum_Energy_Before5G[Province] * \
                              ((1 - K) * Network_Traffic_Before5G / Network_Capacity_Before5G + K)
    Network_Energy_After5G = Maximum_Energy_After5G[Province] * \
                             ((1 - K) * Network_Traffic_After5G / Network_Capacity_After5G + K)
    Network_Energy_Future = Maximum_Energy_After5G[Province] * \
                            ((1 - K) * Network_Traffic_Future / Network_Capacity_Future + K)
    Network_Energy_Before5G_low = (Maximum_Energy_Before5G[Province] - Maximum_Energy_std_Before5G[Province]) * \
                                  ((1 - K) * Network_Traffic_Before5G_low / Network_Capacity_Before5G_high + K)
    Network_Energy_After5G_low = (Maximum_Energy_After5G[Province] - Maximum_Energy_std_After5G[Province]) * \
                                 ((1 - K) * Network_Traffic_After5G_low / Network_Capacity_After5G_high + K)
    Network_Energy_Future_low = (Maximum_Energy_After5G[Province] - Maximum_Energy_std_After5G[Province]) * \
                                ((1 - K) * Network_Traffic_Future_low / Network_Capacity_Future_high + K)

    Network_Carbon_Before5G = Network_Energy_Before5G * gama
    Network_Carbon_After5G = Network_Energy_After5G * gama
    Network_Carbon_Future = Network_Energy_Future * gama
    Network_Carbon_Before5G_low = Network_Energy_Before5G_low * (gama - gama_std)
    Network_Carbon_After5G_low = Network_Energy_After5G_low * (gama - gama_std)
    Network_Carbon_Future_low = Network_Energy_Future_low * (gama - gama_std)
    Network_Carbon_Before5G_high = Network_Energy_Before5G_high * (gama + gama_std)
    Network_Carbon_After5G_high = Network_Energy_After5G_high * (gama + gama_std)
    Network_Carbon_Future_high = Network_Energy_Future_high * (gama + gama_std)

    Network_Carbon_Province[Province] = list(Network_Carbon_After5G) + list(Network_Carbon_Future)
    Network_Carbon_std_Province[Province] = list(Network_Carbon_After5G_high - Network_Carbon_After5G) \
                                            + list(Network_Carbon_Future_high - Network_Carbon_Future)

    Network_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G
    Network_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / Network_Energy_After5G
    Network_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / Network_Energy_Future
    Network_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_low
    Network_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / Network_Energy_After5G_low
    Network_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / Network_Energy_Future_low
    Network_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_high
    Network_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / Network_Energy_After5G_high
    Network_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / Network_Energy_Future_high

    Network_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(Network_Carbon_Before5G)
    Network_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(Network_Carbon_After5G)
    Network_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future)
    Network_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(
        Network_Carbon_Before5G_low)
    Network_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(
        Network_Carbon_Before5G_high)
    Network_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(
        Network_Carbon_After5G_low)
    Network_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(
        Network_Carbon_After5G_high)
    Network_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future_low)
    Network_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(
        Network_Carbon_Future_high)

# Plot d

from pyecharts import options as opts
from pyecharts.charts import Map

ACP = np.load('./data/province/Additional_Carbon_Province.npy', allow_pickle=True).item()['Without Method']
ACeP = np.load('./data/province/Additional_Carbon_std_Province.npy', allow_pickle=True).item()['Without Method']
P4N = np.load('./data/province/Province_4GBS_Num.npy', allow_pickle=True).item()
P5N = np.load('./data/province/Province_5GBS_Num.npy', allow_pickle=True).item()
PNC = np.load('./data/province/Province_Network_Capacity.npy', allow_pickle=True).item()
PUN = np.load('./data/province/Province_User_Num.npy', allow_pickle=True).item()
WEP = np.load('./data/province/Wasted_Energy_Province.npy', allow_pickle=True).item()['Without Method']
WEeP = np.load('./data/province/Wasted_Energy_std_Province.npy', allow_pickle=True).item()['Without Method']

C_keys = {'北京市': 'Beijing', '天津市': 'Tianjing', '河北省': 'Hebei',
          '山西省': 'Shanxi', '内蒙古自治区': 'Inner Mongolia', '辽宁省': 'Liaoning',
          '吉林省': 'Jilin', '黑龙江省': 'Heilongjiang', '上海市': 'Shanghai',
          '江苏省': 'Jiangsu', '浙江省': 'Zhejiang', '安徽省': 'Anhui',
          '福建省': 'Fujian', '江西省': 'Jiangxi', '山东省': 'Shandong',
          '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan', '广东省': 'Guangdong',
          '广西壮族自治区': 'Guangxi', '海南省': 'Hainan', '重庆市': 'Chongqing',
          '四川省': 'Sichuan', '贵州省': 'Guizhou', '云南省': 'Yunnan',
          '西藏自治区': 'Tibet', '陕西省': 'Shaanxi', '甘肃省': 'Gansu', '青海省': 'Qinghai',
          '宁夏回族自治区': 'Ningxia', '新疆维吾尔自治区': 'Xinjiang'}
area = {'北京市': 16412, '天津市': 11903, '河北省': 187159,
        '山西省': 156698, '内蒙古自治区': 1196113, '辽宁省': 148084,
        '吉林省': 190234, '黑龙江省': 439703, '上海市': 6339,
        '江苏省': 102378, '浙江省': 103493, '安徽省': 139615,
        '福建省': 122870, '江西省': 167064, '山东省': 158219,
        '河南省': 166785, '湖北省': 185750, '湖南省': 211833, '广东省': 174246,
        '广西壮族自治区': 237438, '海南省': 30970, '重庆市': 82370,
        '四川省': 491718, '贵州省': 176161, '云南省': 394029,
        '西藏自治区': 1194047, '陕西省': 205629, '甘肃省': 454858, '青海省': 696610,
        '宁夏回族自治区': 66400, '新疆维吾尔自治区': 1640016}
name_list1 = ['Additional Carbon',
              'Num of 4GBSs',
              'Num of 5GBSs',
              'Network Capacity of 4G',
              'Network Capacity of 5G',
              'Network Capacity of 4G and 5G',
              'Network Capacity of 4G STD',
              'Network Capacity of 5G STD',
              'Network Capacity of 4G and 5G STD',
              'Num of Users',
              'Wasted Energy',
              'Additional Carbon per person',
              'Additional Carbon per square kilometer',
              'Additional Carbon after RL']
name_list2 = ['Additional_Carbon',
              '4GBS_Num',
              '5GBS_Num',
              'Network_Capacity_4G',
              'Network_Capacity_5G',
              'Network_Capacity_45G',
              'Network_Capacity_4G_STD',
              'Network_Capacity_5G_STD',
              'Network_Capacity_45G_STD',
              'User_Num',
              'Wasted_Energy',
              'Additional_Carbon_per_person',
              'Additional_Carbon_per_square_kilometer',
              'Additional_Carbon_after_RL']

name_list_csv = ['Additional Carbon (tCO2)',
                 'Additional Carbon errorbar (tCO2)',
                 'Num of 4GBSs',
                 'Num of 5GBSs',
                 'Network Capacity of 4G (PBtye)',
                 'Network Capacity of 5G (PBtye)',
                 'Network Capacity of 4G and 5G (PBtye)',
                 'Network Capacity of 4G STD (PBtye)',
                 'Network Capacity of 5G STD (PBtye)',
                 'Network Capacity of 4G and 5G STD (PBtye)',
                 'Num of Users (Million)',
                 'Wasted Energy (MWh)',
                 'Additional Carbon per person (kgCO2)',
                 'Additional Carbon per square kilometer (tCO2)',
                 'Additional Carbon after RL (tCO2)']
factor_list_csv = [1_000_000, 1, 1_000, 1_000, 1, 1, 1, 1, 1, 1, 1_000_000, 1_000_000, 1, 1]


def dataget(data, C_E):
    data_get = {}
    for p in C_E:
        break
    if type(data[C_E[p]]) == dict:
        for p in C_E:
            for k in data[C_E[p]]:
                if k not in data_get:
                    data_get[k] = {}
                data_get[k][p] = data[C_E[p]][k]
    elif type(data[C_E[p]]) == list:
        for p in C_E:
            data_get[p] = np.mean(data[C_E[p]])
    else:
        for p in C_E:
            data_get[p] = data[C_E[p]]
    return data_get


data_list_plot = []
data_list = [ACP, ACeP, P4N, P5N, PNC, PUN, WEP]
i = 0
for data in data_list:
    dg = dataget(data, C_keys)
    if len(dg) < len(C_keys):
        d1 = {}
        d2 = {}
        for x in dg['4G']:
            d1[x] = dg['4G'][x] + dg['5G'][x]
            d2[x] = dg['4GSTD'][x] + dg['5GSTD'][x]
        data_list_plot.append(dg['4G'])
        i += 1
        data_list_plot.append(dg['5G'])
        i += 1
        data_list_plot.append(d1)
        i += 1
        data_list_plot.append(dg['4GSTD'])
        i += 1
        data_list_plot.append(dg['5GSTD'])
        i += 1
        data_list_plot.append(d2)
        i += 1
    else:
        data_list_plot.append(dg)
        i += 1
dg = {}
for x in data_list_plot[0]:
    dg[x] = data_list_plot[0][x] / (data_list_plot[10][x] * 10)
data_list_plot.append(dg)
i += 1
dg = {}
for x in data_list_plot[0]:
    dg[x] = data_list_plot[0][x] / area[x]
data_list_plot.append(dg)
i += 1
dg = {}
data_list_plot.append(dg)
i += 1
idx_province = []
for x in C_keys:
    idx_province.append(C_keys[x])


def dataget(data, C_E):
    data_get = {}
    for p in C_E:
        break
    if type(data[C_E[p]]) == dict:
        for p in C_E:
            for k in data[C_E[p]]:
                if k not in data_get:
                    data_get[k] = {}
                data_get[k][p] = data[C_E[p]][k]
    elif type(data[C_E[p]]) == list:
        for p in C_E:
            data_get[p] = np.mean(data[C_E[p]])
    else:
        for p in C_E:
            data_get[p] = data[C_E[p]]
    return data_get


s1v = [[0, 400000], [400000, 800000], [800000, 1200000], [1200000, 1600000],
       [1600000, 2000000], [2000000, 2800000], [2800000, 5000000]]
s1n = [['0', '40W'], ['40W', '80W'], ['80W', '120W'], ['120W', '160W'],
       ['160W', '200W'], ['200W', '280W'], ['280W', '500W']]
s2v = [[0, 20000], [20000, 40000], [40000, 60000], [60000, 80000],
       [80000, 100000], [100000, 130000], [130000, 160000]]
s2n = [['0', '20K'], ['20K', '40K'], ['40K', '60K'], ['60K', '80K'],
       ['80K', '100K'], ['100K', '130K'], ['130K', '160K']]
s3v = [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000],
       [40000, 50000], [50000, 60000], [60000, 100000]]
s3n = [['0', '10K'], ['10K', '20K'], ['20K', '30K'], ['30K', '40K'],
       ['40K', '50K'], ['50K', '60K'], ['60K', '100K']]
s4v = [[0, 10], [10, 20], [20, 40], [40, 60],
       [60, 80], [80, 100], [100, 130]]
s4n = [['0', '10'], ['10', '20'], ['20', '40'], ['40', '60'],
       ['60', '80'], ['80', '100'], ['100', '130']]
s5v = [[0, 30], [30, 60], [60, 100], [100, 150],
       [150, 200], [200, 300], [300, 500]]
s5n = [['0', '30'], ['30', '60'], ['60', '100'], ['100', '150'],
       ['150', '200'], ['200', '300'], ['300', '500']]
s6v = [[0, 40], [40, 80], [80, 140], [140, 210],
       [210, 280], [280, 400], [400, 650]]
s6n = [['0', '40'], ['40', '80'], ['80', '140'], ['140', '210'],
       ['210', '280'], ['280', '400'], ['400', '650']]
s7v = [[0, 10], [10, 20], [20, 30], [30, 40],
       [40, 60], [60, 80], [80, 100]]
s7n = [['0', '10'], ['10', '20'], ['20', '30'], ['30', '40'],
       ['40', '60'], ['60', '80'], ['80', '100']]
s8v = [[0, 10], [10, 20], [20, 40], [40, 60],
       [60, 80], [80, 100], [100, 200]]
s8n = [['0', '10'], ['10', '20'], ['20', '40'], ['40', '60'],
       ['60', '80'], ['80', '100'], ['100', '200']]
s9v = [[0, 20], [20, 40], [40, 70], [70, 100],
       [100, 140], [140, 180], [180, 300]]
s9n = [['0', '20'], ['20', '40'], ['40', '70'], ['70', '100'],
       ['100', '140'], ['140', '180'], ['180', '300']]
s10v = [[0, 1000], [1000, 2000], [2000, 3000], [3000, 5000],
        [5000, 8000], [8000, 10000], [10000, 20000]]
s10n = [['0', '1K'], ['1K', '2K'], ['2K', '3K'], ['3K', '5K'],
        ['5K', '8K'], ['8K', '10K'], ['10K', '20K']]
s11v = [[0, 200000], [200000, 400000], [400000, 600000], [600000, 1000000],
        [1000000, 1600000], [1600000, 2400000], [2400000, 5000000]]
s11n = [['0', '200'], ['200', '400'], ['400', '600'], ['600', '1000'],
        ['1000', '1600'], ['1600', '2400'], ['2400', '5000']]
s12v = [[0, 80], [80, 120], [120, 160], [160, 200],
        [200, 250], [250, 350], [350, 500]]
s12n = [['0', '80'], ['80', '120'], ['120', '160'], ['160', '200'],
        ['200', '250'], ['250', '350'], ['350', '500']]
s13v = [[0, 0.3], [0.3, 1.5], [1.5, 3], [3, 7],
        [7, 11], [11, 30], [30, 300]]
s13n = [['0', '0.3'], ['0.3', '1.5'], ['1.5', '3'], ['3', '7'],
        ['7', '11'], ['11', '30'], ['30', '300']]
s14v = [[0, 1], [1, 10000], [10000, 20000], [20000, 30000],
        [30000, 50000], [50000, 100000], [100000, 150000]]
s14n = [['0', ''], ['0', '1W'], ['1W', '2W'], ['2W', '3W'],
        ['3W', '5W'], ['5W', '10W'], ['10W', '15W']]
step_value_list = [s1v, s2v, s3v, s4v, s5v, s6v, s7v,
                   s8v, s9v, s10v, s11v, s12v, s13v, s14v]
step_name_list = [s1n, s2n, s3n, s4n, s5n, s6n, s7n,
                  s8n, s9n, s10n, s11n, s12n, s13n, s14n]


def plot_province(data, name1, name2, step_value, step_name, i):
    province_dis4 = data
    provice = list(province_dis4.keys())
    values = list(province_dis4.values())

    china = (
        Map(opts.InitOpts(width='1200px', height='900px'))
        .add("", [list(z) for z in zip(provice, values)], "china", is_map_symbol_show=False)
        .set_global_opts(title_opts=opts.TitleOpts(title=name1),
                         legend_opts=opts.LegendOpts(type_="scroll", pos_left="left", orient="vertical"),
                         visualmap_opts=opts.VisualMapOpts(max_=step_value[6][1], is_piecewise=True, is_inverse=True,
                                                           pieces=[
                                                               {"min": step_value[0][0], "max": step_value[0][1],
                                                                "label": step_name[0][0] + ' - ' + step_name[0][1]},
                                                               {"min": step_value[1][0], "max": step_value[1][1],
                                                                "label": step_name[1][0] + ' - ' + step_name[1][1]},
                                                               {"min": step_value[2][0], "max": step_value[2][1],
                                                                "label": step_name[2][0] + ' - ' + step_name[2][1]},
                                                               {"min": step_value[3][0], "max": step_value[3][1],
                                                                "label": step_name[3][0] + ' - ' + step_name[3][1]},
                                                               {"min": step_value[4][0], "max": step_value[4][1],
                                                                "label": step_name[4][0] + ' - ' + step_name[4][1]},
                                                               {"min": step_value[5][0], "max": step_value[5][1],
                                                                "label": step_name[5][0] + ' - ' + step_name[5][1]},
                                                               {"min": step_value[6][0], "max": step_value[6][1],
                                                                "label": step_name[6][0] + ' - ' + step_name[6][1]},
                                                               {"max": 0, "label": "No data", "color": "#F8F8FF"}
                                                           ],
                                                           range_color=['#217ed5', '#80b3ae', '#bed389', '#F8FB64',
                                                                        '#FCB245', '#F86D2A', '#E91515'],
                                                           pos_right='90%',
                                                           pos_bottom='50%',
                                                           split_number=8))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )

    china.render("./fig1/" + str(i + 1) + "_" + name2 + ".html")
    china.render_notebook()


i = 2
plot_province(data_list_plot[i + 1], name_list1[i], name_list2[i], step_value_list[i], step_name_list[i], i)

# Plot e

i = 0
plot_province(data_list_plot[i], name_list1[i], name_list2[i], step_value_list[i], step_name_list[i], i)

# Plot f

i = 12
plot_province(data_list_plot[i + 1], name_list1[i], name_list2[i], step_value_list[i], step_name_list[i], i)

########## Figure 2 ##########

# Plot a

T = np.load('./data/T_all.npy', allow_pickle=True).item()
T_RL = np.load('./data/T_RL.npy', allow_pickle=True).item()
CO = np.load('./data/CO.npy', allow_pickle=True).item()
NPe = np.load('./data/NPe.npy', allow_pickle=True).item()
NP = np.load('./data/NP.npy', allow_pickle=True).item()
day_ticks = [1, 48, 96, 144, 192, 240, 288, 335]
day_labels = ['5-20', '5-21', '5-22', '5-23', '5-24', '5-25', '5-26', '5-27']

t4 = T['old']['']['4G'].tolist()
t5 = T['old']['']['5G'].tolist()

pe4 = NPe['old']['']['4G'].tolist()
pe5 = NPe['old']['']['5G'].tolist()

p4 = NP['old']['']['4G'].tolist()
p5 = NP['old']['']['5G'].tolist()

co4 = CO['old']['']['0']['4G'].tolist()
co5 = CO['old']['']['0']['5G'].tolist()

for sen in ['Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9',
            'Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9']:
    t4 += T['old'][sen]['4G'].tolist()
    t5 += T['old'][sen]['5G'].tolist()
    p4 += NP['old'][sen]['4G'].tolist()
    p5 += NP['old'][sen]['5G'].tolist()
    pe4 += NPe['old'][sen]['4G'].tolist()
    pe5 += NPe['old'][sen]['5G'].tolist()
    co4 += CO['old'][sen]['0']['4G'].tolist()
    co5 += CO['old'][sen]['0']['5G'].tolist()

Tm4 = 139583241374.98593
Tm5 = 119444661039.6334
Pem4 = 20654638.68332707
Pem5 = 8679166.897760902
Pm4 = 43160884.4280188
Pm5 = 17675608.0350063
Cm4 = 891.731929149999 - 870.736304999999
Cm5 = 879.430852249999 - 870.736304999999

fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_subplot(111)
ax1.scatter((np.array(t4) + np.array(t5)) / (1024 * 1024 * 1024), (np.array(p4) + np.array(p5)) / (1000000),
            color='#ff4500', label='4G Network Energy', alpha=0.6)
ax1.scatter((Tm4 + Tm5) / (1024 * 1024 * 1024), (Pm4 + Pm5) / (1000000), c='#539158', s=120, marker='*')
ax1.plot([0, (Tm4 + Tm5) / (1024 * 1024 * 1024)], [0, (Pm4 + Pm5) / (1000000)], linestyle='--', color='#539158',
         linewidth=2)
ax1.plot([-20, (Tm4 + Tm5) / (1024 * 1024 * 1024)], [(Pm4 + Pm5) / (1000000), (Pm4 + Pm5) / (1000000)], linestyle='--',
         color='k', linewidth=2)
ax1.plot([(Tm4 + Tm5) / (1024 * 1024 * 1024), (Tm4 + Tm5) / (1024 * 1024 * 1024)], [-5, (Pm4 + Pm5) / (1000000)],
         linestyle='--', color='k', linewidth=2)
ax1.text(13, 25, 'Misalignment Energy', fontsize=12)
ax1.text(225, 64, '(C, E$_M$$_a$$_x$)', fontsize=12)
ax1.text(243, 2, 'Network Capacity', rotation=270, fontsize=12)
ax1.annotate('', xy=(8, 1), xytext=(8, 35), color='k', arrowprops=dict(arrowstyle="<->"))
ax1.annotate('', xy=(80, 19), xytext=(80, 42), color='k', arrowprops=dict(arrowstyle="<->"))
ax1.annotate('', xy=(100, 24), xytext=(100, 44), color='k', arrowprops=dict(arrowstyle="<->"))
ax1.set_xlabel('Network Traffic (TByte)', fontsize=12)
ax1.set_ylabel('Network Energy (MWh)', fontsize=12)
ax1.set_ylim(-2, (Pm4 + Pm5) * 1.2 / (1000000))
ax1.set_xlim(-10, (Tm4 + Tm5) / (1024 * 1024 * 1024) * 1.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./fig2/star.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot b

P4 = np.load('./data/data_old/data_Power_4G.npy', allow_pickle=True).item()
P5 = np.load('./data/data_old/data_Power_5G.npy', allow_pickle=True).item()
A4 = np.load('./data/data_old/data_as_4G.npy', allow_pickle=True).item()
A5 = np.load('./data/data_old/data_as_5G.npy', allow_pickle=True).item()
T4 = np.load('./data/data_old/Traffic_4G.npy', allow_pickle=True).item()
T5 = np.load('./data/data_old/Traffic_5G.npy', allow_pickle=True).item()
B4 = np.load('./data/data_old/BSandCell_4G.npy', allow_pickle=True).item()
B5 = np.load('./data/data_old/BSandCell_5G.npy', allow_pickle=True).item()
CO = np.load('data/CO.npy', allow_pickle=True).item()

TT4 = np.zeros(336)
PP4 = np.zeros(336)
PS4 = np.zeros(336)
TT5 = np.zeros(336)
PP5 = np.zeros(336)
PS5 = np.zeros(336)

for x in T4:
    TT4 += np.array(T4[x])
for x in T5:
    TT5 += np.array(T5[x])
for x in P4:
    if P4[x] != []:
        PP4 += np.array(P4[x]) + np.array(A4[x])
for x in P5:
    if P5[x] != []:
        PP5 += np.array(P5[x]) + np.array(A5[x])

x = np.arange(336)
y1 = np.abs((PP4) / (Pm4) - (TT4) / (Tm4))
y2 = np.abs((PP5) / (Pm5) - (TT5) / (Tm5))
y3 = np.abs((PP4 + PP5) / (Pm4 + Pm5) - (TT4 + TT5) / (Tm4 + Tm5))

y1_smooth = np.convolve(y1, np.arange(3), 'same') / 3
y2_smooth = np.convolve(y2, np.arange(3), 'same') / 3
y3_smooth = np.convolve(y3, np.arange(3), 'same') / 3

fig = plt.figure(figsize=(12, 3))
# plt.plot(x, y1, linewidth=2, label = '4G')
plt.plot(x, y1_smooth, color='#9cce37', linewidth=1.5, label='4G Network')
plt.plot(x, y2_smooth, color='#ff4500', linewidth=1.5, label='5G Network')
plt.plot(x, y3_smooth, color='#20B2AB', linewidth=1.5, label='4G and 5G Network')
plt.ylabel('Misalignment Factor', fontsize=12)
# plt.ylim(0, 1.3*np.max(y1))
plt.ylim(0.35, 0.85)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=12)
plt.yticks(fontsize=12)
# plt.title('Misalignment of 4G Base Stations')
plt.xlim(1, 335)
fig.legend(bbox_to_anchor=(0.03, 0.78, 0.7, .102), ncol=3, fontsize=10)
plt.savefig('./fig2/Misalignment_time.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot i

fig = plt.figure(figsize=(6.8, 3.8))
ax1 = fig.add_subplot(111)
ax1.plot(np.log(TT4 / (1024 * 1024 * 1024)), color='#9cce37', label='4G Network Traffic', linewidth=2)
# ax1.set_xlabel('time slot', fontsize=14)
ax1.set_ylabel('Network Traffic ln(TByte)', fontsize=15)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=15)
plt.yticks(fontsize=16)
ax1.set_xlim(0, 336)
ax1.set_ylim(-0.1, 8)
ax2 = ax1.twinx()
ax2.plot(np.log(PP4 / (1_000_000)), 'k', label='4G Network Energy', linewidth=1.5)
ax2.plot(np.log(TT4 / (1024 * 1024 * 1024)) / np.log(Tm4 / (1024 * 1024 * 1024)) * np.log(Pm4 / (1_000_000)), '#ff4500',
         label='4G Desired Energy', linewidth=1.5)
ax2.fill_between(np.arange(336), np.log(PP4 / (1_000_000)),
                 np.log(TT4 / (1024 * 1024 * 1024)) / np.log(Tm4 / (1024 * 1024 * 1024)) * np.log(Pm4 / (1_000_000)),
                 color='#ff4500', alpha=0.2)
ax2.set_ylabel('Network Energy ln(MWh)', fontsize=15)
ax2.set_ylim(-0.04, 5)
fig.legend(loc=2, bbox_to_anchor=(-0.01, 1), ncol=2, fontsize=13, bbox_transform=ax1.transAxes)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('./fig2/Traffic_and_Power_4G.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot j

fig = plt.figure(figsize=(6.8, 3.8))
ax1 = fig.add_subplot(111)
# ax1.set_xlabel('time slot', fontsize=14)
ax1.set_ylabel('Network Traffic ln(TByte)', fontsize=15)
ax1.set_xlim(0, 336)
ax1.set_ylim(-0.12, 8)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=15)
plt.yticks(fontsize=15)
ax2 = ax1.twinx()
ax2.plot(np.log(PP5 / (1_000_000) + 1), 'k', label='5G Network Energy', linewidth=1.5)
ax2.plot(
    np.log(TT5 / (1024 * 1024 * 1024) + 1) / np.log(Tm5 / (1024 * 1024 * 1024) + 1) * np.log(Pm5 / (1_000_000) + 1),
    '#ffaf1d', label='5G Desired Energy', linewidth=1.5)
ax2.fill_between(np.arange(336), np.log(PP5 / (1_000_000) + 1),
                 np.log(TT5 / (1024 * 1024 * 1024) + 1) / np.log(Tm5 / (1024 * 1024 * 1024) + 1) * np.log(
                     Pm5 / (1_000_000) + 1), color='#ffaf1d', alpha=0.2)
ax2.set_ylabel('Network Energy ln(MWh)', fontsize=15)
ax2.set_ylim(-0.08, 4)
ax1.plot(np.log(TT5 / (1024 * 1024 * 1024) + 1), color='#3872aa', label='5G Network Traffic', linewidth=2)
fig.legend(loc=2, bbox_to_anchor=(-0.01, 1), ncol=2, fontsize=13, bbox_transform=ax1.transAxes)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('./fig2/Traffic_and_Power_5G.pdf', dpi=600, format='pdf')

########## Figure 3 ##########

# Plot a

x = np.arange(336)
Tc = np.load('./data/current/T_all_current.npy', allow_pickle=True).item()
x41 = Tc['4G']
x51 = Tc['5G']
yy1 = NP['current']['']['4G'] / Pm4 - x41 / Tm4
yy2 = NP['current']['']['5G'] / Pm5 - x51 / Tm5
yy3 = NP['old']['']['4G'] / Pm4 - T['old']['']['4G'] / Tm4
yy4 = NP['old']['']['5G'] / Pm5 - T['old']['']['5G'] / Tm5
yy5 = (NP['current']['']['4G'] + NP['current']['']['5G']) / (Pm4 + Pm5) - (x51 + x41) / (Tm4 + Tm5)
yy6 = (NP['old']['']['4G'] + NP['old']['']['5G']) / (Pm4 + Pm5) - (T['old']['']['4G'] + T['old']['']['5G']) / (
        Tm4 + Tm5)
# y3 = np.abs((PP4+PP5)/(np.max(PP4+PP5))-(TT4+TT5)/(np.max(TT4+TT5)))

y1_smooth = np.convolve(yy1, np.arange(3), 'same') / 3
y2_smooth = np.convolve(yy2, np.arange(3), 'same') / 3
y3_smooth = np.convolve(yy3, np.arange(3), 'same') / 3
y4_smooth = np.convolve(yy4, np.arange(3), 'same') / 3
y5_smooth = np.convolve(yy5, np.arange(3), 'same') / 3
y6_smooth = np.convolve(yy6, np.arange(3), 'same') / 3

fig = plt.figure(figsize=(12, 3.5))
plt.plot(x, y5_smooth, color='#3872aa', linewidth=2, label='Energy-Saving Method')
plt.plot(x, y6_smooth, color='k', linewidth=2, label='Network Misalignment')
# plt.plot(x, y3_smooth, linewidth=2, label = '4G and 5G')
# plt.fill_between(x, y5_smooth, facecolor='#20B2AB', label='Network Mobiocm Misalignment')
plt.fill_between(x, y5_smooth, y6_smooth, facecolor='c', label='Misalignment Reduction', alpha=0.3)
# plt.xlabel('time slot')
plt.ylabel('Misalignment Factor', fontsize=22)
plt.ylim(0.15, 0.75)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=22)
plt.yticks(fontsize=22)
# plt.title('Misalignment of 5G Base Stations')
plt.xlim(1, 335)
fig.legend(loc=2, bbox_to_anchor=(-0.09, 0.859), ncol=2, fontsize=20, bbox_transform=ax1.transAxes)
plt.savefig('./fig3/Current_' + sen + '_45G.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot c

day_ticks = [1, 48, 96, 144, 192, 240, 288, 335]
day_labels = ['5-20', '5-21', '5-22', '5-23', '5-24', '5-25', '5-26', '5-27']
sen_list = ['', 'Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9',
            'Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9']
x = np.arange(336)
sen = sen_list[0]
yy1 = NP['mobicom'][sen]['4G'] / Pm4 - T['mobicom'][sen]['4G'] / Tm4
yy2 = NP['mobicom'][sen]['5G'] / Pm5 - T['mobicom'][sen]['5G'] / Tm5
yy3 = NP['old'][sen]['4G'] / Pm4 - T['old'][sen]['4G'] / Tm4
yy4 = NP['old'][sen]['5G'] / Pm5 - T['old'][sen]['5G'] / Tm5
yy5 = (NP['mobicom'][sen]['4G'] + NP['mobicom'][sen]['5G']) / (Pm4 + Pm5) - (
        T['mobicom'][sen]['4G'] + T['mobicom'][sen]['5G']) / (Tm4 + Tm5)
yy6 = (NP['old'][sen]['4G'] + NP['old'][sen]['5G']) / (Pm4 + Pm5) - (T['old'][sen]['4G'] + T['old'][sen]['5G']) / (
        Tm4 + Tm5)

y1_smooth = np.convolve(yy1, np.arange(3), 'same') / 3
y2_smooth = np.convolve(yy2, np.arange(3), 'same') / 3
y3_smooth = np.convolve(yy3, np.arange(3), 'same') / 3
y4_smooth = np.convolve(yy4, np.arange(3), 'same') / 3
y5_smooth = np.convolve(yy5, np.arange(3), 'same') / 3
y6_smooth = np.convolve(yy6, np.arange(3), 'same') / 3

fig = plt.figure(figsize=(12, 3.5))
plt.plot(x, y5_smooth, color='#3872aa', linewidth=2, label='Energy-Saving Method')
plt.plot(x, y6_smooth, color='k', linewidth=2, label='Network Misalignment')
plt.fill_between(x, y5_smooth, y6_smooth, facecolor='c', label='Misalignment Reduction', alpha=0.3)
plt.ylabel('Misalignment Factor', fontsize=22)
plt.ylim(0.15, 0.75)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(1, 335)
fig.legend(loc=2, bbox_to_anchor=(0.12, 1.25), ncol=2, fontsize=20)
plt.savefig('./fig3/mobicom_' + sen + '_45G.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot e

day_ticks = [1, 48, 96, 144, 192, 240, 288, 335]
day_labels = ['5-20', '5-21', '5-22', '5-23', '5-24', '5-25', '5-26', '5-27']
sen_list = ['', 'Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9',
            'Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9']
x = np.arange(336)
sen = sen_list[0]
yy1 = NP['RL'][sen]['4G'] / Pm4 - T_RL[sen]['4G'] / Tm4
yy2 = NP['RL'][sen]['5G'] / Pm5 - T_RL[sen]['5G'] / Tm5
yy3 = NP['old'][sen]['4G'] / Pm4 - T['old'][sen]['4G'] / Tm4
yy4 = NP['old'][sen]['5G'] / Pm5 - T['old'][sen]['5G'] / Tm5
yy5 = (NP['RL'][sen]['4G'] + NP['RL'][sen]['5G']) / (Pm4 + Pm5) - (T_RL[sen]['4G'] + T_RL[sen]['5G']) / (Tm4 + Tm5)
yy6 = (NP['old'][sen]['4G'] + NP['old'][sen]['5G']) / (Pm4 + Pm5) - (T['old'][sen]['4G'] + T['old'][sen]['5G']) / (
        Tm4 + Tm5)

y1_smooth = np.convolve(yy1, np.arange(3), 'same') / 3
y2_smooth = np.convolve(yy2, np.arange(3), 'same') / 3
y3_smooth = np.convolve(yy3, np.arange(3), 'same') / 3
y4_smooth = np.convolve(yy4, np.arange(3), 'same') / 3
y5_smooth = np.convolve(yy5, np.arange(3), 'same') / 3
y6_smooth = np.convolve(yy6, np.arange(3), 'same') / 3

fig = plt.figure(figsize=(12, 3.5))
plt.plot(x, y5_smooth, color='#3872aa', linewidth=2, label='Energy-Saving Method')
plt.plot(x, y6_smooth, color='k', linewidth=2, label='Network Misalignment')
plt.fill_between(x, y5_smooth, y6_smooth, facecolor='c', label='Misalignment Reduction', alpha=0.3)
# plt.xlabel('time slot')
plt.ylabel('Misalignment Factor', fontsize=22)
plt.ylim(0.15, 0.75)
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=22)
plt.yticks(fontsize=22)
# plt.title('Misalignment of 5G Base Stations')
plt.xlim(1, 335)
fig.legend(loc=2, bbox_to_anchor=(0.12, 1.25), ncol=2, fontsize=20)
plt.savefig('./fig3/RL_' + sen + '_45G.pdf', dpi=600, format='pdf', bbox_inches='tight')


# Plot g


def func(x, a, b, c, d):
    return a * x ** 2 + b * x + c * x ** 3 + d


Cell_Capacity_4G = np.load('./data/data_old/Cell_Capacity_4G.npy', allow_pickle=True).item()
Cell_Capacity_5G = np.load('./data/data_old/Cell_Capacity_5G.npy', allow_pickle=True).item()

Capacity_4G = 0
for Cell_ID in Cell_Capacity_4G:
    Capacity_4G = Capacity_4G + Cell_Capacity_4G[Cell_ID]['Capacity']  # (千字节)
Capacity_4G = Capacity_4G / 1024 / 1024 / 1024 / 1024 * 96  # 统计数值是15分钟粒度的 PB 一天的

Capacity_5G = 0
for Cell_ID in Cell_Capacity_5G:
    Capacity_5G = Capacity_5G + Cell_Capacity_5G[Cell_ID]['Capacity']  # (千字节)
Capacity_5G = Capacity_5G / 1024 / 1024 / 1024 / 1024 * 96  # 统计数值是15分钟粒度的 PB 一天的

Date_Before5G = np.linspace(0, 7, 8)
Date_After5G = np.linspace(7, 23, 17)
Date_Future = np.linspace(23, 42, 20)

Network_Capacity_Before5G = []
Network_Capacity_After5G = []
Network_Capacity_Future = []
for index in range(len(Date_Before5G)):
    Network_Capacity_Before5G.append(Capacity_4G)
for index in range(len(Date_After5G)):
    Network_Capacity_After5G.append(Capacity_4G + Capacity_5G)
for index in range(len(Date_Future)):
    Network_Capacity_Future.append(Capacity_4G + Capacity_5G)

Network_Traffic_Before5G = []
for index in range(len(Date_Before5G)):
    Network_Traffic_Before5G.append(1.21042719)

Network_Traffic_After5G_4G = [1.15357101, 1.312790426, 1.399133649, 1.464940381, 1.513061395, 1.524572747, 1.53056248, \
                              1.533335786, 1.473622746, 1.431067853, 1.470064229, 1.498515583, 1.461390619, 1.415919433,
                              1.387327911, 1.382028557, 1.37190577]
Network_Traffic_After5G_5G = [0.056856181, 0.062154632, 0.136420432, 0.167308413, 0.195333782, 0.223651352, 0.231856969,
                              0.240518264, \
                              0.351768637, 0.382075861, 0.386963403, 0.396250525, 0.439388366, 0.488138965, 0.528031143,
                              0.540033872, 0.57771561]
Network_Traffic_After5G = np.array(Network_Traffic_After5G_4G) + np.array(Network_Traffic_After5G_5G)

popt, pcov = curve_fit(func, np.array(Date_After5G), np.array(Network_Traffic_After5G))
Network_Traffic_Future = func(np.array(Date_Future), popt[0], popt[1], popt[2], popt[3])
Network_Traffic_Future[0] = Network_Traffic_After5G[-1]

Network_Capacity_Before5G_log = [np.log(value) for value in Network_Capacity_Before5G]
Network_Capacity_After5G_log = [np.log(value) for value in Network_Capacity_After5G]
Network_Capacity_Future_log = [np.log(value) for value in Network_Capacity_Future]

Network_Traffic_After5G_4G_log = []
Network_Traffic_After5G_5G_log = []
for index in range(len(Network_Traffic_After5G_4G)):
    Network_Traffic_After5G_log = np.log(Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])
    Network_Traffic_After5G_4G_log.append(Network_Traffic_After5G_log * (Network_Traffic_After5G_4G[index] / (
            Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])))
    Network_Traffic_After5G_5G_log.append(Network_Traffic_After5G_log * (Network_Traffic_After5G_5G[index] / (
            Network_Traffic_After5G_4G[index] + Network_Traffic_After5G_5G[index])))

Network_Traffic_Before5G_log = [np.log(value) for value in Network_Traffic_Before5G]
Network_Traffic_Future_log = [np.log(value) for value in Network_Traffic_Future]

Network_Energy_error_Before5G = 0.00758
Current_Energy_error_Before5G = 0.01060
Mobicom_Energy_error_Before5G = 0.00506
RL_Energy_error_Before5G = 0.01590
K = 0.6131
K_std = 0.0061
K1 = 0.5261
K1_std = 0.053
K2 = 0.4387
K2_std = 0.0044
K3 = 0.2856
K3_std = 0.0029
gama = 0.9527
gama_std = 0.0515

Network_Energy_Before5G = (np.array(Network_Traffic_Before5G) / np.array(
    Network_Capacity_Before5G) * 0.454 + 0.546) * 43160884 / 1_000_000 * 24
Network_Energy_After5G = (np.array(Network_Traffic_After5G) / np.array(Network_Capacity_After5G) * (1 - K) + K) * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future = (np.array(Network_Traffic_Future) / np.array(Network_Capacity_Future) * (1 - K) + K) * (
        17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Before5G = (np.array(Network_Traffic_Before5G) / np.array(
    Network_Capacity_Before5G) * 0.587 + 0.423) * 43160884 / 1_000_000 * 24
Current_Energy_After5G = (np.array(Network_Traffic_After5G) / np.array(Network_Capacity_After5G) * (1 - K1) + K1) * (
        17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Future = (np.array(Network_Traffic_Future) / np.array(Network_Capacity_Future) * (1 - K1) + K1) * (
        17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Before5G = (np.array(Network_Traffic_Before5G) / np.array(
    Network_Capacity_Before5G) * 0.576 + 0.424) * 43160884 / 1_000_000 * 24
Mobicom_Energy_After5G = (np.array(Network_Traffic_After5G) / np.array(Network_Capacity_After5G) * (1 - K2) + K2) * (
        17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Future = (np.array(Network_Traffic_Future) / np.array(Network_Capacity_Future) * (1 - K2) + K2) * (
        17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Before5G = (np.array(Network_Traffic_Before5G) / np.array(
    Network_Capacity_Before5G) * 0.747 + 0.253) * 43160884 / 1_000_000 * 24
RL_Energy_After5G = (np.array(Network_Traffic_After5G) / np.array(Network_Capacity_After5G) * (1 - K3) + K3) * (
        17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Future = (np.array(Network_Traffic_Future) / np.array(Network_Capacity_Future) * (1 - K3) + K3) * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Before5G_low = Network_Energy_Before5G - Network_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Network_Energy_After5G_low = Network_Energy_After5G - Network_Energy_error_After5G * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future_low = Network_Energy_Future - Network_Energy_error_Future * (17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Before5G_low = Current_Energy_Before5G - Current_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Current_Energy_After5G_low = Current_Energy_After5G - K1_std * (
        17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Future_low = Current_Energy_Future - K1_std * (17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Before5G_low = Mobicom_Energy_Before5G - Mobicom_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Mobicom_Energy_After5G_low = Mobicom_Energy_After5G - K2_std * (
        17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Future_low = Mobicom_Energy_Future - K2_std * (17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Before5G_low = RL_Energy_Before5G - RL_Energy_error_Before5G * 43160884 / 1_000_000 * 24
RL_Energy_After5G_low = RL_Energy_After5G - K3_std * (17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Future_low = RL_Energy_Future - K3_std * (17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Before5G_high = Network_Energy_Before5G + Network_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Network_Energy_After5G_high = Network_Energy_After5G + Network_Energy_error_After5G * (
        17675608 + 43160884) / 1_000_000 * 24
Network_Energy_Future_high = Network_Energy_Future + Network_Energy_error_Future * (
        17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Before5G_high = Current_Energy_Before5G + Current_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Current_Energy_After5G_high = Current_Energy_After5G + K1_std * (
        17675608 + 43160884) / 1_000_000 * 24
Current_Energy_Future_high = Current_Energy_Future + K1_std * (
        17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Before5G_high = Mobicom_Energy_Before5G + Mobicom_Energy_error_Before5G * 43160884 / 1_000_000 * 24
Mobicom_Energy_After5G_high = Mobicom_Energy_After5G + K2_std * (
        17675608 + 43160884) / 1_000_000 * 24
Mobicom_Energy_Future_high = Mobicom_Energy_Future + K2_std * (
        17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Before5G_high = RL_Energy_Before5G + RL_Energy_error_Before5G * 43160884 / 1_000_000 * 24
RL_Energy_After5G_high = RL_Energy_After5G + K3_std * (17675608 + 43160884) / 1_000_000 * 24
RL_Energy_Future_high = RL_Energy_Future + K3_std * (17675608 + 43160884) / 1_000_000 * 24

Network_Energy_Before5G = np.array(Network_Energy_Before5G)
Network_Energy_After5G = np.array(Network_Energy_After5G)
Network_Energy_Future = np.array(Network_Energy_Future)
Network_Traffic_Before5G = np.array(Network_Traffic_Before5G)
Network_Traffic_After5G = np.array(Network_Traffic_After5G)
Network_Traffic_Future = np.array(Network_Traffic_Future)
Network_Capacity_Before5G = np.array(Network_Capacity_Before5G)
Network_Capacity_After5G = np.array(Network_Capacity_After5G)
Network_Capacity_Future = np.array(Network_Capacity_Future)

Network_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G
Network_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / Network_Energy_After5G
Network_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / Network_Energy_Future
Network_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_low
Network_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / Network_Energy_After5G_low
Network_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / Network_Energy_Future_low
Network_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / Network_Energy_Before5G_high
Network_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / Network_Energy_After5G_high
Network_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / Network_Energy_Future_high

Current_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / Current_Energy_Before5G
Current_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / Current_Energy_After5G
Current_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / Current_Energy_Future
Current_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / Current_Energy_Before5G_low
Current_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / Current_Energy_After5G_low
Current_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / Current_Energy_Future_low
Current_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / Current_Energy_Before5G_high
Current_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / Current_Energy_After5G_high
Current_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / Current_Energy_Future_high

Mobicom_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / Mobicom_Energy_Before5G
Mobicom_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / Mobicom_Energy_After5G
Mobicom_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / Mobicom_Energy_Future
Mobicom_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / Mobicom_Energy_Before5G_low
Mobicom_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / Mobicom_Energy_After5G_low
Mobicom_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / Mobicom_Energy_Future_low
Mobicom_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / Mobicom_Energy_Before5G_high
Mobicom_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / Mobicom_Energy_After5G_high
Mobicom_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / Mobicom_Energy_Future_high

RL_Energy_Efficiency_Before5G = 1024 * Network_Traffic_Before5G / RL_Energy_Before5G
RL_Energy_Efficiency_After5G = 1024 * Network_Traffic_After5G / RL_Energy_After5G
RL_Energy_Efficiency_Future = 1024 * Network_Traffic_Future / RL_Energy_Future
RL_Energy_Efficiency_Before5G_low = 1024 * Network_Traffic_Before5G / RL_Energy_Before5G_low
RL_Energy_Efficiency_After5G_low = 1024 * Network_Traffic_After5G / RL_Energy_After5G_low
RL_Energy_Efficiency_Future_low = 1024 * Network_Traffic_Future / RL_Energy_Future_low
RL_Energy_Efficiency_Before5G_high = 1024 * Network_Traffic_Before5G / RL_Energy_Before5G_high
RL_Energy_Efficiency_After5G_high = 1024 * Network_Traffic_After5G / RL_Energy_After5G_high
RL_Energy_Efficiency_Future_high = 1024 * Network_Traffic_Future / RL_Energy_Future_high

Network_Carbon_error = 0.0515

Network_Carbon_Before5G = (Network_Energy_Before5G / 24 * gama) * 24
Network_Carbon_After5G = (Network_Energy_After5G / 24 * gama) * 24
Network_Carbon_Future = (Network_Energy_Future / 24 * gama) * 24
Network_Carbon_Before5G_low = (Network_Energy_Before5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_After5G_low = (Network_Energy_After5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_Future_low = (Network_Energy_Future_low / 24 * gama) * 24 - Network_Carbon_error * 24
Network_Carbon_Before5G_high = (Network_Energy_Before5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Network_Carbon_After5G_high = (Network_Energy_After5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Network_Carbon_Future_high = (Network_Energy_Future_high / 24 * gama) * 24 + Network_Carbon_error * 24

Current_Carbon_Before5G = (Current_Energy_Before5G / 24 * gama) * 24
Current_Carbon_After5G = (Current_Energy_After5G / 24 * gama) * 24
Current_Carbon_Future = (Current_Energy_Future / 24 * gama) * 24
Current_Carbon_Before5G_low = (Current_Energy_Before5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Current_Carbon_After5G_low = (Current_Energy_After5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Current_Carbon_Future_low = (Current_Energy_Future_low / 24 * gama) * 24 - Network_Carbon_error * 24
Current_Carbon_Before5G_high = (Current_Energy_Before5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Current_Carbon_After5G_high = (Current_Energy_After5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Current_Carbon_Future_high = (Current_Energy_Future_high / 24 * gama) * 24 + Network_Carbon_error * 24

Mobicom_Carbon_Before5G = (Mobicom_Energy_Before5G / 24 * gama) * 24
Mobicom_Carbon_After5G = (Mobicom_Energy_After5G / 24 * gama) * 24
Mobicom_Carbon_Future = (Mobicom_Energy_Future / 24 * gama) * 24
Mobicom_Carbon_Before5G_low = (Mobicom_Energy_Before5G_low / 24 * gama) * 24  - Network_Carbon_error * 24
Mobicom_Carbon_After5G_low = (Mobicom_Energy_After5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
Mobicom_Carbon_Future_low = (Mobicom_Energy_Future_low / 24 * gama) * 24 - Network_Carbon_error * 24
Mobicom_Carbon_Before5G_high = (Mobicom_Energy_Before5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Mobicom_Carbon_After5G_high = (Mobicom_Energy_After5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
Mobicom_Carbon_Future_high = (Mobicom_Energy_Future_high / 24 * gama) * 24 + Network_Carbon_error * 24

RL_Carbon_Before5G = (RL_Energy_Before5G / 24 * gama) * 24
RL_Carbon_After5G = (RL_Energy_After5G / 24 * gama) * 24
RL_Carbon_Future = (RL_Energy_Future / 24 * gama) * 24
RL_Carbon_Before5G_low = (RL_Energy_Before5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
RL_Carbon_After5G_low = (RL_Energy_After5G_low / 24 * gama) * 24 - Network_Carbon_error * 24
RL_Carbon_Future_low = (RL_Energy_Future_low / 24 * gama) * 24 - Network_Carbon_error * 24
RL_Carbon_Before5G_high = (RL_Energy_Before5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
RL_Carbon_After5G_high = (RL_Energy_After5G_high / 24 * gama) * 24 + Network_Carbon_error * 24
RL_Carbon_Future_high = (RL_Energy_Future_high / 24 * gama) * 24 + Network_Carbon_error * 24

Current_Carbon_Reduction_After5G = Network_Carbon_After5G - Current_Carbon_After5G
Current_Carbon_Reduction_After5G_low = Network_Carbon_After5G - Current_Carbon_After5G_high
Current_Carbon_Reduction_After5G_high = Network_Carbon_After5G - Current_Carbon_After5G_low
Current_Carbon_Reduction_Future = Network_Carbon_Future - Current_Carbon_Future
Current_Carbon_Reduction_Future_low = Network_Carbon_Future - Current_Carbon_Future_high
Current_Carbon_Reduction_Future_high = Network_Carbon_Future - Current_Carbon_Future_low

Mobicom_Carbon_Reduction_After5G = Network_Carbon_After5G - Mobicom_Carbon_After5G
Mobicom_Carbon_Reduction_After5G_low = Network_Carbon_After5G - Mobicom_Carbon_After5G_high
Mobicom_Carbon_Reduction_After5G_high = Network_Carbon_After5G - Mobicom_Carbon_After5G_low
Mobicom_Carbon_Reduction_Future = Network_Carbon_Future - Mobicom_Carbon_Future
Mobicom_Carbon_Reduction_Future_low = Network_Carbon_Future - Mobicom_Carbon_Future_high
Mobicom_Carbon_Reduction_Future_high = Network_Carbon_Future - Mobicom_Carbon_Future_low

RL_Carbon_Reduction_After5G = Network_Carbon_After5G - RL_Carbon_After5G
RL_Carbon_Reduction_After5G_low = Network_Carbon_After5G - RL_Carbon_After5G_high
RL_Carbon_Reduction_After5G_high = Network_Carbon_After5G - RL_Carbon_After5G_low
RL_Carbon_Reduction_Future = Network_Carbon_Future - RL_Carbon_Future
RL_Carbon_Reduction_Future_low = Network_Carbon_Future - RL_Carbon_Future_high
RL_Carbon_Reduction_Future_high = Network_Carbon_Future - RL_Carbon_Future_low

Network_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(Network_Carbon_Before5G)
Network_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(Network_Carbon_After5G)
Network_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future)
Network_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Network_Carbon_Before5G_low)
Network_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Network_Carbon_Before5G_high)
Network_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(Network_Carbon_After5G_low)
Network_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(
    Network_Carbon_After5G_high)
Network_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future_low)
Network_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(Network_Carbon_Future_high)

Mobicom_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(Current_Carbon_Before5G)
Current_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(Current_Carbon_After5G)
Current_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(Current_Carbon_Future)
Current_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Current_Carbon_Before5G_low)
Current_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Current_Carbon_Before5G_high)
Current_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(Current_Carbon_After5G_low)
Current_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(
    Current_Carbon_After5G_high)
Current_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(Current_Carbon_Future_low)
Current_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(Current_Carbon_Future_high)

Mobicom_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(Mobicom_Carbon_Before5G)
Mobicom_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(Mobicom_Carbon_After5G)
Mobicom_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(Mobicom_Carbon_Future)
Mobicom_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Mobicom_Carbon_Before5G_low)
Mobicom_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(
    Mobicom_Carbon_Before5G_high)
Mobicom_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(Mobicom_Carbon_After5G_low)
Mobicom_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(
    Mobicom_Carbon_After5G_high)
Mobicom_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(Mobicom_Carbon_Future_low)
Mobicom_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(Mobicom_Carbon_Future_high)

RL_Carbon_Efficiency_Before5G = np.array(Network_Traffic_Before5G) * 1024 / np.array(RL_Carbon_Before5G)
RL_Carbon_Efficiency_After5G = np.array(Network_Traffic_After5G) * 1024 / np.array(RL_Carbon_After5G)
RL_Carbon_Efficiency_Future = np.array(Network_Traffic_Future) * 1024 / np.array(RL_Carbon_Future)
RL_Carbon_Efficiency_Before5G_low = np.array(Network_Traffic_Before5G) * 1024 / np.array(RL_Carbon_Before5G_low)
RL_Carbon_Efficiency_Before5G_high = np.array(Network_Traffic_Before5G) * 1024 / np.array(RL_Carbon_Before5G_high)
RL_Carbon_Efficiency_After5G_low = np.array(Network_Traffic_After5G) * 1024 / np.array(RL_Carbon_After5G_low)
RL_Carbon_Efficiency_After5G_high = np.array(Network_Traffic_After5G) * 1024 / np.array(RL_Carbon_After5G_high)
RL_Carbon_Efficiency_Future_low = np.array(Network_Traffic_Future) * 1024 / np.array(RL_Carbon_Future_low)
RL_Carbon_Efficiency_Future_high = np.array(Network_Traffic_Future) * 1024 / np.array(RL_Carbon_Future_high)

cloud = plt.figure(figsize=(30, 10))
axes = cloud.add_axes([0.1, 0.2, 0.32, 0.5])
# axes.plot(Date_Before5G, Network_Carbon_Efficiency_Before5G, color='olivedrab', linestyle='-', linewidth=2.4)
axes.plot(Date_Before5G, Network_Carbon_Efficiency_Before5G, color='crimson', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Before5G, Network_Carbon_Efficiency_Before5G_low, Network_Carbon_Efficiency_Before5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_After5G[0:9],
          np.concatenate((Network_Carbon_Efficiency_Before5G, Network_Carbon_Efficiency_Before5G))[0:9],
          color='crimson', linestyle=':', linewidth=2)
axes.fill_between(Date_After5G[0:9],
                  np.concatenate((Network_Carbon_Efficiency_Before5G, Network_Carbon_Efficiency_Before5G))[0:9],
                  Network_Carbon_Efficiency_After5G[0:9], facecolor='dimgray', hatch='//', edgecolor="w", linewidth=0.3,
                  alpha=0.5)

axes.plot(Date_After5G, Network_Carbon_Efficiency_After5G, color='crimson', linestyle='-', linewidth=2.4,
          label='W/O Energy Saving')
axes.fill_between(Date_After5G, Network_Carbon_Efficiency_After5G_low, Network_Carbon_Efficiency_After5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_Future, Network_Carbon_Efficiency_Future, color='crimson', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Future, Network_Carbon_Efficiency_Future_low, Network_Carbon_Efficiency_Future_high,
                  facecolor='#346fa9', alpha=0.5)

axes.plot(Date_After5G, Current_Carbon_Efficiency_After5G, color='#9cce37', linestyle='-', linewidth=2.4,
          label='Threshold-Based')
axes.fill_between(Date_After5G, Current_Carbon_Efficiency_After5G_low, Current_Carbon_Efficiency_After5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_Future, Current_Carbon_Efficiency_Future, color='#9cce37', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Future, Current_Carbon_Efficiency_Future_low, Current_Carbon_Efficiency_Future_high,
                  facecolor='#346fa9', alpha=0.5)

axes.plot(Date_After5G, Mobicom_Carbon_Efficiency_After5G, color='#20B2AB', linestyle='-', linewidth=2.4,
          label='Greedy')
axes.fill_between(Date_After5G, Mobicom_Carbon_Efficiency_After5G_low, Mobicom_Carbon_Efficiency_After5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_Future, Mobicom_Carbon_Efficiency_Future, color='#20B2AB', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Future, Mobicom_Carbon_Efficiency_Future_low, Mobicom_Carbon_Efficiency_Future_high,
                  facecolor='#346fa9', alpha=0.5)

axes.plot(Date_After5G, RL_Carbon_Efficiency_After5G, color='#CCCCFF', linestyle='-', linewidth=2.4,
          label='DeepEnergy')
axes.fill_between(Date_After5G, RL_Carbon_Efficiency_After5G_low, RL_Carbon_Efficiency_After5G_high,
                  facecolor='#346fa9', alpha=0.5)
axes.plot(Date_Future, RL_Carbon_Efficiency_Future, color='#CCCCFF', linestyle='-', linewidth=2.4)
axes.fill_between(Date_Future, RL_Carbon_Efficiency_Future_low, RL_Carbon_Efficiency_Future_high, facecolor='#346fa9',
                  alpha=0.5, label='Confidence interval')

axes.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')

axes.spines['bottom'].set_linewidth('1.0')
axes.spines['left'].set_linewidth('1.0')
axes.spines['right'].set_linewidth('1.0')
axes.spines['top'].set_linewidth('1.0')
axes.axvline(Date_After5G[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.axvline(Date_Future[0], c="purple", ls="--", alpha=0.8, lw=3)
axes.set_xticks([7, 23, 31, 42])
axes.set_xticklabels(['2021-01 (Launching 5G)', '2022-05', '2023-1', '2024-12'], fontsize=28)
axes.set_xlim(0, 31)
axes.tick_params(axis="y", labelsize=28)
axes.set_ylabel('Carbon Efficiency (TByte/tCO$_2$)', fontsize=28)
axes.set_ylim(0.9 * np.min(Network_Carbon_Efficiency_After5G), 4.8)
axes.text(Date_After5G[0] + 2, 1 * Network_Carbon_Efficiency_After5G[0], 'Carbon Efficiency Trap', fontsize=26)
plt.legend(fontsize=27, bbox_to_anchor=(-0.05, 1.45), loc=2, ncol=2)
cloud.savefig(r"fig3\Method Carbon Efficiency Temporal.pdf", format='pdf', bbox_inches='tight')

# Plot h

from pyecharts import options as opts
from pyecharts.charts import Map

ACP = np.load('./data/province/Additional_Carbon_Province.npy', allow_pickle=True).item()['Without Method']
ACPRL = np.load('./data/province/Additional_Carbon_Province.npy', allow_pickle=True).item()['RL-Based Method']
ACeP = np.load('./data/province/Additional_Carbon_std_Province.npy', allow_pickle=True).item()['Without Method']
ACePRL = np.load('./data/province/Additional_Carbon_std_Province.npy', allow_pickle=True).item()['RL-Based Method']
P4N = np.load('./data/province/Province_4GBS_Num.npy', allow_pickle=True).item()
P5N = np.load('./data/province/Province_5GBS_Num.npy', allow_pickle=True).item()
PNC = np.load('./data/province/Province_Network_Capacity.npy', allow_pickle=True).item()
PUN = np.load('./data/province/Province_User_Num.npy', allow_pickle=True).item()
WEP = np.load('./data/province/Wasted_Energy_Province.npy', allow_pickle=True).item()['Without Method']
WEeP = np.load('./data/province/Wasted_Energy_std_Province.npy', allow_pickle=True).item()['Without Method']

C_keys = {'北京市': 'Beijing', '天津市': 'Tianjing', '河北省': 'Hebei',
          '山西省': 'Shanxi', '内蒙古自治区': 'Inner Mongolia', '辽宁省': 'Liaoning',
          '吉林省': 'Jilin', '黑龙江省': 'Heilongjiang', '上海市': 'Shanghai',
          '江苏省': 'Jiangsu', '浙江省': 'Zhejiang', '安徽省': 'Anhui',
          '福建省': 'Fujian', '江西省': 'Jiangxi', '山东省': 'Shandong',
          '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan', '广东省': 'Guangdong',
          '广西壮族自治区': 'Guangxi', '海南省': 'Hainan', '重庆市': 'Chongqing',
          '四川省': 'Sichuan', '贵州省': 'Guizhou', '云南省': 'Yunnan',
          '西藏自治区': 'Tibet', '陕西省': 'Shaanxi', '甘肃省': 'Gansu', '青海省': 'Qinghai',
          '宁夏回族自治区': 'Ningxia', '新疆维吾尔自治区': 'Xinjiang'}
area = {'北京市': 16412, '天津市': 11903, '河北省': 187159,
        '山西省': 156698, '内蒙古自治区': 1196113, '辽宁省': 148084,
        '吉林省': 190234, '黑龙江省': 439703, '上海市': 6339,
        '江苏省': 102378, '浙江省': 103493, '安徽省': 139615,
        '福建省': 122870, '江西省': 167064, '山东省': 158219,
        '河南省': 166785, '湖北省': 185750, '湖南省': 211833, '广东省': 174246,
        '广西壮族自治区': 237438, '海南省': 30970, '重庆市': 82370,
        '四川省': 491718, '贵州省': 176161, '云南省': 394029,
        '西藏自治区': 1194047, '陕西省': 205629, '甘肃省': 454858, '青海省': 696610,
        '宁夏回族自治区': 66400, '新疆维吾尔自治区': 1640016}
name_list1 = ['Additional Carbon',
              'Num of 4GBSs',
              'Num of 5GBSs',
              'Network Capacity of 4G',
              'Network Capacity of 5G',
              'Network Capacity of 4G and 5G',
              'Network Capacity of 4G STD',
              'Network Capacity of 5G STD',
              'Network Capacity of 4G and 5G STD',
              'Num of Users',
              'Wasted Energy',
              'Additional Carbon per person',
              'Additional Carbon per square kilometer',
              'Additional Carbon after RL']
name_list2 = ['Additional_Carbon',
              '4GBS_Num',
              '5GBS_Num',
              'Network_Capacity_4G',
              'Network_Capacity_5G',
              'Network_Capacity_45G',
              'Network_Capacity_4G_STD',
              'Network_Capacity_5G_STD',
              'Network_Capacity_45G_STD',
              'User_Num',
              'Wasted_Energy',
              'Additional_Carbon_per_person',
              'Additional_Carbon_per_square_kilometer',
              'Additional_Carbon_after_RL']

name_list_csv = ['Additional Carbon (tCO2)',
                 'Additional Carbon errorbar (tCO2)',
                 'Num of 4GBSs',
                 'Num of 5GBSs',
                 'Network Capacity of 4G (PBtye)',
                 'Network Capacity of 5G (PBtye)',
                 'Network Capacity of 4G and 5G (PBtye)',
                 'Network Capacity of 4G STD (PBtye)',
                 'Network Capacity of 5G STD (PBtye)',
                 'Network Capacity of 4G and 5G STD (PBtye)',
                 'Num of Users (Million)',
                 'Wasted Energy (MWh)',
                 'Additional Carbon per person (kgCO2)',
                 'Additional Carbon per square kilometer (tCO2)',
                 'Additional Carbon after RL (tCO2)']
factor_list_csv = [1_000_000, 1, 1_000, 1_000, 1, 1, 1, 1, 1, 1, 1_000_000, 1_000_000, 1, 1]


def dataget(data, C_E):
    data_get = {}
    for p in C_E:
        break
    if type(data[C_E[p]]) == dict:
        for p in C_E:
            for k in data[C_E[p]]:
                if k not in data_get:
                    data_get[k] = {}
                data_get[k][p] = data[C_E[p]][k]
    elif type(data[C_E[p]]) == list:
        for p in C_E:
            data_get[p] = np.mean(data[C_E[p]])
    else:
        for p in C_E:
            data_get[p] = data[C_E[p]]
    return data_get


data_list_plot = []
data_list = [ACP, ACeP, P4N, P5N, PNC, PUN, WEP]
i = 0
for data in data_list:
    dg = dataget(data, C_keys)
    if len(dg) < len(C_keys):
        d1 = {}
        d2 = {}
        for x in dg['4G']:
            d1[x] = dg['4G'][x] + dg['5G'][x]
            d2[x] = dg['4GSTD'][x] + dg['5GSTD'][x]
        data_list_plot.append(dg['4G'])
        i += 1
        data_list_plot.append(dg['5G'])
        i += 1
        data_list_plot.append(d1)
        i += 1
        data_list_plot.append(dg['4GSTD'])
        i += 1
        data_list_plot.append(dg['5GSTD'])
        i += 1
        data_list_plot.append(d2)
        i += 1
    else:
        data_list_plot.append(dg)
        i += 1
dg = {}
for x in data_list_plot[0]:
    dg[x] = data_list_plot[0][x] / (data_list_plot[10][x] * 10)
data_list_plot.append(dg)
i += 1
dg = {}
for x in data_list_plot[0]:
    dg[x] = data_list_plot[0][x] / area[x]
data_list_plot.append(dg)
i += 1
dg = {}
for x in data_list_plot[0]:
    dg[x] = ACPRL[C_keys[x]]
data_list_plot.append(dg)
i += 1
idx_province = []
for x in C_keys:
    idx_province.append(C_keys[x])

s1v = [[0, 400000], [400000, 800000], [800000, 1200000], [1200000, 1600000],
       [1600000, 2000000], [2000000, 2800000], [2800000, 5000000]]
s1n = [['0', '40W'], ['40W', '80W'], ['80W', '120W'], ['120W', '160W'],
       ['160W', '200W'], ['200W', '280W'], ['280W', '500W']]
s2v = [[0, 20000], [20000, 40000], [40000, 60000], [60000, 80000],
       [80000, 100000], [100000, 130000], [130000, 160000]]
s2n = [['0', '20K'], ['20K', '40K'], ['40K', '60K'], ['60K', '80K'],
       ['80K', '100K'], ['100K', '130K'], ['130K', '160K']]
s3v = [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000],
       [40000, 50000], [50000, 60000], [60000, 100000]]
s3n = [['0', '10K'], ['10K', '20K'], ['20K', '30K'], ['30K', '40K'],
       ['40K', '50K'], ['50K', '60K'], ['60K', '100K']]
s4v = [[0, 10], [10, 20], [20, 40], [40, 60],
       [60, 80], [80, 100], [100, 130]]
s4n = [['0', '10'], ['10', '20'], ['20', '40'], ['40', '60'],
       ['60', '80'], ['80', '100'], ['100', '130']]
s5v = [[0, 30], [30, 60], [60, 100], [100, 150],
       [150, 200], [200, 300], [300, 500]]
s5n = [['0', '30'], ['30', '60'], ['60', '100'], ['100', '150'],
       ['150', '200'], ['200', '300'], ['300', '500']]
s6v = [[0, 40], [40, 80], [80, 140], [140, 210],
       [210, 280], [280, 400], [400, 650]]
s6n = [['0', '40'], ['40', '80'], ['80', '140'], ['140', '210'],
       ['210', '280'], ['280', '400'], ['400', '650']]
s7v = [[0, 10], [10, 20], [20, 30], [30, 40],
       [40, 60], [60, 80], [80, 100]]
s7n = [['0', '10'], ['10', '20'], ['20', '30'], ['30', '40'],
       ['40', '60'], ['60', '80'], ['80', '100']]
s8v = [[0, 10], [10, 20], [20, 40], [40, 60],
       [60, 80], [80, 100], [100, 200]]
s8n = [['0', '10'], ['10', '20'], ['20', '40'], ['40', '60'],
       ['60', '80'], ['80', '100'], ['100', '200']]
s9v = [[0, 20], [20, 40], [40, 70], [70, 100],
       [100, 140], [140, 180], [180, 300]]
s9n = [['0', '20'], ['20', '40'], ['40', '70'], ['70', '100'],
       ['100', '140'], ['140', '180'], ['180', '300']]
s10v = [[0, 1000], [1000, 2000], [2000, 3000], [3000, 5000],
        [5000, 8000], [8000, 10000], [10000, 20000]]
s10n = [['0', '1K'], ['1K', '2K'], ['2K', '3K'], ['3K', '5K'],
        ['5K', '8K'], ['8K', '10K'], ['10K', '20K']]
s11v = [[0, 200000], [200000, 400000], [400000, 600000], [600000, 1000000],
        [1000000, 1600000], [1600000, 2400000], [2400000, 5000000]]
s11n = [['0', '200'], ['200', '400'], ['400', '600'], ['600', '1000'],
        ['1000', '1600'], ['1600', '2400'], ['2400', '5000']]
s12v = [[0, 80], [80, 120], [120, 160], [160, 200],
        [200, 250], [250, 350], [350, 500]]
s12n = [['0', '80'], ['80', '120'], ['120', '160'], ['160', '200'],
        ['200', '250'], ['250', '350'], ['350', '500']]
s13v = [[0, 0.3], [0.3, 1.5], [1.5, 3], [3, 7],
        [7, 11], [11, 30], [30, 300]]
s13n = [['0', '0.3'], ['0.3', '1.5'], ['1.5', '3'], ['3', '7'],
        ['7', '11'], ['11', '30'], ['30', '300']]
s14v = [[0, 1], [1, 10000], [10000, 20000], [20000, 30000],
        [30000, 50000], [50000, 100000], [100000, 150000]]
s14n = [['0', ''], ['0', '1W'], ['1W', '2W'], ['2W', '3W'],
        ['3W', '5W'], ['5W', '10W'], ['10W', '15W']]
step_value_list = [s1v, s2v, s3v, s4v, s5v, s6v, s7v,
                   s8v, s9v, s10v, s11v, s12v, s13v, s14v]
step_name_list = [s1n, s2n, s3n, s4n, s5n, s6n, s7n,
                  s8n, s9n, s10n, s11n, s12n, s13n, s14n]


def plot_province(data, name1, name2, step_value, step_name, i):
    province_dis4 = data
    provice = list(province_dis4.keys())
    values = list(province_dis4.values())

    china = (
        Map(opts.InitOpts(width='1200px', height='900px'))
        .add("", [list(z) for z in zip(provice, values)], "china", is_map_symbol_show=False)
        .set_global_opts(title_opts=opts.TitleOpts(title=name1),
                         legend_opts=opts.LegendOpts(type_="scroll", pos_left="left", orient="vertical"),
                         visualmap_opts=opts.VisualMapOpts(max_=step_value[6][1], is_piecewise=True, is_inverse=True,
                                                           pieces=[
                                                               {"min": step_value[0][0], "max": step_value[0][1],
                                                                "label": step_name[0][0] + ' - ' + step_name[0][1]},
                                                               {"min": step_value[1][0], "max": step_value[1][1],
                                                                "label": step_name[1][0] + ' - ' + step_name[1][1]},
                                                               {"min": step_value[2][0], "max": step_value[2][1],
                                                                "label": step_name[2][0] + ' - ' + step_name[2][1]},
                                                               {"min": step_value[3][0], "max": step_value[3][1],
                                                                "label": step_name[3][0] + ' - ' + step_name[3][1]},
                                                               {"min": step_value[4][0], "max": step_value[4][1],
                                                                "label": step_name[4][0] + ' - ' + step_name[4][1]},
                                                               {"min": step_value[5][0], "max": step_value[5][1],
                                                                "label": step_name[5][0] + ' - ' + step_name[5][1]},
                                                               {"min": step_value[6][0], "max": step_value[6][1],
                                                                "label": step_name[6][0] + ' - ' + step_name[6][1]},
                                                               {"max": 0, "label": "No data", "color": "#F8F8FF"}
                                                           ],
                                                           range_color=['#217ed5', '#80b3ae', '#bed389', '#F8FB64',
                                                                        '#FCB245', '#F86D2A', '#E91515'],
                                                           pos_right='90%',
                                                           pos_bottom='50%',
                                                           split_number=8))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )

    china.render("./fig3/" + str(i + 1) + "_" + name2 + ".html")
    china.render_notebook()


i = 13
plot_province(data_list_plot[i + 1], name_list1[i], name_list2[i], step_value_list[i], step_name_list[i], i)

########## Figure ##########

# Plot a

T = np.load('./data/T_all.npy', allow_pickle=True).item()
T_RL = np.load('./data/T_RL.npy', allow_pickle=True).item()
CO = np.load('./data/CO.npy', allow_pickle=True).item()
NP = np.load('./data/NP.npy', allow_pickle=True).item()
PC = np.load('./data/PC.npy', allow_pickle=True).item()
T['RL'] = T_RL

day_ticks = [1, 48, 96, 144, 192, 240, 288, 335]
day_labels = ['5-20', '5-21', '5-22', '5-23', '5-24', '5-25', '5-26', '5-27']
color_list = ['#9cce37', '#20B2AB', '#ff4500', '#FFBB78']

fig = plt.figure(figsize=(10, 3.5))
x = np.arange(336)
n = 0
for sen in ['', 'Max0.5', 'Max0.7', 'Max0.9']:
    if len(sen):
        label = sen[-1] + '0% of Capacity'
    else:
        label = 'Real World Traffic'
    y = 1 - (CO['RL'][sen]['0']['4G'] + CO['RL'][sen]['0']['5G']) / (
            CO['old'][sen]['0']['4G'] + CO['old'][sen]['0']['5G'])
    plt.plot(x[:48] / 2, y[:48], color=color_list[n], linewidth=3, label=label)
    n += 1
plt.xlabel('Time of a Day', fontsize=28)
plt.ylabel('Net Zero Rate', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.ylim(0.25, 0.5)
fig.legend(loc=2, bbox_to_anchor=(0.05, 1.35), ncol=2, fontsize=24)
plt.savefig(r'fig4\Carbon_Reduction_rate_RL.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot b

day_ticks = [1, 48, 96, 144, 192, 240, 288, 335]
day_labels = ['5-20', '5-21', '5-22', '5-23', '5-24', '5-25', '5-26', '5-27']
color_list = ['#9cce37', '#20B2AB', '#ff4500', '#FFBB78']

y = 1 - (CO['old']['']['5']['4G'] + CO['old']['']['5']['5G']) / (CO['old']['']['0']['4G'] + CO['old']['']['0']['5G'])
y1 = np.convolve(y, np.arange(3), 'same') / 3
plt.plot(x[:48], y1[:48], color=color_list[0], linewidth=3, label='5 m$^2$')
y = 1 - (CO['old']['']['10']['4G'] + CO['old']['']['10']['5G']) / (CO['old']['']['0']['4G'] + CO['old']['']['0']['5G'])
y1 = np.convolve(y, np.arange(3), 'same') / 3
plt.plot(x[:48], y1[:48], color=color_list[1], linewidth=3, label='10 m$^2$')
y = 1 - (CO['old']['']['15']['4G'] + CO['old']['']['15']['5G']) / (CO['old']['']['0']['4G'] + CO['old']['']['0']['5G'])
y1 = np.convolve(y, np.arange(3), 'same') / 3
plt.plot(x[:48], y1[:48], color=color_list[2], linewidth=3, label='15 m$^2$')
y = 1 - (CO['old']['']['20']['4G'] + CO['old']['']['20']['5G']) / (CO['old']['']['0']['4G'] + CO['old']['']['0']['5G'])
y1 = np.convolve(y, np.arange(3), 'same') / 3
plt.plot(x[:48], y1[:48], color=color_list[3], linewidth=3, label='20 m$^2$')
plt.xlabel('Time of a Day', fontsize=28)
plt.ylabel('Net Zero Rate', fontsize=28)
plt.xticks(fontsize=28)
# plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=16)
plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8], labels=['0', '0.2', '0.4', '0.6', '0.8'], fontsize=28)
# plt.xlim(0, 336)
plt.ylim(-0.05, 0.8)
fig.legend(loc=2, bbox_to_anchor=(0.25, 1.35), ncol=2, fontsize=24)
plt.savefig(r'fig4\Carbon_Reduction_rate_PV.pdf', dpi=600, format='pdf', bbox_inches='tight')

# Plot c

color_list = ['#72ABB3', '#20B2AB', '#9cce37', '#ABCB93', '#DFEA73', '#FADC56', '#FFBB78', '#FB9F3D', '#F66026',
              '#E81514']
traffic_list = ['Real World Traffic', '50% of Capacity', '70% of Capacity', '90% of Capacity']
x = {}
y = {}
md = {'old': 'W/O Energy-Saving', 'current': 'Threshold-Based', 'RL': 'DeepEnergy'}
for mode in ['old', 'RL']:
    if mode == 'old':
        n = 1
    else:
        n = 2
    for area in ['0', '5', '10', '15', '20']:
        x[md[mode] + ' (' + area + 'm$^2$)'] = [11 * i + n for i in range(4)]
        y[md[mode] + ' (' + area + 'm$^2$)'] = []
        n += 2
        if mode == 'RL':
            for sen in ['', 'Max0.5', 'Max0.7', 'Max0.9']:
                CE = np.mean(T[mode][sen]['4G'] + T[mode][sen]['5G']) / (1024 ** 3) / np.mean(
                    CO[mode][sen][area]['4G'] + CO[mode][sen][area]['5G'])
                y[md[mode] + ' (' + area + 'm$^2$)'].append((CE))
        if mode == 'old':
            for sen in ['', 'Max0.5', 'Max0.7', 'Max0.9']:
                CE = np.mean(T[mode][sen]['4G'] + T[mode][sen]['5G']) / (1024 ** 3) / np.mean(
                    CO[mode][sen][area]['4G'] + CO[mode][sen][area]['5G'])
                y[md[mode] + ' (' + area + 'm$^2$)'].append((CE))
        if mode == 'current':
            for sen in ['', 'Min-max0.5', 'Min-max0.7', 'Min-max0.9']:
                CE = np.mean(T[mode][sen]['4G'] + T[mode][sen]['5G']) / (1024 ** 3) / np.mean(
                    CO[mode][sen][area]['4G'] + CO[mode][sen][area]['5G'])
                y[md[mode] + ' (' + area + 'm$^2$)'].append((CE))

fig = plt.figure(figsize=(24, 8))
n = 0
for name in x:
    plt.bar(x[name], y[name], width=1, color=color_list[n], label=name)
    n += 1
xl = [11 * i + 5.5 for i in range(4)]
yl = [0 for i in range(4)]
plt.bar(xl, yl, width=0.01, tick_label=traffic_list)
plt.ylabel('Carbon Efficiency (TByte/tCO$_2$)', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(-3, 46)
plt.ylim(0., 10)
fig.legend(loc=2, bbox_to_anchor=(0.13, 0.88), ncol=2, fontsize=24)
plt.savefig(r'fig4\Carbon_Efficiency_bar.pdf', dpi=600, format='pdf', bbox_inches='tight')


# Plot d

def cost_calculator(PV_area, P_m, N_PV, N_B, i, d, E_L, E_m=0, num=0):
    C_PV = PV_area * 1500 * 0.16

    C_B = 12 * E_m / (12 * 0.8 * 0.94 * 0.85)

    C_Inv = (129 + 0.29 * P_m * 1.5)

    C_c = 150 * num

    if N_PV % N_B == 0:
        n = N_PV // N_B - 1
    else:
        n = N_PV // N_B
    C_BPW_arr = np.zeros(n)
    for j in range(n):
        C_BPW_arr[j] = C_B * (((1 + i) / (1 + d)) ** ((j + 1) * N_B))
    C_BPW = np.sum(C_BPW_arr)

    C_Ins = 0.1 * C_PV

    yita_year = (1 + i) / (1 + d)
    yita_all = yita_year ** N_PV
    C_MPW = 0.02 * C_PV * yita_year * (1 - yita_all) / (1 - yita_year)

    cost = C_PV + C_B + C_Inv + C_c + C_BPW + C_Ins + C_MPW

    c = cost * (1 - (1 + i) / (1 + d)) / (1 - (((1 + i) / (1 + d)) ** N_PV)) / E_L

    return cost, c


def profitcalculator(CER, ac_up):
    profit = CER * 78 + ac_up * 0.3598
    return profit


sen_list1 = ['Max0.4', 'Max0.5', 'Max0.6', 'Max0.7', 'Max0.8', 'Max0.9']
sen_list2 = ['Min-max0.4', 'Min-max0.5', 'Min-max0.6', 'Min-max0.7', 'Min-max0.8', 'Min-max0.9']

fig = plt.figure(figsize=(24, 8))
area = [0, 5, 10, 15, 20]
ax1 = fig.add_subplot(111)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
ax2 = ax1.twinx()

cost1 = []
cost2 = []
profit_list = []
clear1 = []
clear2 = []
rate = []
Carbon_Reduction = []
std = []
std_CR = []
PC_list = []
mode = 'old'
sen = ''
for k in range(5):
    co1, c1 = cost_calculator(area[k], P_m=200 * area[k], N_PV=20, N_B=5, i=0.035, d=0.04, E_L=1)
    if 0 <= area[k] < 5:
        a = '0'
    else:
        a = str(area[k])
    co2, c2 = cost_calculator(int(a), P_m=200 * area[k], N_PV=20, N_B=5, i=0.035, d=0.04, E_L=1)
    profit = profitcalculator(
        np.sum(CO[mode][''][a]['4G'] + CO[mode][''][a]['5G'] - CO['old']['']['0']['4G'] + CO['old']['']['0']['5G']), 0)
    profit_list.append(profit)
    cost1.append(co1 * 21035 / 20)
    cost2.append(co2 * 21035 / 20)
    clear1.append(cost1[k] * int(bool(area[k])))
    clear2.append(cost2[k] * int(bool(area[k])))
    rate1 = []
    CR = []
    for sen in CO[mode]:
        rate1.append(1 - np.sum(CO[mode][sen][a]['4G'] + CO[mode][sen][a]['5G']) / np.sum(
            CO['old'][sen]['0']['4G'] + CO['old'][sen]['0']['5G']))
        CR.append(np.sum((CO['old'][sen]['0']['4G'] + CO['old'][sen]['0']['5G']) - (
                CO[mode][sen][a]['4G'] + CO[mode][sen][a]['5G'])) / (cost1[k]) * 1000)
    rate.append(np.mean(rate1))
    std.append(np.max(np.abs(rate1 - np.mean(rate1))))
    Carbon_Reduction.append(np.mean(CR))
    std_CR.append(np.std(CR))
    PC_list.append(np.sum(PC[mode][''][a]['4G'] + PC[mode][''][a]['5G']) * 2)
ax1.errorbar(np.array(clear2) / 1000_000, rate, color='#9cce37', yerr=std, linewidth=3.5, elinewidth=2, capsize=8,
             capthick=2)
p1, = ax1.plot(np.array(clear2) / 1000_000, rate, color='#9cce37', linewidth=3.5,
               label='Net Zero Rate (W/O Energy-Saving)')
ax1.scatter(np.array(clear2) / 1000_000, rate, color='#9cce37', s=PC_list, alpha=0.6)
Carbon_Reduction[0] = Carbon_Reduction[1] / 5 * area[0]
ax2.errorbar(np.array(clear1)[1:] / 1000_000, Carbon_Reduction[1:], color='#ff4500', yerr=std_CR[1:], linewidth=3.5,
             label='Carbon Reduction per Yuan (W/O Energy-Saving)', elinewidth=2, capsize=8, capthick=2)
p2 = ax2.scatter(np.array(clear1)[1:] / 1000_000, Carbon_Reduction[1:], color='#ff4500', s=150, marker='v')

cost1 = []
cost2 = []
profit_list = []
clear1 = []
clear2 = []
rate = []
Carbon_Reduction = []
std = []
std_CR = []
PC_list = []
mode = 'RL'
sen = ''
for k in range(5):
    co1, c1 = cost_calculator(area[k], P_m=200 * area[k], N_PV=20, N_B=5, i=0.035, d=0.04, E_L=1)
    if 0 <= area[k] < 5:
        a = '0'
    else:
        a = str(area[k])
    co2, c2 = cost_calculator(int(a), P_m=200 * area[k], N_PV=20, N_B=5, i=0.035, d=0.04, E_L=1)
    profit = profitcalculator(
        np.sum(CO[mode][''][a]['4G'] + CO[mode][sen][a]['5G'] - CO['old']['']['0']['4G'] + CO['old']['']['0']['5G']), 0)
    profit_list.append(profit)
    cost1.append(co1 * 21035 / 20)
    cost2.append(co2 * 21035 / 20)
    clear1.append(cost1[k] * int(bool(area[k])))
    clear2.append(cost2[k] * int(bool(area[k])))
    rate1 = []
    CR = []
    for sen in CO[mode]:
        rate1.append(1 - np.sum(CO[mode][sen][a]['4G'] + CO[mode][sen][a]['5G']) / np.sum(
            CO['old'][sen]['0']['4G'] + CO['old'][sen]['0']['5G']))
        CR.append(np.sum((CO['old'][sen]['0']['4G'] + CO['old'][sen]['0']['5G']) - (
                CO[mode][sen][a]['4G'] + CO[mode][sen][a]['5G'])) / (cost1[k]) * 1000)
    rate.append(np.mean(rate1))
    std.append(np.max(np.abs(rate1 - np.mean(rate1))))
    Carbon_Reduction.append(np.mean(CR))
    std_CR.append(np.std(CR))
    PC_list.append(np.sum(PC[mode][''][a]['4G'] + PC[mode][''][a]['5G']) * 2)
ax1.errorbar(np.array(clear2) / 1000_000, rate, color='#20B2AB', yerr=std, linewidth=3.5, elinewidth=2, capsize=8,
             capthick=2)
p3, = ax1.plot(np.array(clear2) / 1000_000, rate, color='#20B2AB', linewidth=3.5, label='Net Zero Rate (Deep Energy)')
ax1.scatter(np.array(clear2) / 1000_000, rate, color='#20B2AB', s=PC_list, alpha=0.4)
Carbon_Reduction[0] = Carbon_Reduction[1] / 5 * area[0]
ax2.errorbar(np.array(clear1)[1:] / 1000_000, Carbon_Reduction[1:], color='#FFBB78', linewidth=3.5, yerr=std_CR[1:],
             label='Carbon Reduction per Yuan (Deep Energy)', elinewidth=2, capsize=8, capthick=2)
p4 = ax2.scatter(np.array(clear1)[1:] / 1000_000, Carbon_Reduction[1:], color='#FFBB78', s=150, marker='v')

p5 = ax1.scatter(-1, 0.5, color='#C0C0C0', s=100 * 8, label='100MWh PV Energy Curtailment')
fig.legend((p1, p3, p5, p2, p4), (
    'Net Zero Rate (W/O Energy-Saving)', 'Net Zero Rate (DeepEnergy)', '100MWh PV Energy Curtailment',
    'Carbon Reduction per Yuan (W/O Energy-Saving)', 'Carbon Reduction per Yuan (DeepEnergy)'),
           bbox_to_anchor=(0.15, 1.25),
           loc=2, ncol=2, fontsize=24, borderpad=0.9, labelspacing=1.1)
ax1.set_ylim(-0.05, 0.7)
ax1.set_xlim(-0.5, 10)
ax2.set_ylim(-0.1, 1.4)
# ax1.set_xlim(0, 14)
ax1.set_ylabel('Net Zero Rate', fontsize=28)
ax2.set_ylabel('Carbon Reduction per Yuan (kgCO$_2$/Yuan)', fontsize=28)
ax1.set_xlabel('Cost of PV System (Million Yuan)', fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.savefig(r'fig4\Carbon_Reduction_and_PV_Curtailment.pdf', dpi=600, format='pdf', bbox_inches='tight')
