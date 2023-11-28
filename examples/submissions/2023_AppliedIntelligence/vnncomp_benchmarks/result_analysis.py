import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_res_by_cs(results: pd.DataFrame, case_study: str):

    res_over = results.loc[(results['NET_ID'] == case_study) & (results['VER_MODE'] == 'overapprox')].reset_index()
    res_mix = results.loc[(results['NET_ID'] == case_study) & (results['VER_MODE'] == 'mixed')].reset_index()
    res_comp = results.loc[(results['NET_ID'] == case_study) & (results['VER_MODE'] == 'complete')].reset_index()

    reformat_df = pd.DataFrame()
    reformat_df.insert(0, 'SPEC_ID', res_over['SPEC_ID'])
    reformat_df.insert(1, 'RES_OVER', res_over['RESULT'])
    reformat_df.insert(2, 'RES_MIX', res_mix['RESULT'])
    reformat_df.insert(3, 'RES_COMP', res_comp['RESULT'])
    reformat_df.insert(4, 'TIME_OVER', res_over['TIME'].replace('-', value=3600))
    reformat_df.insert(5, 'TIME_MIX', res_mix['TIME'].replace('-', value=3600))
    reformat_df.insert(6, 'TIME_COMP', res_comp['TIME'].replace('-', value=3600))
    reformat_df.to_csv(f"logs/res_{case_study}.csv")

    comp_timeouts = np.count_nonzero(res_comp['RESULT'].to_numpy() == '-')

    errors_over = res_over.__len__() - \
                  np.count_nonzero(res_over['RESULT'].to_numpy() == res_comp['RESULT'].to_numpy()) - comp_timeouts

    errors_mix = res_mix.__len__() - \
                 np.count_nonzero(res_mix['RESULT'].to_numpy() == res_comp['RESULT'].to_numpy()) - comp_timeouts

    print(f"NUMBER OF TIMEOUTS by COMPLETE = {comp_timeouts}")
    print(f"NUMBER OF ERRORS by OVERAPPROX = {errors_over}")
    print(f"NUMBER OF ERRORS by MIXED = {errors_mix}")

    over_times = res_over['TIME'].to_numpy(dtype='float')
    mix_times = res_mix['TIME'].to_numpy(dtype='float')
    comp_times = res_comp['TIME'].replace('-', value=3600).to_numpy(dtype='float')

    ord_over_times = np.sort(over_times[over_times < 3600])
    ord_mix_times = np.sort(mix_times[mix_times < 3600])
    ord_comp_times = np.sort(comp_times[comp_times < 3600])

    times = np.array([over_times, mix_times, comp_times]).transpose()
    labels = np.array(['overapprox', 'mixed', 'complete'])

    plt.figure()
    plt.plot(ord_over_times, label='overapprox', color='red', marker='o', markevery=5)
    plt.plot(ord_mix_times, label='mixed', color='green', marker='v', markevery=5)
    plt.plot(ord_comp_times, label='complete', color='blue', marker='s', markevery=5)
    plt.yscale('log')
    plt.legend()
    plt.title(f'CS: {case_study}')
    plt.tight_layout()
    plt.savefig(f"graphs/{case_study}_times_plot.eps")
    plt.show()

    return ord_over_times, ord_mix_times, ord_comp_times


res_df = pd.read_csv("logs/VerResults_final.txt")
cp_times = check_res_by_cs(res_df, 'cartpole')
ll_times = check_res_by_cs(res_df, 'lunarlander')
dr_times = check_res_by_cs(res_df, 'dubinsrejoin')

plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=15)
plt.rcParams['text.usetex'] = True

linestyles = ['-', '--', ':']
colors = ['red', 'green', 'blue']
labels = ['overapprox', 'mixed', 'complete']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
for i in range(len(cp_times)):
    ax1.plot(cp_times[i], label=labels[i], color=colors[i], linestyle=linestyles[i])
# ax1.grid()
ax1.legend()
ax1.set_yscale('log')
ax1.set_ylabel('t')
ax1.set_title("CS: cartpole")

for i in range(len(cp_times)):
    ax2.plot(ll_times[i], label=labels[i], color=colors[i], linestyle=linestyles[i])
# ax1.grid()
ax2.legend()
ax2.set_yscale('log')
ax2.set_title("CS: lunarlander")

for i in range(len(cp_times)):
    ax3.plot(dr_times[i], label=labels[i], color=colors[i], linestyle=linestyles[i])
# ax1.grid()
ax3.legend()
ax3.set_yscale('log')
ax3.set_title("CS: dubinsrejoin")

plt.tight_layout()
plt.savefig("graphs/res_by_cs.eps")
plt.savefig("graphs/res_by_cs.pdf")
plt.show()
