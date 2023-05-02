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

    times = np.array([over_times, mix_times, comp_times]).transpose()
    labels = np.array(['overapprox', 'mixed', 'complete'])

    plt.figure()
    plt.boxplot(times, labels=labels)
    plt.tight_layout()
    plt.savefig(f"graphs/{case_study}_times_boxplot.eps")
    plt.show()


res_df = pd.read_csv("logs/VerResults_final.txt")
check_res_by_cs(res_df, 'cartpole')
check_res_by_cs(res_df, 'lunarlander')
check_res_by_cs(res_df, 'dubinsrejoin')
