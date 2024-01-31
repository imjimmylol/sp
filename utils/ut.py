import numpy as np
from tqdm import tqdm

def Monte_GBM(S,v,mu,n,T, nstep=5):
    dt = T/nstep
    e = np.random.normal(loc=0.0, scale=1.0, size=(n,nstep))
    S_T = S*np.exp((mu - v ** 2 / 2) * dt + v * np.sqrt(dt) * e).cumprod(axis=1)
    return [np.mean(S_T[:,i]) for i in range(nstep)]


def run_simulation(df, Estimation_window, step_forward, 
                   S,v,mu,n,T, nstep):
    
    total = 0
    act = 0

    date_li = []
    res_li = []
    params_estimation_dict = {'vol':[], 'drift':[]}

    for i in tqdm(range(len(df)-Estimation_window+1)):
    # for i in range(5):
        total+=1
        df_estimation = df.iloc[i:i+Estimation_window]

        drift = df_estimation['ret'].mean()
        vol = df_estimation['ret'].std()
        S_t = df_estimation['Close'][-1]

        params_estimation_dict['vol'] = vol
        params_estimation_dict['drift'] = drift

        df_test = df.iloc[i:i+Estimation_window+step_forward]
        S_tp1 = df_test['Close'][-step_forward:].mean()


        # pred dir
        pred = np.mean(Monte_GBM(S = S_t, v=vol, mu=drift, n=n, T=1, nstep=step_forward))
        dir_num = pred - S_t
        dir_dummy = 1 if dir_num > 0 else -1

        # act dir
        dir_num_act = S_tp1 - S_t
        dir_dummy_act = 1 if dir_num_act > 0 else -1

        if dir_dummy-dir_dummy_act==0:
            act+=1
            date_li.append(df.iloc[i:i+Estimation_window+step_forward][-step_forward:].index[-1])
            res_li.append(1)

        if dir_dummy-dir_dummy_act!=0:
            date_li.append(df.iloc[i:i+Estimation_window+step_forward][-step_forward:].index[-1])
            res_li.append(-1)

        return (act/total, date_li, res_li, params_estimation_dict)