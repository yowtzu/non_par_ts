
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import npts

import matplotlib.pyplot as plt


# ### Download data 

# In[2]:

# month_ends = pd.date_range(start='1995-12-31', end=pd.datetime.today(), freq='M')
# month_starts = month_ends + pd.Timedelta('1d')
# month_ends = month_ends[1:]
# month_starts = month_starts[:-1]

# tides = pd.DataFrame()
# for start, end in zip(month_starts, month_ends):
#     start_str = f'{start.year}{start.month:02d}{start.day:02d}'
#     end_str = f'{end.year}{end.month:02d}{end.day:02d}'
    
#     print(f'downloading from {start_str} to {end_str}')
    
#     df = pd.read_csv(f'https://tidesandcurrents.noaa.gov/api/datagetter?product=water_level'+
#                 f'&application=NOS.COOPS.TAC.WL&begin_date={start_str}&end_date={end_str}&'+
#                 f'datum=MLLW&station=9415020&time_zone=lst&units=metric&format=csv', index_col=0)
    
#     tides = pd.concat([tides, df])

# tides.index = pd.to_datetime(tides.index)
# tides.to_csv(f'point_reyes_tides_{start_str}_{end_str}.csv.gz', compression='gzip')


# ### Download predictions 

# In[3]:

# start_day = '1995-12-31'
# end_day = pd.datetime.today()
# month_ends = pd.date_range(start=start_day, end=end_day, freq='M')
# month_starts = month_ends + pd.Timedelta('1d')
# month_ends = month_ends[1:]
# month_starts = month_starts[:-1]

# noaa_predictions = pd.DataFrame()
# for start, end in zip(month_starts, month_ends):
#     start_str = f'{start.year}{start.month:02d}{start.day:02d}'
#     end_str = f'{end.year}{end.month:02d}{end.day:02d}'
    
#     print(f'downloading from {start_str} to {end_str}')
    
#     df = pd.read_csv(f'https://tidesandcurrents.noaa.gov/api/datagetter?product=predictions'+
#                 f'&application=NOS.COOPS.TAC.WL&begin_date={start_str}&end_date={end_str}&'+
#                 f'datum=MLLW&station=9415020&time_zone=lst&units=metric&format=csv', index_col=0)
    
#     noaa_predictions = pd.concat([noaa_predictions, df])

# noaa_predictions.index = pd.to_datetime(noaa_predictions.index)
# noaa_predictions.to_csv(f'point_reyes_noaa_predictions_{start_day}_{end_day}.csv.gz', compression='gzip')


# ### Load data from disk 

# In[3]:

tides = pd.read_csv('data/point_reyes_tides_1995-12-31_2018-01-31.csv.gz', parse_dates=[0], index_col=0, usecols=[0,1,7])


# I couldn't find the specification of the data. I assume the the has values "p" and "v" in the "Quality" field stand
# for "preliminary" and "verified". I discard preliminar data.

# In[4]:

water_level = tides[tides[' Quality '] == 'v'][' Water Level']
del tides


# ### Make independent test set 

# In[5]:

print('we have years:', set(water_level.index.year))
data_used = water_level[water_level.index.year <= 2015]
indep_test = water_level[water_level.index.year > 2015]


# In[6]:

predictions = pd.read_csv('data/point_reyes_noaa_predictions_1995-12-31_2018-02-28.csv.gz',
                          parse_dates=[0], index_col=0)

predictions_test = predictions[(predictions.index.year > 2015) & (predictions.index <= water_level.index[-1])]
predictions_test = predictions_test.iloc[:,0]


# In[7]:

indep_test_rmse = pd.DataFrame()
time_taken = pd.DataFrame()
model_objs = pd.DataFrame()


# In[11]:

models_modelnames = [
#     (npts.Baseline(), 'constant', 1.),
    
    (npts.Baseline(npts.DayOfYear(lambdas=[1E-8]),
                   npts.DaysSinceNewMoon(lambdas=[1E-8])), 'week-of-year and days-since-new-moon avg.', 1.),

    (npts.Baseline(npts.IntervalOfDay(n_seconds=360, lambdas=[1E-8]),
                   npts.DaysSinceNewMoon(lambdas=[1E-8])), 'interval-of-day and days-since-new-moon avg.', 1.),
    
    (npts.Baseline(npts.IntervalOfDay(n_seconds=360, lambdas=[1E-8]),
                   npts.DayOfYear(lambdas=[1E-8])), 'interval-of-day and week-of-year avg.', 1.),
    
    (npts.Baseline(npts.DayOfYear(lambdas=[1E-8]),
               npts.IntervalOfDay(n_seconds=360, lambdas=[1E-8]),
                     npts.DaysSinceNewMoon(lambdas=[1E-8]),
              verbose=True), 'days-since-new-moon and interval-of-day and week-of-year avg.', 1.),

    (npts.Baseline(npts.DayOfYear(lambdas=np.logspace(-8,-5, 4)),
               npts.IntervalOfDay(n_seconds=360, lambdas=np.logspace(-8,-5, 4)),
               npts.DaysSinceNewMoon(lambdas=np.logspace(-8,-5, 4)),
              verbose=True), 'days-since-new-moon and interval-of-day and week-of-year bas.', .75)
]


# In[12]:

np.random.seed(0)

import time 
import copy

def sparsify_data(data, frac):
    return data[np.random.uniform(size=len(data)) < frac]

for data, dataname in [(sparsify_data(data_used, frac), f'{100*frac:.1f}% data') 
                       for frac in [.001, .01,]]:
    print(len(data),dataname)
    for model, modelname, train_frac in models_modelnames:
        model_used = copy.copy(model)
        print(f'fitting {modelname} using {100*train_frac:.0f}% train data')
        model_objs.loc[dataname, modelname] = model_used
        s = time.time()
        model_used.fit(data,train_frac=train_frac)
        time_taken.loc[dataname, modelname] = time.time() - s
        pred = model_used.predict(indep_test.index)
        indep_test_rmse.loc[dataname, modelname] = np.sqrt(np.mean((indep_test - pred)**2)) 


# In[13]:

with open('../latex/tides_RMSE_table.tex', 'w') as f:
    f.write(indep_test_rmse.to_latex(float_format="%.3f"))


# In[23]:

# model_used.iloc[]

# test_window = indep_test[:240*2]

# baselines[0].predict(test_window.index).plot(label='1% baseline')
# baselines[1].predict(test_window.index).plot(label='100% baseline')

# test_window.plot(label='real')
# plt.legend(loc='lower left')



# 

# ## Prediction on 1% data

# In[35]:

# np.random.seed(0)
# mask = np.random.uniform(size=len(data_used)) < .01
# small_data = data_used[mask]

# baseline = npts.Baseline(npts.HourOfDay(),#lambdas=np.logspace(-6,2, 5)),
#                           npts.DayOfYear(),#lambdas=np.logspace(-6,2, 5)),
#                          npts.LunarPhase(n_periods=16))#,lambdas=np.logspace(-6,2, 5)))

# baseline.fit(small_data)


# # In[37]:

# print('data size', len(small_data))
# print('model size', len(baseline.theta))


# # In[42]:

# poll = 240*7
# win_len = 240*7
# window = indep_test[poll:poll+win_len]
# plt.figure(figsize=(8,5))
# baseline.predict(window.index).plot(label='prediction' )
# window.plot(label='real')
# import matplotlib.pyplot as plt
# plt.legend()
# plt.savefig('../../non_par_ts/tides_prediction.pdf')


# ## Baseline 


# In[8]:

# test = water_level[-len(water_level)//20:]
# train = water_level[:-len(water_level)//20]


# # In[9]:

# train = water_level[:len(train)//6]


# In[11]:

# baseline = npts.Baseline(npts.IntervalOfDay('6min'),#, lambdas=np.logspace(-5,-2, 5)), 
#                          npts.MonthOfYear(),##lambdas=np.logspace(-9,-2, 5)), 
#                          npts.LunarPhase(n_periods=48))#, lambdas=np.logspace(-8,-2, 5)))#np.logspace(-22,-18, 4)))


# # In[12]:

# baseline.fit(train, compute_tr_costs=True)


# In[13]:

# baseline.val_costs


# In[19]:

# import matplotlib.pyplot as plt
# def plot_RMSE(baseline):
#     ref = ['intraday', 'day', 'moonphase']
#     for pat in [[0,1], [0,2], [1,2]]:
#         for cost_dict, title in [[baseline.val_costs, 'test'], 
#                                  [baseline.tr_costs, 'train']]:
#             fig = plt.figure(figsize=(8,4))
#             cax = plt.hexbin(*np.array([[*(np.log10(k)[pat]), np.sqrt(v)] for k, v 
#                                   in cost_dict.items()]).T,
#                       gridsize=5)
#             axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
#             plt.plot(*np.log10(baseline.best_lambda)[pat], 'ro', markersize=10)
#             plt.xlabel(f'log10(λ_{ref[pat[0]]})')
#             plt.ylabel(f'log10(λ_{ref[pat[1]]})')
#             plt.title(title)
    
# plot_RMSE(baseline)


# # In[15]:

# import matplotlib.pyplot as plt
# def plot_RMSE(baseline):
#     ref = ['intraday', 'day', 'moonphase']
#     for pat in [[0,1], [0,2], [1,2]]:
#         for cost_dict, title in [[baseline.val_costs, 'test'], 
#                                  [baseline.tr_costs, 'train']]:
#             fig = plt.figure(figsize=(8,4))
#             cax = plt.hexbin(*np.array([[*(np.log10(k)[pat]), np.sqrt(v)] for k, v 
#                                   in cost_dict.items()]).T,
#                       gridsize=5)
#             axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
#             plt.plot(*np.log10(baseline.best_lambda)[pat], 'ro', markersize=10)
#             plt.xlabel(f'log10(λ_{ref[pat[0]]})')
#             plt.ylabel(f'log10(λ_{ref[pat[1]]})')
#             plt.title(title)
    
# plot_RMSE(baseline)


# # In[16]:

# plt.figure()
# plt.imshow(np.median(baseline.theta.reshape(*reversed(baseline.n_periods)),0), origin='lower', aspect='auto')
# plt.ylabel('month')
# plt.xlabel('interval')


# plt.figure()
# plt.imshow(np.median(baseline.theta.reshape(*reversed(baseline.n_periods)),1), origin='lower', aspect='auto')
# plt.ylabel('moonphase')
# plt.xlabel('interval')

# plt.figure()
# plt.imshow(np.median(baseline.theta.reshape(*reversed(baseline.n_periods)),2), origin='lower', aspect='auto')
# plt.xlabel('month')
# plt.ylabel('moonphase')


# # ## Experiments 

# # In[27]:

# import scipy.sparse as sp
# import numpy as np


# # In[32]:

# a = np.arange(1000.)
# b = np.arange(1000.)


# # In[33]:

# get_ipython().magic('timeit a*b')


# # In[38]:

# get_ipython().magic('timeit sp.diags(a)@b')


# # In[40]:

# get_ipython().magic('timeit c = np.array(a)')


# # In[41]:

# get_ipython().magic('timeit c = np.array(a, copy=False)')


# # In[42]:

# d = np.array([1,2])


# # In[43]:

# get_ipython().magic('timeit e = np.array(d)')


# # In[44]:

# get_ipython().magic('timeit e = np.array(d, copy=False)')


# # In[45]:

# get_ipython().magic('timeit e = d')


# # In[ ]:



