
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

import npts


# ## Load Data (Grid Load in Northern California)

# Data from 
# http://www.caiso.com/planning/Pages/ReliabilityRequirements/Default.aspx#Historical ,
# the webpage seems down, however.

# In[2]:

caiso_df = pd.read_excel("data/HistoricalEMSHourlyLoad_2014-2016.xlsx")
caiso_df.index = caiso_df.Dates

PGE_loads = caiso_df.PGE
#SCE_loads = caiso_df.SCE
PGE_loads.index.name = ''


# In[3]:

PGE_loads.plot()


# In[4]:

print('we have years:', set(PGE_loads.index.year))
data_used = PGE_loads[PGE_loads.index.year < 2016]
indep_test = PGE_loads[PGE_loads.index.year >= 2016]


# ## Fit and test the models 

# In[5]:

indep_test_rmse = pd.DataFrame()
time_taken = pd.DataFrame()
model_objs = pd.DataFrame()


# In[15]:

models_modelnames = []#(npts.Baseline(), 'constant avg.', 1.)]

features = [npts.HourOfDay, npts.MonthOfYear, npts.DayOfWeek, npts.USHoliday]

for n_features in [4]:

#     for features_used in itertools.combinations(features,n_features):
#         models_modelnames.append((npts.Baseline(*(f(lambdas=[1E-6]) for f in features_used)), 
#                                   ' and '.join(f.__name__ for f in features_used) + ' avg.',
#                                   1.))

    for features_used in itertools.combinations(features,n_features):
        models_modelnames.append((npts.Baseline(*(f(lambdas=np.logspace(-6,2, 20)) for f in features_used)), 
                                  ' and '.join(f.__name__ for f in features_used) + ' bas.',
                                  .75))


# In[16]:

np.random.seed(0)

import time 

def sparsify_data(data, frac):
    return data[np.random.uniform(size=len(data)) < frac]

for data, dataname in [(sparsify_data(data_used, frac), f'{100*frac:.1f}% data') 
                       for frac in [1, .01]]:
    print(len(data),dataname)
    for model, modelname, train_frac in models_modelnames:
        print(f'fitting {modelname} using {100*train_frac:.0f}% train data')
        model_objs.loc[dataname, modelname] = model
        s = time.time()
        model.fit(data,train_frac=train_frac)
        time_taken.loc[dataname, modelname] = time.time() - s
        pred = model.predict(indep_test.index)
        indep_test_rmse.loc[dataname, modelname] = np.sqrt(np.mean((indep_test - pred)**2)) 


# In[13]:

indep_test_rmse.T


# In[12]:

indep_test_rmse.T


# In[9]:

indep_test_rmse.T


# In[8]:

indep_test_rmse


# In[12]:

win_s = 0
win_e = 1000
plt.figure(figsize=(12,7))
model_objs.loc['100.0% data',
               'HourOfDay and MonthOfYear and Weekend bas.'].predict(indep_test.index[win_s:win_e]).plot()
indep_test[win_s:win_e].plot()


# In[39]:

# models_modelnames = [
#     (npts.Baseline(), 'constant', 1.),
#     (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                    npts.MonthOfYear(lambdas=[1E-6])), 'hour and month-of-year avg.', 1.),
#     (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                    npts.MonthOfYear(lambdas=np.logspace(-6,2, 20))), 'hour and month-of-year bas.', .75),
#         (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 'hour and day-of-week avg.', 1.),
#         (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 'hour and day-of-week bas.', .75),
#         (npts.Baseline(npts.MonthOfYear(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 'month-of-year and day-of-week avg.', 1.),
#         (npts.Baseline(npts.MonthOfYear(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 'month-of-year and day-of-week bas.', .75),
#         (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                        npts.MonthOfYear(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 
#                        'hour and month-of-year and day-of-week avg.', 1.),
#             (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                        npts.MonthOfYear(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 
#                        'hour and month-of-year and day-of-week bas.', .75),
#  ]
# features = [npts.HourOfDay, npts.MonthOfYear, npts.DayOfWeek]
# models_modelnames = [
#     (npts.Baseline(), 'constant', 1.),
#     (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                    npts.MonthOfYear(lambdas=[1E-6])), 'hour and month-of-year avg.', 1.),
#     (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                    npts.MonthOfYear(lambdas=np.logspace(-6,2, 20))), 'hour and month-of-year bas.', .75),
#         (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 'hour and weekend avg.', 1.),
#         (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 'hour and weekend bas.', .75),
#         (npts.Baseline(npts.MonthOfYear(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 'month-of-year and weekend avg.', 1.),
#         (npts.Baseline(npts.MonthOfYear(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 'month-of-year and weekend bas.', .75),
#         (npts.Baseline(npts.HourOfDay(lambdas=[1E-6]),
#                        npts.MonthOfYear(lambdas=[1E-6]),
#                    npts.DayOfWeek(lambdas=[1E-6])), 
#                        'hour and month-of-year and day-of-week avg.', 1.),
#             (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
#                        npts.MonthOfYear(lambdas=np.logspace(-6,2, 20)),
#                    npts.DayOfWeek(lambdas=np.logspace(-6,2, 20))), 
#                        'hour and month-of-year and day-of-week bas.', .75),
#  ]



np.random.seed(0)

import time 

def sparsify_data(data, frac):
    return data[np.random.uniform(size=len(data)) < frac]

for data, dataname in [(sparsify_data(data_used, frac), f'{100*frac:.1f}% data') 
                       for frac in [1,.1, .01]]:
    print(len(data),dataname)
    for model, modelname, train_frac in models_modelnames:
        print(f'fitting {modelname} using {100*train_frac:.0f}% train data')
        model_objs.loc[dataname, modelname] = model
        s = time.time()
        model.fit(data,train_frac=train_frac)
        time_taken.loc[dataname, modelname] = time.time() - s
        pred = model.predict(indep_test.index)
        indep_test_rmse.loc[dataname, modelname] = np.sqrt(np.mean((indep_test - pred)**2)) 


# In[40]:

indep_test_rmse


# In[50]:

indep_test_rmse


# In[34]:

model_objs.iloc[-1,-1].best_lambda


# In[30]:

indep_test_rmse  ## this was with 10x10x10 lambdas


# In[66]:

win_s = 800
win_e = 1000
plt.figure(figsize=(12,7))
model_objs.loc['100.0% data',
               'hour and month-of-year and day-of-week bas.'].predict(indep_test.index[win_s:win_e]).plot()
indep_test[win_s:win_e].plot()


# In[5]:

# train = PGE_loads[(PGE_loads.index<"2015-01-01")]
# validation = PGE_loads[(PGE_loads.index>="2015-01-01")&(PGE_loads.index<"2016-01-01")]
# test = PGE_loads[(PGE_loads.index>="2016-01-01")&(PGE_loads.index<"2017-01-01")]
# test = test[~((test.index.month==2)&(test.index.day==29))]

# fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=FIGBIG)
# cax = ax1.imshow(train.values.reshape(365,24).T, origin='lower',aspect='auto')
# _ = ax1.set_title('2014')
# _ = ax2.imshow(validation.values.reshape(365,24).T, origin='lower',aspect='auto')
# _ = ax2.set_title('2015')
# _ = ax3.imshow(test.values.reshape(365,24).T, origin='lower',aspect='auto')
# _ = ax3.set_title('2016')
# _ = ax3.set_xlabel('Day')
# _ = ax1.set_ylabel('Hour')
# _ = ax2.set_ylabel('Hour')
# _ = ax3.set_ylabel('Hour')
# axc = fig.colorbar(cax, ax=[ax1, ax2, ax3], shrink=.6)
# axc.ax.set_ylabel('MW')
# plt.savefig(GRAPHDIR+'norcal_loads_2d.pdf')


# In[6]:

# indexers = lambda index: index.hour, \
#         lambda index: index.dayofyear-1,\

# n_periods =  24, 366
# lambdas = [[.01, .02, .05, .1, .2, .5, 1., 2., 5., 10., 20.],
#           [10., 20., 50.]]

# b, val_rmse, indexer = smooth_cyclic_baseline(pd.concat([train,validation]), 
#                                      indexers, n_periods,
#                                      lambdas, train_fraction = .75)


# In[21]:

# val_rmse_df = pd.DataFrame()
# val_rmse_df.columns.name = '$\lambda_{diu}$'
# val_rmse_df.index.name = '$\lambda_{smooth}$'
# for lambdas_used in val_rmse:
#     val_rmse_df.loc[lambdas_used] = val_rmse[lambdas_used]
# #val_rmse_df


# In[22]:

# fig = plt.figure(figsize=FIGREGULAR)
# cax = plt.imshow(val_rmse_df, origin='lower', aspect='auto')
# cax.axes.set_xlabel('$\lambda_{diu}$', size='large')
# cax.axes.set_ylabel('$\lambda_{smooth}$', size='large')
# plt.yticks(np.arange(len(lambdas[0])), lambdas[0])
# plt.xticks(np.arange(len(lambdas[1])), lambdas[1])
# # colorbar
# axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
# axc.ax.set_ylabel('RMSE')


# In[23]:

fig = plt.figure(figsize=FIGWIDE)
cax = plt.imshow(b.reshape(366, 24).T, origin='lower', aspect='auto')
#plt.title('$b$')
axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
cax.axes.set_xlabel('Day', size='large')
cax.axes.set_ylabel('Hour', size='large')
axc.ax.set_ylabel('MW')
plt.savefig(GRAPHDIR+'norcal_baseline.pdf')


plot_index = test.iloc[:24*7].index
fig=plt.figure(figsize=FIGREGULAR)
test[plot_index].plot(label='Load')
pd.Series(data=b[indexer(plot_index)],
         index = plot_index).plot(label='Baseline')
fig.gca().set_ylabel('MW')
plt.legend()
plt.savefig(GRAPHDIR+'norcal_loads.pdf')


# ### Diagnostic 

# In[14]:

plot_index = test.iloc[24*110:24*130].index
fig=plt.figure(figsize=FIGWIDE)
test[plot_index].plot(label='Load')
pd.Series(data=b[indexer(plot_index)],
         index = plot_index).plot(label='Baseline')
fig.gca().set_ylabel('MW')
plt.legend()
#plt.savefig(GRAPHDIR+'norcal_loads_week.pdf')

test_residuals = test - b[indexer(test.index)]
from pandas.plotting import autocorrelation_plot
plt.figure()
autocorrelation_plot(test_residuals)
plt.xlim([0,24*32])


# ## Month/week fit 

# In[24]:

indexers = lambda index: index.hour,     lambda index: index.month-1,    lambda index: index.dayofweek,

n_periods =  24, 12, 7
#lambdas = [[.01, .02, .05, .1, .2, .5, 1., 2., 5., 10., 20.]]*3
lambdas = [[.01, .02, .05, .1, .2, .5, 1.,2.]]*3


b, val_rmse, indexer = smooth_cyclic_baseline(pd.concat([train,validation]), 
                                     indexers, n_periods,
                                     lambdas, train_fraction = .75)


# In[16]:

fig = plt.figure(figsize=FIGREGULAR)
cax = plt.imshow(b[:288].reshape(12, 24).T, origin='lower', aspect='auto')
plt.title('Monday', size='large')
axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
cax.axes.set_xlabel('Month', size='large')
cax.axes.set_ylabel('Hour', size='large')
axc.ax.set_ylabel('MW')

fig = plt.figure(figsize=FIGREGULAR)
plt.title('Sunday', size='large')
cax = plt.imshow(b[-288:].reshape(12, 24).T, origin='lower', aspect='auto')
#plt.title('$b$')
axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
cax.axes.set_xlabel('Month', size='large')
cax.axes.set_ylabel('Hour', size='large')
axc.ax.set_ylabel('MW')
#plt.savefig(GRAPHDIR+'norcal_baseline.pdf')


# In[18]:

plt.plot(b[:288],label='monday')
#plt.plot(b[-288*3:-288*2])
#plt.plot(b[-288*2:-288])
plt.plot(b[-288:],label='sunday')
plt.legend()


# ### Diagnostic

# In[25]:

plot_index = test.iloc[24*110:24*130].index
fig=plt.figure(figsize=FIGWIDE)
test[plot_index].plot(label='Load')
pd.Series(data=b[indexer(plot_index)],
         index = plot_index).plot(label='Baseline')
fig.gca().set_ylabel('MW')
plt.legend()
#plt.savefig(GRAPHDIR+'norcal_loads_week.pdf')

test_residuals = test - b[indexer(test.index)]
from pandas.plotting import autocorrelation_plot
plt.figure()
autocorrelation_plot(test_residuals)
plt.xlim([0,24*32])


# ## Old 

# In[ ]:

# residuals = []

# K = 10
# steps = np.zeros((N,K))
# xs = np.zeros((N,K))

# Q = np.ones((K+1, K+1))
# p = np.concatenate([np.zeros(K),[1.]])
# Q[-1,-1] = 0.

# # Anderson acceleration
# def Anderson(steps, xs, i, K):
#     if i <= K:
#         return x
#     Q[:K,:K] = 2 * steps.T@steps
#     return xs @ np.linalg.solve(Q, p)[:-1]

# import time
# start = time.time()
# for i in range(5000):
#     x = Anderson(steps, xs, i, K)
#     step = AAT @ x - Ab
#     x -= lambda_step * step
#     steps[:,i%K] = step
#     xs[:,i%K] = x
#     residuals.append(np.sqrt(np.mean(step**2)))
#     if residuals[-1] < 1E-5:
#         break
# print (time.time() - start)

# plt.semilogy(residuals)


# Anderson with K=25 takes about 40s, 2500 iters, to get to 1E-5 residual avg.
# 
# Anderson with K=10 takes about 25s, 3300 iters, to get to 1E-5 residual avg.
# 
# Anderson with K=7 takes about 22s, 4000 iters, to get to 1E-5 residual avg.
# 
# Anderson with K=5 takes about 14s, 3500 iters, to get to 1E-5 residual avg.

# In[ ]:

# import cvxpy as cvx

# days, times = windpow_2011.shape
# baseline = cvx.Variable(days, times)

# objective = cvx.sum_squares(baseline - windpow_2011.as_matrix()) #+ \
#    # cvx.sum_squares(baseline - windpow_2012.as_matrix())

# lambda_time = 30.
# objective += lambda_time * cvx.sum_squares(baseline[:,:-1] - baseline[:,1:])
# objective += lambda_time * cvx.sum_squares(baseline[:,-1] - baseline[:,0])

# lambda_date = 100.
# objective += lambda_date * cvx.sum_squares(baseline[:-1,:] - baseline[1:,:])
# objective += lambda_date * cvx.sum_squares(baseline[-1,:] - baseline[0,:])


# problem = cvx.Problem(cvx.Minimize(objective), [])
# #problem.solve(solver=cvx.LS, verbose=True)
# problem.solve(solver=cvx.SCS, verbose=True, use_indirect=True, 
#               acceleration_lookback = 25, max_iters = 2000)


# In[ ]:




# In[ ]:



