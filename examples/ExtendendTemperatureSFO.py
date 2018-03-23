
# coding: utf-8

# In[1]:

# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import npts

DATA_DIR = 'data/'


# ## (Down)load SFO weather data 

# In[2]:

# import urllib.request

# for year in range(1987, 2019):
#     print(f"Downloading data for year {year}")
#     urllib.request.urlretrieve(f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/72494023234.csv", 
#                                DATA_DIR + f"{year}-SFO-weather.csv")

def load_year(year):
    ## Look at the document in DATA_DIR for spec.
    data=pd.read_csv(DATA_DIR+f'{year}-SFO-weather.csv.gz', usecols=[1,13])
    data.index = pd.to_datetime(data.DATE, format='%Y-%m-%dT%H:%M:%S')
    data = data.dropna()
    data['temp'] = data.TMP.apply(lambda el: float(el.split(',')[0])/10.)
    data['qual_code'] = data.TMP.apply(lambda el: (el.split(',')[1]))
    # time zone
    import pytz
    pacific = pytz.timezone('US/Pacific')
    data.index = data.index.tz_localize(pytz.utc).tz_convert(pacific)

    return data.temp[data.qual_code.isin(['1','5'])]

temperatures = pd.Series()
for year in range(1987, 2019):
    temperatures = temperatures.append(load_year(year))

# ## Make independent test set

# In[3]:

print('we have years:', set(temperatures.index.year))
data_used = temperatures[temperatures.index.year <= 2015]
indep_test = temperatures[temperatures.index.year > 2015]


# In[4]:

len(data_used)


# ## Train models on data 

# In[5]:

# columns: constant, daily avg., annual avg., daily-annual avg., daily-annual bas.
# rows: full data, 50% data, 10% data, 1% data, 0.1% data, 0.01% data


# In[6]:

models_modelnames = [
    (npts.Baseline(), 'const.', 1.),
    (npts.Baseline(npts.HourOfDay(lambdas=[1E-8])), 'hour', 1.),
    (npts.Baseline(npts.DayOfYear(lambdas=[1E-8])), 'day', 1.),
    (npts.Baseline(npts.HourOfDay(lambdas=[1E-8]),
                   npts.DayOfYear(lambdas=[1E-8])), 'hour and day', 1.),
    (npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 20)),
                   npts.DayOfYear(lambdas=np.logspace(-6,2, 20))), 'hour and day bas.', .75)
]


np.random.seed(0)
indep_test_rmse = pd.DataFrame()
time_taken = pd.DataFrame()
model_objs = pd.DataFrame()

import time 

def sparsify_data(data, frac):
    return data[np.random.uniform(size=len(data)) < frac]

for data, dataname in [(sparsify_data(data_used, frac), f'{100*frac:.1f}%') 
                       for frac in [1, .1, .01, .001]]:
    print(len(data),dataname)
    for model, modelname, train_frac in models_modelnames:
        print(f'fitting {modelname} using {100*train_frac:.0f}% train data')
        model_objs.loc[dataname, modelname] = model
        s = time.time()
        model.fit(data,train_frac=train_frac)
        time_taken.loc[dataname, modelname] = time.time() - s
        pred = model.predict(indep_test.index)
        indep_test_rmse.loc[dataname, modelname] = np.sqrt(np.mean((indep_test - pred)**2)) 


# In[7]:

with open('../latex/temperature_RMSE_table.tex', 'w') as f:
    f.write(indep_test_rmse.to_latex(float_format="%.3f"))


# In[8]:

# time_taken


# ## Inspect baseline fit on 1% of data 

# In[9]:

np.random.seed(0)
small_data = sparsify_data(data_used, .01)

baseline_small_data = npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-8,5, 30)), 
                         npts.DayOfYear(lambdas=np.logspace(-8,5, 30)))

baseline_small_data.fit(small_data)


# In[10]:

def plot_2d_model(theta):
    
    plt.figure()
    plt.plot(theta)
    plt.xlabel('hours since start of year')

    fig = plt.figure(figsize=(8,4))
    cax = plt.imshow(theta.reshape((366,24)).T, 
                     aspect='auto',origin='lower',interpolation='gaussian')
    axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7, label='degrees C')
    plt.xlabel('days')
    plt.ylabel('hours')

plot_2d_model(baseline_small_data.theta)
plt.savefig('../fig/temperatures_baseline.pdf')


# In[11]:

def plot_RMSE(baseline, cost_dict, title):
    fig = plt.figure(figsize=(8,4))
    cax = plt.hexbin(*np.array([[*k,np.sqrt(v)] for k, v in cost_dict.items() ]).T,
                     xscale='log',
                     yscale='log',
              gridsize=20)
    axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)
    plt.loglog(*baseline.best_lambda, 'ro', markersize=10)
    plt.xlabel('$λ_1$')
    plt.ylabel('$λ_2$')
    plt.title(title)
    
plot_RMSE(baseline_small_data, baseline_small_data.val_costs, 'RMSE validation set')
plt.savefig('../fig/temperatures_rmse.pdf')


# ## Error

# In[12]:

# residuals = np.abs(train - baseline.predict(train.index))

# res_baseline = npts.Baseline(npts.HourOfDay(lambdas=[baseline.best_lambda[0]]),#np.logspace(-6,-1, 20)), 
#                              npts.DayOfYear(lambdas=[baseline.best_lambda[1]]))#np.logspace(-7,-1, 20)))

# res_baseline.fit(residuals)#, initial_lambda = baseline.best_lambda)


# In[13]:

# residuals = train - baseline.predict(train.index)
# abs_sigmas = (res_baseline.predict(train.index))

# (residuals/abs_sigmas).kurtosis()


# In[14]:

# #res_baseline.theta = np.sqrt(res_baseline.theta)

# plt.plot((res_baseline.theta))

# fig = plt.figure()
# cax = plt.imshow((res_baseline.theta.reshape((366,24))).T, aspect='auto',origin='lower')
# axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)


# ## Error squared 

# In[20]:

# np.random.seed(0)
# large_data = sparsify_data(data_used, 1.)

# baseline_large_data = npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-8,5, 30)), 
#                          npts.DayOfYear(lambdas=np.logspace(-8,5, 30)))

# baseline_large_data.fit(large_data)


# # In[27]:

# residuals_squared = np.abs(large_data - baseline_large_data.predict(large_data.index))

# res_sq_baseline = npts.Baseline(npts.HourOfDay(lambdas=np.logspace(-6,2, 30)), 
#                              npts.DayOfYear(lambdas=np.logspace(-6,2, 30)))

# res_sq_baseline.fit(residuals_squared, train_frac=.75)


# # In[28]:

# plot_RMSE(res_sq_baseline, res_sq_baseline.val_costs, 'test')
# plot_2d_model(np.sqrt(res_sq_baseline.theta))


# In[17]:

# #res_baseline.theta = np.sqrt(res_baseline.theta)

# plt.plot(np.sqrt(res_sq_baseline.theta))

# fig = plt.figure()
# cax = plt.imshow(np.sqrt(res_sq_baseline.theta.reshape((366,24))).T, aspect='auto',origin='lower')
# axc = fig.colorbar(cax, ax=fig.gca(), shrink=.7)


# ### Prediction on sample days 

# In[19]:

# win_data = indep_test[24*0:24*3]

# plt.figure(figsize=(12,8))
# one_pc_baseline.predict(win_data.index).plot(label='baseline')
# (one_pc_baseline.predict(win_data.index) + np.sqrt(res_sq_baseline.predict(win_data.index))).plot(label='$\pm$ residual baseline',
#                                                                                  style='r--')
# (one_pc_baseline.predict(win_data.index) - np.sqrt(res_sq_baseline.predict(win_data.index))).plot(style='r--')
# win_data.plot(label='real')
# plt.legend()

# win_data = indep_test[24*180:24*183]

# plt.figure(figsize=(12,8))
# one_pc_baseline.predict(win_data.index).plot(label='baseline')
# (one_pc_baseline.predict(win_data.index) + np.sqrt(res_sq_baseline.predict(win_data.index))).plot(label='$\pm$ residual baseline',
#                                                                                  style='r--')
# (one_pc_baseline.predict(win_data.index) - np.sqrt(res_sq_baseline.predict(win_data.index))).plot(style='r--')
# win_data.plot(label='real')
# plt.legend()



# In[ ]:

# residuals = test - baseline.predict(test.index)
# sigmas = np.sqrt(res_sq_baseline.predict(test.index))

# normalized = (residuals/sigmas)

# normalized.kurtosis()


# In[ ]:

# residuals.hist(bins=400)


# In[ ]:

# import matplotlib.mlab as mlab

# xs = np.arange(-4,4,.01)

# bins=normalized.hist(bins=400)

# l = plt.plot(xs, mlab.normpdf( xs, 0, 1)*1000, 'r--', linewidth=2)


# In[ ]:

# residuals.kurtosis()


# In[ ]:

# xs = np.arange(-10,10,.01)

# bins=residuals.hist(bins=400)

# l = plt.plot(xs, mlab.normpdf( xs, residuals.mean(), residuals.std())*28000, 'r--', linewidth=2)


# ## Fourier Experiment

# In[ ]:

## FOURIER EXPERIMENT

# data = pd.DataFrame(temperatures)

# data['dayofyear'] = data.index.dayofyear
# data['hour'] = data.index.hour

# matrix = data[:].groupby(('dayofyear', 'hour')).mean().unstack().values
# plt.imshow(matrix, aspect='auto')

# plt.figure()
# plt.imshow(matrix, aspect='auto')

# transf = np.fft.fftshift(np.fft.fft2(matrix))

# plt.figure()
# plt.imshow(np.log(np.abs(transf)), aspect='auto', interpolation='gaussian')

# plt.figure()
# plt.imshow(np.angle(transf), aspect='auto', interpolation='gaussian')

# win_year = 7
# win_day = 5

# low_pass = np.zeros_like(transf)
# low_pass[183-win_year:183+win_year, 12-win_day:12+win_day] = \
# transf[183-win_year:183+win_year,12-win_day:12+win_day]

# plt.figure()
# plt.imshow(np.log(np.abs(low_pass[183-win_year:183+win_year, 12-win_day:12+win_day])), aspect='auto', interpolation='gaussian')

# plt.figure()
# plt.imshow(np.log(np.abs(low_pass)), aspect='auto', interpolation='gaussian')

# retransform = np.fft.ifft2(np.fft.ifftshift(low_pass))

# plt.figure()
# plt.imshow(np.real(retransform),aspect='auto')

