import numpy as np 
from itertools import product


def split_data_random(data, train_frac, seed=0):
    assert (train_frac >= 0) and (train_frac <= 1)
    mask = np.random.uniform(size=len(data)) <= train_frac
    return data[mask], data[~mask]


def split_data_contiguous(data, train_frac):
    assert (train_frac >= 0) and (train_frac <= 1)
    l = int(len(data)*train_frac)
    return data[:l], data[l:]


def split_data(data, train_frac, last_is_test, seed):
    if last_is_test:
        return split_data_contiguous(data, train_frac)
    else:
        return split_data_random(data, train_frac, seed)


def min_val_key(d):
    return min(d, key=d.get)


def RMSE(x, y):
    assert x.shape == y.shape
    return np.sqrt(np.mean(np.square(x-y)))


def prediction_cost(model, data):
    pred = model.predict(data.index)
    assert pred.shape == data.shape
    return RMSE(pred, data)




def hyperparameter_grid_search(fit_func, val_cost_func, *hp_iterables, no_cache_models=False):
    """Run a function, on iterables for the values of each hyperpar."""

    val_costs = {}
    models = {}

    for hyperparameters in itertools.product(*hp_iterables):

        model = fit_func(*hyperparameters)

        val_costs[hyperparameters] = val_cost_func(model)

        if not no_cache_models:
            models[hyperparameters] = model

    best_hp = min_val_key(val_costs)
    best_model = fit_func(*best_hp) if no_cache_models else models[best_hp]

    return best_model, val_costs




class Model(object):

    def fit(data, *hyperparameters):
        pass

    def predict(index):
        pass


