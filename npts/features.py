import pandas as pd
import numpy as np


class Feature(object):

    def __init__(self, **kwargs):
        self.lambdas = kwargs.pop('lambdas', [None])

# DO HOLIDAY

# DO WEEKEND WEEKDAY

#from pandas.tseries.holiday import USFederalHolidayCalendar

# class USHoliday(Feature):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.n_periods = 2
#         self.cal = USFederalHolidayCalendar()


#     #NOT INDEXER, THEY'RE FEATURES
#     def indexer(self, index, column = None):
#         holidays = self.cal.holidays(start=index.min(),
#                                      end=index.max())
#         return index.isin(holidays)


class HourOfDay(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 24

    # NOT INDEXER, THEY'RE FEATURES

    def indexer(self, index, column=None):
        return index.hour


class DayOfWeek(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 7

    # NOT INDEXER, THEY'RE FEATURES

    def indexer(self, index, column=None):
        return index.dayofweek


class DayOfWorkWeek(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 5

    # NOT INDEXER, THEY'RE FEATURES

    def indexer(self, index, column=None):
        return index.dayofweek


class DayOfMonth(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 31

    def indexer(self, index, column=None):
        return index.day - 1


class DayOfYear(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 366

    def indexer(self, index, column=None):
        return index.dayofyear - 1


class MonthOfYear(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 12

    def indexer(self, index, column=None):
        return index.month - 1


class LunarPhase(Feature):

    def __init__(self, n_periods=4, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = n_periods

    def indexer(self, index, column=None):
        # TODO
        pass


class QuarterOfYear(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 4

    def indexer(self, index, column=None):
        return index.quarter - 1


class DayOfQuarter(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 92

    def indexer(self, index, column=None):
        return np.array([(date - ts.start_time).days
                         for date, ts in zip(index,
                                             pd.PeriodIndex(index, freq='Q'))])

from pandas.tseries.offsets import BDay


class BDayOfYear(Feature):
    # TODO merge with bdayofquarter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 262

    def indexer(self, index, column=None):
        # TODO fix this hacky thing (corner case not handled)
        if len(index) == 0:
            return np.array([])
        bdays = pd.date_range(index.min(), index.max(), freq=BDay())
        daycount = np.zeros(len(bdays))
        count = 0
        y = 0
        for i, el in enumerate(bdays):
            if el.year != y:
                count = 0
                y = el.year
            daycount[i] = count
            count += 1
        result = pd.DataFrame(index=bdays, data=daycount).reindex(
            index).values[:, 0]
        return np.array(result, dtype=int)


class BDayOfQuarter(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 66

    def indexer(self, index, column=None):
        # TODO fix this hacky thing (corner case not handled)
        if len(index) == 0:
            return np.array([])
        bdays = pd.date_range(index.min(), index.max(), freq=BDay())
        daycount = np.zeros(len(bdays))
        count = 0
        q = 5
        for i, el in enumerate(bdays):
            if el.quarter != q:
                count = 0
                q = el.quarter
            daycount[i] = count
            count += 1
        result = pd.DataFrame(index=bdays, data=daycount).reindex(
            index).values[:, 0]
        return np.array(result, dtype=int)


# class Columns(Feature):
#     def __init__(self, columns):
#         self.columns = columns
#         self.n_periods = len(columns)
#
#     def indexer(self, index, column = None):


class IntervalOfDay(Feature):

    def __init__(self, freq='5min', **kwargs):
        super().__init__(**kwargs)
        """Return indexer at given (pandas) frequency."""
        self.freq = freq
        self.times = list(pd.date_range('2018-01-01', '2018-01-02',
                                        freq=self.freq, closed='left').time)
        self.n_periods = len(self.times)

    def indexer(self, index, column=None):
        return np.array([self.times.index(el) for el in index.time])
