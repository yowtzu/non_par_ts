import pandas as pd
import numpy as np
import ephem  # for moon phases

from pandas.tseries.holiday import USFederalHolidayCalendar


class Feature(object):

    def __init__(self, **kwargs):
        self.lambdas = kwargs.pop('lambdas', [None])

# DO HOLIDAY

# DO WEEKEND WEEKDAY

#from pandas.tseries.holiday import USFederalHolidayCalendar


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


class Weekend(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 2

    def indexer(self, index, column=None):
        res = index.dayofweek.isin((5,6))
        return np.array(res, dtype=int)


class USHoliday(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 2
        self.cal = USFederalHolidayCalendar()

    def indexer(self, index, column = None):
        holidays = self.cal.holidays(start=index.min()-pd.Timedelta('2d'),
                                     end=index.max()+pd.Timedelta('2d'))
        return np.isin(index.date, holidays.date)


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


class WeekOfYear(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 53

    def indexer(self, index, column=None):
        return index.week - 1


class MonthOfYear(Feature):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 12

    def indexer(self, index, column=None):
        return index.month - 1


# class LunarPhase(Feature):

#     def __init__(self, n_periods=4, **kwargs):
#         super().__init__(**kwargs)
#         self.n_periods = n_periods

#     def indexer(self, index, column=None):
#         # TODO
#         pass


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

# class LunarPhase(Feature):

#     def __init__(self, n_periods=8, **kwargs):
#         super().__init__(**kwargs)
#         self.n_periods = n_periods

#     def lunations(self, timestamp):
#         """Lunations since Jan 1, 2001."""
#         diff = timestamp - pd.datetime(2001, 1, 1)
#         days = diff.days + diff.seconds / 86400
#         return 0.20439731 + days * 0.03386319269

#     def indexer(self, index, column=None):
#         lunations = self.lunations(index)
#         return (np.floor(self.n_periods * lunations + .5
#                          ).astype(int) % self.n_periods)


class DaysSinceNewMoon(Feature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_periods = 30

    def indexer(self, index):

        res = np.empty_like(index, dtype=int)

        moon_intervals = []
        for i, ts in enumerate(index):

            for interval in moon_intervals:
                if ts >= interval[0] and ts <= interval[1]:
                    res[i] = (ts - interval[0]).days
                    break

            else:
                moon_intervals.append(
                    (ephem.previous_new_moon(ts).datetime(),
                        ephem.next_new_moon(ts).datetime())
                )
                res[i] = (ts - moon_intervals[-1][0]).days

        return res


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
