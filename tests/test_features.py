import unittest
from npts.features import *

test_time_1 = pd.DatetimeIndex([pd.Timestamp(2000,1,1,0,0,0)])
test_time_2 = pd.DatetimeIndex([pd.Timestamp(2017,12,25,23,50,0)])

class TimeFeatures(unittest.TestCase):
    """Tests for `features.py`."""

    def test_hour_of_day(self):
        self.assertEqual(0, HourOfDay().indexer(test_time_1))
        self.assertEqual(23, HourOfDay().indexer(test_time_2))

    def test_day_of_year(self):
        self.assertEqual(0, DayOfYear().indexer(test_time_1))
        self.assertEqual(358, DayOfYear().indexer(test_time_2))

    # def test_us_holiday(self):
    #     self.assertEqual(0, USHoliday().indexer(test_time_1))
    #     self.assertEqual(1, USHoliday().indexer(test_time_2))

    def test_day_of_week(self):
        self.assertEqual(5, DayOfWeek().indexer(test_time_1))
        self.assertEqual(0, DayOfWeek().indexer(test_time_2))

    def test_quarter_of_year(self):
        self.assertEqual(0, QuarterOfYear().indexer(test_time_1))
        self.assertEqual(3, QuarterOfYear().indexer(test_time_2))

    def test_day_of_quarter(self):
        self.assertEqual(0, DayOfQuarter().indexer(test_time_1))
        self.assertEqual(85, DayOfQuarter().indexer(test_time_2))

    def test_interval_of_day(self):
        self.assertEqual(0, IntervalOfDay().indexer(test_time_1))
        self.assertEqual(286, IntervalOfDay().indexer(test_time_2))


if __name__ == '__main__':
    unittest.main()
