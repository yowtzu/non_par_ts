import unittest
from features import *

test_time_1 = pd.datetime(2000,1,1,0,0,0)
test_time_2 = pd.datetime(2017,12,31,23,59,59)

class TimeFeatures(unittest.TestCase):
    """Tests for `features.py`."""

    def test_hour_of_day(self):
        self.assertEqual(0, HourOfDay().indexer(test_time_1))
        self.assertEqual(23, HourOfDay().indexer(test_time_2))

if __name__ == '__main__':
    unittest.main()
