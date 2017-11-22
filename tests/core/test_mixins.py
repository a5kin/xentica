import unittest

from xentica.core.mixins import BscaDetectorMixin
from xentica.core.base import XenticaException


class TestBscaDetector(unittest.TestCase):

    def test_not_detected(self):
        class Dummy(BscaDetectorMixin):
            pass
        with self.assertRaises(XenticaException):
            dummy = Dummy()
            dummy._bsca
