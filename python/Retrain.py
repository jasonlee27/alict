# This script is to re-test models using failed cases
# found from testsuite results

from typing import *
from pathlib import Path

from checklist.test_suite import TestSuite as suite

from Macros import Macros
from Utils import Utils
from Testsuite import Testsuite
from Model import Model

import os
import sys


class Retrain:

    @classmethod
    def get_failed_cases_from_test_results(cls):
        pass

    @classmethod
    def load_model(cls):
        pass

    @classmethod
    def train(cls):
        pass

    @classmethod
    def evaluate(cls):
        pass

    @classmethod
    def test(cls):
        pass
    
