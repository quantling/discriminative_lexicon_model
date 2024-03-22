import pytest
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyldl.mapping as pm
import xarray as xr
from pyldl.classldl import LDL

TEST_ROOT = Path('.')
#TEST_ROOT = Path(__file__).parent
RESOURCES = TEST_ROOT / 'resources'

def test_empty_initialize ():
    ldl = LDL()
    assert isinstance(ldl, LDL)
    assert ldl.__dict__ == dict()

