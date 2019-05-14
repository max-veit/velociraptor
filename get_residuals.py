"""Script to compute the residuals of a previously-stored model"""


import argparse
import logging
import sys

import ase

import fitutils
import transform


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Get the residuals for a previously-stored model on new data")
