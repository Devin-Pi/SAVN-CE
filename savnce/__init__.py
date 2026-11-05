# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from savnce.sims.savnce_simulator import SAVNCE_Simulator
from savnce.datasets.savnce_dataset import SAVNCE_Dataset
from savnce.tasks.savnce_task import SAVNCE_Task

# Keep rendering imports independent of optional policy-training dependencies.
def __getattr__(name):
    if name == 'Challenge':
        from savnce.challenge import Challenge
        return Challenge
    if name == 'Benchmark':
        from savnce.benchmark import Benchmark
        return Benchmark
    raise AttributeError(name)
