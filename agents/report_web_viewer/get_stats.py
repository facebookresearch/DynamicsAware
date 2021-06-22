# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import submitit
import os
import numpy as np


def gather_numbers(dir_):
    os.chdir(dir_)
    results = {}
    for dirname in os.listdir("."):
        if os.path.isdir(dirname):
            results[dirname] = []

            os.chdir(dirname)
            for rundir in os.listdir("."):
                try:
                    data = open(os.path.join(rundir, "results.json"),
                                "r").read().split(":")[-1][:-1]
                    results[dirname].append(float(data))
                except FileNotFoundError:
                    pass
            os.chdir("..")

    return (results)


# res = gather_numbers("/private/home/ekahmed/explore_phyre/phyre/results/dev/dqn_10k")
from collections import defaultdict

vals = defaultdict(lambda: [])

# for network in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
# res = gather_numbers(f"/checkpoint/ekahmed/phyre/action_cond/dev")
base_dir = "/checkpoint/ekahmed/phyre/"
# for dir_ in os.listdir(base_dir):
if 1:
    res = gather_numbers(base_dir)
    print("*" * 10, dir, "*"*10)
    for run, val in res.items():
        print(run, np.mean(val), np.std(val), len(val))
        vals[run].append(np.mean(val))
        vals[run + "_err"].append(np.std(val))

    for run, val in res.items():
        print(run, np.mean(val), np.std(val), len(val))

    # for run in [
            # "ball_within_template", "ball_cross_template",
            # "two_balls_within_template", "two_balls_cross_template"
    # ]:
        # print(f"{np.mean(res[run])*100:.1f} ± {np.std(res[run]) * 100:.1f}",
              # end="\t")
