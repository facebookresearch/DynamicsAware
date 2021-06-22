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

import subprocess
from itertools import chain
import re
import pathlib
import submitit
import os
import sys
import numpy as np

from itertools import product
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
from run_experiment import DummyExecutor

class Checkpointable:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


exec_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
base_command = "python " + exec_ + " --use-test-split=TEST --output-dir=OUTPUT --eval-setup-name=EVALSETUP --fold-id=SEED --dqn-save-checkpoints-every=5000 --dqn-updates=UPDATES --dqn-cosine-scheduler=1 --dqn-learning-rate=3e-4  --dqn-train-batch-size=BATCHSIZE --dqn-balance-classes=1 --dqn-rank-size=RANKSIZE --dqn-num-auccess-actions=10000 --dqn-eval-every=5000 --dqn-num-auccess-actions=10000 --dqn-tensorboard-dir=TENSORBOARD --dqn-network-type=NETWORK"  # --dqn-label-smoothing-buffer=BUFFER --dqn-label-smoothing-row-norm=RN --dqn-label-smoothing-sigma=SIGMA --dqn-coord-conv=COORD"

PLACEHOLDER = "$%^XVBRF*"
MANDATORY_PARAMETERS = {
    "updates": "UPDATES",
    "network": "NETWORK",
    "eval-setup": "EVALSETUP",
    "dir": "OUTPUT",
    "tensorboard": "TENSORBOARD",
    "seed": "SEED",
    #DEfault value params
    "use-test-split": "TEST",
    "dqn-train-batch-size": "BATCHSIZE",
    "dqn-rank-size": "RANKSIZE"
}

DEFAULT_PARAMETERS = {
    "use-test-split": "0",
    "dqn-rank-size": "10000"
    # "dqn-train-batch-size": "512"
}
optimal_rank_sizes = dict(ball_cross_template="1000", ball_within_template="10000", two_balls_within_template="100000", two_balls_cross_template="100000")
def subsitute_tokens(str_, tokens, keys, values):
    output = str_
    for param, alias in tokens.items():
        if param in keys:
            output = output.replace(alias, values[keys.index(param)])
    return output


def build_command(base_command, parameter_vals_dict, parameter_fmt_names):
    for param in MANDATORY_PARAMETERS.keys():
        if not param in parameter_vals_dict:
            if param in DEFAULT_PARAMETERS:
                print("using default value")
                parameter_vals_dict[param] = [DEFAULT_PARAMETERS[param]]
            else:
                raise ValueError(f"Missing Parameter {param}")

    tb = parameter_vals_dict["tensorboard"][0]
    dir_ = parameter_vals_dict["dir"][0]
    parameter_fmt_names.update(MANDATORY_PARAMETERS)
    for key, values in parameter_vals_dict.items():
        if len(values) > 1:
            if not all([parameter_fmt_names[key] in i for i in (dir_, tb)]):
                raise ValueError(
                    "Runs with different " + key +
                    " will overwrite each other in tensorboard or savedir")

    keys, values = zip(*parameter_vals_dict.items())
    commands = []
    table = []
    for value_combination in product(*values):
        command = base_command
        table.append({"aliases": tuple(sorted(parameter_fmt_names.items()))})
        instance_parameters = dict(zip(keys, value_combination))
        for key, value in instance_parameters.items():
            value = subsitute_tokens(value, parameter_fmt_names, keys,
                                     value_combination)
            table[-1][key] = value
            if key == "fmt":
                continue
            if key in MANDATORY_PARAMETERS:
                if key == "dqn-rank-size":
                    if value == "optimal":
                        command = command.replace("RANKSIZE", optimal_rank_sizes[instance_parameters["eval-setup"]])
                    else:
                        command = command.replace("RANKSIZE", value)
                else:
                    command = command.replace(MANDATORY_PARAMETERS[key], str(value))
            else:
                command += " --" + key + "=" + str(value)
        # Default value params
        # command = command.replace("TEST", "0")
        # command.replace("BATCHSIZE", "512")
        commands.append(command)

    return commands, table


def parse_config_file(config_file):
    if "===" in config_file:
        return sum([parse_config_file(i) for i in config_file.split("===")],[])

    values_dict = {}
    alias_dict = {}
    lines = config_file.split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        line = line.split()
        if len(line) == 3:
            param, alias, vals = line
        elif len(line) == 2:
            param, vals = line
            alias = None
        else:
            raise ValueError(f"Invalid syntax line {i} \"{' '.join(line)}\"")
        if alias is not None:
            alias_dict[param] = alias
        if param == "fmt":
            vals = vals.replace(",", PLACEHOLDER)
        values_dict[param] = vals.split(",")
    return [(values_dict, alias_dict)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep according to a config file")
    parser.add_argument("conf-file", type=str, nargs=1)
    parser.add_argument("--base-dir", required=True, type=str, default="", help="Path of directory to write output to.")
    parser.add_argument("-o", "--output", const=True, default=False, nargs="?", help="Output summary of job results")
    parser.add_argument("-d", "--dry", const=True, default=False, nargs="?", help="Dry run. Do not submit jobs to SLURM."
                                                                                  "Useful for viewing results of "
                                                                                  "completed jobs or generating job "
                                                                                  "commands.")
    parser.add_argument("--make-dirs", const=True, default=False, nargs="?", help="Create output directories. Is "
                                                                                  "automatically turned on when dry "
                                                                                  "run is off.")
    parser.add_argument("--failed-only", const=True, default=False, nargs="?", help="Relaunch only failed jobs")
    parser.add_argument("--n-gpu", type=int, default=4, help="Number of GPUs.")
    parser.add_argument("--partition", type=str, help="Which SLURM partition to submit the jobs to.")
    parser.add_argument("--comment", type=str, default=None, help="SLURM job comment")
    parser.add_argument("-v", "--verbose", const=True, default=False, nargs="?", help="Output the job commands to the "
                                                                                      "terminal.")
    parser.add_argument("-l", "--local", const=True, default=False, nargs="?", help="Run jobs on local machine instead"
                                                                                    " of submitting to SLURM.")
    args = parser.parse_args()
    print(vars(args))
    while args.base_dir and args.base_dir[-1] in ("/", "\\"):
        args.base_dir = args.base_dir[:-1]
    with open(vars(args)["conf-file"][0]) as file_io:
        conf = file_io.read()
    parsed_params = parse_config_file(conf)
    commands, tables = zip(
        *[build_command(base_command, *i) for i in parsed_params])
    for entry in chain(*tables):
        entry["dir"] = entry["dir"].replace("basedir", args.base_dir)

    commands = [c.replace("basedir", args.base_dir) for c in chain(*commands)]

    if args.failed_only:
        search_reg= re.compile(r"output-dir=.*?\s")
        clean_reg = re.compile(r"/.*\S")
        new_commands = []
        for command in commands:
            output_dir = search_reg.findall(command)[0]
            output_dir = clean_reg.findall(output_dir)[0]
            if os.path.exists(output_dir) and "results.json" in os.listdir(output_dir):
                continue
            else:
                new_commands.append(command)
        commands = new_commands


    if args.verbose:
        for command in commands:
            print(command, flush=True)
    if not args.dry or args.make_dirs:
        for entry in chain(*tables):
            pathlib.Path(entry["dir"]).mkdir(parents=True, exist_ok=True)

    if not args.dry:
        logdir = os.path.commonpath([x["dir"] for x in chain(*tables)])
        logdir = os.path.join(logdir, "logdir")
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
        print("logdir : " + logdir)
        print(f"Submitting to {args.partition} with {args.n_gpu}gpu/task")
        with open(os.path.join(logdir, "sweep_params.conf"), "a+") as file_io:
            file_io.write("\n===\n" + conf)
        n_commands = len(commands)
        if args.local:
            executor = DummyExecutor()
        else:
            executor = submitit.AutoExecutor(logdir, max_num_timeout=4 * n_commands)
        executor.update_parameters(gpus_per_node=args.n_gpu,
                                   mem_gb=300 * args.n_gpu // 7,
                                   cpus_per_task=args.n_gpu * 10,
                                   partition=args.partition,
                                   timeout_min=72 * 60,
                                   signal_delay_s=90,
                                   name="phyre",
                                   array_parallelism=64,
                                   comment=args.comment)
        jobs = executor.map_array(Checkpointable(subprocess.check_call),
                                  map(lambda x: x.split(), commands))
        print("Submitted jobs, waiting ...")
        res = [job.result() for job in jobs]

    #reduce runs which differ in only in seed into one entry
    reduced = defaultdict(lambda: [])
    reduce_by = ["seed", "tensorboard", "dir", "dqn-load-from", "dqn-checkpoint-dir"]
    if args.output:
        for entry in chain(*tables):
            dir_ = entry["dir"]
            [entry.pop(key, None) for key in reduce_by]
            try:
                with open(os.path.join(dir_, "results.json")) as output_file:
                    score = output_file.read().split(":")[-1].strip()[:-1]
            except:
                score = float("nan")
            reduced[tuple(sorted(entry.items()))].append(float(score))

        functions = {
            "MAX": max,
            "MIN": min,
            "MED": np.median,
            "MEAN": np.mean,
            "STD": np.std
        }
        report = ""
        for entry, values in reduced.items():
            entry = dict(entry)
            fmt = entry["fmt"].replace(PLACEHOLDER, ",")
            aliases = dict(entry["aliases"])
            report_line = subsitute_tokens(fmt, aliases, *zip(*entry.items()))
            for alias, fn in functions.items():
                report_line = report_line.replace(alias, str(fn(values)))
            report += report_line + "\n"
        report = report.replace("_template", "")
        print(report)
