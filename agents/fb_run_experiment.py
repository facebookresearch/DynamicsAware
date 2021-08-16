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

import logging

import submitit

import agents.run_experiment as run_experiment


def main(params):
    if params.local_run:
        executor = run_experiment.DumyExecutor()
    else:
        executor = submitit.AutoExecutor(folder=run_experiment.ROOT_DIR /
                                         'submitit')
        executor.update_parameters(comment='PHYRE project',
                                   gpus_per_node=1,
                                   mem_gb=52,
                                   cpus_per_task=10,
                                   partition='learnfair',
                                   timeout_min=24 * 60)
    return run_experiment.main(params, executor)


if __name__ == "__main__":
    logging.basicConfig(format=('%(asctime)s %(levelname)-8s'
                                ' {%(module)s:%(lineno)d} %(message)s'),
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-run', type=int, default=0)
    parser.add_argument('--num-seeds', type=int)
    parser.add_argument('--use-test-split', type=int, required=True)
    parser.add_argument('--arg-generator',
                        required=True,
                        choices=run_experiment.ARG_GENERATORS.keys())
    main(parser.parse_args())
