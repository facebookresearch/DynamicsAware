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

import phyre
from phyre import Evaluator


class DetailedEvaluator(Evaluator):

    def __init__(self, task_ids):
        super().__init__(task_ids)
        self._per_task_evaluator = [
            Evaluator((task_id,)) for task_id in self._task_ids
        ]
        self._per_task_auccess_cache = {}
        self.templates = list(set(map(lambda x: x.split(":")[0], task_ids)))
        self._action_log = [[] for i in task_ids]

    def maybe_log_attempt(self, task_index, status, action=None):
        self._per_task_evaluator[task_index].maybe_log_attempt(0, status)
        super().maybe_log_attempt(task_index, status)
        if action is not None and status != phyre.SimulationStatus.INVALID_INPUT:
            self._action_log[task_index].append(action)

    def get_per_task_auccess(self, task_index):
        auccess = self._per_task_auccess_cache.get(task_index)
        if auccess is None:
            auccess = self._per_task_evaluator[task_index].get_auccess()
            self._per_task_auccess_cache[task_index] = auccess
        return auccess

    def get_task_index(self, task_id):
        return self._task_ids.index(task_id)

    def get_template_auccess(self, template):
        n = 0
        sum_auccess = 0 
        for i, task_id in enumerate(self._task_ids):
            if task_id.startswith(template + ":"):
                sum_auccess += self.get_per_task_auccess(i)
                n += 1
        return sum_auccess / n 
    
    def dump_report(self):
        report = ""
        for task_id in sorted(self._task_ids):
            index = self.get_task_index(task_id)
            auccess = self.get_per_task_auccess(index)
            report += f"{task_id},{auccess:.2f}"
            if self._action_log[index]:
                report += ","
                for action in self._action_log[index]:
                    report += "-".join(map(lambda x: f"{x:.5f}", action)) + ";"
                report = report[:-1]
            report += "\n"

        for template in sorted(self.templates):
            auccess = self.get_template_auccess(template)
            report += f"{template},{auccess:.2f}\n"
        return report

if __name__ == "__main__":
    import phyre
    import numpy as np
    task_ids, _, _ = phyre.get_fold(phyre.MAIN_EVAL_SETUPS[0], 0)
    evaluator = DetailedEvaluator(task_ids)
    rng = np.random.RandomState(42)
    print(evaluator.templates)
    for i, task_id in enumerate(task_ids):
        for j in range(100):
            if rng.random() < 0.1 and (not task_id.startswith("00007:") or rng.random() < 0.3):
                evaluator.maybe_log_attempt(i, phyre.SimulationStatus.SOLVED, rng.random(size=3))
            else:
                evaluator.maybe_log_attempt(i, phyre.SimulationStatus.NOT_SOLVED, rng.random(size=3))

    # print(evaluator.get_auccess())
    print([(template, evaluator.get_template_auccess(template)) for template in evaluator.templates])
    print(np.mean([evaluator.get_per_task_auccess(i) for i,j in enumerate(task_ids)]))
    print(evaluator.dump_report())
