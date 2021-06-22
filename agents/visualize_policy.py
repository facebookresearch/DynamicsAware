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

import numpy as np
import phyre
from PIL import Image, ImageDraw, ImageOps
import math
import functools
from skimage.color import rgb2lab, lab2rgb

SQRT2 = math.sqrt(2)

def adaptive_threshold(scores, ground_truth):
    ground_truth = ground_truth == 1
    candidate_thresholds = np.linspace(scores.min(), scores.max(), 20)
    best_n_matches = 0
    best_threshhold = 0
    for threshold in candidate_thresholds:
        n_matches  = ((scores > threshold ) == ground_truth).astype(int).sum()
        if n_matches > best_n_matches:
            best_n_matches = n_matches
            best_threshhold = threshold
    return threshold

red_lab = rgb2lab(np.array((1.0, 0, 0)).reshape(1,1,3))
white_lab = rgb2lab(np.array((1.0, 1.0, 1.0)).reshape(1,1,3))
red_lab[0,0,2] /= (1.7)
red_lab[0,0,0] -= 20

def lab_color(intensity, end_color=None, offset=0, scale=1):
    if end_color is None:
        end_color = red_lab
    else:
        end_color = rgb2lab(np.array(end_color).reshape(1, 1, 3)/255)
        end_color[0, 0, 2] /= scale
        end_color[0, 0, 0] += offset

    lab_color = (intensity * end_color + (1 - intensity) * white_lab)
    color = 1 -  lab2rgb(lab_color).reshape(-1)
    return tuple((255 * color).astype(int).reshape(3))
    if intensity < 0.05:
        return (0, 0, 0)
    color = red_lab.copy()
    color[0, 0, 0]  = intensity * -75 + 100
    color = lab2rgb(color)
    return tuple(255 - (color *254).reshape(-1).astype(int))


def visualize_policy(simulator, task_ind, actions, scores, threshold=0.5, r=5, vary_intensity=False, text=None, sigmoid=True, rescale=True, end_color=None, end_color_offset=0, end_color_scale=1):
    initial_scene = simulator.initial_scenes[task_ind]
    scene = phyre.observations_to_uint8_rgb(initial_scene)

    canvas = Image.new('RGB', (phyre.SCENE_HEIGHT, phyre.SCENE_WIDTH))
    draw = ImageDraw.Draw(canvas)
    if vary_intensity:
        rank = np.argsort(scores)
        scores = scores[rank]
        actions = actions[rank]
        if sigmoid:
            intensity_factors = 1 / (1 + np.exp(- 0.3 * scores))
        else:
            if rescale:
                intensity_factors = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                intensity_factors = scores
    for i, (action, score) in enumerate(zip(actions, scores)):
        if score > threshold:

            if len(action) == 3:
                centers = [action[:2]]
                colors = [functools.partial(lab_color, end_color=end_color, scale=end_color_scale, offset=end_color_offset)]#[lambda x: (0, x, x)]
            else:
                centers = [action[:2], action[3:5]]
                centers.sort(key=lambda x: x[0])
                colors = [lambda x: (0, x, x), lambda x:(0, x // 3 , x)]

            for point, color in zip(centers, colors):
                scaled_point = point * phyre.SCENE_WIDTH
                upper_corner = scaled_point + r / SQRT2
                lower_corner = scaled_point - r / SQRT2
                upper_corner = int(upper_corner[0]), int(upper_corner[1])
                lower_corner = int(lower_corner[0]), int(lower_corner[1])

                if vary_intensity:
                    intensity = intensity_factors[i]# int(255 * intensity_factors[i])
                else:
                    intensity = 1 
                draw.ellipse([lower_corner, upper_corner], fill=color(intensity))
    
    canvas = ImageOps.flip(canvas)
    draw = ImageDraw.Draw(canvas)
    if text is not None:
        draw.text((phyre.SCENE_WIDTH * 3 // 5, phyre.SCENE_HEIGHT // 7), text, align="left")

    visualization = (np.array(canvas) ).astype(np.uint8)
    visualization = (scene - visualization).clip(0, 255)
    original_pixels = (scene != 255).any(axis=2)
    visualization[original_pixels] = scene[original_pixels]
    return visualization

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import neural_agent

    tier, eval_setup = [("two_balls", "two_balls_cross_template"), ("ball", "ball_cross_template")][0]
    cache = phyre.get_default_100k_cache(tier)
    train_ids, _, _ = phyre.get_fold(eval_setup, 0)
    sim = phyre.initialize_simulator(train_ids[:11], tier)

    n = 10000
    actions = cache.action_array[:n]
    scores = cache.load_simulation_states(train_ids[10])[:n]

    img = visualize_policy(sim, 10, actions, scores, text="Ground Truth")
    plt.imsave("rollout/gt.jpg", img)
    
    agent = neural_agent.load_agent_from_folder("phyre/results/dev/dqn_10k/two_balls_cross_template/0/")
    q_scores = neural_agent.eval_actions(agent, actions, 128, sim.initial_scenes[0])
    q_pol = visualize_policy(sim, 0, actions, q_scores, threshold=0, vary_intensity=True, text="History Agent")
    plt.imsave("rollout/qpol.jpg", q_pol)
