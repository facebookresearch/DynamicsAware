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

import flask
import os
from flask import abort, request
from flask.templating import render_template
template_folder = os.path.dirname(os.path.abspath(__file__))
app = flask.app.Flask(
    __name__, template_folder=template_folder)
app.config["CACHE_TYPE"] = "null"
from flask import send_from_directory, send_file
from pathlib import Path
import numpy as np
import phyre
import io
import imageio

from visualize_policy import visualize_policy

def is_image(file_path):
    return file_path.split(".")[-1] in ("jpeg", "jpg", "png", "gif")


img_dir = "visuals/ground_truths/all"
gif_dir = "visuals/ground_truths/gifs/all"

def text_dump(file_path, request):
    with open(file_path, "r") as file_io:
        return file_io.read()

def auccess_report(file_path, request):
    stride = int(request.args.get("stride", 1))
    template_group = request.args.get("Template Group", False) == "on"
    one_per_template = request.args.get("Template Results", False) == "on"
    num_actions_to_plot = int(request.args.get("num_to_plot", 50))
    num_actions_to_anim = int(request.args.get("num_to_anim", 4))
    template_to_display  = int(request.args.get("template", -1))

    with open(file_path) as file_io:
        report = file_io.read()
    table = {}
    actions_table ={}
    templates = set()
    for line in report.split("\n"):
        if not line:
            continue
        line = line.split(",")
        identifier, score = line[0], line[1]
        score = float(score)
        if ":" in identifier:
            template, task_no = identifier.split(":")
            table[(template, task_no)] = score
            templates.add(template)
            if len(line) > 2:
                actions = line[2].replace("-", ",").split(";")
                actions_table[(template, task_no)] = actions

    template_means = {
        template: np.mean(
            list(
                map(lambda x: x[1],
                    filter(lambda x: x[0][0] == template, table.items()))))
        for template in templates
    }

    lexographic = [(template_means[key[0]], key[0], score, key[1])
                   for key, score in table.items()]
    
    if template_to_display!= -1:
        template_to_display = "%05d" % template_to_display
        lexographic = [entry for entry in lexographic if entry[1] == template_to_display]
    if template_group:
        lexographic.sort(reverse=True)
    else:
        lexographic.sort(key=lambda x: (x[2], x[1], x[3]), reverse=True)

    lexographic = lexographic[::stride]

    if one_per_template:
        new_records = []
        for record in lexographic:
            if not any(x[1] == record[1] for x in new_records):
                new_records.append(record)
        lexographic = new_records
    
    if template_group:
        i = 0
        lexographic.insert(0, "sep")
        while i < len(lexographic) - 1:
            if lexographic[i][1] != lexographic[i + 1][1] and ("sep" not in (
                    lexographic[i], lexographic[i + 1])):
                lexographic.insert(i + 1, "sep")
            i += 1

    actions_to_anim = {key: ';'.join(actions[:num_actions_to_anim]) for key, actions in actions_table.items()}
    actions_to_plot = {key: ";".join(actions[:num_actions_to_plot]) for key, actions in actions_table.items()}
    # import ipdb; ipdb.set_trace()
    return render_template("auccess_report.html",
                           img_dir=img_dir,
                           gif_dir=gif_dir,
                           join=os.path.join,
                           enumerate=enumerate,
                           records=lexographic,
                           template_results=one_per_template,
                           num_to_plot=num_actions_to_plot,
                           num_to_anim=num_actions_to_anim,
                           actions_to_anim=actions_to_anim,
                           actions_to_plot=actions_to_plot)


special_files = {"csv": auccess_report, "json": text_dump}


def is_servable(file_path):
    return is_image(file_path) or file_path.split(
        ".")[-1] in special_files.keys()


@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>', methods=["GET", "POST"])
def dir_listing(req_path):
    # raise Exception("Use python pathlib to make sure no escaping")
    BASE_DIR = "/home/eltayeb"
    # req_path = "/" + req_path
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)
    # if abs_path[-1] == "/" and len(abs_path) > 1:
    # abs_path = abs_path[:-1]
    # if "." not in abs_path:
    # abs_path += "/"
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    if ".." in abs_path:
        print("..")
        raise Exception("No")
    path_obj = Path(abs_path).resolve()
    if BASE_DIR not in map(lambda x: x.as_posix(),
                           (path_obj, *path_obj.parents)):
        print(path_obj.as_posix(), "causing issues")
        raise Exception("No")

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if is_image(abs_path):
            return send_from_directory(BASE_DIR, req_path, cache_timeout=0.001)
        else:
            ext = abs_path.split(".")[-1]
            if ext in special_files.keys():
                return special_files[ext](abs_path, request)
            else:
                return "No"
    # Show directory contents
    files = os.listdir(abs_path)
    files.sort()
    images = [file for file in files if is_image(file)]
    return render_template('files.html',
                           files=files,
                           images=images,
                           base_path=req_path,
                           join=os.path.join)


one_ball_task_ids = phyre.get_fold("ball_within_template", 0)
one_ball_task_ids = sum(one_ball_task_ids, ())
one_ball_sim = phyre.initialize_simulator(one_ball_task_ids, "ball")

two_ball_task_ids = phyre.get_fold("two_balls_within_template", 0)
two_ball_task_ids = sum(two_ball_task_ids, ())
two_ball_sim = phyre.initialize_simulator(two_ball_task_ids, "two_balls")

@app.route("/visualize_policy", methods=["GET"])
def generate_plot():
    task_id = request.args.get("task_id")
    actions = request.args.get("action")
    actions = np.array([list(map(float, action.split(","))) for action in actions.split(";")])
    if task_id.startswith("001"):
        sim = two_ball_sim
    else:
        sim = one_ball_sim
    task_idx = sim.task_ids.index(task_id)
    scores = len(actions) - np.arange(len(actions))
    vis = visualize_policy(sim, task_idx, actions, scores, threshold=-1, vary_intensity=True, sigmoid=False)
    io_buffer = io.BytesIO()
    io_buffer.seek(0)
    imageio.imwrite(io_buffer, vis, format="png")
    io_buffer.seek(0)
    return send_file(io_buffer, attachment_filename="vis.png")

@app.route("/get_sim", methods=["GET"])
def generate_simulation():
    task_id = request.args.get("task_id")
    actions = request.args.get("action")
    actions = [list(map(float, action.split(","))) for action in actions.split(";")]
    if task_id.startswith("001"):
        sim = two_ball_sim
    else:
        sim = one_ball_sim
    task_idx = sim.task_ids.index(task_id)
    imgs_batch = []
    statuses = []
    blank = np.zeros((1, 50, 256, 256)).astype(int)
    for action in actions:
        status, imgs = sim.simulate_single(task_idx, action, stride=20)
        if imgs is None:
            imgs = blank
        else:
            imgs = np.concatenate([imgs, blank[0, :50 -len(imgs)]]) 
            imgs = imgs[None, :, :, :]
        imgs_batch.append(imgs)
        statuses.append(status == 1)
    
    imgs_batch = np.concatenate(imgs_batch)
    io_stream = io.BytesIO()
    io_stream.seek(0)
    phyre.vis.save_observation_series_to_gif(imgs_batch, io_stream, statuses)
    io_stream.seek(0)
    return send_file(io_stream, attachment_filename="anim.gif")


if __name__ == "__main__":
    app.run(processes=20, threaded=False, host="0.0.0.0", port=8166, debug=False)
