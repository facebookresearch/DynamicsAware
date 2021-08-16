
[![GitHub license](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

# Dynamics-Aware Models

Code for reproducing [Physical Reasoning Using Dynamics-Aware Models](https://arxiv.org/abs/2102.10336).
This branch trains models that use the hand crafted loss. 

# Getting Started
## Installation
A [Conda](https://docs.conda.io/en/latest/) virtual enviroment is provided contianing all necessary dependencies.
```(bash)
git clone https://github.com/facebookresearch/dynamics-aware-embeddings
cd dynamics-aware-embeddings
git checkout hand_crafted_loss
conda env create -f env.yml
source activate dynamics_aware
# You might need to replace next command with the correct
# command to install pytorch on your system if you are not running linux
# and cuda 10.2  
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -e src/python
```
# Running Experiments
To run an experiments locally 
```(bash)
cd agents
python  python run_sweep_file.py  <experiment_file>  --base-dir=<basedir> -o -l
```
Where `<experiment_file>` is the experiment file to run and `<basedir>` is the directory where the experiment output should be 
stored. For more details see [agents](agents/)
# License
Dynamics-Aware Models is released under the Apache license. See [LICENSE](LICENSE) for additional details.

# Citation
If you use `dynamics-aware-embeddings` or the baseline results, please cite it

```bibtex
@article{ahmed2021physical,
  title={Physical Reasoning Using Dynamics-Aware Models},
  author={Ahmed, Eltayeb and Bakhtin, Anton and van der Maaten, Laurens and Girdhar, Rohit},
  journal={arXiv preprint arXiv:2102.10336},
  year={2021}
}

```