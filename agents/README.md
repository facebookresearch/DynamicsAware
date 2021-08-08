# Dynamics-Aware models on PHYRE
For installation, see the [README](../README.md)

# Training an agent

All of the agents are trained from a configuration file. Configurations are included for all experiments included in the paper in the `agents/experiments` folder. The entry point is `agents/run_sweep_file.py` which will train an agent based off the configuration file.
E.g., the following command will launch a sweep training self-supervised models with different
values for the hyperparameter temperature, running 4 seeds at each value. All output files
will be placed in a new folder called `output_folder` under the current directory.
This can be replaced with any path to a folder (which be created if it does not exist). Relative or absolute paths can be used

```(bash)
cd agents
python  python run_sweep_file.py  phyre_agents_final/agents/experiments/self_supervised_experiments/11_temperature_table_3e.conf  --base-dir=output_folder -o -l
```

The `-o` flag will cause the results of the experiment to be printed to the terminal when the experiment is over.
The `-l` flag will cause the jobs to be run locally, omitting this flag will 
result in the jobs being sent to the `submitit` library for submission to 
a compute cluster.


To view the results of job after it finished you can use the `-d` flag.
For example to print the results of the previous experiment after it 
finishes 

```(bash)
cd agents
python  python run_sweep_file.py  phyre_agents_final/agents/experiments/self_supervised_experiments/11_temperature_table_3e.conf  --base-dir=output_folder -o -d
```


To view the bash commands that correspond to the jobs to be run you can
use the `-v` flag, for example

```(bash)
cd agents
python  python run_sweep_file.py  phyre_agents_final/agents/experiments/self_supervised_experiments/11_temperature_table_3e.conf  --base-dir=output_folder -d -v
```

Each experiment file corresponds to table in the paper and the table number is at the end of the file name.
