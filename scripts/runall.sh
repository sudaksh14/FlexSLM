#!/usr/bin/bash
mkdir -p ~/jobs
python3 -m training.run_experiment listcommand $1 | while read config
do
    echo $config
    sbatch $config
done