# Style Generation

## Set-Up

I'm running on Python 3.6.6.

These scripts require [fast.ai](https://github.com/fastai/fastai) version 0.7. This is pretty confusing, as they just released version 1.0 but most of their tutorial still run on 0.7. I can't seem to directly install it with conda or pip; I actually needed some code from their repository. I've included in this repository the code for the old (0.7) and new (1.0) fastai. We're working with the old, but, hey, maybe we'll want the new at some point. 

There are instructions to install fastai version 0.7 [here](http://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652) but I don't *think* you need it.

I did used their configuration file to set up the environment. I used `conda` to manage the packages and environment. Note that the code below is for cpu -- I've included the environment file for gpu `enviroment.yml` but haven't tested it.

```
conda env create -f environment-cpu.yml
conda activate fastai-cpu
```

Or for updating the environment:

`conda env update -f environment-cpu.yml`

I had to manually update `pytorch`, not sure why it didn't install the most recent version:

`conda install -c pytorch pytorch`

These scripts are set up to run as Jupyter Notebooks, so open jupyter with:

`jupyter notebook`