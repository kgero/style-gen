# Style Generation

## Set-up (short)

I'm running on Python 3.6.6.

These scripts require [fast.ai](https://github.com/fastai/fastai) version 0.7. I've included in this repository the code for the old (0.7) and new (1.0) fastai. We're working with the old, but, hey, maybe we'll want the new at some point. 

I did used their configuration file to set up the environment. I used `conda` to manage the packages and environment. Note that the code below is for cpu, but I've included the environment file for gpu `enviroment.yml`. 

```
conda env create -f environment-cpu.yml
conda activate fastai-cpu
```

Or for updating the environment:

`conda env update -f environment-cpu.yml`

These scripts are set up to run as Jupyter Notebooks, so open jupyter with:

`jupyter notebook`

## Set-up (commentary)

I'm a little disappointed with fast.ai because it wasn't trivial to get set up working with their code. Actually, working with their code was really confusing, as they just released version 1.0 but most of their tutorials still run on 0.7. Additionally I can't seem to directly install fastai with conda or pip; I actually needed some code from their repository. 

There are instructions to install fastai version 0.7 [here](http://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652) but I don't *think* you need it.

Also note that their configuration files don't work out of the box; I had to modify them. What they have just patently does not work. I've had to add some packages they were missing, as well as make sure it installs pytorch 0.4 -- for some reason they are installing pytorch < 0.4, which is wild because at least some of their code requires pytorch 0.4.