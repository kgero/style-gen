# Style Generation

## Set-Up

You need to install fastai version 0.7. Instructions [here](http://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652). I'm running on Python 3.6.6.

Since I don't want to have the whole fastai repository in here, I just used their configuration file to set up the environment.

I used `conda` to manage the packages and environment:

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