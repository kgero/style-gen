# Style Generation

This repository hosts scripts for fine-tuning generic language models to produce text of certain linguistic styles. It is based on the fast.ai tutorial for fine-tuning a language model for classification tasks, which you can find [here](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb).

## Set-up

This runs on Python 3.6.6.

These scripts require [fast.ai](https://github.com/fastai/fastai) version 0.7. We've included in this repository the code for the old (0.7) and new (1.0) fastai. We're working with the old, but, hey, maybe we'll want the new at some point. 

We did used their configuration file to set up the environment. We used `conda` to manage the packages and environment. Note that the code below is for cpu, but We've included the environment file for gpu `enviroment.yml`. We slightly modified the `.yml` files to deal with some install issues; so they are not exactly the same as on the fastai repository.

```
conda env create -f environment-cpu.yml
conda activate fastai-cpu
```

Or for updating the environment:

`conda env update -f environment-cpu.yml`

Some of these scripts are set up to run as Jupyter Notebooks, so open jupyter with:

`jupyter notebook`