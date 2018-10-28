# Style Generation

This repository hosts scripts for fine-tuning generic language models to produce text of certain linguistic styles. It is based on the fast.ai tutorial for fine-tuning a language model for classification tasks, which you can find [here](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb).

Below are some results, in which we compare simply training a language model with the available stylistic data (raw) to using a pre-trained language model and fine-tuning with the stylistic data (pre-train).

| Style  | Tokens | Condition | Example Outputs |
|-----|----|----|----|
|imaginative | 25k | raw | saying my head -- that which it . t not rabbit ; went near ‘ us a business ! i , and under , -- ' violently '|
|imaginative | 25k | pre-train | ‘ do you know what you're saying ! ' said alice . caterpillar lobsters ! --   |
|highbrow | 5k | raw | the  steak holes immersed the " to are the . it like it there thanks lobster cerebral an . |
|highbrow | 5k | pre-train | ( there also appear to be some touristic stimuli that make the crab turn carnivore , though some do b\&bs dislike this demotic thing . ) |
|poetry | 25k | raw | the  chair pleasure <br> length that passed bird n't . so smiling <br> he dissent a ? in . , stand |
|poetry | 25k | pre-train |  after god , surpasses god have <br>no nearer than heaven ; <br> but heaven had not daffodils , |

## Fine-tuning a model with your own data

To understand the process, we recommend using the `style_gen.ipynb` python notebook, which has explanations of how the pre-processing and training works. Included in this repository are some example data sets in the `data_style/` directory. 

Note that you will have to download the pre-trained language model. We use the wikitext103 language created by Stephen Merity @ Salesforce research. [Here](http://files.fast.ai/models/wt103/) is a link from fast.ai, and here is a command for the terminal:

`$ wget -nH -r -np -P {PATH} http://files.fast.ai/models/wt103/`

Then you can use `generate_text.ipynb` to generate text with the models you trained in `style_gen.ipynb`.

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
