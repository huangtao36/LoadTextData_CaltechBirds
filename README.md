# LoadTextData_CaltechBirds
Two Method For Load Text Data， load Caltech200_Birds dataset as an example.

两种方法总体结构上差不多，只是一个需要构建词库，一个使用fastText编码.

## Method1
这个方法需要先统计所有文本，构建自己的词库，
输出的是文本是对应词库位置的'编号'， 以 0 作为 padding 项统一句子长度。 

> run `vocab_util.py` to build your vocabulary  
> run `dataloader.py` to load the caltech_birds datasets

对应的方法： attnGAN

## Method2
这个方法使用 fastText 对文本进行预编码， 输出的文本是编码后的概率值。

> run `dataloader.py` to load the caltech_birds datasets

对应的方法： ReedICML2016

## Download
- fastText 预训练模型：  [fastText Model Download](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip)
- Caltech-200 birds: [images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [captions](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing)

The caption data is from [this repository](https://github.com/reedscot/icml2016).

## File Structure
Root is 'Caltech200_birds' 
> |-- CUB_200_2011  
> |-- cub_icml  
