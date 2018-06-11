# Text-Generation

## Papers related to Text generation. 
### - ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `#1589F0` Jianguo Zhang, updated on June 11, 2018

### Note: I write this Readme file in mixed English and Chinese, If needed, I will add separate English and/or Chinese version. 

* One [technical blog](https://zhuanlan.zhihu.com/p/33956907) of NLG on Taobao Recommendation gives some instructions on tackling multi-source and noisy data.  

* Two papers ([Alibaba-1a](https://github.com/jianguoz/Text-Generation/blob/master/Alibaba-18-A%20Multi-task%20Learning%20Approach%20for%20Improving%20Product%20Title%20Compression%20with%20User%20Search%20Log%20Data.pdf), [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)) of **Alibaba, 2018** focus on `short title` generation.

* A classical paper of Facebook, named [ICLR-16-Sequence Level training with...](https://github.com/jianguoz/Text-Generation/blob/master/ICLR-16-Sequence%20Level%20training%20with%20Recurrent%20Neural%20Networks.pdf), for sequence level generation


### [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)

**Summary:** 这篇文章用了RNN结构，主要贡献是将不同的features (e.g., content features, attention features, TF-IDF features, NER features, where NER is used to label entities like color, style, etc.) combine 到了一起， 前两种features利用了模型的深度 (deep) 信息, 后两种features直接对输入数据进行处理, 利用的宽度 (wide) 信息, 所以 模型特点 deep & wide. 另外一个贡献是作者 <打算> 开源 youhaohuo short text summarization dataset. Moreover, this paper treats short title summarization as an extractive summarization task, 另外一种常见的做法是将短标题生成任务当作abstractive summarization, which has the ability to generate text beyond the original input text, in most cases, can produce more coherent and concise summaries. 但是对于淘宝数据集抽取标题来说, 我们没有必要在原始标题上创造新的单词, 所以 extractive summarization task更为适合. 

**Weekness**: This paper regards the extractive summary as a sequence classification problem, and aims to maximize the likelihood of aoll word labels $Y=(y_1,...,y_n)$, given input product title $X$ and model parameters $\theta$. 这种做法容易使得相近单词很难被分开, for instance,　“皮衣”　和　“皮夹克”　可能会在生成的标题中重复出现．另外，本文在模型或者方法上的创新很小，只是把现有的方法多加了谢features换了个场景应用了一下而已． 

