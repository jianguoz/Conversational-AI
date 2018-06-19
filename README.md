# Text-Generation

## Papers related to Text generation. 
###  ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `updated on June 19, 2018`.

### Note: I write this Readme file in mixed English and Chinese, If needed, I will add separate English and/or Chinese version. 

* One [technical blog](https://zhuanlan.zhihu.com/p/33956907) of NLG on Taobao Recommendation gives some instructions on tackling multi-source and noisy data.  

* Two papers ([Alibaba-1a](https://github.com/jianguoz/Text-Generation/blob/master/1-Alibaba-18-A%20Multi-task%20Learning%20Approach%20for%20Improving%20Product%20Title%20Compression%20with%20User%20Search%20Log%20Data.pdf), [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/2-Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)) of **Alibaba, 2018** focus on `short title` generation.

* One recent paper, named [3-self-attention Generative ...](https://github.com/jianguoz/Text-Generation/blob/master/3-Self-Attention%20Generative%20Adversarial%20Networks.pdf), focus on using attention mechanism to generate images with fine-grained details, which is quite helpful for text generation. 

* A classical paper of Facebook, named [4-ICLR-16-Sequence Level training with...](https://github.com/jianguoz/Text-Generation/blob/master/4-Sequence%20Level%20Training%20with%20Recurrent%20Neural%20Networks.pdf), for sequence level generation

### Comments for papers

#### [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/2-Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)

**Summary:** 这篇文章用了RNN结构，主要贡献是将不同的features (e.g., content features, attention features, TF-IDF features, NER features, where NER is used to label entities like color, style, etc.) combine 到了一起， 前两种features利用了模型的深度 (deep) 信息, 后两种features直接对输入数据进行处理, 利用的宽度 (wide) 信息, 所以 模型特点 deep & wide. 另外一个贡献是作者 <打算> 开源 youhaohuo short text summarization dataset. Moreover, this paper treats short title summarization as an `extractive summarization` task, 另外一种常见的做法是将短标题生成任务当作 `abstractive summarization`, which has the ability to generate text beyond the original input text, in most cases, can produce more coherent and concise summaries. 但是对于淘宝数据集抽取标题来说, 我们没有必要在原始标题上创造新的单词, 所以 extractive summarization task更为适合. 

**Weekness**: This paper regards the extractive summary as a sequence classification problem, and aims to maximize the likelihood of aoll word labels $Y=(y_1,...,y_n)$, given input product title $X$ and model parameters $\theta$. 这种做法容易使得相近单词很难被分开, for instance,　“皮衣”　和　“皮夹克”　可能会在生成的标题中重复出现．另外，本文在模型或者方法上的创新很小，只是把现有的方法多加了谢features换了个场景应用了一下而已． 

#### [3-self-attention Generative ...](https://github.com/jianguoz/Text-Generation/blob/master/3-Self-Attention%20Generative%20Adversarial%20Networks.pdf)

**Summary:** This paper uses self-attention to improve the Inception score to a large extent. Self-attention has not yet been explored in the context of GANs, and although some researchers use attention over word embedding within an input sequence in GAN, but not self-attention over internal model states. In this paper, SAGAN learns to efficiently find global, long-range dependencies within internal representations of images.  Besides, SAGAN also uses two tricks: Spectral normalization and separate learning rates (TTUR) to futher stabilize GAN training, which should be useful for further research in natural language generation. 



## GAN
- SeqGAN - SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- MaliGAN - Maximum-Likelihood Augmented Discrete Generative Adversarial Networks
- RankGAN - Adversarial ranking for language generation
- LeakGAN - Long Text Generation via Adversarial Training with Leaked Information
- TextGAN - Adversarial Feature Matching for Text Generation
- GSGAN - GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution

## Reinforcement Learning
- (*****) Deep Reinforcement Learning For Sequence to Sequence Models
> 常用RL方法应用在Seq2Seq中，包含Tensorflow写的源代码。
- Maximum Entropy Inverse Reinforcement Learning
- (*****) Towards Diverse Text Generation with Inverse Reinforcement Learning
> IRL本质上和GAN等价，Maximum Entropy IRL是一种比较常用的IRL方法。
- (*****) Reinforcement Learning and Control as Probabilistic Inference-Tutorial and Review
> 从概率图的角度对RL进行了推导。可以考虑在图模型中加入latent variable。VAE＋maximum entropy RL进行fine-grained text generation.

## Transfer Learning/Meta-Learning
- (*****) Model-Agnostic Meta-Learning
> 最常用的一种meta-learning方法之一。
- A spect-augmented Adversarial Networks for Domain Adaptation
- (*****) Natural Language to Structured Query Generation via Meta-Learning
> 很好的将seq2seq的task定义成了meta-learning task.
- (*****) Learning a Prior over Intent via Meta-Inverse Reinforcement Learning 
> 当expert demonstration比较少的时候很难求reward function。考虑将其它tasks里面的信息作为prior引入进来。


==========

# (Application)
## Dialog Generation
- A Hierarchical Latent Structure for Variational Conversation Modeling
- A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues
- (*****) Improving Variational Encoder-Decoders in Dialogue Generation
> 以上三篇都是利用Variational Encoder-Decoders进行open domain中的dialogue generation
- (*****) DialogWAE- Multimodal Response Generation with Conditional Wasserstein Auto-Encoder
> Dialog generation, VAE+GAN, 在latent variable加入了mixture Gaussion。
- Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders

## QA
- Cross-Dataset Adaptation for Visual Question Answering
- Joint Image Captioning and Question Answering
- Learning Answer Embeddings for Visual Question Answering
- Question Answering through Transfer Learning from Large Fine-grained Supervision Data

## VQA (Visual Question Answering)
- (**) Visual Question Answering- A Survey of Methods and Datasets
> survey，快速了解VQA。
- Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning
- (****) Visual Question Answering as a Meta Learning Task
> 首次将VQA这个task定义成了一个Meta Learning Task。

## VisDialog (Visual Dialog)
- (**) Visual Dialog
> 快速了解VisDialog这个task。
- (****) Are You Talking to Me-Reasoned Visual Dialog Generation through Adversarial Learning
> 用GAN做visDiag
- Two can play this Game- Visual Dialog with Discriminative Question Generation
- Zero-Shot Dialog Generation with Cross-Domain Latent Actions
- Adversarial Learning of Task-Oriented Neural Dialog Models
- Best of Both Worlds- Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model


