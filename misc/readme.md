
## Note: I write this Readme file in mixed English and Chinese, If needed, I will add separate English and/or Chinese version. 


## Schedule

### 1. Conversational AI Tutorial Review and Reinforcement Learning Review

- Time-1: September 19, 2018, Wednesday, 4:00 - 10:30 PM
- Reading-1: [Natural Language Generation: A Review](https://github.com/jianguoz/Natural-Language-Generation/blob/master/qa-review-09.20-wanyao-upload.pdf) 
- Time-2: September 25, 2018, Tuesday, 2:30 - 5:00 PM
- Reading-2: [Model-free Policy Gradient Method](https://github.com/jianguoz/Natural-Language-Generation/blob/master/model-free%20PB%20on%20RL%20.pdf) 

- Time-3: September 28, 2018, Friday, 2:30 - 5:30 PM
- Reading-3: [Conversational AI 1](https://github.com/jianguoz/Text-Generation/blob/master/Tutorial-1-Conversational%2BAI.pdf) and [Conversational AI 2](https://github.com/jianguoz/Natural-Language-Generation/blob/master/Tutorial-1-2-COLING18_Tutorial.pdf)

> Descriptions for Reading-1: 

> In this pdf, we briefly review deep (reinforcement) learning based Text Summarization, Question Answering, Dialog Generation, etc. Moreover, we introduce several representative papers. The roadmap can be summarized as: **summarization/caption -> TextQA -> Text Dialog -> VQA -> Video QA -> Visual Dialog -> Conversational IR**. 

> Descriptions for Reading-2:

>  We  review **model-free policy gradient methods** and related papers as well as provide some related matetrials in the pdf. It includes policy gradient method, DQN method, PPO, etc.

> Descriptions for Reading-3:

>  We give a tutorial on deep learning for conversational AI. It mainly includes **task-oriented** conversational AI (e.g., Apple SiRi, Microsoft Cortana) and **fully data-driven** convesational AI (e.g., Chit-Chat bot).

### 2. Variational Autoencoder for Dialogue Generation

- Time: September 30, 2018, Sunday, 4:00 - 6:30 PM
- Reading (Required): [A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues](https://arxiv.org/pdf/1605.06069.pdf) and [Improving Variational Encoder-Decoders in Dialogue Generation](https://arxiv.org/pdf/1802.02032.pdf)

- Reading (Optional): [No Metrics Are Perfect Adversarial Reward Learning](https://arxiv.org/pdf/1804.09160.pdf)
and [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/pdf/1805.00909.pdf)

### 3. VAE +GAN for Dialogue Generation
- Time: October 8, 2018, Monday, 9:00-12:00am
- Reading: Review Week-2, [Vae+Gan (version 1)](https://github.com/jianguoz/Natural-Language-Generation/blob/master/VAE%2BGAN.pdf) and [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/pdf/1805.00909.pdf)

> Descriptions for Reading-3:

>  We give a brief introduction for GAN and VAE, and discuss **issues** related to GAN (e.g., training instability, mode collapse) and VAE (e.g., safe response, the posterior collapse problem), as well as **methods** to mitigate these issues.


### 4. Inverse Reinforcement Learning for NLG
- Time: October 11, 2018, Thursday, 4:00-6:30am
- Reading: [InverseRL for NLG (Part-I)](https://github.com/jianguoz/Natural-Language-Generation/blob/master/Inverse%20Reinforcemen%20Learning.pdf) 

> Descriptions for Reading-4:

>  This is an introduction of Inverse reinforcement learning for natural language generation.

### 5. Discussion

- Time: November 18, 2018, Sunday, 5:00-7:00pm

- Reading-5: [QA-Advanced](https://www.dropbox.com/home/Materials?preview=qa-advances.pptx)

> Descriptions for Reading-5:

>  This is an discussion for some new papers and directions in QA.

### [Difference betweek VQA and Visual dialog](https://github.com/jianguoz/Natural-Language-Generation/blob/master/ICCV-17-Visual%20Dialog.pdf)

Despite rapid progress at the intersection of vision and lan- guage – in particular, in image captioning and visual question answering (VQA) – it is clear that we are far from this grand goal of an AI agent that can ‘see’ and ‘communicate’. In captioning, the human-machine interaction consists of the machine simply talking at the human (‘Two people are in a wheelchair and one is holding a racket’), with no dialog or input from the human. While **VQA** takes a significant step towards human-machine interaction, it still represents `only a single round of a dialog` – unlike in human conversations, there is no scope for follow-up questions, no memory in the system of previous questions asked by the user nor consistency with respect to previous answers provided by the system (Q: ‘How many people on wheelchairs?’, A: ‘Two’; Q: ‘How many wheelchairs?’, A: ‘One’).

## Mis

* A [short summary](https://github.com/jianguoz/Text-Generation/blob/master/0-Short-Summary-GAN-Discrete-Texts.pdf) of GAN for discrete texts

* One [technical blog](https://zhuanlan.zhihu.com/p/33956907) of NLG on Taobao Recommendation gives some instructions on tackling multi-source and noisy data.  

* Two papers ([Alibaba-1a](https://github.com/jianguoz/Text-Generation/blob/master/1-Alibaba-18-A%20Multi-task%20Learning%20Approach%20for%20Improving%20Product%20Title%20Compression%20with%20User%20Search%20Log%20Data.pdf), [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/2-Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)) of **Alibaba, 2018** focus on `short title` generation.

* One recent paper, named [3-self-attention Generative ...](https://github.com/jianguoz/Text-Generation/blob/master/3-Self-Attention%20Generative%20Adversarial%20Networks.pdf), focus on using attention mechanism to generate images with fine-grained details, which is quite helpful for text generation. 

* A classical paper of Facebook, named [4-ICLR-16-Sequence Level training with...](https://github.com/jianguoz/Text-Generation/blob/master/4-Sequence%20Level%20Training%20with%20Recurrent%20Neural%20Networks.pdf), for sequence level generation

### Comments for papers

#### [Alibaba-1b](https://github.com/jianguoz/Text-Generation/blob/master/2-Alibaba-18-Automatic%20Generation%20of%20Chinese%20Short%20Product%20Titles%20for%20Mobile%20Display.pdf)

**Summary:** 这篇文章用了RNN结构，主要贡献是将不同的features (e.g., content features, attention features, TF-IDF features, NER features, where NER is used to label entities like color, style, etc.) combine 到了一起， 前两种features利用了模型的深度 (deep) 信息, 后两种features直接对输入数据进行处理, 利用的宽度 (wide) 信息, 所以 模型特点 deep & wide. 另外一个贡献是作者 <打算> 开源 youhaohuo short text summarization dataset. Moreover, this paper treats short title summarization as an `extractive summarization` task, 另外一种常见的做法是将短标题生成任务当作 `abstractive summarization`, which has the ability to generate text beyond the original input text, in most cases, can produce more coherent and concise summaries. 但是对于淘宝数据集抽取标题来说, 我们没有必要在原始标题上创造新的单词, 所以 extractive summarization task更为适合. 

**Weekness**: This paper regards the extractive summary as a sequence classification problem, and aims to maximize the likelihood of aoll word labels $Y=(y_1,...,y_n)$, given input product title $X$ and model parameters $\theta$. 这种做法容易使得相近单词很难被分开, for instance,　“皮衣”　和　“皮夹克”　可能会在生成的标题中重复出现．另外，本文在模型或者方法上的创新很小，只是把现有的方法多加了谢features换了个场景应用了一下而已． 

#### [3-self-attention-Generative ...](https://github.com/jianguoz/Text-Generation/blob/master/3-Self-Attention%20Generative%20Adversarial%20Networks.pdf)

**Summary:** This paper uses self-attention to improve the Inception score to a large extent. Self-attention has not yet been explored in the context of GANs, and although some researchers use attention over word embedding within an input sequence in GAN, but not self-attention over internal model states. In this paper, SAGAN learns to efficiently find global, long-range dependencies within internal representations of images.  Besides, SAGAN also uses two tricks: Spectral normalization and separate learning rates (TTUR) to futher stabilize GAN training, which should be useful for further research in natural language generation. 

#### [4-ICLR-16-Sequence Level training with...](https://github.com/jianguoz/Text-Generation/blob/master/4-Sequence%20Level%20Training%20with%20Recurrent%20Neural%20Networks.pdf)

**Summary:** 这篇论文的背景写的很好，对 exposure bias, 以及 cross-entropy, beam-search, reinforce, etc., 的缺陷解释的很清楚。 This paper 提出了一个方法, dubbed MIXER, 将cross-entropy和reinforce结合到了一起， mitigate了 只用cross-entropy会出现training and inference inconsistent 以及只用 reinforce在做文本生成时search space过大的问题。具体上是 刚开始用 XENT (cross-entropy) 训练一定的epochs， 然后对剩下的epochs, 用 XENT训练前 S 步， 用 Reinforce 训练 后 （T-S）步。 其中 S=T (T represents the length of a sequence), S--， 直到只用reinforce来训练整个句子。

**Note:** 现在大部分用 RL 的基本上都有一个预训练， 一般情况下都是先训练一定的epochs的XENT，然后剩余的epochs全部用论文提出的方法进行训练， 这篇论文的不同之处是采用了混合训练的方式。

![image1](https://github.com/jianguoz/Text-Generation/blob/master/misc/4-ICLR-16-Sequence%20Level%20training%20with.png)

#### [0-short-summary-of-GAN](https://github.com/jianguoz/Text-Generation/blob/master/0-Short-Summary-GAN-Discrete-Texts.pdf)

This ia a short summary of GAN writen by myself for text generation, and will be updated soon.
