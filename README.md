## Natural Language Generation Reading Group.


### ![#f03c15](https://placehold.it/15/f03c15/000000?text=+)  `Updated on Sep 26, 2018`.

## Schedule

### 1. Conversational AI Tutorial Review and Reinforcement Learning Review

- Yao Wan
- Time-1: September 19, 2018, Wednesday, 4:00 - 10:30 PM
- Reading-1: [Natural Language Generation: A Review](https://github.com/jianguoz/Natural-Language-Generation/blob/master/qa-review-09.20-wanyao-upload.pdf) 
- Ye Liu
- Time-2: September 25, 2018, Tuesday, 2:30 - 5:00 PM
- Reading-2: [Model-free Policy Gradient Method](https://github.com/jianguoz/Natural-Language-Generation/blob/master/model-free%20PB%20on%20RL%20.pdf) 

- Jianguo Zhang
- Time-3: September 28, 2018, Friday, 2:30 - 5:30 PM
- Reading-3: [Conversational AI](https://github.com/jianguoz/Text-Generation/blob/master/Tutorial-1-Conversational%2BAI.pdf)


> Descriptions for Reading-1: 

> In this pdf, we briefly review deep (reinforcement) learning based Text Summarization, Question Answering, Dialog Generation, etc. Moreover, we introduce several representative papers. The roadmap can be summarized as: **summarization/caption -> TextQA -> Text Dialog -> VQA -> Video QA -> Visual Dialog -> Conversational IR**. 

> (在这个pdf中，我主要对基于深度强化学习的文本生成技术在当前比较火的Summarization, QA, Dialog等几个领域中应用进行了简单回顾，介绍了我觉得比较好的几篇论文，希望能给读者一些启发。
整个思维为：summarization/caption -> TextQA -> Text Dialog -> VQA -> Video QA -> Visual Dialog -> Conversational IR.)

> Descriptions for Reading-2:

>  We  review **model-free policy gradeint methods and related papers** as well as provide some related matetrials in the pdf. It includes policy gradient method, DQN method, PPO, etc.

> Descriptions for Reading-3:

>  We give a tutorial on deep learning for conversational AI. It mainly includes **task-oriented** conversational AI (e.g., Apple SiRi, Microsoft Cortana) and **fully data-driven** convesational AI (e.g., Chit-Chat bot).


## <========================================================>

## ==> Tutorials

#### ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `Updated on Sep 13, 2018`

Three tutorials on [deep generative models](https://github.com/jianguoz/Text-Generation/blob/master/Tutorial-2-ijcai_ecai_tutorial_deep%20generative%20model%20_copy.pdf) in `IJCAL 2018`, [conversational AI](https://github.com/jianguoz/Text-Generation/blob/master/Tutorial-1-Conversational%2BAI.pdf) in `ACL 2018`, and a recent tutorial on a [unified view of deep generative models](https://github.com/jianguoz/Text-Generation/blob/master/Tutorial-3-A%20univied%20view%20of%20Deep%20Generative%20models%20.pdf) in `ICML 2018`.

## ==> GAN
- SeqGAN - SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient
- MaliGAN - Maximum-Likelihood Augmented Discrete Generative Adversarial Networks
- RankGAN - Adversarial ranking for language generation
- LeakGAN - Long Text Generation via Adversarial Training with Leaked Information
- TextGAN - Adversarial Feature Matching for Text Generation
- GSGAN - GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution

## ==> Reinforcement Learning
- (*****) Deep Reinforcement Learning For Sequence to Sequence Models
> 常用RL方法应用在Seq2Seq中，包含Tensorflow写的源代码。
- Maximum Entropy Inverse Reinforcement Learning
- (*****) Towards Diverse Text Generation with Inverse Reinforcement Learning
> IRL本质上和GAN等价，Maximum Entropy IRL是一种比较常用的IRL方法。
- (*****) Reinforcement Learning and Control as Probabilistic Inference-Tutorial and Review
> 从概率图的角度对RL进行了推导。可以考虑在图模型中加入latent variable。VAE＋maximum entropy RL进行fine-grained text generation.
- (*****) Policy gradient methods for reinforcement learning with function approximation
- (*****) Proximal policy optimization algorithms
> Change the on-policy gradient method to off-policy gradient method to improve the data efficiency. And use trust region optimization method to search for the better result.  
- (*****) Continuous control with deep reinforcement learning
> Use the deterministic policy instead of stochastic policy to deal with the continuous action space. The architecture of framework is Actor-Critic. 

## ==> Transfer Learning/Meta-Learning
- (*****) Model-Agnostic Meta-Learning
> 最常用的一种meta-learning方法之一。
- A spect-augmented Adversarial Networks for Domain Adaptation
- (*****) Natural Language to Structured Query Generation via Meta-Learning
> 很好的将seq2seq的task定义成了meta-learning task.
- (*****) Learning a Prior over Intent via Meta-Inverse Reinforcement Learning 
> 当expert demonstration比较少的时候很难求reward function。考虑将其它tasks里面的信息作为prior引入进来。

## ==> Multi-Agent Learning
###  ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ` Yao Wan, updated on Sep. 13, 2018`.
- Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
- Multi-agent cooperation and the emergence of (natural) language
- (*****)Counterfactual Multi-Agent Policy Gradients (AAAI best paper)
> 解决multi-agent中一个比较重要的credit assignment issue。
==========

# (Application)
## ==> (Textual) Dialog Generation
- A Hierarchical Latent Structure for Variational Conversation Modeling
- A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues
- (*****) Improving Variational Encoder-Decoders in Dialogue Generation
> 以上三篇都是利用Variational Encoder-Decoders进行open domain中的dialogue generation
- (*****) DialogWAE- Multimodal Response Generation with Conditional Wasserstein Auto-Encoder
> Dialog generation, VAE+GAN, 在latent variable加入了mixture Gaussion。
- Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders

## ==> VQA (Visual Question Answering)
- (**) Visual Question Answering- A Survey of Methods and Datasets
> survey，快速了解VQA。
- (****) Visual Question Answering as a Meta Learning Task
> 首次将VQA这个task定义成了一个Meta Learning Task。
- Cross-Dataset Adaptation for Visual Question Answering
- Joint Image Captioning and Question Answering
- Learning Answer Embeddings for Visual Question Answering
- Question Answering through Transfer Learning from Large Fine-grained Supervision Data


## ==> Visual Dialog
###  ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ` Yao Wan, updated on Sep. 13, 2018`.
- (**) Visual Dialog
> 快速了解VisDialog这个task。
- (****) Are You Talking to Me-Reasoned Visual Dialog Generation through Adversarial Learning
> 用GAN做visDiag
- (****)Two can play this Game-Visual Dialog with Discriminative Question Generation
- Zero-Shot Dialog Generation with Cross-Domain Latent Actions
- Adversarial Learning of Task-Oriented Neural Dialog Models
- (*****)Best of Both Worlds-Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model
- (*****)Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning
- (****)Mind Your Language-Learning Visually Grounded Dialog in a Multi-Agent Setting
> 本人比较看好
> 1. Visual dialog + multi-agent. dialog本质上存在两个agent（qbot and abot），这很适合multi-agent learning.
> 2. visual dialog + meta-learning/transfer learning. visual dialog基于一个dataset进行训练，但在test data上可能会出现一个全新的图片或者句子，类似于zero-shot learning中出现一个新的class。meta-learing在vqa中已经有人做过了。

## ==> Embodied Question Answering
- (*****)Embodied Question Answering (https://embodiedqa.org/)
> 一个全新的QA场景，有公开数据集和代码。VQA＝Visual understanding + language understanding + answer generation. Embodied Question Answering = Visual understanding + language understanding + navigation + answer generation。

## ==> Video Question Answering
###  ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ` Yao Wan, updated on Sep. 14, 2018`.
- Open-Ended Long-form Video Question Answering via Adaptive Hierarchical Reinforced Networks.
- Multi-Turn Video Question Answering via Multi-Stream Hierarchical Attention Context Network.
> visual做的人很多，video做的人相对少一点。video和visual的区别就在于video是个temporal-spatial的，主要是encoder不同。这两篇论文我可以要到数据集。
- TVQA-Localized, Compositional Video Question Answering

## ==> Conversational Information Retrieval
###  ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ` Yao Wan, updated on Sep. 13, 2018`.
> 个人觉得dialog+information retrieval是一个很重要的应用场景，打破了传统keyword作为query的IR方式，这个在IR领域的影响力应该会很大。近两年sigir也有相关workshop出现，很适合www/sigir/kdd这类会议。（https://sites.google.com/view/cair-ws/cair-2018）
- Dialog-based Interactive Image Retrieval （NIPS2018）
> 作者十月会公布代码／数据集
- Improving Search through A3C Reinforcement Learning based Conversational Agent
