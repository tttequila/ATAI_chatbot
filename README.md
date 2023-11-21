# ATAI_chatbot
<span style="color:gray"> 23HS ATAI Proj </span> [Github](https://github.com/tttequila/ATAI_chatbot)


❗VERY IMPORTANT: to fully operate the agent or test it, there are still some steps that need to be followed to download the data needed. 

-----

## 🕑Updates

### 11.21 Evaluation3 Update
Re-structurize the whole demo, now we have separate modules to handle different functions: 

重新对代码进行了结构化，将这几个功能分别装在不同的模组里，希望能让代码的结构和逻辑更清晰

- closed question/open question
- recommendation
- crowdsource

**Common Modules**

`LM.py`: module for language process. Contains various language models.

语言模型模组，里面装了各种可能用到的语言模型

`KG_handler.py`: module for knowledge graph. Store all member variables relevant to the KG. Each variable relevant to the KG (e.g. rel2lbl, ent2lbl, rel_emb, etc.) should be grabbed from this module and from the class with the same name as the module file (e.g. KG_handler)

知识图谱模组，负责初始化各种知识图谱相关的变量。也提供了各种判断实体或者关系的函数。

`demo.py`: module for the main body of our bot. The main function is to handle various modules and assemble them as a real bot. 

bot的主体，负责整合各个模块，负责将问题分类，并发布给各个功能模组进行处理

**Functional Modules**

`Recommendor.py`: module specially designed for handling recommendation questions. The public API is `recommend(sentence, mode)`. The mode argument is supposed to be within *["feature", "movie"]*. To be more specific, *"feature"* means return common features of given movies. While *"movie"* mean recommending a movie directly based on graph embedding

负责处理推荐问题的模组

`Que.py`: module for natural language query. **Haven't finished yet**

负责处理自然语言query的模组

`CrowdSource.py`: module for crowdsource task. **Haven't finished yet**

负责处理众包任务的模组

-----
## Set up

##### step 1. Download packages needed

Please instead listed packages in `requirement.txt`. Notice that due to some technical issues, `pip install requirement.txt` may not work properly. If so, please manually install them.

##### step 2. Download core package

Depends on whether to utilize GPUs or not, you may need to install different versions.
Please follow the instruction in offical website of [spacy](https://spacy.io/usage) **and** following commend for models needed: 
```commend
pip install spacy-transformers
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

##### step 3. Install pretrained language model
Please install fasttext module following [link](https://github.com/facebookresearch/fastText)
For the pretrained model, download [weight](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)  and put the the decompressed file `cc.en.300.bin` in the given folder `utils`.

##### step 4. Download other data
Download other data may needed from [link](https://drive.google.com/file/d/1a6re-lhl6B9ebVBfsihmF65Wma8gCssk/view). 
And also decompress it into given folder `utils` 





