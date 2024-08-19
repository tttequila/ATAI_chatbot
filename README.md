# ATAI_chatbot
<span style="color:gray"> 23HS ATAI Proj </span> [Github](https://github.com/tttequila/ATAI_chatbot)

[Final evaluation paper](https://github.com/tttequila/ATAI_chatbot/blob/main/ATAI_Final_Evaluation.pdf)

❗VERY IMPORTANT: to fully operate the agent or test it, there are still some steps that need to be followed to download the data needed. 

-----

## 🕑Updates

### 11.21 Evaluation3 Update
Re-structurize the whole demo, now we have separate modules to handle different functions: 

- closed question/open question
- recommendation
- crowdsource

### 12.11 Evaluation Final Update
Adding **multimedia** module and **Que** module

### Further Update (Probably not)
- Parsing restrictions in input.
- Improving humaness for responisng.
- Better relation extraction 
  - Dependency based
  - ...
- ...

---

**Common Modules**

`LM.py`: module for language process. Contains various language models. Basically for *NER*, *POS tagging* and *word vector* generation.


`KG_handler.py`: module for knowledge graph. Store all member variables relevant to the KG. Each variable relevant to the KG (e.g. rel2lbl, ent2lbl, rel_emb, etc.) should be grabbed from this module and from the class with the same name as the module file (e.g. KG_handler)


`demo.py`: module for the main body of our bot. The main function is to handle various modules and assemble them as a real bot. 


**Functional Modules**

`Recommendor.py`: module specially designed for handling recommendation questions. The public API is `recommend(sentence, mode)`. The mode argument is supposed to be within *["feature", "movie"]*. To be more specific, *"feature"* means return common features of given movies. While *"movie"* mean recommending a movie directly based on graph embedding


`Que.py`: handling natural language based 1. factual questions; 2. embedding question; 3. crowdsourcing question. The user input will be firstly queried by crowdsourcing, if failed then be queried by factual querier. And finally embedding querier if failed again.

`Multimedia.py` handling image query. Parsing sentence, extracting entity involveing and finally return the most matched images from [MovieNet](https://movienet.github.io/)

----

#### Necessary Installation

**fasttext**
Please install fasttext module following [link](https://github.com/facebookresearch/fastText)
For the pretrained model, download [weight](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)  and put the the decompressed file `cc.en.300.bin` in the given folder `utils`.

**speakeasy-client**

**requirement.txt**


****








