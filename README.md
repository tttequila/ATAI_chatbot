# ATAI_chatbot
<span style="color:gray"> 23HS ATAI Proj </span> [Github](https://github.com/tttequila/ATAI_chatbot)


❗VERY IMPORTANT: to fully operate the agent or test it, there are still some steps needed to be followed to download data needed. 

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

---
The final result of folder `utils` supposed to be like:

> utils
>  > 14_graph.nt
cc.en.300.bin
entity_ids.del
ent_lbl2vec.pkl
relation_embeds.npy
relation_ids.del
rel_lbl2vec.pkl




