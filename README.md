# Recipe Ingredient Embeddings

A word embeddings model trained on recipe ingredients.

This repository contains the code used to train an embeddings model specifically for use with recipe ingredients. The [floret](https://github.com/explosion/floret) library is used to create the embeddings using the recipes in the [RecipeNLG](https://www.kaggle.com/datasets/saldenisov/recipenlg/data) corpus.

The code here should be readily adaptable to using other libraries to create the embeddings model, such as FastText or Word2Vec.

### How do download the model?

The pretrained model can be found on the [releases](https://github.com/strangetom/ingredient-embeddings/releases) page.

### How do I use the model?

```python
import floret

# Load model
model = floret.load_model("ingredient-embeddings.300d.floret.bin")
# Get vector for word
vec = model["apple"]
```

The floret library is based on FastText, so the [FastText documentation](https://fasttext.cc/) applies.

There are a couple of limitations that apply due to how the model has been trained.

* The model was trained on lowercase stems of words (e.g. using NLTK's `PorterStemmer`).
* The model was trained on nouns, verbs and adjectives only.
* Numbers, punctuation, white space, stop words and single character words were removed prior to training.

### How was the model trained?

The model training process is based off the process described in [^1] and [^2].

The [RecipeNLG](https://www.kaggle.com/datasets/saldenisov/recipenlg/data) corpus of recipes was used as it provides 2.23 million recipes for a variety of sources [^3]. Each recipe in the corpus was prepared as follows:

* The recipe ingredients and instructions were appended together to create a single string.
* That string was split into tokens to separate each word by white space and punctuation.
* For each token, the lowercase stem was kept if the following conditions were met:
  * The part of speech was a noun, verb or adjective[^4],
  * The token was not punctuation, numeric, white space,
  * The token was not a stop word,
  * The length of the token was greater than 1
* These pre-processed recipes were written to a text file, with one recipe per line.
  * These pre-processing steps result in a corpus of 70,000 words.

* The text file was passed to `floret.train_unsupervised()` to initiate the training.
  * The selected hyper-parameters for the model are based of those selected in [^1].


> [!NOTE]
>
> Both [^1] and [^2] refer to using word lemmas instead of stems. however the examples in the papers show stemmed words. Therefore stemming has been used here instead of lemmatization.

### How do I train the model myself?

```bash
$ git clone https://github.com/strangetom/ingredient-embeddings.git
$ cd ingredient-embeddings
$ python -m venv venv
$ source venv/bin/activate
$ python -m pip install -r requirements.txt
$ python main.py
```

### References

[^1]: A. Morales-Garzón, J. Gómez-Romero, and M. J. Martin-Bautista, ‘A Word Embedding Model for Mapping Food Composition Databases Using Fuzzy Logic’, in *Information Processing and Management of Uncertainty in Knowledge-Based Systems*, vol. 1238, M.-J. Lesot, S. Vieira, M. Z. Reformat, J. P. Carvalho, A. Wilbik, B. Bouchon-Meunier, and R. R. Yager, Eds., in Communications in Computer and Information Science, vol. 1238. , Cham: Springer International Publishing, 2020, pp. 635–647. doi: [10.1007/978-3-030-50143-3_50](https://doi.org/10.1007/978-3-030-50143-3_50).

[^2]: A. Morales-Garzon, J. Gomez-Romero, and M. J. Martin-Bautista, ‘A Word Embedding-Based Method for Unsupervised Adaptation of Cooking Recipes’, *IEEE Access*, vol. 9, pp. 27389–27404, 2021, doi: [10.1109/ACCESS.2021.3058559](https://doi.org/10.1109/ACCESS.2021.3058559).

[^3]: M. Bień, M. Gilski, M. Maciejewska, W. Taisner, D. Wisniewski, and A. Lawrynowicz, ‘RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation’, in *Proceedings of the 13th International Conference on Natural Language Generation*, Dublin, Ireland: Association for Computational Linguistics, 2020, pp. 22–28. doi: [10.18653/v1/2020.inlg-1.4](https://doi.org/10.18653/v1/2020.inlg-1.4).

[^4]: G. Ispirova, G. Popovski, E. Valenčič, N. Hadzi-Kotarova, T. Eftimov, and B. K. Seljak, ‘Food Data Normalization Using Lexical and Semantic Similarities Heuristics’, in *Biomedical Engineering Systems and Technologies*, vol. 1400, X. Ye, F. Soares, E. De Maria, P. Gómez Vilda, F. Cabitza, A. Fred, and H. Gamboa, Eds., in Communications in Computer and Information Science, vol. 1400. , Cham: Springer International Publishing, 2021, pp. 468–485. doi: [10.1007/978-3-030-72379-8_23](https://doi.org/10.1007/978-3-030-72379-8_23).
