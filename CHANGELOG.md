## 2.1.0

* Further improvements to pre-processing of recipes to allow tokens ending with "%", remove more URLS, use a customized stop word list to avoid removing semantically usable tokens.
* Change the minimum vocab count to 15.

## 2.0.0

* Change embeddings model to GloVe from floret.

* Denoise embeddings by removing top principal components.

* Exclude recipes from cookbooks.com because the recipe quality seems extremely variable and includes entries that aren't real recipes.

* Improve pre-processing of recipes to remove quotes from words, remove symbols, remove html tags and remove URLs.

## 1.0.0

* Initial release