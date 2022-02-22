# MARRO
MARRO: Multi-headed Attention for Rhetorical Role Classification in Legal Documents

This repository contains the code for MARRO : a multi headed attention based model for rhetoric role classification. MARRO has two variants, where we can use either pretraine dmebddings from the legal documents OR use LEGAL-BERT-SMALL model to generate embeddings form the sentences. The model LEGAL-BERT-SMALL can be swapped for other BERT based models

Every sentence in a court case document can be assigned a rhetorical (semantic) role, such as 'Arguments', 'Facts', 'Ruling by Present Court', etc. The task of assigning rhetorical roles to individual sentences in a document is known as semantic segmentation. We have developed MARRO for automatic segmentation of Indian court case documents. A single document is represented as a sequence of sentences. We have used 7 labels for this task: Arguments, Precedent, Statutes, Facts, Ratio Decidendi, Ruling of Lower Court, Ruling of Present Court.

## TRAINING
For training a model on an annotated dataset

Input Data format
For training and validation, data is placed inside "data/text" folder. Each document is represented as an individual text file, with one sentence per line. The format is:

```
  text <TAB> label
 ```

If you wish to use pretrained embeddings variant of the model, data is placed inside "data/pretrained_embeddings" folder. Each document is represented as an individual text file, with one sentence per line. The format is:

```
emb_f1 <SPACE> emb_f2 <SPACE> ... <SPACE> emb_f200 <TAB> label  (For 200 dimensional sentence embeddings)
```
"in_categories.txt"  and "uk_categories.txt" contains the category information of documents in the format:

```
category_name <TAB> doc <SPACE> doc <SPACE> ...
```

### Usage
To run experiments with default setup, use: 
  ```
  python run.py --data_path data/text/IN-train-set/ --use_marro True                                                              (no pretrained variant)
  python run.py --pretrained True --data_path data/pretrained_embeddings/  --use_marro True   (pretrained variant)
  ```
  
Other default values and hyperparamters are given in run.py

By default, the model employs 5 fold cross-validation, where folds are manually constructed to have balanced category distribution across each fold.


