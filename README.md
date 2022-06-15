# Summarization of Dutch Legal Cases
This repository contains code that was used for my thesis "On Automatic Summarization of Dutch Legal cases". For this thesis, I studied the feasibility of automatic summarization of Dutch Legal Cases.   

This repository has four main components:
1. Obtaining the Dutch Legal Cases dataset
2. Computing feature sets
3. Clustering of the dataset
4. Modelling the summarization system

## 1. Obtaining the Dutch Legal Cases dataset 
Digitized Dutch Legal cases are published by 'Raad van de Rechtspraak' using the European Case Law Identifier2 (ECLI) standard. The cases are publicly available and published under the 'Public Domain Mark 1.0' license[^1]. The cases are published as a collection of XML files, where each XML files desribes a single case.

In this project we describe four versions of this dataset:
1. The <strong>external dataset</strong>; which is the collection of XML files;
2. The <strong>raw dataset</strong>; obtained by parsing the XML files from the external dataset;
3. The <strong>interim dataset</strong>; obtained by removing non-viable cases from the raw dataset;
4. The <strong>processed dataset</strong>; these are the train/dev/test splits that are used as input to the 

Depending on your needs, you can follow below steps to obtain one or more of these datasets. For example, if you are interested in only obtaining a parsed copy of the complete collection of digitized Dutch legal cases, then the <strong>raw dataset</strong> is the dataset you need.

### 1.1 The external dataset
The external dataset can be downloaded from the website of 'Raad van de rechtspraak": https://www.rechtspraak.nl/Uitspraken/paginas/open-data.aspx

### 1.2 The raw dataset
To obtain the raw dataset, first the external dataset, which is downloaded as a zip file, should be extracted to ```/this_repo/data/open_data_uitspraken/external```. This folder should look like this:
```
1905
1911
1913
...
```
Each of these year folders contains 12 zip files; one for each month. These zip files contain the case XML files. Don't extract these zips; they can be kept as-is.

Now run ```/this_repo/src/data/rechtspraak_make_dataset.py``` to parse the external dataset and obtain the raw dataset consisting of four parquet files. This process can take up to 45 hours, as there are millions of cases to be parsed. The dataset is stored in ```/this_repo/data/open_data_uitspraken/raw```.

### 1.3 The interim dataset
The interim dataset only contains the cases that are viable; they should have both a case text and a case summary and this summary should have a length of at most 10 words. Run ```/this_repo/src/data/rechtspraak_make_interim_dataset.py``` to generate the interim dataset. It will be stored as 10 parquet files in ```/this_repo/data/open_data_uitspraken/interim```.

### 1.4 The processed dataset
<!-- zo maken dat de default manier niet stratified gebruikt, met de opmerking: 'om de thesis te reproducen, maak dan eerst de cluster modellen'-->
<strong>Important</strong>: This step can only be done after the [cluster model](#22-clustering-features) scripts have been run. In the thesis we studied what influence clustering has on subsequent summarization. Therefore, the processed dataset is generated in a stratified manner depending on the cluster each case belong to.

We chose to use the following distribution of cases: 70% train/ 20% dev / 10% test. Now, run ```/this_repo/src/data/generate_splits_stratified.py```. Two things have happened:
- The dataset splits for the full dataset were generated and stored in ```/this_repo/data/open_data_uitspraken/processed```
- For each of the clusters, dataset splits for that specific cluster are stored in ```/this_repo/data/open_data_uitspraken/processed/cluster_subsets```


[^1]: See https://data.overheid.nl/en/dataset/uitspraken-rechtspraak-nl

## 2. Computing feature sets

In the thesis, we used two extra sets of features to achieve two different goals:
1. Descriptive features; to explore the dataset
2. Clustering features; to cluster the datset and use this as input for the summarization models

### 2.1 Descriptive features
First, as part of an exploratory analysis of the dataset, a set of features by Bommasani and Cardie (xxxx), was computed. One of these features requires us to derive topics for the cases using LDA. Therefore, we first need to train the LDA model on the complete corpus. To do this, run ```/this_repo/src/models/create_lda_model.py```. 

Now that we have the LDA model, we can compute the set of descriptive features by running ```/this_repo/src/features/compute_bommasani_features.py```. This can take up to N hours. The results will be stored in ```/this_repo/reports/descriptive_features_full_1024.csv```.

### 2.2 Clustering features
The clustering features are used to test whether clustering as a prior step to automatic summarization can improve automatic summarization.  

To obtain the cluster features for all the dataset's cases, run ```/this_repo/src/features/compute_clustering_features.py```. The features will be stored in ```/this_repo/data/open_data_uitspraken/features/clustering_features_full_1024.csv``` and will be used as input to [the clustering component](#3clustering-the-dataset).
## 3. Clustering the dataset
Clustering simply uses k-means. To cluster the data, run ```/this_repo/src/features/learn_k_means.py```, the clustered data will be stored as a mapping of case ids and assigned clusters. This mapping is required when [generating the processed datasets](#14-the-processed-dataset) to obtain stratified splits. 

## 4. Modelling the summarization system

### 4.1 Training the BART tokenizer

### 4.2 Pretraining BART

### 4.3 Fine-tuning BART to obtain summarization models

## Notes 
- Scattered over the repository, you might find other scripts. Most of these are used to evaluate the generated summaries and create graphs of the results.
- In the thesis' experiment, for training of the BART models a TPU was used (for [free](https://sites.research.google/trc/about/), thanks Google!). This was not only necessary to speed up training, but it also enabled me to load the BART model in the first place; the GPU-server, that I also had access too, unfortunately could not work with the large model.    