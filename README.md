# OPTIC

The KnOwledge graPh-augmented enTity lInking approaCh (OPTIC) is an approach for the disambiguation step of the Entity Linking (EL) task, based on deep neural network and knowledge and word embedding. Different from other approaches, we train the knowledge and word embedding simultaneously by using the model [fastText](https://github.com/facebookresearch/fastText) [2-3].

## Technical requirements

The approach was developed in python 3.7, working fine on python 3.6. The code depends on the following packages:
* pynif 0.1.4
* flask 1.1.1
* bcolz 1.2.1
* numpy 1.17.2
* torchvision 0.4.0
* pytorch 1.2.0
* fuzzysearch 0.6.2
* nltk 3.4.5
* elasticsearch 7.0.4

Moreover, OPTIC uses the version 0.3.2 of the [Tweet NLP](http://www.cs.cmu.edu/~ark/TweetNLP/) tool [4-5]. Besides the Tweet NLP, all remaining packages can be obtained via pip. 

## Data Generation

This repository contains a script that transforms the resulting file of the fastText model, i.e., the embedding file on .vec format, to the format usable by OPTIC. 
Foremost, we consider that the fastText model has already trained the knowledge and word embeddings. In [1], we detail the process to train them jointly. 

Before the usage of the script, it is necessary to change the value of the variables EMBEDDING_DIM (line 8) and data_path (line 9) according to your needs. EMBEDDING_DIM refers to the dimension size of the embeddings, while data_path refers to the embeddings file path. 

To use the script, execute the command:

```
python3.7 prepare_embedding_matrix.py
```

### Build entity candidates index

The entity candidates employed on OPTIC is the same index employed on [AGDISTIS/MAG](https://github.com/DiegoMoussallem/AGDISTIS) [6]. Please consult their [wiki](https://github.com/dice-group/AGDISTIS/wiki/3-Running-the-webservice) to learn how to recreate the index. Moreover, we provide our mapping for the ElasticSearch to help the replication.

## Neural Network model training

The model folder presents that code to train our Neural Network model based on knowledge and word embedding trained jointly. To train the model, execute the command:

```
python3.7 train.py
```

Moreover, it is necessary to declare the following arguments:

	--batch 				Size of batch
	--output_dim 				Dimension of the output layer (always 1 for our model)
	--embedding 				Dimension of the embeddings
	--dropout 				Dropout value
	--epoch					Number of epochs that the training will run
	--hidden 				Number of cells for hidden layer(s)
	--layer 				Number of hidden layers
	--datapath 				Data files path
	--dataset 				Dataset name
	--ws 					Size of the window context
	--rank 					Flag to consider or not the popularity (Not implemented yet, always consider the popularity)

Lastly, in our case, we use the following pattern for, respectively, our train and test set: train_DATASET.json and test_DATASET.json. The DATASET refers to the dataset name specified by the argument --dataset.

## OPTIC usage

OPTIC can run both locally or as a Web Service. Example of the command to run locally:

```
python3.7 run.py
```

Example of the command to run as Web Service:
```
export FLASK_APP=run.py
flask run
```

OPTIC requires several arguments, detailed as following:
	
	General type arguments:
		--mode 				Type of experiment that OPTIC will execute (a2kb,d2kb). Currently only supporting d2kb
		--input 			Folder path with NIF files to be disambiguated
		--data 				Other data files (like model) path
		--verbose 			show progress messages (yes, no)

	Neural Network type arguments:
		--embed				Dimension of the embeddings
		--hidden 			Number of cells for hidden layer(s)
		--layer 			Number of hidden layers
		--dropout 			Dropout value
		--batch 			Size of batch
		--extra 			Flag for extra attributes in the Neural Network model (0 = None, 1 = popularity)

	ElasticSearch type arguments:
		--type 				Type of elasticsearch query
		--max 				Max of documents returned by elasticsearch in queries
		--boost 			Boost for exact match in multi-match queries

	Disambiguation specific arguments:
		--threshold 		Threshold to consider the miminum score of the disambiguation step
		--ws 				Size of the window context

These arguments can also be declared on the configuration file. An example of such a file is present on this repository.

### Input and Output

The input of OPTIC is a microblog text, with the named entity mentions already recognized. The input must follow the NIF standard. 
The output is the microblog text with the named entity mentions disambiguated. The output is a file following the NIF standard.

When running locally, OPTIC will disambiguate all NIF files contained in the specified folder.
Running as Web Service, it is necessary to pass each NIF file individually.  

## Developer notes

To submit bugs and feature requests, please report at [project issues](https://github.com/ItaloLopes/optic/issues).

## Bibliography

[1] Italo Lopes Oliveira, Luís Paulo Faina Garcia, Diego Moussallem and Renato Fileto. (2020). OPTIC: KnOwledge graPh-augmented enTity lInking approaCh. To be published.

[2] Piotr Bojanowski, Edouard Grave, Armand Joulin and Tomas Mikolov. (2017). Transactions of the Association for Computational Linguistics. (5)135-146.

[3] Armand Joulin, Edouard Grave, Piotr Bojanowski, Maximilian Nickel and Tomas Mikolov. (2017). Fast Linear Model for Knowledge Graph Embeddings.

[4] Kevin Gimpel, Nathan Schneider, Brendan O'Connor, Dipanjan Das, Daniel Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey Flanigan and Noah A. Smith.  Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments. (2011). Proceedings of ACL. 

[5] Olutobi Owoputi, Brendan O'Connor, Chris Dyer, Kevin Gimpel, Nathan Schneider and Noah A. Smith. (2013). Improved Part-of-Speech Tagging for Online Conversational Text with Word Clusters. Proceedings of NAACL.   

[6] Diego Moussallem, Ricardo Usbeck, Michael Röeder, Axel-Cyrille Ngonga Ngomo. (2017). MAG: A multilingual, knowledge-base agnostic and deterministic Entity Linking approach. Proceedings of the Knowledge Capture Conference.
