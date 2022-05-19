$ \ color {red} {### Since the e-katalog.ru has ceased to exist,  need to reorient the parser to other source} $
## PC assembly assistant bot

The goal of this project was to create a PC build tool, wich will help the user to choose pc-parts and notebooks with good quality and lower price.

Tool consist of five parts:
* Parser wich can download product characteristics and prices everyday.
* Preprocessing unit is converting raw data to datasets for the model
* Anomaly detector finds interesting products from the datasets
* NLP model based on BERT transformer as a sentiment analyser
* And finally telegram bot as a user interface

### Directory content

* AnomalyDetection.ipynb - notebook with a demonstration of the anomaly detector training
* AnomalyPCbot.py - bot based on aiogramm
* parser.py - code with data scraping, that also runs preprocess unit
* preprocess.ipynb - notebook with data preprocessing
* preprocess.py - script made from previous file to run from parser
* rating_predictor.py - here is stored model to run in bot
* RatingPrediction.ipynb - notebook with model training to recognize the user's attitude to the product
