### Code for our paper on applying generative FSL for Covid  prediction. 

Please feel free to use for any humanatarian application. Please cite this repository if you decide to use. You can find the preprint of the paper on https://www.researchgate.net/publication/351021021_Generative_Transfer_Learning_from_Few_Shots_Attempting_Covid-19_Classification_with_few_Chest_X-ray_Images


## Train custom source generative model 

### Steps

* Dowload and save dataset from https://github.com/shervinmin/DeepCovid in data/ folder
* Train source generative model 

`python3 main.py configs/src_chestxray.json`

* Create target model from borrowed weights from generative model's encoder part 
* Finetune target model with DeepCovid dataset (train split). 

`python3 main.py configs/target_deepcovid_tune_all.json`

* Compute performance metrics Sensitivity, Specificity, F1 score.

`python3 main.py configs/target_deepcovid_test_all.json`



### Creating random splits of k-shots of target dataset

`python3 -m utils.random_training_splits` 
