
## Train custom source generative model 

### Steps

* Dowload and save dataset in data/ folder
* Train source generative model 

`python3 main.py configs/src_chestxray.json`

* Create target model from borrowed weights from generative model's encoder part 
* Finetune target model with DeepCovid dataset (train split). 

`python3 main.py configs/target_deepcovid_tune_all.json`

* Compute performance metrics Sensitivity, Specificity, F1 score.

`python3 main.py configs/target_deepcovid_test_all.json`



### Creating random splits of k-shots of target dataset

`python3 -m utils.random_training_splits` 