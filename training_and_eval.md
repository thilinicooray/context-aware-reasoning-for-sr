# Training procedure for CAQ model for semantic role prediction

* Run main_top_down_baseline.py to train TDA model. Under our hyper-parameter setup we were able to obtain a converged model around 30 epochs
* Use the pretrained TDA model as an input and train the CAQ model using main_top_down_query_context.py (Maximum 60 epochs for a converged model)

# Training procedure for RE-VGG model for verb prediction

* Run main_vgg_verb_classifier.py to obtain the baseline VGG verb classifier
* Run main_top_down_baseline_addemb.py to otain the modified TDA model for agent and place only semantic role predictions
* Use the pretrained modified TDA model as input and train main_top_down_verb.py to obtain VQA based verb predictor.

All models converge around 30 epochs

# Inference

* Use all above pretrained models and execute main_revggverb_caqrole_eval.py to get evaluation or test set performance for the the RE-VGG + CAQ situation prediction model

## Special note about the provided conda environment

We have provided the exact environment with dependency versions we used for our experiments. Please consider updating the dependency versions in case they are not compatible with your setup