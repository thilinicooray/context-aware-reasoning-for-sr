# Attention-based Context Aware Reasoning for Situation Recognition

## Preparing the environment

1. Our implementation is in PyTorch running on GPUs. Use the provided [car4sr.yml](env/car4sr.yml) to create a virtual environment using Anaconda.
2. Download imSitu image set from http://imsitu.org (we use the resized image set)
3. Annotations updated with top 2000 nouns are in [imSitu](imSitu) folder.

## Implementation Details

This repository contains implementations for all methods we have used in our paper. We explain below what each file is responsible for

* main_ggnn_baseline.py - our implementation of https://arxiv.org/abs/1708.04320
* main_revggverb_caqrole_eval.py - reasoning enhanced VGG based verb model joint with CAQ role model for entire situation prediction.
* main_top_down_baseline.py - TDA model
* main_top_down_baseline_addemb.py - modified TDA model used for agent and place predictions for verb model
* main_top_down_image_context.py - CAI model
* main_top_down_img_recons.py - CAIR model
* main_top_down_query_context.py - CAQ model
* main_top_down_verb.py - TDA model for verb predictions
* main_vgg_verb_classifier.py - VGG verb classifier
* main_vggverb_caqrole_joint_eval.py - VGG verb model joint with CAQ role model for entire situation prediction.
* main_vggverb_ggnnrole_joint_eval.py - VGG verb model joint with GGNN role model for entire situation prediction.
* main_vggverb_tdarole_joint_eval.py - VGG verb model joint with TDA role model for entire situation prediction.

All required arguments to be passed to each file are provided in each of their argument list respectively.

For any enquiry, please contact me via <a href="mailto:thilini_cooray@mymail.sutd.edu.sg">thilini_cooray@mymail.sutd.edu.sg</a> or <a href="mailto:thilinicooray.ucsc@gmail.com">thilinicooray.ucsc@gmail.com</a>

