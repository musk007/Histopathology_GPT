# Histopathology_GPT
A histopathology report generator that takes an image as input and generates a report based on the observed resulta

## Abstract
The emergence of deep learning networks, commonly referred to as foundation models, trained on extensive datasets, has ushered in a new era for the integration of AI in the field of medicine. These models have demonstrated
remarkable capabilities in addressing various downstream tasks, often yielding results comparable to or even surpassing state-of-the-art methods in several domains. This paper introduces a foundation model designed specifically for
generating reports based on histopathology images, widely recognized as among the most challenging medical images. The presented model excels in producing reports that illustrate observations derived from input images in a format
comprehensible to the medical professionals utilizing it

![plot](https://github.com/musk007/Histopathology_GPT/blob/main/arch.png)

## Environment
Create an environment then install the dependencies

```sh
conda create --name hgpt python==3.9.18
conda activate hgpt
pip install -r requirements.txt
```

## Data
The data structure is similar to that followed by https://github.com/Vision-CAIR/MiniGPT-4 for finetuning their model. You can download the data from https://github.com/UCSD-AI4H/PathVQA.
QA-pairs were preprocessed and you can find the train and test annotation at: https://mbzuaiac-my.sharepoint.com/:f:/g/personal/roba_majzoub_mbzuai_ac_ae/EhpnjnNGfTBBrGzzaSAtVp4BasrFFk0JJ3BItg2Seg1eHQ?e=25oQFY

## Weights
Model weights can be found at https://mbzuaiac-my.sharepoint.com/:u:/g/personal/roba_majzoub_mbzuai_ac_ae/ETIGuq5rJ35LvqPMfxGpe8gBGeRjHUpch-x_qfULONnygQ?e=OdXTOa, which contains checkpoints for the original model as well as the LLM 
model weights and the final checkpoint of the trained model on the PathVQA dataset.

## To evaluate the model on a specific dataset, set eval=True in the minigpt4_stage2_finetune.yaml file and follow the instructions at https://github.com/Vision-CAIR/MiniGPT-4 to load the checkpoints. Afterwards
run the code 
```sh
torchrun train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```
  


