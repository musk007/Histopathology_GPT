This Code is based on the code of MiniGPt-4 https://github.com/Vision-CAIR/MiniGPT-4
  <h3 align="center">Histopathology-GPT</h3>

  <p align="center">
    A histopathology report generator that takes an image as input and generates a report based on the observed results

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About Histopathology-GPT

The emergence of deep learning networks, commonly referred to as foundation models, trained on extensive datasets, has ushered in a new era for the integration of AI in the field of medicine. These models have demonstrated remarkable capabilities in addressing various downstream tasks, often yielding results comparable to or even surpassing state-of-the-art methods in several domains. This paper introduces a foundation model designed specifically for generating reports based on histopathology images, widely recognized as among the most challenging medical images. The presented model excels in producing reports that illustrate observations derived from input images in a format comprehensible to the medical professionals utilizing it.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

![plot](https://github.com/musk007/Histopathology_GPT/blob/main/arch.png)






<!-- GETTING STARTED -->
## Getting Started

1. Creating environment
  ```sh
  conda create --name H_gpt python==3.9.18
  conda activate hgpt
  pip install -r requirements. txt 
  ```
2. Download Dataset
  You can download the data from https://github.com/UCSD-AI4H/PathVQA
  The preprocessed annotations are avilable at : https://mbzuaiac-my.sharepoint.com/:f:/g/personal/roba_majzoub_mbzuai_ac_ae/EhpnjnNGfTBBrGzzaSAtVp4BasrFFk0JJ3BItg2Seg1eHQ?e=3fqHGH
  Following the data structure by https://github.com/Vision-CAIR/MiniGPT-4, images must be in the same folder with a json file containing the image_ids and annotations for each image.

4. Pre-trained weights
   Pretrained weights can be found at:
   
   Follow the same structure for weight loading as that on https://github.com/Vision-CAIR/MiniGPT-4

5. For evaluation:
   Set evaluate=True in  minigpt4_stage2_finetune.yaml as well as the test split
   Then run the following line in the terminal:
   ```sh
    torchrun train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml 
    ```
6. For training set evaluation=False in the same file and run the same command again
