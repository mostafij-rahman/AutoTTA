# AutoTTA: Automatic Test Time Augmentation

There are two components to the code:

1. **Controller:** a recurrent neural network that suggests transformations
2. **Child Network:** the baseline pretrained neural network.

How to run?
1. First create a conda environment and then activate the environment. Run pip install -r requirements.txt inside the invironment to install required libraries.
2. Data - Please download the data from this following link and put into the 'data' folder.
https://drive.google.com/drive/folders/1yvZK-DZBiFtWfnLLqPo8Gjt5a2hTkjO8?usp=sharing
3. Please download the pre-trained model from the following link and put into the 'child_models' folder. 
https://drive.google.com/drive/folders/1A7o8uWVUVWKFc2Kki8XTjAyf15_ACaP_?usp=sharing
4. Now type 'python run_autotta.py' to run the program.

