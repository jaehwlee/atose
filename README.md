# JETATAG : Joint Embeddings Training Algorithm for Auto-tagging

This is the repository for the method presented in the paper: "Audio Auto-tagging with Joint Embeddings" by J. Lee, J. Kim and M. Cho. It can learn not only acoustic features of audio but also semantic information of tags without additional data. As a result, our model achevies state of the art in MTAT, DCASE2017-task4, Speech Commands datasets.

</br>

![image](https://github.com/jaehwlee/jetatag/blob/main/assets/model_architecture.png)
* **Tag Autoencoder** : Module for extracting tag domain features from tags
* **Feature Extractor** : Module for extracting audio domain features from source data. Our joint embedding technique utilizes feature extractors used in conventional tagging models as a general approach applicable to other models that already exist. For more readable feature extractor, please check [this repository](https://github.com/jaehwlee/music-auto-tagging-models)
* **Projection Head** : Module for mapping features of a music domain to embedded vectors projected into the tag domain. 
* **Classifier** : Module for classifying features in the extracted music domain into tags using a pre-trained feature extractor in stage 1.

</br>

Usage
--

**Preparing Dataset**

* MTAT : [link](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
* DCASE2017-task4 : [link](https://dcase.community/challenge2017/download)
* Speech Command : [link](https://www.tensorflow.org/datasets/catalog/speech_commands)


**Installation**

<pre>
<code>
conda env create -n $ENV_NAME -- file environment.yaml
conda activate $ENV_NAME
</code>
</pre>

**Preprocessing**

<pre>
<code>
cd preprocessing/$DATASET
python -u preprocess.py run $DATASET_PATH
python -u split.py run $DATASET_PATH
</code>
</pre>

**Training**

<pre>
<code>
cd training
python main.py
</code>
</pre>

**Options**
<pre>
<code>
# If you want to use the hyperparameter in paper, refer to the contents of 'train_model.sh'
'--gpu'            # GPU to be used
'--data_path'      # Path of datasets 
'--dataset'        # Types of datasets to learn, choose among 'mtat', 'dcase', and 'keyword'
'--batch_size'     # batch size
'--isTest'         # Check if the model is working
'--encoder_type    # Types of feature extractor, choose among 'HC'(HarmonicCNN), 'MS'(TagSincNet), and 'SC' (SampleCNN)
'--block'          # Block types of SampleCNN, choose among 'basic', 'se', 'res', and 'rese'
'--latent'         # Dimensions of latent vectors to be joint embedded
'--withJE'         # Options for deciding to apply joint embedding
</code>
</pre>

</br>



## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation. 

</br>
  
## Author
  
* Jaehwan Lee [@jaehwlee](https://github.com/jaehwlee)
* Contacts: jaehwlee@gmail.com
