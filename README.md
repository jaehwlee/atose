# JETATAG : Joint Embeddings Training Algorithm for Auto-tagging

This is the repository for the method presented in the paper: "Music Auto-tagging with Joint Embeddings" by J. Lee and M. Cho. It can learn not only acoustic features of music but also semantic information of tags without additional data. As a result, our model achevies state of the art in MTAT datasets. Other tasks (e.g. Acoustic scene classification, Keyword spotting) are also going to be added soon. [(pdf)](https://github.com/jaehwlee/jetatag/blob/main/assets/paper.pdf)

</br>

![image](https://github.com/jaehwlee/jetatag/blob/main/assets/model_architecture.png)
* **Tag Autoencoder** : Module for extracting tag domain features from tags
* **Feature Extractor** : Module for extracting music domain features from source data. Our joint embedding technique utilizes feature extractors used in conventional tagging models as a general approach applicable to other models that already exist. For more readable feature extractor, please check [this repository](https://github.com/jaehwlee/music-auto-tagging-models)
* **Projection Head** : Module for mapping features of a music domain to embedded vectors projected into the tag domain. 
* **Classifier** : Module for classifying features in the extracted music domain into tags using a pre-trained feature extractor in stage 1.



Usage
--

**Preparing Dataset**

* MTAT : [Check this page](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
* DCASE2017-task4 : Upcoming dataset
* Speech Command : Upcoming dataset


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
cd preprocessing
python -u preprocess.py run ../dataset
python -u split.py run ../dataset
</code>
</pre>

**Training**

<pre>
<code>
cd training
python main.py
</code>
</pre>

Options
<pre>
<code>
'--gpu', type=str, default='0'
'--encoder_type', type=str, default='HC', choices=['HC', 'MS', 'SC']
'--block', type=str, default='rese', choices=['basic', 'se', 'res', 'rese']
'--withJE', type=bool, default=True
</code>
</pre>

Results
--

**Performance**

We confirmed that the performance of withJE has improved compared to the previous one. In particular, even in the current SOTA model(Harmonic-CNN), ROC-AUC and PR-AUC showed performance gains of 1.5% and 4.5%. Below is a summary.

<table>
  <tr>
    <td rowspan="2">Feature Extractor</td>
    <td colspan="2"><center>noJE</center></td>
    <td colspan="2"><center>withJE(ours)</center></td>
  </tr>
  <tr>
    <td>ROC-AUC</td>
    <td>PR-AUC</td>
    <td>ROC-AUC</td>
    <td>PR-AUC</td>
  </tr>
  <tr>
    <td>Harmonic-CNN</td>
    <td>0.9034</td>
    <td>0.5137</td>
    <td>0.9185</td>
    <td>0.5595</td>
  </tr>
  <tr>
    <td>Music-SincNet</td>
    <td>0.8616</td>
    <td>0.3937</td>
    <td>0.8823</td>
    <td>0.4551</td>
  </tr>
  <tr>
    <td>Sample-CNN</td>
    <td>0.8744</td>
    <td>0.4214</td>
    <td>0.8933</td>
    <td>0.4853</td>
  </tr>
  <tr>
    <td>+SE</td>
    <td>0.8809</td>
    <td>0.4456</td>
    <td>0.8840</td>
    <td>0.4581</td>
  </tr>
  <tr>
    <td>+Res</td>
    <td>0.8758</td>
    <td>0.4271</td>
    <td>0.8859</td>
    <td>0.4915</td>
  </tr>
  <tr>
    <td>+Res+SE</td>
    <td>0.8775</td>
    <td>0.4358</td>
    <td>0.8951</td>
    <td>0.4903</td>
  </tr>
</table>

</br>

**Analysis**

Pre-trained feature extractor with JE learned semantic information of tags.

![image](https://github.com/jaehwlee/jetatag/blob/main/assets/analysis_results.png)
