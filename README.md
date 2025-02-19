<h1 align="center">
  <img src="pic/logo.png" alt="Logo" width="50" height="50" style="vertical-align: middle;">
  MMKcat
</h1>
<h2 align="center">A Multimodal Deep Learning Framework for Enzyme Turnover Prediction with Missing Modality</h2>

![The Overall Architecture of MMKcat.](pic/model.png)

Accurate prediction of the turnover number ($k_{\rm cat}$), which quantifies the maximum rate of substrate conversion at an enzyme's active site, is essential for assessing catalytic efficiency and understanding biochemical reaction mechanisms. Traditional wet-lab measurements of $k_{\rm cat}$ are time-consuming and resource-intensive, making deep learning (DL) methods an appealing alternative. However, existing DL models often overlook the impact of reaction products on $k_{\rm cat}$ due to feedback inhibition, resulting in suboptimal performance. The multimodal nature of this $k_{\rm cat}$ prediction task, involving enzymes, substrates, and products as inputs, presents additional challenges when certain modalities are unavailable during inference due to incomplete data or experimental constraints, leading to the inapplicability of existing DL models. To address these limitations, we introduce **MMKcat**, a novel framework employing a prior-knowledge-guided missing modality training mechanism, which treats substrates and enzyme sequences as essential inputs while considering other modalities as maskable terms. Moreover, an innovative auxiliary regularizer is incorporated to encourage the learning of informative features from various modal combinations, enabling robust predictions even with incomplete multimodal inputs. We demonstrate the superior performance of MMKcat compared to state-of-the-art methods, including DLKcat, TurNup, UniKP, EITLEM-Kinetic, DLTKcat and GELKcat, using BRENDA and SABIO-RK. Our results show significant improvements under both complete and missing modality scenarios in RMSE, $R^2$, and SRCC metrics, with average improvements of 6.41\%, 22.18\%, and 8.15\%, respectively.

<h2 id="overview"> ⚙️ Install Necessary Dependencies </h2>

- Firstly, please make sure that you have installed **ESM2** and **ESMFold** correctly with their corresponding pre-trained checkpoints. Concretely, you can follow the official instructions in this [repository](https://github.com/facebookresearch/esm) to prepare. Please note that we select **ESM2_t33_650M_UR50D** for **ESM2** in our experiments.
- After installing **EMS2** and **ESMFold** correctly, run the following command to complete this part:
```
pip install -r requirements.txt
```
- ⚠️ You may encounter the issue of the use of dssp for a error like *FileNotFoundError: \[Errno 2\] No such file or directory: 'mkdssp'*. If this doesn't happen, please ignore this. If so, to solve this problem, we use the following commands:
```
conda install -c ostrokach dssp
which mkdssp  # Here, we denote this path as 'dssp_path'
cd dssp_path
cp mkdssp dssp
```
Then we add this path to environmental variable **PATH** to make it work. It can also be added like the codes in model/test_examplt.py.

<h2 id="overview"> 🧪 Perform $k_{\rm cat}$ Prediction for Chemical Reactions</h2>

Use the code in model/test_example.py for $k_{\rm cat}$ prediction.
