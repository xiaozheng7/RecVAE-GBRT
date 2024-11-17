# RecVAE-GBRT
![License](https://img.shields.io/badge/license-Apache-green)![Python](https://img.shields.io/badge/-Python-blue)![PyTorch](https://img.shields.io/badge/-PyTorch-red)

Implementation of RecVAE-GBRT: Memory-Fused XGBoost for Time-Series Forecasting. This work was published in the 2024 International Joint Conference on Neural Networks (IJCNN).

RecVAE-GBRT is a hybrid method for time series forecasting. In this method, we design a memory mechanism for GBRT, namely, Recursive Variational AutoEncoder (RecVAE), which can generate compressed representations of historical sequences by recursively summarizing a section of input time series and preceding internal outputs into current internal outputs. This compensates for the limitation of the GBRT in incorporating long historical sequences for time series forecasting. 

<p align="center">
<img src=".\image\RecVAE structure..svg" height = "600" alt="" align=center />
<br><br>
<b>Figure 1.</b> RecVAE structure.
</p>


<p align="center">
<img src=".\image\Development of the downstream XGBoost of RecVAE-GBRT..svg" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Development of the downstream XGBoost of RecVAE-GBRT.
</p>



## <span id="resultslink">Results</span>
<p align="center">
<img src=".\image\results.png" height = "420" alt="" align=center />
<br><br>
<b>Figure 3.</b> Accuracy comparison.
</p>



## <span id="citelink">Cite this work</span>
@inproceedings{zheng2024recvae,
  title={RecVAE-GBRT: Memory-Fused XGBoost for Time-Series Forecasting},
  author={Zheng, Xiao and Bagloee, Saeed Asadi and Sarvi, Majid},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}
