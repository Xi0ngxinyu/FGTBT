# Frequency-Guided Task-Balancing Transformer for Unified Facial Landmark Detection
<p align="left">
  <a href="https://github.com/Xi0ngxinyu/FGTBT">
    <img
      src="https://img.shields.io/badge/FGTBT-Project%20Page-0A66C2?logo=safari&logoColor=white"
      alt="FGTBT Project Page"
    />
  </a>
  <a href="https://arxiv.org/abs/2601.12863">
    <img
      src="https://img.shields.io/badge/FGTBT-Paper-red?logo=arxiv&logoColor=red"
      alt="FGTBT Paper on arXiv"
    />
  </a>
</p>


![teaser](assets/model.png)




**Official Implementation of Information Sciences 2026 Accepted Paper** 
<!-- | [Project Page](https://github.com/CIawevy/FreeFine) | [arXiv Paper](https://arxiv.org/pdf/2507.23300) | [GeoBench Dataset](https://huggingface.co/datasets/CIawevy/GeoBench)  | [GeoBenchMeta Dataset](https://huggingface.co/datasets/CIawevy/GeoBenchMeta)   -->




## 🌟 Introduction  
>Recently, deep learning based facial landmark detection (FLD) methods have achieved considerable success. However, in challenging scenarios such as large pose variations, illumination changes, and facial expression variations, they still struggle to accurately capture the geometric structure of the face, resulting in performance degradation. Moreover, the limited size and diversity of existing FLD datasets hinder robust model training, leading to reduced detection accuracy. To address these challenges, we propose a Frequency-Guided Task-Balancing Transformer (FGTBT), which enhances facial structure perception through frequency-domain modeling and multi-dataset unified training. Specifically, we propose a novel Fine-Grained Multi-Task Balancing loss (FMB-loss), which moves beyond coarse task-level balancing by assigning weights to individual landmarks based on their occurrence across datasets. This enables more effective unified training and mitigates the issue of inconsistent gradient magnitudes. Additionally, a Frequency-Guided Structure-Aware (FGSA) model is designed to utilize frequency-guided structure injection and regularization to help learn facial structure constraints. Extensive experimental results on popular benchmark datasets demonstrate that the integration of the proposed FMB-loss and FGSA model into our FGTBT framework achieves performance comparable to state-of-the-art methods.


## 🚀 Setup
The required Python version is 3.6.13 and the Pytorch version is 1.8.1.

Additional required packages are listed in the requirements file.

## 📚 Datasets
Datasets are available at: 
[<a href="https://drive.google.com/drive/folders/1CG-6OxaJpkwmzHNnPHEAU7y_fVjMjIAl?usp=sharing">**AFLW**</a>]
[<a href="https://drive.google.com/drive/folders/1shd-yo8OkNzi9HkGulUI4v6lG38qUOWs?usp=sharing">**WFLW**</a>]
[<a href="https://drive.google.com/drive/folders/1lHvLzYnsqziZw7FI8AbmCOfmZSYvIZ0b?usp=sharing">**COFW**</a>]
[<a href="https://drive.google.com/drive/folders/1uNnaVCoB5cUaT6JZVQtSU5FcYA967GWk?usp=sharing">**300W**</a>].


## ⏬ Download Model
Our trained model is avaliable at <a href="https://drive.google.com/file/d/1kzzGiGhjdCXDW6pck_Wn1KKI5ya3_wOX/view?usp=drive_link">**here**</a>.




## 📜 Citation  
```bibtex
@inproceedings{wan2026fgtbt,
  title={FGTBT: Frequency-guided task-balancing transformer for unified facial landmark detection},
  author={Wan, Jun and Xiong, Xinyu and Chen, Ning and Lai, Zhihui and Zhou, Jie and Min, Wenwen},
  journal={Information Sciences},
  pages={123130},
  year={2026},
  publisher={Elsevier}
}
