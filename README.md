# QLIP

[\[📂 GitHub\]](https://github.com/NVlabs/QLIP)
[\[📃 QLIP Tech Report\]](http://arxiv.org/abs/2502.yyyyy)
[\[🔗 Project Page\]](http://nvlabs.github.io/QLIP/)
[\[🤗 HF Collections\]](https://huggingface.co/collections/nvidia/qlip-67a478054fce07a7be99d5cd)

[**QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation**](http://arxiv.org/abs/2502.yyyyy)  
Yue Zhao<sup>1,&ast;</sup>,
Fuzhao Xue<sup>2,&dagger;</sup>,
Scott Reed</a><sup>2</sup>,
Linxi Fan<sup>2</sup>,
Yuke Zhu<sup>2</sup>,
Jan Kautz<sup>2</sup>,
Zhiding Yu<sup>2</sup>,
Philipp Kr&auml;henb&uuml;hl<sup>1</sup>
De-An Huang<sup>2</sup>  
<sup>1</sup> UT Austin, <sup>2</sup>NVIDIA  
<sup>&ast;</sup>The work was done during an internship at NVIDIA Research.  
<sup>&dagger;</sup>Now at Google DeepMind.  
[arxiv](http://arxiv.org/abs/2502.yyyyy) | [bibtex](#citing-qlip) 

## Introduction

We introduce Quantized Language-Image Pretraining (**QLIP**), a visual tokenization method that combines state-of-the-art reconstruction quality with state-of-the-art zero-shot image understanding.
QLIP trains a binary-spherical-quantization-based autoencoder with reconstruction and language-image alignment objectives.
We are the first to show that the two objectives do not need to be at odds.
We balance the two loss terms dynamically during training and show that a two-stage training pipeline effectively mixes the large-batch requirements of image-language pre-training with the memory bottleneck imposed by the reconstruction objective.
We validate the effectiveness of QLIP for multimodal understanding and text-conditioned image generation with a single model.
Specifically, QLIP serves as a drop-in replacement for the visual encoder for LLaVA and the image tokenizer for LlamaGen with comparable or even better performance.
Finally, we demonstrate that QLIP enables a unified mixed-modality auto-regressive model for understanding and generation.

## Model Zoo

We provide the following models:
| model name    | #bits  |  CR<sub>&uarr;<sub>   | 0-shot<sub>&uarr;<sub> | rFID<sub>&darr;<sub> | HF Link |
| ------------- | ------ | ----- | ------ | ---- | ------- |
| QLIP-B-16-256 |   28   | 219.4 |  74.3  | 3.21 | [🤗 link](https://huggingface.co/NVIDIA/QLIP-B-16-256) |
| QLIP-B-8-256  |   28   |  54.8 |  75.6  | 0.70 | [🤗 link](https://huggingface.co/NVIDIA/QLIP-B-8-256)  |
| QLIP-L-14-392 |   28   | 168   |  79.1  | 1.46 | [🤗 link](https://huggingface.co/NVIDIA/QLIP-L-14-392) |

Note:  
- **CR**: compression ratio = 24/(#bits)*patch_size^2;
- **0-shot**: zero-shot classification accuracy on IN-1k-val;
- **rFID**: reconstruction FID on IN-1k-val.

## Usage

Please refer to [notebook](QLIP/example.ipynb).

## Citing QLIP

```bibtex
@article{zhao2025qlip,
  title={QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation},
  author={Zhao, Yue and Xue, Fuzhao and Reed, Scott and Fan, Linxi and Zhu, Yuke and Kautz, Jan and Yu, Zhiding and Krähenbühl, Philipp and Huang, De-An},
  journal={arXiv preprint arXiv:2502.yyyyy},
  year={2025}
}
```

## Acknowledgement
The project builds upon the following open-source efforts:
- [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP/rei): We use EVA-CLIP as initialization which significantly speeds up the training convergence.

- [LLaVA](https://github.com/haotian-liu/LLaVA): We use LLaVA to evaluate the multimodal understanding performance.

- [LlamaGen](https://github.com/FoundationVision/LlamaGen): We build the text-to-image generation evaluation on top of LlamaGen.

- [Lingua](https://github.com/facebookresearch/lingua): We build the unified multimodal model on top of Lingua.