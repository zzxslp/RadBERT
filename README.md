## RadBERT

RadBERT is a series of models trained with millions (more to come!) radiology reports, which achieves stronger medical language understanding performance than previous bio-medical domain models such BioBERT, Clinical-BERT, BLUE-BERT and BioMed-RoBERTa.

For details, check out the paper here: 
[RadBERT: Adapting transformer-based language models to radiology](https://pubs.rsna.org/doi/abs/10.1148/ryai.210258)

### Pretrained Models
```RadBERT-RoBERTa-4m``` is trained with RoBERTa initialization and 4 million VA hospital reports. You can access the model on huggingface from [here](https://huggingface.co/zzxslp/RadBERT-RoBERTa-4m).

### How to use

Here is an example of how to use this model to extract the features of a given medical report in PyTorch:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel
config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=config)
text = "Replace me by any medical text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

### BibTeX entry and citation info

If you find this repository helpful, please cite our paper:

```bibtex
@article{yan2022radbert,
  title={RadBERT: Adapting transformer-based language models to radiology},
  author={Yan, An and McAuley, Julian and Lu, Xing and Du, Jiang and Chang, Eric Y and Gentili, Amilcare and Hsu, Chun-Nan},
  journal={Radiology: Artificial Intelligence},
  volume={4},
  number={4},
  pages={e210258},
  year={2022},
  publisher={Radiological Society of North America}
}
```