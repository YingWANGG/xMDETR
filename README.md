# Adapting Grounded Visual Question Answering Models to Low Resource Languages 

We propose xMDETR, a multi-lingual grounded vision-language model based on the state-of-the-art model [MDETR](https://github.com/ashkamath/mdetr), by adapting it to new languages without machine-translated data, while also keeping most of the pre-trained weights frozen (only the embedding layer and adapters in the text encoders are updated).

![image](https://github.com/YingWANGG/xMDETR/blob/main/diagram.png)
There are two or three streams of data in the proposed cross-lingual transfer. The first stream only includes textual data in the target language and is fed to the text encoder to compute the MLM loss. The second stream consists of images from GQA and corresponding code-switched questions, fed into pre-trained MDETR for QA loss. For languages with existing image-caption datasets (such as German and Chinese), we have an additional data stream to compute the contrastive loss.

For more details, please see the paper: [Adapting Grounded Visual Question Answering Models to Low Resource Languages](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Wang_Adapting_Grounded_Visual_Question_Answering_Models_to_Low_Resource_Languages_CVPRW_2023_paper.pdf) by Ying Wang, Jonas Pfeiffer, Nicolas Carion, Yann LeCun, Aishwarya Kamath.

### Checkpoints
We provided [checkpoints](https://drive.google.com/drive/folders/1tcYUfkEYGR2fGH1l0F--x8x2uN000hqk?usp=sharing) trained with MLM and code-switch QA (and contrastive loss if applicable). The test accuracy is reported as zero-shot results in the paper. 

### Data Preparation
1. Update the path of the GQA dataset in gqa.json. "vg_img_path" should point to the directory where GQA images are stored. The GQA annotation files (to obtain these files, see [MDETR](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md)) should be stored in the folder ```annotations```.
2. Obtain annotation files from [xGQA](https://github.com/Adapter-Hub/xGQA). Store them under ```annotations/fewshot```.


### Training (MLM + code-switch QA)
To facilitate training and evaluation, we provided a copy of a subset of MDETR code in this repo. The code for the model and dataloader has been modified for cross-lingual transfer implemented in this repo.
1. We have provided bilingual dictionaries for each language from xGQA in ```data/fasttext/```. The word-level translations are based on bilingual dictionaries from [MUSE](https://github.com/facebookresearch/MUSE), and then we use Google Translate to obtain translations for those words that are not present in MUSE but are included in the annotations of the GQA training split.
2. Download text datasets (e.g. OSCAR) or use HuggingFace's datasets library. Change <mlm_dir> in the command below.
3. Run the command below.
```
torchrun --nproc_per_node=<num_gpu> scripts/train.py \
--ema --do_qa --split_qa_heads --backbone timm_tf_efficientnet_b5_ns \
--load https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth?download=1 \
--no_aux_loss --qa_loss_coef 25 --no_contrastive_align_loss \
--text_tokenizer_type xlm-roberta-base \
--code_switch_path data/fasttext/en-<language_code>.txt \
--dataset_config gqa.json \
--preprocessing_num_workers <preprocessing_num_workers> \
--num_workers <num_workers> \
--mlm_dir <path to the text data for the MLM steps> \
--lang <language_code> \
--output-dir <language code> \
--lr 0.0001 \
--warmup_steps 0 \
--max_steps 100000 \
--batch_size 16 \
--gradient_accumulation 2 \
--max_seq_length 128 \
--reduction_factor 16 \
--n_shot 0 \
--distributed \
```
### Training (Contrastive)
1. Prepare dataset for contrastive learning. Download annotations and images from [COCO-CN](https://github.com/li-xirong/coco-cn) for Chinese and [Multi30K](https://github.com/multi30k/dataset) for German. Prepare the data in the following format.
```
[
  {
    "caption_new": "Zwei junge wei\u00dfe M\u00e4nner sind im Freien in der N\u00e4he vieler B\u00fcsche.",
    "caption_en": "Two young, White males are outside near many bushes.",
    "image": "/Flick30k/train/1000092795.jpg",
    "image_id": "1000092795"
  },
  ...
]
```
2. Run the command below.
```
torchrun --nproc_per_node=<num_gpu> scripts/contrastive.py \
--dataset_config gqa.json \
--ema --do_qa --split_qa_heads --backbone timm_tf_efficientnet_b5_ns \
--no_aux_loss --contrastive_loss
--num_workers <num_workers> \
--load <path to checkpoint> \
--text_tokenizer_type xlm-roberta-base \
--contrastive_path <path to the annotation file above> \
--lang <language code> \
--output-dir <output path> \
--lr <learning rate. Recommend small value> \
--warmup_steps 0 \
--batch_size <Recommend larger value> \
--reduction_factor 16 \
--distributed \
```
### Finetuning (xGQA Fewshot learning)
```
torchrun --nproc_per_node=<num_gpu> scripts/finetune.py \
--dataset_config gqa.json \
--ema --do_qa --split_qa_heads --backbone timm_tf_efficientnet_b5_ns \
--load <path to checkpoint> \
--no_aux_loss --qa_loss_coef 25 --no_contrastive_align_loss \
--num_workers <num_workers> \
--text_tokenizer_type xlm-roberta-base \
--lang <language code> \
--output-dir <output path> \
--lr <learning rate. Recommend slightly larger value> \
--epochs 10 \
--batch_size 16 \
--max_seq_length 128 \
--reduction_factor 16 \
--n_shot <num of images in fewshot>  \
--distributed
```
### Evaluation
```
python3 scripts/evaluate.py \
--dataset_config gqa.json \
--ema --eval --do_qa  --no_contrastive_align_loss --split_qa_heads \
--load <path to checkpoint> \
--backbone timm_tf_efficientnet_b5_ns \
--text_tokenizer_type xlm-roberta-base \
--batch_size 16 \
--lang <language code>
```

# Citations
```
@InProceedings{Wang_2023_CVPR_xMDETR,
    author    = {Wang, Ying and Pfeiffer, Jonas and Carion, Nicolas and LeCun, Yann and Kamath, Aishwarya},
    title     = {Adapting Grounded Visual Question Answering Models to Low Resource Languages},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2595-2604}
}
```
