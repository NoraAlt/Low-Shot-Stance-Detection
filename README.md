# Low-Shot Stance Detection (Arabic)

Code and data for **"Cross-Task Knowledge Transfer for Arabic Low-Shot Stance Detection"** (N. Alturayeif, H. Luqman, M. Ahmed).

A prompt-based multi-task model (mT0-large, text-to-text) is fine-tuned on six Arabic tasks (stance detection, sentiment analysis, NLI, claim verification, offensive language detection, sarcasm detection) with English prompts over Arabic text. It is evaluated on stance detection in full-shot, cross-task (stance hidden) and cross-target (one Mawqif target hidden) settings with 0, 5, 10 and 20 shots.

## Repository structure

```
PMTL/                   Parallel multi-task learning (AraBERT-twitter): stance + sentiment + sarcasm
  PMTL.ipynb, PMTL_UW.ipynb, PMTL_SW.ipynb, PMTL_RW.ipynb, PMTL_HW.ipynb
SMTL/                   Sequential multi-task learning (AraBERT-twitter)
  SMTL_Sarc_Sent.ipynb, SMTL_Sent_Sarc.ipynb      Task order variants
  SMTL_UW_Sar_Sent.ipynb, SMTL_SW_Sar_Sent.ipynb, SMTL_RW_Sar_Sent.ipynb, SMTL_HW_Sar_Sent.ipynb, SMTL_HW_Sent_Sar.ipynb
code/
  run_seq2seq_M.py      Training / evaluation script (modified Hugging Face seq2seq example)
  run_multitask.sh      Multi-task and low-shot training runs
  run_task.sh           Single-task (stance only) training run
  eval_multitask.sh     Prediction on the Mawqif stance test set
  my_metric.py          Metric used during prediction (accuracy, F1, favor/against F1)
  All_metrics.py        Favor/against F1 per target and overall
  utils/
    clean_data.py       Text cleaning
    Prepare_data.py     Cleaning + prompting + merging all datasets
    Data_split.py       80/20 stratified split (datasets without an official split)
    balance_data.py     Per-task cap for the balanced variant (set here to 3,000 for the test set; training used 13,000)
    Low_shot.py         N-way K-shot support-set sampling
Data/
  Mawqif/               Mawqif official split (Mawqif_AllTargets_Train.csv, Mawqif_AllTargets_Test.csv), used by PMTL/SMTL
  train_prompted.json   All tasks, prompted (95,599 rows): prompt, response, dataset
  test_prompted.json    All tasks, prompted (21,375 rows)
  train.json, test.json All tasks, without prompts (context, response; original labels), used for the no-prompt baseline
  test_Mawqif.json      Mawqif stance test set, prompted (619 rows: WE 210, DT 203, CV 206)
  low-shot/
    rebuild_lowshot.py  Rebuilds the 17 low-shot training files
    support_sets/       Exact support sets used in each low-shot run
```

Each row in the prompted files has a `dataset` field naming its source split (e.g. `Mawqif_stance_train`, `AraCovidVac_traindev`, `XGLUE-xnli_train`).

## PMTL and SMTL notebooks

These notebooks implement the parallel and sequential multi-task models (stance, sentiment and sarcasm on Mawqif) with different loss-weighting schemes (suffixes UW, SW, RW, HW; HW = hierarchical weighting), from "Enhancing stance detection through sequential weighted multi-task learning" (Alturayeif, Luqman, Ahmed, 2023). `SMTL/SMTL_HW_Sar_Sent.ipynb` (SMTL-HW) is the baseline compared against in the low-shot experiments.

To run them, put `Data/Mawqif/Mawqif_AllTargets_Train.csv` and `Mawqif_AllTargets_Test.csv` next to the notebook (or edit `TrainData_name` / `TestData_name`). They log to Weights & Biases: run `wandb login` with your own key, or set `WANDB_MODE=disabled`. Cell outputs from the original runs are kept. Three SMTL notebooks (`SMTL_HW_Sent_Sar`, `SMTL_SW_Sar_Sent`, `SMTL_UW_Sar_Sent`) include a Google Colab cell; skip it when running locally.

## Low-shot training files

The low-shot training files (about 45 to 50 MB each) are not stored. Rebuild them with:

```
python Data/low-shot/rebuild_lowshot.py
```

This writes `Data/low-shot/train_3N_{0,5,10,20}K_{All,Women,Digital,Covid}.json` and `train_3N_0K_Covid_withAracovid.json`. Each file is `train_prompted.json` with the held-out stance entries removed, plus the support set from `support_sets/`. The rebuilt files are identical (as sets of rows) to the files used in the experiments.

| File suffix | Setting | Stance data removed | Support set |
| --- | --- | --- | --- |
| `All` | Cross-task | All stance entries (Mawqif + ArCovidVac) | Drawn from all three targets |
| `Women`, `Digital`, `Covid` | Cross-target | Entries for that target (for Covid: Mawqif and ArCovidVac) | Drawn from the held-out target |
| `Covid_withAracovid` | Cross-target, zero-shot | Mawqif Covid entries only | None |

Support-set sizes per run: 45 (5-shot), 90 (10-shot), 180 (20-shot).

## Running

1. Install the requirements (`pip install -r requirements.txt`). The runs used `transformers` 4.24.0 on one GPU with 48 GB memory.
2. Edit the paths in the scripts. They contain the original server paths (`/home/nora_alturayeif/STANCE/My_T/...`); `data/Prompted_data_EN/train.json` there corresponds to `Data/train_prompted.json` here.
3. Train: set `task_name`, `train_file` and `validation_file` in `code/run_multitask.sh`, then run it.
4. Predict: set `task_name`, the model directory and `test_file` (`Data/test_Mawqif.json`) in `code/eval_multitask.sh`. Also set `file_name` and `file_dir` in `code/my_metric.py`, which writes the report.
5. Score per target: set `data_name` in `code/All_metrics.py` to the generated predictions file.

The single-task run (`code/run_task.sh`) trains on Mawqif stance only. Create its training file with:

```
python -c "import json; f=open('Data/train_Mawqif.json','w',encoding='utf-8'); [f.write(l) for l in open('Data/train_prompted.json',encoding='utf-8') if json.loads(l)['dataset']=='Mawqif_stance_train']"
```

Main hyperparameters: max source length 1024, max target length 16, batch size 32, gradient accumulation 4, learning rate 3e-4, Adafactor, 10 epochs, early stopping patience 5.

## Evaluation metric

Stance is scored with the average of the F1 scores of the favor and against classes, computed per target and over all targets (`All_metrics.py`).

## Datasets

The data combines the following public datasets. Please cite the original papers and follow their licenses:

- Stance: Mawqif, ArCovidVac
- Sentiment: ASTD, AJGT, MASC, UMSAB, Mawqif (sentiment labels), ArSarcasm-v2 (sentiment labels)
- NLI / claim verification: XNLI (Arabic), AraStance, ANS
- Offensive language: OSACT 2020, OSACT 2022
- Sarcasm: ArSarcasm-v2, iSarcasmEval (Arabic), Mawqif (sarcasm labels)

## Contact

Nora Alturayeif
