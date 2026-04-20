# MMBO-Attack
# AudioTextJailbreak
## 📖 Introduction
This is a official PyTorch implementation of the paper：MMBO-Attack on Divide and Bypass: Unveiling Cross-Modal Vulnerabilities in Diffusion Models.

<img width="958" height="682" alt="MMBO_Attack" src="https://github.com/user-attachments/assets/61ba7e7c-406b-4229-9f7c-c82a6da327a8" />


## 🚀 Run
To run MJO

```bash
python filter.py
```

To run MMBO-Attack

```bash
pip install -r Requirements.txt

# Run Audio-Text Jailbreak
python both_optimize.py
```




## 👍 Contributing


We welcome contributions and suggestions!

## 🤔Experiment

### To run the asr experiment

```bash
# Run ASR experiment
python eval.py
# Run violence/illegal ASR experiment
python harm_eval.py
```
<img width="1214" height="963" alt="Experiment (2)" src="https://github.com/user-attachments/assets/8ae4cf66-d384-4ff0-9193-a4728c883846" />


### To run the extra experiment
```bash
# Run Clip experiment
python compute_Clip.py
# Run PPL experiment
python compute_PPL.py
```
<img width="597" height="654" alt="experiment (3)" src="https://github.com/user-attachments/assets/a2d690d1-978a-475c-9a10-9016422b6140" />


