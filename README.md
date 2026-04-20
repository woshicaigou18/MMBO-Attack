# MMBO-Attack
# AudioTextJailbreak
## 📖 Introduction
This is a official PyTorch implementation of the paper：Audio-Text Jailbreak Attack on Large Audio-Language Models: Towards Generality and Stealthiness.

![Uploading 1d6f87774f86de85a9377ef6f0035187.png…](https://github.com/woshicaigou18/MMBO_Attack/blob/main/MMBO_Attack.png)


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

```bash
# Run ASR experiment
python eval.py
# Run violence/illegal ASR experiment
python harm_eval.py
```



```bash
# Run Clip experiment
python compute_Clip.py
# Run PPL experiment
python compute_PPL.py
```



![image-20251220122559161](https://github.com/woshicaigou18/MMBO_Attack/blob/main/Experiment.png)
