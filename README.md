# Bayesformer

> This project is based on the work of **Sankararaman et al. (2022)**,  
> *BayesFormer: Transformer with Uncertainty Estimation*,  
> available on **arXiv** ([arXiv:2206.00826](https://arxiv.org/abs/2206.00826)).


## Project Overview

This project explores **active learning strategies** using **Bayesian methods** within transformer-based architectures. Specifically, we compare a standard Transformer model with **BayesFormer**, a Bayesian-enhanced Transformer that integrates uncertainty estimation. The goal is to evaluate how Bayesian methods impact **sample efficiency, uncertainty quantification, and model performance** in an active learning setting.

This project was developed collaboratively by:
- **Gaetano Agazzotti** (`gaetano(dot)agazzotti(at)ens-paris-saclay(dot)fr`)
- **Marion Chabrol** (`marion(dot)chabrol(at)ensae(dot)fr`)
- **Jules Chapon** (`jules(dot)b(dot)chapon(at)gmail(dot)com`)
- **Suzie Grondin** (`suzie(dot)grondin(at)ensae(dot)fr`)

---

## Installation and Usage

### Prerequisites
- Python 3.10
- Required libraries: `torch`, `numpy`, `matplotlib`, `scipy`, `pandas`

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Suzie14/BayesFormer.git
   cd BayesFormer
   ```

2. Run the notebooks `overfitting.ipynb`, `bayesian_classification.ipynb` and `active_learning.ipynb`.

## Code architecture
```plaintext
Bayesian-Active-Learning-Project/
├── analysis/
│   ├── notebooks/
│   │   ├── active_learning.ipynb
│   │   ├── bayesian_classification.ipynb
│   │   ├── overfitting.ipynb
├── data/
├── src/
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── constants.py
│   ├── libs/
│   │   ├── __init__.py
│   │   ├── active_learning.py
│   │   ├── preprocessing.py
│   │   ├── preprocessing_classif.py
│   │   ├── tokenizer.py
│   │   ├── utils.py
│   │   ├── visualization.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── bayesian_classification.py
│   │   ├── transformer.py
├── .gitignore
├── README.md
```
---
## Key Concepts

- **Active Learning**: A technique where models actively query the most informative data points to improve learning efficiency.
- **Uncertainty Estimation**: Bayesian methods allow us to quantify prediction confidence, improving sample selection.
- **BayesFormer**: A Transformer model incorporating Bayesian dropout for uncertainty estimation.
- **Transformers in Active Learning**: We compare standard Transformers and BayesFormers under multiple selection strategies.

The implementation consists of:

### 1. Model Architectures:
- **Transformer**:
  - Standard attention-based model.
  - Uses fixed dropout and weight updates.
- **BayesFormer**:
  - Incorporates Bayesian uncertainty estimation.
  - Utilizes dropout-based Monte Carlo sampling for active learning.

### 2. Active Learning Strategies:
We evaluate different strategies for selecting the most informative samples:
- **Max Uncertainty:** Prioritizing samples with the highest uncertainty.
- **Margin Uncertainty:** Selecting samples where the model is most uncertain between two classes.
- **Entropy-based Selection:** Choosing samples with the highest entropy in output probability distributions.

---

## Repository Structure

Find the code and notebooks in our [GitHub repository](https://github.com/Suzie14/BayesFormer).


---

## Results

### **Key Takeaways**
- Bayesian uncertainty improves active learning efficiency.
- BayesFormer provides better uncertainty quantification but requires more computational resources.
- Entropy-based selection outperforms margin and max uncertainty methods.

