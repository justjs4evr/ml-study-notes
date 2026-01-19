# Deep Learning for Biology

[![KMLA](https://img.shields.io/badge/KMLA-30th%20Wave-blue?style=flat-square)](http://www.minjok.hs.kr/)
[![Focus](https://img.shields.io/badge/Focus-CompBio%20Practice-green?style=flat-square)](https://github.com/)
[![Stack](https://img.shields.io/badge/Tech-JAX%20%7C%20PyTorch%20%7C%20GNN-orange?style=flat-square)](https://github.com/google/jax)

# Hello, this is CDY

This part of my repository is for some molecular, computational biology hands-on projects using **JAX, CNN, GNN, autoencoder, transformer, NN**, etc. It counts as 'practice', following through a book.

> **Why I am learning this:**
> Computational biology recently caught my attention. My KSEF research with Hyunjoon Seo was related to **audio-based Parkinson's detection** using CNNs with diffusion models as a key tool for solving data scarcity and generalization problems.

---

# Deep Learning For Biology

Welcome to **Deep Learning for Biology**, an O’Reilly book that explores how modern machine learning methods can be applied to DNA, RNA, proteins, and cellular data to investigate and model biological problems.

* **Chapter 2:** *Learning the Language of Proteins*
* **Chapter 3:** *Learning from DNA Sequences*
* **Chapter 4:** *Understanding Drug–Drug Interactions Using Graphs*
* **Chapter 5:** *Detecting Skin Cancer in Medical Images*
* **Chapter 6:** *Learning Spatial Organization Patterns Within Cells*

## Troubleshooting

### Chapter 2's EsmModel not loading

Occasionally you will need to **Restart Session** in the Runtime dropdown if an import is not working as expected (e.g. `numpy`).

This has been an issue for some users in Chapter 2, where loading the `EsmModel` would give this error:
```text
ImportError: cannot import name '_center' from 'numpy._core.umath' (/usr/local/lib/python3.12/dist-packages/numpy/_core/umath.py)