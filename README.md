# Machine Learning Study Log

## 👨‍💻 About Me

**Hi! I am a KMLA student, 30th waver, CDY.**

I am currently diving deep into the intersection of Computer Science and Natural Sciences, focusing on:

* **Machine Learning** (Statistical Learning Theory, Reinforcement Learning)
* **Physics** (Mechanics, Electromagnetism)
* **Mathematics** (Calculus, Linear Algebra)

My long-term goal involves developing next-generation chip architectures optimized for AI operations.

> This repository is a **learning log** (code-alongs and practice), not original research. My own projects are listed under [Related original work](#-related-original-work).

---

## 📚 Current Study Roadmap

I am simultaneously reviewing and learning from three core texts to bridge theory, implementation, and application:

| Focus Area | Textbook / Resource | Folder | Status |
| --- | --- | --- | --- |
| **Core ML & DL** | *Machine Learning with PyTorch and Scikit-Learn* (Raschka et al.) | `ml_w_ptc_skl/` | **Main focus of this repo** |
| **Finance** | *Machine Learning for Algorithmic Trading* (Stefan Jansen) | `ml4t_jansen/` | *Supplementary study* |
| **Biology** | *Deep Learning for Biology* (Ravarani et al.) | `dl_4bio/` (+ `dlfb-clone` submodule) | *Supplementary study* |
| **LLMs** | LLM book notes | `honllm/` | *Just started (chapter 1)* |

---

## 📂 Repository Focus: Raschka's ML

`ml_w_ptc_skl/` is a code-along and practice log for **"Machine Learning with PyTorch and Scikit-Learn"**.

My goal here is **fluency**. Rather than just copying code, I am rewriting practice examples to focus on **memorizing and implementing Scikit-Learn/PyTorch code patterns** without reliance on external documentation.

### Structure

Each chapter folder contains some of the following:

1. **`practice*`**: My own re-implementations of specific `sklearn` patterns (pipelines, transformers) or PyTorch training loops, written from memory.
2. **`snip.py`**: Key concepts, mathematical derivations, or API nuances I need to remember.
3. **`whole*`**: The whole code used in the book, coded alongside reading it.

### Chapter progress

Topics follow the book's table of contents. Only files that exist are listed.

| Chapter | Topic | What is in the folder |
| --- | --- | --- |
| **Ch 02** | Simple classifiers: Perceptron, Adaline | `whole.py`, `snip.py` |
| **Ch 03** | Tour of scikit-learn classifiers | `whole.py`, `practice.py`, `snip.py` |
| **Ch 04** | Data preprocessing (`StandardScaler`, `SimpleImputer`, categorical data) | `whole.py`, `practice.py`, `snip.py` |
| **Ch 05** | Dimensionality reduction | `whole.ipynb`, `practice.ipynb` |
| **Ch 06** | Model evaluation and tuning (K-fold CV, learning curves, `GridSearchCV`) | `whole.ipynb`, `practice.ipynb`, `snip.py` |
| **Ch 07** | Ensemble learning | `whole.ipynb`, `practice.ipynb` |
| **Ch 08** | Sentiment analysis (text) | `practice.ipynb` only (`whole.ipynb` not done yet) |
| **Ch 12-14** | PyTorch: training pipeline, tensors/autograd/`nn.Module`, CNNs | `whole.ipynb`, `ch13_whole.ipynb`, `14whole.ipynb`, `practice.ipynb` |
| **Ch 15** | Sequence modeling (RNNs) | `whole.ipynb` |

---

## 🗂️ Other folders

| Folder | Contents |
| --- | --- |
| `dl_4bio/` | *Deep Learning for Biology*: chapter 1 intro and lessons, chapter 2-3 notebooks, `requirements.txt` |
| `ml4t_jansen/` | *ML for Algorithmic Trading*: parsing NASDAQ ITCH messages and rebuilding the order book (two notebooks) |
| `honllm/` | LLM book, chapter 1 intro notebook and questions |

---

## 🔗 Related original work

* [`kast2026-sam3-glioma-08-prog-log`](https://github.com/justjs4evr/kast2026-sam3-glioma-08-prog-log): SAM3 glioma sub-region segmentation study (interim snapshot from the KAST 2026 mentorship program).
* [`sat-words`](https://github.com/justjs4evr/sat-words): SAT vocabulary study website.
