# Hello, this is CDY
This part of my repository is for some molecular, computational biology hands-on projects using JAX, CNN, GNN, autoencoder, transformer, NN, etc.
It can be count as a 'practice', following through a book.

The main reason why I am learning this is because computational biology recently caught my attention and my ksef with Hyunjoon Seo was related to audio-based Parkinson's detection CNN with diffusion models as a key tool for solving data scarcity and generalization problems.

# Deep Learning For Biology

Welcome to  **Deep Learning for Biology**, an O’Reilly book that explores how modern machine learning methods can be applied to DNA, RNA, proteins, and cellular data to investigate and model biological problems.

  - Chapter 2: *Learning the Language of Proteins*
  - Chapter 3: *Learning from DNA Sequences*
  - Chapter 4: *Understanding Drug–Drug Interactions Using Graphs*
  - Chapter 5: *Detecting Skin Cancer in Medical Images*
  - Chapter 6: *Learning Spatial Organization Patterns Within Cells*

## Troubleshooting

### Chapter 2's EsmModel not loading

Occasionally you will need to _Restart Session_ in the Runtime dropdown if an import is not working as expected (e.g. `numpy`). This has been an issue for some users in Chapter 2, where loading the EsmModel would give this error `ImportError: cannot import name '_center' from 'numpy._core.umath' (/usr/local/lib/python3.12/dist-packages/numpy/_core/umath.py)`. Restarting and running cells again fixes the issue.

## Personal Notes (Bottleneck Log)
Use this section to record findings relevant to future hardware design.

(Example): "Chapter 4: Constructing the adjacency matrix for molecular graphs on the CPU is 3x slower than the forward pass on the GPU. Potential for hardware acceleration here."