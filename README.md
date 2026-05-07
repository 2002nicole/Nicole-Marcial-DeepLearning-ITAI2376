# Nicole Marcial — ITAI 2376 Deep Learning Portfolio

## Overview

This repository contains my course portfolio for **ITAI 2376: Deep Learning** at Houston City College. It includes my major hands-on labs, midterm blueprint, portfolio presentation, and final AI agent project summary.

This portfolio shows my learning progression from neural network foundations to CNNs, transformers, generative models, reinforcement learning, and AI agents.

## Portfolio Links

- **GitHub Profile:** https://github.com/2002nicole
- **ITAI 2376 Portfolio Repository:** https://github.com/2002nicole/Nicole-Marcial-DeepLearning-ITAI2376
- **Full Final Project Repository:** https://github.com/2002nicole/NicoleMarcial_Solo_ITAI2376

## Featured Final Project

### Cookie Recipe Adaptation Agent

The featured project for this course is my **Cookie Recipe Adaptation Agent**, a single AI agent that helps beginner bakers adapt cookie recipes through ingredient substitutions, batch scaling, dietary changes, and texture guidance.

The full final project is stored in its own complete repository:

```text
https://github.com/2002nicole/NicoleMarcial_Solo_ITAI2376
```

I kept the final project in a separate repository because it contains the complete working agent, including the notebook implementation, requirements file, environment template, reflection, architecture diagram, data folder, and demo video.

### Final Project Highlights

- Built as a **single AI agent**
- Uses **LangChain** for agent workflow and custom tools
- Uses **OpenAI** as the language model provider
- Uses **Retrieval-Augmented Generation** for baking and recipe context
- Uses **Hugging Face sentence-transformer embeddings**
- Uses **Chroma** as the local vector store
- Includes an optional **Gradio** demo interface
- Includes custom tools for:
  - recipe retrieval
  - ingredient scaling
  - substitution guidance
  - full recipe adaptation

## Learning Journey

This course moved from foundational deep learning concepts into applied AI agent design.

| Course Area | Modules | Main Concepts |
|---|---:|---|
| Neural Network Foundations | Modules 02–03 | MLPs, CNNs, image classification, transfer learning |
| Sequence and Language Models | Modules 04–05 | RNNs, NLP, transformers, attention |
| Generative AI | Modules 06–08 | VAEs, GANs, U-Net, diffusion models, CLIP evaluation |
| AI Agents and Reasoning | Modules 09–12 | RL, RLHF, RAG, LangChain, tools, single-agent design |
| Final Build | Final Project | Cookie Recipe Adaptation Agent |

## Repository Structure

```text
Nicole-Marcial-DeepLearning-ITAI2376/
├── README.md
├── FinalProject-CookieRecipeAgent/
│   ├── README.md
│   ├── REQUIREMENTS.md
│   ├── DATA_ACCESS.md
│   └── Final_Agent_Nicole_Marcial_ITAI2376.ipynb
├── Labs/
│   ├── L02-NeuralNetworkFoundations/
│   │   ├── README.md
│   │   ├── REQUIREMENTS.md
│   │   ├── DATA_ACCESS.md
│   │   └── L02_Nicole_Marcial_ITAI2376.ipynb
│   ├── L03-ConvolutionalNeuralNetworks/
│   │   ├── README.md
│   │   ├── REQUIREMENTS.md
│   │   ├── DATA_ACCESS.md
│   │   └── L03_Nicole_Marcial_ITAI2376.ipynb
│   ├── L08-DiffusionModels/
│   │   ├── README.md
│   │   ├── REQUIREMENTS.md
│   │   ├── DATA_ACCESS.md
│   │   ├── L08_Notebook_Nicole_Marcial_ITAI2376.ipynb
│   │   └── L08_Report_Nicole_Marcial_ITAI2376.pdf
│   └── L09-ReinforcementLearning/
│       ├── README.md
│       ├── REQUIREMENTS.md
│       ├── DATA_ACCESS.md
│       └── L09_Nicole_Marcial_ITAI_2376.ipynb
├── MidtermBlueprint/
│   ├── README.md
│   └── MidTerm_Blueprint_Marcial_Nicole_ITAI2376.pdf
├── Presentation/
│   └── Pf_NicoleMarcial_ITAI2376.pdf
└── .gitignore
```

## Supporting Labs

The folders under `Labs/` are guided course labs, not independent full-scale projects. Most of the notebooks were provided with starter code as part of the course. My work included completing required coding sections, running the notebooks, analyzing outputs, answering reflection questions, completing student challenges, and documenting the results for GitHub.

### L02 — Neural Network Foundations

This lab focused on building, training, and evaluating basic neural networks using **PyTorch** and **TensorFlow/Keras**. It used the Fashion-MNIST dataset and included hyperparameter tuning, dropout regularization, and comparison between frameworks.

**Main skills practiced:**

- MLP training and evaluation
- PyTorch training loops
- TensorFlow/Keras model building
- Hyperparameter tuning
- Dropout and overfitting analysis

### L03 — Convolutional Neural Networks

This lab focused on binary image classification using the **Puppy or Bagel** dataset. It included a custom CNN, transfer learning with MobileNetV2, model evaluation, and fine-tuning.

**Main skills practiced:**

- CNN architecture
- Image classification
- Data augmentation
- Transfer learning
- Fine-tuning pre-trained models
- Confusion matrix and classification report interpretation

### L08 — Diffusion Models

This lab focused on training a class-conditioned diffusion model on MNIST. The notebook included U-Net architecture components, forward and reverse diffusion, denoising visualizations, and CLIP-based evaluation. A separate PDF report explains the results and reflection questions.

**Main skills practiced:**

- Generative modeling
- U-Net denoising architecture
- Forward and reverse diffusion
- Time and class conditioning
- CLIP evaluation
- Comparing numerical loss to visual quality

### L09 — Reinforcement Learning and AI Agents

This lab used the CartPole-v1 environment to demonstrate reinforcement learning with a Q-learning agent. It included a random agent baseline, 500-episode Q-learning training, exploration-rate experiments, and reflections connecting RL concepts to my final Cookie Recipe Adaptation Agent.

**Main skills practiced:**

- Reinforcement learning loop
- Q-learning
- Q-table updates
- Exploration vs. exploitation
- Learning curve interpretation
- RLHF and agent-design reflection

## Midterm Blueprint

The `MidtermBlueprint/` folder contains my AI agent planning document. This blueprint proposed the Cookie Recipe Adaptation Agent before the final implementation was built.

The blueprint covers:

- problem statement
- chosen option: single AI agent
- proposed architecture
- deep learning connection
- LangChain framework choice
- planned tools and data
- build timeline
- expected challenges

## Portfolio Presentation

The `Presentation/` folder contains my final portfolio summary presentation for ITAI 2376.

The presentation summarizes:

- my learning journey through the course
- the Cookie Recipe Adaptation Agent
- supporting labs
- skills and competencies
- reflection and next steps

## Skills Demonstrated

### Deep Learning Skills

- Neural network training and evaluation
- CNN image classification
- Transfer learning
- Transformer-based language model concepts
- Diffusion model concepts
- Reinforcement learning foundations

### Agent Development Skills

- Single-agent design
- LangChain workflow setup
- Tool-based reasoning
- Retrieval-Augmented Generation
- Embeddings and vector stores
- Gradio interface development

### Professional Skills

- GitHub documentation
- Project scoping
- Debugging and testing
- Technical reflection
- Communicating AI results clearly
- Connecting AI tools to real user needs

## Notes About Authorship

This repository includes both guided course labs and my final independent project work.

For the labs, starter code and guided notebook structure were provided by the course. My contributions included completing required coding sections, running and debugging notebooks, interpreting model outputs, completing student challenge sections, and writing analysis/reflection responses.

The final Cookie Recipe Adaptation Agent was my individual final project. It is linked as a separate full repository because it contains the complete working implementation and demo materials.

## Contact

**Nicole Marcial**  
GitHub: https://github.com/2002nicole  
Email: 2002.28.nicole@gmail.com
