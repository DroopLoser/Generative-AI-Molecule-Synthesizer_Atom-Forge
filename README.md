# Generative AI Molecule Synthesizer

A generative AI–powered molecular design platform that generates novel molecular structures based on user-defined molecular properties.

The system combines **Variational Autoencoders (VAEs)** and **Graph Neural Networks (GNNs)** to generate and evaluate molecular structures. Generated molecules can also be searched against **PubChem**, a public chemical database, to retrieve information about known compounds.

---

## Overview

Designing molecules with specific chemical properties is a challenging and computationally intensive task. This project aims to assist the molecular discovery process by using generative artificial intelligence to propose molecular structures according to properties specified by the user.

The application consists of two major components:

* **Frontend** — A web-based interface through which users specify molecular requirements and interact with generated results.
* **Backend** — A Python-based AI and cheminformatics pipeline responsible for molecular generation, property prediction, optimization, molecular processing, and database lookup.

The backend combines deep learning and cheminformatics techniques to represent, generate, evaluate, and process molecular structures.

---

## Key Features

* Generate molecular structures using generative AI.
* Property-guided molecular generation.
* Variational Autoencoder (VAE) for molecular representation and generation.
* Graph Neural Network (GNN) for molecular property modeling.
* Molecular optimization based on target properties.
* Molecular validity and chemistry-related processing.
* Molecular tokenization and representation processing.
* Molecular naming functionality.
* Search generated molecules against **PubChem**.
* Retrieve information about known compounds from public chemical databases.
* Web-based interface for interacting with the molecular generation system.
* FastAPI backend for communication between the frontend and AI pipeline.
* Pre-trained VAE and GNN models for molecular generation and property prediction.

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    │ Desired Properties   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Frontend        │
                    │       Next.js        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             ┌──────────────┐       ┌──────────────┐
             │     VAE      │       │     GNN      │
             │  Generation  │       │  Properties  │
             └──────┬───────┘       └──────┬───────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Molecular Processing │
                    │   & Optimization     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       PubChem        │
                    │  Chemical Database   │
                    └──────────────────────┘
```

---

## Repository Structure

```text
Generative-AI-Molecule-Synthesizer/
│
├── Atom-Forge-Backend/
│   │
│   ├── data/
│   │   └── zinc250k.csv
│   │
│   ├── models/
│   │   ├── gnn.py
│   │   └── vae.py
│   │
│   ├── utils/
│   │   ├── chemistry.py
│   │   ├── naming.py
│   │   ├── optimization.py
│   │   └── tokenizer.py
│   │
│   ├── blabber.py
│   ├── cli.py
│   ├── data.py
│   ├── Dockerfile
│   ├── fastapi_app.py
│   ├── fix_scalar.py
│   ├── generate.py
│   ├── gnn_scaler.pkl
│   ├── gnn.pt
│   ├── requirements.txt
│   ├── train_gnn.py
│   ├── train_vae.py
│   └── vae.pt
│
├── atomforge-frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── .gitignore
│   ├── eslint.config.mjs
│   ├── next.config.ts
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
├── .gitignore
└── README.md
```

---

## Backend

The backend contains the machine learning, molecular processing, optimization, and API components of the project.

### Machine Learning Models

The `models/` directory contains the core deep learning models:

* `vae.py` — Variational Autoencoder used for molecular representation and generation.
* `gnn.py` — Graph Neural Network used for molecular property modeling.

Pre-trained model artifacts are included in the backend:

* `vae.pt`
* `gnn.pt`
* `gnn_scaler.pkl`

### Molecular Processing

The `utils/` directory contains utilities supporting the molecular generation pipeline:

* `chemistry.py` — Molecular and chemistry-related processing.
* `naming.py` — Molecular naming functionality.
* `optimization.py` — Optimization of generated molecules based on target properties.
* `tokenizer.py` — Processing and tokenization of molecular representations.

### Dataset

The project includes the **ZINC250K** molecular dataset:

```text
Atom-Forge-Backend/data/zinc250k.csv
```

The dataset is used as part of the machine learning pipeline for molecular modeling and generation.

### API

`fastapi_app.py` provides the FastAPI application that connects the frontend with the backend's molecular generation and processing functionality.

### Model Training

The repository includes scripts for training the machine learning models:

```text
train_vae.py
train_gnn.py
```

---

## Frontend

The frontend is built with **Next.js**, **React**, and **TypeScript**.

It provides the user interface through which users can:

* Specify desired molecular properties.
* Interact with the molecular generation system.
* View generated molecular results.
* Access information returned by the backend.

The main frontend directories are:

* `app/` — Next.js application and routes.
* `components/` — Reusable UI components.
* `public/` — Static assets.

---

## Tech Stack

### Frontend

* Next.js
* React
* TypeScript
* PostCSS
* ESLint

### Backend

* Python
* FastAPI
* PyTorch
* Graph Neural Networks
* Variational Autoencoders
* Cheminformatics tooling

### Machine Learning

* Variational Autoencoder (VAE)
* Graph Neural Network (GNN)
* Molecular representation learning
* Property-guided molecular generation
* Molecular optimization

### External Resources

* ZINC250K molecular dataset
* PubChem

---

## Getting Started

### Prerequisites

Make sure the following are installed:

* Python
* Node.js
* npm

---

## Backend Setup

Navigate to the backend directory:

```bash
cd Atom-Forge-Backend
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI development server:

```bash
uvicorn fastapi_app:app --reload
```

The backend will start using the FastAPI development server.

---

## Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd atomforge-frontend
```

Install the Node.js dependencies:

```bash
npm install
```

Start the Next.js development server:

```bash
npm run dev
```

The frontend will then be available through the local Next.js development server.

---

## Molecular Generation Workflow

The general workflow of the system is:

```text
1. User specifies desired molecular properties
                    ↓
2. Frontend sends requirements to backend
                    ↓
3. Backend processes the requested properties
                    ↓
4. VAE generates candidate molecular structures
                    ↓
5. GNN evaluates relevant molecular properties
                    ↓
6. Candidate molecules are optimized
                    ↓
7. Molecular structures undergo chemistry validation
                    ↓
8. Molecular naming / identification is performed
                    ↓
9. PubChem is searched for known compounds
                    ↓
10. Results are returned to the frontend
```

---

## Data and Models

The project includes pre-trained model artifacts so that the trained models can be used without retraining from scratch.

```text
Atom-Forge-Backend/
├── gnn.pt
├── gnn_scaler.pkl
└── vae.pt
```

The training scripts are also included:

```text
train_gnn.py
train_vae.py
```

This allows the machine learning pipeline to be further developed or retrained when required.

---

## Important Note

This project is intended as a computational molecular design and research tool.

Generated molecular structures and predicted properties are computational results and should not be considered experimentally validated chemical compounds. Any molecule intended for real-world chemical or pharmaceutical applications requires appropriate expert review and experimental validation.

---

## Future Improvements

Potential future improvements include:

* Supporting additional molecular properties.
* Improving molecular generation quality and validity.
* Incorporating additional molecular datasets.
* Adding more molecular property prediction models.
* Improving molecular optimization strategies.
* Expanding chemical database integrations.
* Adding richer molecular visualization.
* Improving model training and inference performance.
* Supporting additional generative architectures.

---

## Project Status

This project is an experimental generative-AI platform for computational molecular design and exploration.

It combines generative deep learning, graph-based molecular modeling, cheminformatics, and public chemical database search into a single application.

---

## License

MIT License
