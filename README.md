---
title: ColPali
app_file: app.py
sdk: gradio
sdk_version: 4.41.0
---
# RAG-based PDF Search and Keyword Extraction using Qwen2VL

This repository contains an implementation of a **RAG (Retrieval-Augmented Generation)** based PDF search system using **Copali's implementation** of the Byaldi library and **Qwen2VL** for creating the RAG pipeline. Additionally, the repository includes a Gradio app that allows users to extract text from images and highlight searched keywords using **Qwen2VL**.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [RAG PDF Search](#rag-pdf-search)
  - [Gradio App for Keyword Extraction](#gradio-app-for-keyword-extraction)
- [License](#license)

## Overview

### RAG PDF Search

In `copali-qwen.ipynb`, you will find the complete implementation of the **RAG-based PDF search**. The pipeline is built using the **Copali** implementation of the Byaldi library, along with **Qwen2VL**. By default, the code indexes and searches through an image (`image.png`), but you can easily modify the path to a PDF file or any other desired document.

### Gradio App for Keyword Extraction

The `app.py` file contains a **Gradio app** that utilizes only **Qwen2VL** to extract text from an image and highlight the keywords matching the user's search query. This app is an easy-to-use interface for real-time keyword extraction from images.

## Installation

To run this project, you will need to install the following dependencies:

```bash
pip install transformers byaldi qwen-vl-utils gradio pillow torch
