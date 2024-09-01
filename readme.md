# Resume QA Bot

## Overview

This project involves a fine-tuned QA bot that answers questions based on resume data. 

## Folder Structure

- `data/`: Contains the resume data and custom Q&A files.
- `src/`: Contains scripts for data extraction, preparation, fine-tuning, and interacting with the QA bot.
- `models/`: Directory to save the fine-tuned model.

## Setup

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Extract resume data and prepare data:
    ```sh
    python src/extract_data.py
    python src/prepare_data.py
    ```

3. Fine-tune the model:
    ```sh
    python src/fine_tune_model.py
    ```

4. Interact with the QA bot:
    ```sh
    streamlit run app.py
    ```

## Notes

-To run the base model just run streamlit run app.py withhout making any changes except the env file