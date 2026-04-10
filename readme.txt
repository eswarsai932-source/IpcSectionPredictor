go to this path:
    cd "project_folder_location"

if no virtual environment is created the use:
        python -m venv venv

to Activate  virtual environment use:
        venv\Scripts\activate

Install required packages:
        pip install streamlit pandas numpy scikit-learn joblib sentence-transformers torch
        pip install scipy

best to install if any this is not installed:
    pip install -r requirements.txt

run expand_dataset.py to get dataset:
    python expand_dataset.py

TRAIN THE MODEL:
        python train_model.py

to run the application:
        python -m streamlit run app.py
            (or)
        streamlit run app.py

to install the llm model locally follow:
        1. download ollama from web browser (or) use this link: https://ollama.com/download/windows
        2.open your command prompt and enter:"ollama pull mistral"

if possible run your local llm model in another terminal using:
        ollama run mistral

