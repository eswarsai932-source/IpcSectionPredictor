go to this path:
    cd "C:\Users\saies\Downloads\IPC Section Project"

if no virtual environment is created the use:
        python -m venv venv

Activate environment:
        venv\Scripts\activate

Install required packages:
        pip install streamlit pandas numpy scikit-learn joblib sentence-transformers torch
        pip install scipy

best to install if any this is not installed:
    pip install -r requirements.txt

TRAIN THE MODEL:
        python train_model.py

to run the application:
        python -m streamlit run app.py
        streamlit run app.py

if possible run your model in another terminal:
        ollama run mistral

