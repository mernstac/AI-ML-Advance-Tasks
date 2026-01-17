# AI-ML-Advance-Tasks

**Task 1: News Classification Using Transformer Models**

**Objective**

The objective of this task is to build an automated system that classifies news articles into predefined categories using a Transformer-based deep learning model. This task demonstrates the application of transfer learning in Natural Language Processing.

**Methodology**

The dataset was first cleaned to remove missing and noisy records. Text data was tokenized using a pre-trained BERT tokenizer, and labels were encoded into numerical form. A pre-trained bert-base-uncased model was fine-tuned by adding a classification head for multi-class prediction.

The model was trained using the AdamW optimizer and cross-entropy loss. Training was performed in mini-batches using the HuggingFace Trainer API to ensure efficient optimization.

**Evaluation**

Model performance was evaluated using accuracy and F1-score to ensure both overall correctness and balanced class prediction.

Metric	Value
Accuracy	92.05%
F1-Score	92.14%

**Key Observations**

The Transformer-based model achieved strong generalization performance, confirming the effectiveness of transfer learning for text classification tasks.


**Task 2: End-to-End Customer Churn Prediction Pipeline**

**Objective**

The objective of this task is to develop a complete and reusable machine learning pipeline that predicts customer churn using structured business data. The system is designed for direct deployment in production environments.

**Methodology**

The **IBM Telco Customer Churn** dataset was used. Missing values were handled, categorical features were encoded, and numerical features were scaled. A ColumnTransformer was implemented to manage preprocessing steps.

A unified Scikit-learn Pipeline was constructed that combines preprocessing and model training. Logistic Regression and Random Forest classifiers were trained, and hyperparameter tuning was performed using GridSearchCV.

The final trained pipeline was saved using joblib for future deployment without retraining.

**Evaluation**

Metric	Value
Accuracy	80.55%
F1-Score (Churn Class)	87%

**Key Observations**

The pipeline architecture ensures consistent preprocessing and model inference, making the system suitable for real-world production use.


**Task 3: Multimodal Housing Price Prediction (Images + Tabular Data)**

**Objective**

The objective of this task is to predict house prices by combining structured numerical features with unstructured image data using a multimodal deep learning model.

**Methodology**

House images were resized and normalized, while numerical housing attributes were standardized. A Convolutional Neural Network (CNN) was used to extract visual features from images. A dense neural network processed tabular data.

Both feature sets were merged using a concatenation layer and passed through fully connected layers for final price prediction.

The model was trained using the Adam optimizer and Mean Squared Error loss function.

**Evaluation**
Metric	Value
MAE	30,209
RMSE	36,980

**Key Observations**

The multimodal approach improved prediction accuracy by learning complementary patterns from both visual and numerical data sources.

**Task 4: Context-Aware Chatbot Using Retrieval-Augmented Generation (RAG)**

**Objective**

The objective of this task is to build a conversational AI system that can answer user queries based on custom documents while maintaining conversational context. This system reduces hallucinations by grounding responses in external knowledge.

**Methodology**

PDF documents were loaded and split into semantic text chunks. SentenceTransformer models were used to generate vector embeddings. These embeddings were stored in a FAISS vector database for fast similarity search.

For each user query, the most relevant document chunks are retrieved and passed to a language model, forming a Retrieval-Augmented Generation (RAG) pipeline.

The chatbot interface was deployed using Streamlit for real-time interaction.

**Evaluation**

The system was evaluated based on answer relevance, factual grounding, and multi-turn conversational consistency.

**Key Observations**

RAG significantly improves chatbot reliability by combining information retrieval with generative modeling, reducing incorrect or fabricated responses.


**Technologies Used**

Python, Scikit-learn, PyTorch, TensorFlow, HuggingFace Transformers, FAISS, SentenceTransformers, Streamlit, Gradio.
