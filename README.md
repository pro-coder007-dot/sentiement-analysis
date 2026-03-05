# sentiement-analysis
This project is a machine learning-based natural language processing (NLP) application designed to detect and classify human emotions from text. The core of the system is a Support Vector Machine (SVM) model trained with TF-IDF features, which converts raw text into numerical representations that the model can interpret. The model is capable of predicting six primary emotions: sadness 😢, anger 😠, love ❤️, surprise 😲, fear 😨, and joy 😄. Each prediction is accompanied by a confidence score that reflects the probability assigned by the model to the predicted emotion, as well as an emoji to make results more intuitive and visually appealing.

The frontend is built using Streamlit, providing an interactive and user-friendly interface. Users can type or paste text into a text box, click a button to get predictions, and immediately see the predicted emotion along with its confidence. Additionally, the application maintains a history of all predictions in the current session, allowing users to track multiple inputs and compare results over time. This makes it easy to experiment with different sentences and observe how the model responds to subtle differences in wording or emotional tone.

Since the system is ML-based, it may occasionally make mistakes, particularly for sentences that are phrased differently from the training data or contain ambiguous emotional content. High confidence scores do not always guarantee correctness, as the model’s understanding is limited to the patterns it learned during training. Despite this, it provides a strong baseline for emotion detection and can be useful for educational, research, or experimental purposes.

The repository includes:

app.py: The Streamlit application for interacting with the model.

svm_model.pkl: The pre-trained SVM model along with the TF-IDF vectorizer.

requirements.txt: Python dependencies required to run the application.

This project demonstrates how machine learning and NLP techniques can be combined into a practical, interactive web application. It serves as a foundation for further exploration, such as training on larger datasets, adding more emotion categories, integrating with chatbots, or building more advanced emotion recognition systems. By visualizing predictions, confidence, and emojis, the app makes text-based emotion classification accessible and engaging for users, while also highlighting the challenges and limitations of ML-based NLP models.
