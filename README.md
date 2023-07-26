# GFG Olympic Prediction 2023

## Description
GFG Olympic Prediction 2023 is a web application that predicts the medal type (Gold or Silver) for participants in the upcoming 2023 Olympic Games. The prediction is made using a Machine Learning model based on the participant's discipline title, event gender, and participant type.

The Summer Olympic Games, also known as the Games of the Olympiad, are a major international multi-sport event held once every four years. As athletes from all over the world prepare to compete in various sports, this project aims to predict the medal type for each participant, adding an exciting element to the games.

## Features
- Predicts medal type (Gold or Silver) for Olympic participants.
- User-friendly web interface for input and prediction.
- Utilizes Machine Learning techniques, including LSTM model and Tokenization.
- Provides insights into the participant's discipline title, event gender, and participant type.
- Built with Django for backend and Bootstrap for frontend.

## Technologies Used
- Django (Python web framework)
- Bootstrap (Frontend framework)
- Keras (Deep Learning library)
- Pandas (Data manipulation library)
- HTML/CSS/JavaScript

## Dataset
The project uses a labeled dataset containing information about the participant's discipline title, event gender, and participant type. The data is preprocessed, tokenized, and used to train the LSTM model for predicting medal types.

## How to Use
1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Django development server with `python manage.py runserver`.
4. Access the application on `http://127.0.0.1:8000/medal/` in your web browser.
5. Select the discipline title, event gender, and participant type for prediction.
6. Click on the "Predict" button to view the predicted medal type.

## Note
The predictions are based on the model's training data, and the accuracy may vary depending on the quality of the dataset and model training. The application is intended for educational and demonstration purposes only.

## Contribution
Contributions to this project are welcome! Feel free to open issues, suggest improvements, or submit pull requests.

## Credits
This project was developed as part of the Great Learning Academy's Full Stack Development program, and all credits go to the developers who contributed to its creation.

