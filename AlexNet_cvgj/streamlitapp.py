import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = tf.keras.models.load_model('my_model.keras')

def display_model_analysis():
    # Capture the model summary
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    summary_str = "\n".join(summary)
    
    # Display the model summary in Streamlit
    st.text(summary_str)

# Call the function to display the model analysis
display_model_analysis()

# Example: Display model architecture
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
st.image('model.png')

def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction

st.title('CNN Model Analysis and Prediction')

st.header('Model Analysis')
display_model_analysis()

st.header('Make a Prediction')
user_input = st.text_input('Enter input data:')
if st.button('Predict'):
    prediction = make_prediction(user_input)
    st.write('Prediction:', prediction)

