import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image

def main():
    st.title("NLP fake news")

    im = Image.open('trumpgame.jpg')
    st.image(im)

    newmodel = joblib.load('fakenews.pkl')
    newmodel

    x1 = st.text_input('Inserisci una notizia')
    pred = newmodel.predict([x1])
    st.write(pred[0])

    st.write('0 stands for Fake, 1 stands for True')

    
        
    
if __name__ == "__main__":
    main()