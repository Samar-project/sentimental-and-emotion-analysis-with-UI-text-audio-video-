# line 74-80,90
import matplotlib.pyplot as plt
import preprocessor as p
import numpy as np
import pandas as pd
import emoji
import pickle
import speech_recognition as sr
import tkinter as tk
from keras.utils.data_utils import pad_sequences
from keras.models import model_from_json

# Misspelled data
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

a_file = open("./data/text/aspell.pkl", "rb")
miss_corr = pickle.load(a_file)


def misspelled_correction(val):
    for x in val.split():
        if x in miss_corr.keys():
            val = val.replace(x, miss_corr[x])
    return val


# Contractions
contractions = pd.read_csv("./data/text/contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))


def cont_to_meaning(val):
    for x in val.split():
        if x in cont_dic.keys():
            val = val.replace(x, cont_dic[x])
    return val


# Punctuations and emojis

def punctuation(val):
    punctuations = r'''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
    return val

def clean_text(val):
    val = misspelled_correction(val)
    val = cont_to_meaning(val)
    val = p.clean(val)
    val = ' '.join(punctuation(emoji.demojize(val)).split())

    return val


# assigning id to each sentiment
sent_to_id = {"empty": 0, "sadness": 1, "enthusiasm": 2, "neutral": 3, "worry": 4,
              "surprise": 5, "love": 6, "fun": 7, "hate": 8, "happiness": 9, "boredom": 10,
              "relief": 11, "anger": 12}

token = pickle.load(open("./data/text/token.pkl", "rb"))
max_len = 160


# def speech_to_text():
#     r = sr.Recognizer()
#     file = glob.glob('./audio/*')[0]
#     with sr.AudioFile(file) as source:
#         audio = r.record(source)
#         text = r.recognize_google(audio)
#     return text

def get_text_sentiment(Text):
    json_file = open('./data/text/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./data/text/model.h5")

    if Text:
        text = Text
    # else:
    # text = speech_to_text()
    else:
        text = "happy to see you again and how is your mother?"

    text = clean_text(text)
    # tokenize
    twt = token.texts_to_sequences([text])
    twt = pad_sequences(twt, maxlen=max_len, dtype='int32')
    #  twt = sequence.pad_sequences(twt, maxlen=max_len, dtype='int32')
    sentiment = model.predict(twt, batch_size=1, verbose=2)
    sent = np.round(np.dot(sentiment, 100).tolist(), 0)[0]
    result = pd.DataFrame([sent_to_id.keys(), sent]).T
    result.columns = ["sentiment", "percentage"]
    result = result[result.percentage != 0]

    return result


def displayPlot(plots_frame, Result):
    # Display pie chart
    sentiment_labels = Result["sentiment"]
    percentage_values = Result["percentage"]

    for widget in plots_frame.winfo_children():
        widget.destroy()

    # Plot pie chart
    fig_pie = plt.figure(figsize=(4, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf", "#9b59b6", "#34495e", "#e74c3c"]
    plt.pie(percentage_values, labels=sentiment_labels, colors=colors, autopct='%1.1f%%')
    plt.title("Sentiment Distribution")
    # plt.show()

    # Create a canvas to display the pie chart
    canvas_pie = FigureCanvasTkAgg(fig_pie, master=plots_frame)
    canvas_pie.draw()
    canvas_pie.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    # Create a toolbar for the pie chart
    toolbar_pie = NavigationToolbar2Tk(canvas_pie, plots_frame)
    toolbar_pie.update()
    toolbar_pie.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
