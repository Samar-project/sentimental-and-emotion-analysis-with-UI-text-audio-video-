import tkinter as tk
from tkinter import filedialog

from PIL import ImageTk, Image

import textAnalysis
import audioAnalysis
import videoAnalysis
from textAnalysis import displayPlot


def textAction():
    Result = textAnalysis.get_text_sentiment(INput1.get())
    output_text.insert(tk.END, " Text Result: \n" + str(Result) + "\n\n")
    displayPlot(plots_frame, Result)


def audioAction(file_path=''):
    Result = audioAnalysis.runn(file_path)
    output_text.insert(tk.END, " Audio Result: \n" + Result + "\n\n")


def audioSelection(event):
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    audioAction(file_path)


def videoAction(file_path=''):
    Result = videoAnalysis.runn(file_path)
    output_text.insert(tk.END, "\n\nVideo Result: \n" + str(Result) + "\n\n")


def videoSelection(event):
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    videoAction(file_path)


# Create the main window
window = tk.Tk()
window.title("sentEmoProject")
winheight = 650
winwidth = 850
# Set the window size
window.geometry(f"{winwidth}x{winheight}")

# Load and resize the background image
background_image = Image.open("bgg.png")

# displaying background image
# background_image = background_image.resize((winwidth, winheight), Image.ANTIALIAS)
# bg = PhotoImage(file = "Your_image.png")
background_photo = ImageTk.PhotoImage(background_image)

# Create a label to display the background image
background_label = tk.Label(window, image=background_photo)
window.wm_state('zoomed')
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Input label and text entry
label_input1 = tk.Label(window, text="Text Input :")
label_input1.place(x=30, y=20)
INput1 = tk.Entry(window, width=80)
INput1.place(x=110, y=20)

label_input2 = tk.Label(window, text="click to add Audio file :")
label_input2.place(x=180, y=85)
label_input2.bind("<Button-1>", audioSelection)

label_input3 = tk.Label(window, text="click to add Video file :")
label_input3.place(x=180, y=115)
label_input3.bind("<Button-1>", videoSelection)

# Output label and text box
label_output = tk.Label(window, text="OUTPUT :")
label_output.place(x=80, y=160)
label_output.config(font=("Arial", 14))
output_text = tk.Text(window, height=19, width=30)
output_text.config(font=("Arial", 14))
output_text.configure(bg="lightblue", fg="black")
output_text.place(x=30, y=190)

# for graph plotting
plots_frame = tk.Frame(window)
plots_frame.place(x=380, y=190)

# Create buttons
button1 = tk.Button(window, text="Text Analyse", command=textAction)
button1.place(x=30, y=50)
button1.configure(bg="red", fg="white")
button2 = tk.Button(window, text="Audio Analyse", command=audioAction)
button2.place(x=30, y=80)
button2.configure(bg="red", fg="white")
button3 = tk.Button(window, text="Video Analyse", command=videoAction)
button3.place(x=30, y=110)
button3.configure(bg="red", fg="white")

window.mainloop()
