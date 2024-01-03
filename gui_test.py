import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Loading the pre-trained model for Age and Gender Detection
model = load_model('Age_Gender_Detection.h5')

# GUI setup
top = tk.Tk()
top.geometry("800x600")
top.title("Age & Gender Detector")
top.configure(background='#CDCDCD')

# Labels and image display setup
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label2 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def Detect(file_path):
    global label1, label2
    # Preprocess image for prediction
    image = Image.open(file_path)
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
    image = np.resize(image, (48, 48, 3))

    # Make prediction using the model
    pred = model.predict(np.array([image]))
    age = int(np.round(pred[1][0]))
    sex = "Male" if int(np.round(pred[0][0])) == 0 else "Female"

    # Display prediction results
    label1.configure(foreground='#011638', text="Predicted Age: " + str(age))
    label2.configure(foreground='#011638', text="Predicted Gender: " + sex)

def show_detect_button(file_path):
    Detect_b = Button(top, text="Make Prediction", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background='#364156', foreground='White', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        # Display selected image
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text="")
        label2.configure(text="")
        show_detect_button(file_path)
    except Exception as e:
        print(e)

# Upload button setup
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='White', font=('arial', 10, 'bold'))

upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)

# Packing labels and heading
label1.pack(side='bottom', pady=50)
label2.pack(side='bottom', pady=50)
heading = Label(top, text="Age & Gender Detector", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground='#364156')
heading.pack()

# Main loop for GUI
top.mainloop()
