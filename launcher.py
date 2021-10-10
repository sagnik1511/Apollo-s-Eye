import os
from tkinter import *
from PIL import Image, ImageTk

def launch():
    os.system("streamlit run app.py")


root = Tk()

root.title('App launcher')
root.geometry("380x280")
root.config(bg = 'lightskyblue')

load= Image.open("static/launcher_logo.jpg")
render = ImageTk.PhotoImage(load)
img = Label(root, image=render)
img.pack()

launcher_button = Button( root , text = 'Launch' ,
                             height = 1 , width = 5 ,
                             command = launch ,
                             bg = 'royalblue' , bd = 4)

launcher_button.pack()
root.mainloop()