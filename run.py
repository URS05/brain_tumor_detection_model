import os
import tkinter as tk
from tkinter import filedialog, messagebox
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Brain Tumor Recognition')
        self.root.geometry('800x600')
        self.model_path = 'C:/drive D/experiment/brain tumor/resources/converted_keras/keras_model.h5'
        self.labels_file = 'C:/drive D/experiment/brain tumor/resources/converted_keras/labels.txt'
        self.model = load_model(self.model_path)
        self.labels = self.load_labels()
        self.init_gui()

    def load_labels(self):
        with open(self.labels_file, 'r') as f:
            return [line.strip().split(' ')[1] for line in f if line.strip()]

    def init_gui(self):
        self.frame = tk.Frame(self.root, bg='lightgrey')
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.upload_btn = tk.Button(self.frame, text='Upload Images', command=self.upload_images)
        self.upload_btn.pack(pady=20)

        self.result_text = tk.Text(self.frame, wrap=tk.WORD, height=20)
        self.result_text.pack(padx=20)

        self.save_btn = tk.Button(self.frame, text='Save Results', command=self.save_results)
        self.save_btn.pack(pady=10)

    def upload_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[('JPEG files', '*.jpg')])
        if not file_paths:
            return

        self.result_text.delete(1.0, tk.END)

        for path in file_paths:
            img = self.process_image(path)
            prediction, confidence = self.predict(img)
            result = f'{os.path.basename(path)}: {prediction} ({confidence:.2f}%)\n'
            self.result_text.insert(tk.END, result)

    def process_image(self, img_path):
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_array):
        predictions = self.model.predict(img_array)
        label_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        return self.labels[label_index], confidence

    def save_results(self):
        results = self.result_text.get(1.0, tk.END).strip()
        if not results:
            messagebox.showwarning('No Results', 'No results to save.')
            return
        with open('results.txt', 'w') as file:
            file.write(results)
        messagebox.showinfo('Saved', 'Results have been saved to results.txt')

if __name__ == '__main__':
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
