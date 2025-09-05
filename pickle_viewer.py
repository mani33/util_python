# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 11:36:52 2025

@author: ChatGPT
"""

import tkinter as tk
from tkinter import Listbox, Scrollbar, END, filedialog, Button, Label
import os
import pickle
import numpy as np

class PickleFigureLoader:
    def __init__(self, root):
        self.root = root
        self.root.title("Pickle Figure Loader")
        self.current_directory = os.getcwd()

        # Label to show current directory
        self.dir_label = Label(root, text=f"Current Folder: {self.current_directory}", wraplength=500, anchor='w', justify='left')
        self.dir_label.pack(padx=10, pady=(10, 0), fill='x')

        # Browse button
        self.browse_button = Button(root, text="Browse Folder", command=self.browse_folder)
        self.browse_button.pack(padx=10, pady=5)

        # Listbox + Scrollbar
        self.scrollbar = Scrollbar(root)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = Listbox(root, yscrollcommand=self.scrollbar.set, width=60, height=20)
        self.listbox.pack(padx=10, pady=10)

        self.scrollbar.config(command=self.listbox.yview)

        # Load initial files
        self.populate_listbox(self.current_directory)

        # Double-click to open
        self.listbox.bind("<Double-1>", self.load_figure)

    def browse_folder(self):
        selected_folder = filedialog.askdirectory(initialdir=self.current_directory)
        if selected_folder:
            self.current_directory = selected_folder
            self.dir_label.config(text=f"Current Folder: {self.current_directory}")
            self.populate_listbox(self.current_directory)

    def populate_listbox(self, folder):
        self.listbox.delete(0, END)
        try:
            files = [f for f in os.listdir(folder) if f.endswith('.pickle')]
            files = np.sort(files)
            for file in files:
                self.listbox.insert(END, file)
        except Exception as e:
            print(f"Error reading directory {folder}: {e}")

    def load_figure(self, event):
        selected_index = self.listbox.curselection()
        if not selected_index:
            return

        filename = self.listbox.get(selected_index)
        filepath = os.path.join(self.current_directory, filename)

        try:
            with open(filepath, 'rb') as f:
                fig = pickle.load(f)
                fig.show()
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PickleFigureLoader(root)
    root.mainloop()
