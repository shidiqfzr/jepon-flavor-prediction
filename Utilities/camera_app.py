import os
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox
from picamera2 import Picamera2


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.counter = 0

        # Relative path dari folder root
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Load the trained KNN model and scaler
        path = self.current_directory.replace('Utilities', 'Model')
        self.knn = joblib.load(os.path.join(path, 'knn_model.pkl'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))

        # Initialize the Raspberry Pi camera
        self.camera = Picamera2()
        self.camera.preview_configuration.main.size=(640,480)
        self.camera.preview_configuration.main.format="RGB888"
        self.camera.start()
        time.sleep(0.1)

        # Create left frame for buttons
        self.left_frame = ctk.CTkFrame(window, width=300)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Create right frame for video feed
        self.right_frame = ctk.CTkFrame(window)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for resizing
        self.window.grid_columnconfigure(0, weight=0)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # Create a canvas that can fit the above video source size
        self.canvas = ctk.CTkCanvas(self.right_frame)
        self.canvas.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Frame to display the prediction results
        self.results_frame = ctk.CTkFrame(self.left_frame, width=1000)
        self.results_frame.pack(pady=10, fill=ctk.BOTH, expand=True)

        # Label for displaying the captured Lab value
        self.label_lab_value = ctk.CTkLabel(self.results_frame, text="Nilai Lab: ", anchor="w")
        self.label_lab_value.pack(pady=5, fill=ctk.X)

        # Label for displaying the captured pH value
        self.label_ph_value = ctk.CTkLabel(self.results_frame, text="Nilai pH: ", anchor="w")
        self.label_ph_value.pack(pady=5, fill=ctk.X)

        # Label for displaying the predicted flavor
        self.label_flavor = ctk.CTkLabel(self.results_frame, text="Prediksi Rasa: ", anchor="w")
        self.label_flavor.pack(pady=5, fill=ctk.X)
        
        # Button that lets the user capture a frame
        self.btn_snapshot = ctk.CTkButton(self.left_frame, text="Prediksi", command=self.capture_image)
        self.btn_snapshot.pack(pady=10, fill=ctk.X)

        # Button to quit the application
        self.btn_quit = ctk.CTkButton(self.left_frame, text="Keluar", command=self.window.quit)
        self.btn_quit.pack(pady=10, fill=ctk.X)

        # Start the video stream
        self.update()

        self.window.mainloop()

    def capture_image(self):
        frame = self.camera.capture_array()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Define the center and radius of the circle
        center = (width // 2, height // 2)
        radius = min(center) // 2

        # Create a mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw a filled circle on the mask
        cv2.circle(mask, center, radius, 255, -1)

        # Convert the frame to Lab
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Extract the region of interest in Lab using the mask
        masked_lab = cv2.bitwise_and(frame_lab, frame_lab, mask=mask)

        # Calculate the average Lab values of the region inside the circle
        avg_lab = cv2.mean(masked_lab, mask=mask)[:3]
        lab_value = [int(avg_lab[0]), int(avg_lab[1]), int(avg_lab[2])]

        # Scale the values
        lab_value_scaled = self.scaler.transform([lab_value])

        # Predict the flavor using the KNN model
        flavor = self.knn.predict(lab_value_scaled)[0]

        # Update the labels with the prediction results
        self.label_lab_value.configure(text=f"Nilai Lab: {lab_value}")
        self.label_ph_value.configure(text=f"Nilai pH: 4.28")  # Static value for demonstration
        self.label_flavor.configure(text=f"Prediksi Rasa: {flavor}")

        messagebox.showinfo("Prediksi Rasa", f"Nilai Lab: {lab_value} \nNilai pH: 4.28 \nPrediksi Rasa: {flavor}")

    def update(self):
        frame = self.camera.capture_array()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Define the center and radius of the circle
        center = (width // 2, height // 2)
        radius = min(center) // 2

        # Create a mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw a circle on the frame
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)

        # Extract the region of interest using the mask
        # masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        # masked_frame = cv2.resize(masked_frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))

        # Resize the frame to fit the canvas size
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        
        # Convert the image format from OpenCV BGR to PIL HSV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=ctk.NW)
        self.window.after(10, self.update)

    def __del__(self):
        self.camera.stop()

    def save_pandas_dataframe(self, filename='basisData.csv') -> None:
        """Menyimpan data sensor ke csv format.

        Args:
            filename (str): nama file (default: basisData.csv)
        """
        frame = self.camera.capture_array()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to HSV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the dimensions of the frame
        height, width, _ = frame_rgb.shape

        # Define the center and radius of the circle
        center = (width // 2, height // 2)
        radius = min(center) // 2

        # Create a mask with the same dimensions as the frame
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw a filled circle on the mask
        cv2.circle(mask, center, radius, (255,255,255), -1)

        # Calculate the average color of the region inside the circle
        avg_color = cv2.mean(frame_rgb, mask=mask)[:3]
        hsv_value = [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]
        self.label_rgb.configure(text=f"Captured RGB Value: {hsv_value}")

        data = {'id': self.counter, 'h':int(avg_color[0]), 's':int(avg_color[1]), 'v':int(avg_color[2])}
        df = pd.DataFrame([data])

        path = self.current_directory.replace('Utilities', 'Datasets')
        full_path = os.path.join(path, filename)
        
        if not os.path.isfile(full_path):
            df.to_csv(full_path, mode='w', header=True, index=False)
        else:
            df.to_csv(full_path, mode='a', header=False, index=False)
        
        self.counter+=1