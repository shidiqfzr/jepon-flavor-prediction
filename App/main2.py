import customtkinter as ctk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import joblib
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load the trained KNN model and scaler
        self.knn = joblib.load('knn_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

        # Initialize the Raspberry Pi camera
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.raw_capture = PiRGBArray(self.camera, size=(640, 480))
        time.sleep(0.1)  # Allow the camera to warm up

        # Create left frame for buttons
        self.left_frame = ctk.CTkFrame(window, width=300)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Create right frame for video feed
        self.right_frame = ctk.CTkFrame(window)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid weights for resizing
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)

        # Create a canvas that can fit the above video source size
        self.canvas = ctk.CTkCanvas(self.right_frame)
        self.canvas.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Frame to display the prediction results
        self.results_frame = ctk.CTkFrame(self.left_frame, width=300)
        self.results_frame.pack(pady=10, fill=ctk.BOTH, expand=True)

        # Label for displaying the captured RGB value
        self.label_rgb = ctk.CTkLabel(self.results_frame, text="Captured RGB Value: ", anchor="w")
        self.label_rgb.pack(pady=5, fill=ctk.X)

        # Label for displaying the predicted flavor
        self.label_flavor = ctk.CTkLabel(self.results_frame, text="Predicted Flavor: ", anchor="w")
        self.label_flavor.pack(pady=5, fill=ctk.X)
        
        # Button that lets the user capture a frame
        self.btn_snapshot = ctk.CTkButton(self.left_frame, text="Predict", command=self.capture_image)
        self.btn_snapshot.pack(pady=10, fill=ctk.X)

        # Button to quit the application
        self.btn_quit = ctk.CTkButton(self.left_frame, text="Quit", command=self.window.quit)
        self.btn_quit.pack(pady=10, fill=ctk.X)

        # Start the video stream
        self.update()

        self.window.mainloop()

    def capture_image(self):
        self.camera.capture(self.raw_capture, format="bgr")
        frame = self.raw_capture.array

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate the average color of the image
        avg_color_per_row = frame_rgb.mean(axis=0)
        avg_color = avg_color_per_row.mean(axis=0)
        rgb_value = [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]

        # Scale the RGB values
        rgb_value_scaled = self.scaler.transform([rgb_value])

        # Predict the flavor using the KNN model
        flavor = self.knn.predict(rgb_value_scaled)[0]

        # Update the labels with the prediction results
        self.label_rgb.configure(text=f"Captured RGB Value: {rgb_value}")
        self.label_flavor.configure(text=f"Predicted Flavor: {flavor}")

        messagebox.showinfo("Predicted Flavor", f"Captured RGB Value: {rgb_value}\nPredicted Flavor: {flavor}")

        self.raw_capture.truncate(0)

    def update(self):
        self.camera.capture(self.raw_capture, format="bgr")
        frame = self.raw_capture.array

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the frame to fit the canvas size
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))

        # Convert the image format from OpenCV BGR to PIL RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=ctk.NW)

        self.raw_capture.truncate(0)
        self.window.after(10, self.update)

    def __del__(self):
        # Release the video source when the object is destroyed
        self.camera.close()

# Create a window and pass it to the Application object
window = ctk.CTk()
ctk.set_appearance_mode("Dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"
CameraApp(window, "Fruit Flavor Detection")
