import customtkinter as ctk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import joblib

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 1  # Use 0 for primary camera, or change to 1 for external camera

        # Load the trained KNN model and scaler
        self.knn = joblib.load('knn_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

        # Open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(self.video_source)

        # Create left frame for buttons
        self.left_frame = ctk.CTkFrame(window, width=200)
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

        # Button that lets the user capture a frame
        self.btn_snapshot = ctk.CTkButton(self.left_frame, text="Capture", command=self.capture_image)
        self.btn_snapshot.pack(pady=10, fill=ctk.X)

        # Button to quit the application
        self.btn_quit = ctk.CTkButton(self.left_frame, text="Quit", command=self.window.quit)
        self.btn_quit.pack(pady=10, fill=ctk.X)

        # Start the video stream
        self.update()

        self.window.mainloop()

    def capture_image(self):
        ret, frame = self.vid.read()
        if ret:
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

            messagebox.showinfo("Predicted Flavor", f"Captured RGB Value: {rgb_value}\nPredicted Flavor: {flavor}")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Resize the frame to fit the canvas size
            frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))

            # Convert the image format from OpenCV BGR to PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=ctk.NW)

        self.window.after(10, self.update)

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
window = ctk.CTk()
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"
CameraApp(window, "Fruit Flavor Detection")
