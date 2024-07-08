import cv2
import numpy as np
import joblib
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

class TestMain:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 1  # Use 0 for primary camera, or change to 1 for external camera

        # Load the trained KNN model and scaler
        self.knn = joblib.load('Model/knn_model.pkl')
        self.scaler = joblib.load('Model/scaler.pkl')

        # Open video source (by default this will try to open the computer webcam)
        self.vid = cv2.VideoCapture(self.video_source)

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
        ret, frame = self.vid.read()
        if ret:
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the image from BGR to RGB
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
            rgb_value = [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]

            # Scale the RGB values
            rgb_value_scaled = self.scaler.transform([rgb_value])

            # Predict the flavor using the KNN model
            flavor = self.knn.predict(rgb_value_scaled)[0]

            # Update the labels with the prediction results
            self.label_rgb.configure(text=f"Captured RGB Value: {rgb_value}")
            self.label_flavor.configure(text=f"Predicted Flavor: {flavor}")

            messagebox.showinfo("Predicted Flavor", f"Captured RGB Value: {rgb_value}\nPredicted Flavor: {flavor}")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
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


            # Extract the region of interest using the mask
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            masked_frame = cv2.resize(masked_frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))

            # Convert the image format from OpenCV BGR to PIL RGB
            frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=ctk.NW)

        self.window.after(10, self.update)

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()