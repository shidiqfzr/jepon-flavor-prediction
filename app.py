import customtkinter as ctk
from Utilities.flavorPrediction import flavorPrediction
from Utilities.camera_app import CameraApp

if __name__ == "__main__":
    window = ctk.CTk()
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    CameraApp(window, "Fruit Flavor Detection")
    # flavorPrediction(window, "Fruit Flavor Detection")