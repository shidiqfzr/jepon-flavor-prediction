import customtkinter as ctk
from Utilities.test_main import TestMain


if __name__ == "__main__":
    window = ctk.CTk()
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    TestMain(window, "Fruit Flavor Detection")