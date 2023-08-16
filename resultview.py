import os
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, filedialog

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.current_image = None
        self.image_label = Label(root)
        self.image_label.pack()

        self.next_button = Button(root, text="Next Image", command=self.next_image)
        self.next_button.place(x=50, y=270)  # 버튼 위치 고정

        self.prev_button = Button(root, text="Previous Image", command=self.prev_image)
        self.prev_button.place(x=180, y=270)  # 버튼 위치 고정

        self.image_files = self.get_image_files()
        self.current_index = 0
        self.load_current_image()

    def get_image_files(self):
        folder_path = r".\results\SRF_4"  # 원하는 폴더 경로
        image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if any(file.lower().endswith(ext) for ext in image_extensions)]
        return image_files

    def load_current_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_index]
            self.current_image = Image.open(image_path)
            self.display_image()

    def display_image(self):
        if self.current_image:
            self.current_image.thumbnail((250, 250))
            photo = ImageTk.PhotoImage(self.current_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_current_image()

    def prev_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_current_image()

if __name__ == "__main__":
    root = Tk()
    app = ImageViewer(root)
    root.geometry("300x380")
    root.resizable(False, False)
    root.mainloop()
