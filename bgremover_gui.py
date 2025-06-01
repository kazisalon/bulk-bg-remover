import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from rembg import remove
from PIL import Image
import glob

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class BGRemoverApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Katmatic Remover")
        self.geometry("600x600")
        self.resizable(False, False)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="Ready to process images")
        self.progress = tk.DoubleVar(value=0)

        self.create_widgets()

    def create_widgets(self):
        # Title and Instructions
        title_frame = ctk.CTkFrame(self)
        title_frame.pack(pady=(20, 10), padx=20, fill="x")
        
        ctk.CTkLabel(
            title_frame, 
            text="Katmatic Background Remover", 
            font=("Helvetica", 24, "bold")
        ).pack(pady=(10, 5))
        
        ctk.CTkLabel(
            title_frame,
            text="Remove backgrounds from multiple images at once",
            font=("Helvetica", 12)
        ).pack(pady=(0, 10))

        # Instructions
        instructions = """
        How to use:
        1. Select input folder containing your images
        2. Choose output folder for processed images
        3. Click 'Start' to begin processing
        4. Wait for completion
        
        Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP
        """
        
        instruction_frame = ctk.CTkFrame(self)
        instruction_frame.pack(pady=(0, 20), padx=20, fill="x")
        
        ctk.CTkLabel(
            instruction_frame,
            text=instructions,
            justify="left",
            font=("Helvetica", 11)
        ).pack(pady=10, padx=10)

        # Input Folder Selection
        input_label = ctk.CTkLabel(self, text="Input Folder:", font=("Helvetica", 12, "bold"))
        input_label.pack(pady=(0, 5))
        
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(padx=20, fill="x")
        
        input_entry = ctk.CTkEntry(input_frame, textvariable=self.input_dir, width=320)
        input_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        input_button = ctk.CTkButton(
            input_frame, 
            text="Browse", 
            command=self.browse_input,
            width=100
        )
        input_button.pack(side="left")
        
        # Add tooltip for input
        self.create_tooltip(input_entry, "Select folder containing images to process")

        # Output Folder Selection
        output_label = ctk.CTkLabel(self, text="Output Folder:", font=("Helvetica", 12, "bold"))
        output_label.pack(pady=(20, 5))
        
        output_frame = ctk.CTkFrame(self)
        output_frame.pack(padx=20, fill="x")
        
        output_entry = ctk.CTkEntry(output_frame, textvariable=self.output_dir, width=320)
        output_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        output_button = ctk.CTkButton(
            output_frame, 
            text="Browse", 
            command=self.browse_output,
            width=100
        )
        output_button.pack(side="left")
        
        # Add tooltip for output
        self.create_tooltip(output_entry, "Select folder where processed images will be saved")

        # Start Button
        start_button = ctk.CTkButton(
            self, 
            text="Start Processing", 
            command=self.start_processing,
            width=200,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        start_button.pack(pady=30)
        
        # Add tooltip for start button
        self.create_tooltip(start_button, "Click to begin background removal process")

        # Progress Section
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(pady=(0, 10), padx=20, fill="x")
        
        ctk.CTkProgressBar(progress_frame, variable=self.progress, width=400).pack(pady=(10, 5))
        ctk.CTkLabel(
            progress_frame, 
            textvariable=self.status,
            font=("Helvetica", 11)
        ).pack(pady=(0, 10))

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                tooltip, 
                text=text, 
                justify='left',
                background="#ffffe0", 
                relief='solid', 
                borderwidth=1,
                font=("Helvetica", "9", "normal")
            )
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
        
        widget.bind('<Enter>', show_tooltip)

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_dir.set(folder)
            self.status.set("Input folder selected")

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir.set(folder)
            self.status.set("Output folder selected")

    def start_processing(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        
        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return
            
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", "Input folder does not exist.")
            return
            
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create output folder: {str(e)}")
                return
            
        self.status.set("Processing images...")
        self.progress.set(0)
        threading.Thread(target=self.process_images, args=(input_dir, output_dir), daemon=True).start()

    def process_images(self, input_dir, output_dir):
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        files = [f for f in glob.glob(os.path.join(input_dir, "*")) if f.lower().endswith(image_extensions)]
        total = len(files)
        
        if total == 0:
            self.status.set("No images found in input folder.")
            return
            
        for idx, file_path in enumerate(files, 1):
            try:
                with Image.open(file_path) as img:
                    if img is None:
                        raise ValueError("Failed to load the image.")
                    # Remove background
                    out = remove(img)
                    if out is None:
                        raise ValueError("Background removal failed for this image.")
                    
                    # Create a white background image
                    white_bg = Image.new('RGBA', out.size, (255, 255, 255, 255))
                    # Composite the transparent image over white background
                    final_image = Image.alpha_composite(white_bg, out)
                    
                    base = os.path.basename(file_path)
                    out_path = os.path.join(output_dir, base)
                    if not os.access(os.path.dirname(out_path), os.W_OK):
                        raise PermissionError("No permission to save the output file.")
                    final_image.save(out_path)
                self.status.set(f"Processed {idx}/{total}: {os.path.basename(file_path)}")
            except Exception as e:
                self.status.set(f"Error: {os.path.basename(file_path)} - {str(e)}")
            self.progress.set(idx / total)
            
        self.status.set("Processing complete! Check the output folder for results.")
        self.progress.set(1)
        messagebox.showinfo("Success", "Background removal completed successfully!")

if __name__ == "__main__":
    app = BGRemoverApp()
    app.mainloop()