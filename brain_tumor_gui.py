import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from tkinterdnd2 import TkinterDnD, DND_FILES
import pandas as pd
from datetime import datetime
import json
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from keras.models import load_model

class BrainTumorDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.load_model_and_labels()
        self.setup_styles()
        self.create_widgets()
        self.history = []
        self.load_history()
        
    def setup_window(self):
        self.root.title("Brain Tumor Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
        
    def load_model_and_labels(self):
        try:
            self.model_path = r"C:\drive D\experiment\brain tumor\resources\converted_keras\keras_model.h5"
            self.labels_file = r"C:\drive D\experiment\brain tumor\resources\converted_keras\labels.txt"
            
            # Load model
            self.model = load_model(self.model_path, compile=False)
            
            # Load labels
            self.labels = []
            with open(self.labels_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(' ', 1)
                        if len(parts) > 1:
                            self.labels.append(parts[1])
                        
            print(f"Model loaded successfully. Labels: {self.labels}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or labels: {str(e)}")
            self.root.destroy()
            
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'light': '#ecf0f1',
            'dark': '#34495e'
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', 
                           font=('Arial', 16, 'bold'),
                           foreground=self.colors['primary'])
        
        self.style.configure('Heading.TLabel',
                           font=('Arial', 12, 'bold'),
                           foreground=self.colors['dark'])
        
        self.style.configure('Custom.TButton',
                           font=('Arial', 10),
                           padding=10)
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🧠 Brain Tumor Detection System", 
                              style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Detection tab
        self.detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detection_frame, text="Detection")
        
        # History tab
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="History")
        
        self.create_detection_tab()
        self.create_history_tab()
        
    def create_detection_tab(self):
        # Left panel for controls
        left_panel = ttk.Frame(self.detection_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Upload section
        upload_frame = ttk.LabelFrame(left_panel, text="Upload Images", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.upload_btn = ttk.Button(upload_frame, text="📁 Select Images", 
                                   command=self.upload_images, style='Custom.TButton')
        self.upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Drag and drop area
        self.drop_frame = tk.Frame(upload_frame, bg='#ddd', relief=tk.SUNKEN, bd=2)
        self.drop_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.drop_label = tk.Label(self.drop_frame, text="📥 Drag & Drop Images Here\n(JPG format only)", 
                                 bg='#ddd', font=('Arial', 10), height=4)
        self.drop_label.pack(fill=tk.BOTH, expand=True)
        
        # Enable drag and drop
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(upload_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(upload_frame, text="Ready", foreground=self.colors['success'])
        self.status_label.pack()
        
        # Action buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(action_frame, text="🔍 Analyze", 
                                    command=self.analyze_images, style='Custom.TButton')
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.clear_btn = ttk.Button(action_frame, text="🗑️ Clear Results", 
                                  command=self.clear_results, style='Custom.TButton')
        self.clear_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.save_btn = ttk.Button(action_frame, text="💾 Save Results", 
                                 command=self.save_results, style='Custom.TButton')
        self.save_btn.pack(fill=tk.X)
        
        # Right panel for results
        right_panel = ttk.Frame(self.detection_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results section
        results_frame = ttk.LabelFrame(right_panel, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        self.results_text = ScrolledText(results_frame, wrap=tk.WORD, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with welcome message
        welcome_text = """🧠 Brain Tumor Detection System Ready!

This system can detect the following conditions:
• Glioma - A type of brain tumor
• Meningioma - A type of brain tumor  
• Pituitary Tumor - A type of brain tumor
• No Tumor - Healthy brain tissue

Instructions:
1. Upload JPG images using the 'Select Images' button or drag & drop
2. Click 'Analyze' to process the images
3. View results with confidence percentages
4. Save results for future reference

Model Status: ✅ Loaded and Ready
"""
        self.results_text.insert(tk.END, welcome_text)
        
        # Store selected files
        self.selected_files = []
        
    def create_history_tab(self):
        # History controls
        history_controls = ttk.Frame(self.history_frame)
        history_controls.pack(fill=tk.X, pady=(0, 10))
        
        self.refresh_history_btn = ttk.Button(history_controls, text="🔄 Refresh", 
                                            command=self.refresh_history)
        self.refresh_history_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_history_btn = ttk.Button(history_controls, text="📊 Export to CSV", 
                                           command=self.export_history)
        self.export_history_btn.pack(side=tk.LEFT)
        
        # History display
        self.history_text = ScrolledText(self.history_frame, wrap=tk.WORD, font=('Courier', 10))
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
    def upload_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Brain Scan Images",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.selected_files = list(file_paths)
            self.update_status(f"Selected {len(self.selected_files)} images")
            self.show_selected_files()
            
    def on_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        jpg_files = [f for f in files if f.lower().endswith('.jpg')]
        
        if jpg_files:
            self.selected_files = jpg_files
            self.update_status(f"Dropped {len(jpg_files)} images")
            self.show_selected_files()
        else:
            messagebox.showwarning("Invalid Files", "Please drop only JPG files.")
            
    def show_selected_files(self):
        if self.selected_files:
            file_list = "\n".join([f"📄 {os.path.basename(f)}" for f in self.selected_files])
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Selected Files ({len(self.selected_files)}):\n{file_list}\n\nClick 'Analyze' to process these images.")
            
    def analyze_images(self):
        if not self.selected_files:
            messagebox.showwarning("No Images", "Please select images first.")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🔍 Analyzing images...\n\n")
        self.root.update()
        
        results = []
        total_files = len(self.selected_files)
        
        for i, file_path in enumerate(self.selected_files):
            try:
                # Update progress
                progress = (i / total_files) * 100
                self.progress_var.set(progress)
                self.update_status(f"Processing {os.path.basename(file_path)}...")
                self.root.update()
                
                # Process image
                img_array = self.preprocess_image(file_path)
                prediction, confidence = self.predict_tumor(img_array)
                
                # Format result
                result = {
                    'filename': os.path.basename(file_path),
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Display result
                confidence_bar = "█" * int(confidence / 5)
                result_text = f"📄 {result['filename']}\n"
                result_text += f"   🔍 Detection: {prediction}\n"
                result_text += f"   📊 Confidence: {confidence:.2f}% {confidence_bar}\n\n"
                
                self.results_text.insert(tk.END, result_text)
                self.root.update()
                
            except Exception as e:
                error_text = f"❌ Error processing {os.path.basename(file_path)}: {str(e)}\n\n"
                self.results_text.insert(tk.END, error_text)
                
        # Complete analysis
        self.progress_var.set(100)
        self.update_status(f"Analysis complete! Processed {len(results)} images.")
        
        # Save to history
        self.history.extend(results)
        self.save_history()
        
        # Summary
        summary_text = f"\n{'='*50}\n"
        summary_text += f"📊 ANALYSIS SUMMARY\n"
        summary_text += f"{'='*50}\n"
        summary_text += f"Total Images: {len(results)}\n"
        
        # Count predictions
        prediction_counts = {}
        for result in results:
            pred = result['prediction']
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
        for pred, count in prediction_counts.items():
            summary_text += f"{pred}: {count} images\n"
            
        summary_text += f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        self.results_text.insert(tk.END, summary_text)
        
        # Reset progress
        self.root.after(2000, lambda: self.progress_var.set(0))
        
    def preprocess_image(self, img_path):
        """Preprocess image for model prediction"""
        try:
            # Load and resize image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize to model input size (typically 224x224 for Teachable Machine)
            img = img.resize((224, 224))
            
            # Convert to array and normalize
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0
            
            return img_array
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
            
    def predict_tumor(self, img_array):
        """Make prediction using the loaded model"""
        try:
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get the class with highest probability
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # Map to label
            if predicted_class < len(self.labels):
                prediction = self.labels[predicted_class]
            else:
                prediction = f"Unknown (Class {predicted_class})"
                
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
            
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.selected_files = []
        self.progress_var.set(0)
        self.update_status("Ready")
        
    def save_results(self):
        results_content = self.results_text.get(1.0, tk.END).strip()
        if not results_content:
            messagebox.showwarning("No Results", "No results to save.")
            return
            
        # Save as text file
        filename = f"brain_tumor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialname=filename
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(results_content)
                messagebox.showinfo("Success", f"Results saved to: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                
    def update_status(self, message):
        self.status_label.config(text=message)
        
    def load_history(self):
        try:
            history_file = "brain_tumor_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.history = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            self.history = []
            
    def save_history(self):
        try:
            history_file = "brain_tumor_history.json"
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
            
    def refresh_history(self):
        self.history_text.delete(1.0, tk.END)
        
        if not self.history:
            self.history_text.insert(tk.END, "No analysis history found.")
            return
            
        history_content = "🕒 ANALYSIS HISTORY\n"
        history_content += "="*60 + "\n\n"
        
        # Group by date
        history_by_date = {}
        for record in self.history:
            date = record['timestamp'][:10]  # Extract date part
            if date not in history_by_date:
                history_by_date[date] = []
            history_by_date[date].append(record)
            
        for date, records in sorted(history_by_date.items(), reverse=True):
            history_content += f"📅 {date}\n"
            history_content += "-" * 40 + "\n"
            
            for record in records:
                time = record['timestamp'][11:19]  # Extract time part
                history_content += f"  {time} - {record['filename']}\n"
                history_content += f"    🔍 {record['prediction']} ({record['confidence']:.2f}%)\n\n"
                
        self.history_text.insert(tk.END, history_content)
        
    def export_history(self):
        if not self.history:
            messagebox.showwarning("No History", "No history to export.")
            return
            
        filename = f"brain_tumor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialname=filename
        )
        
        if filepath:
            try:
                df = pd.DataFrame(self.history)
                df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"History exported to: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export history: {str(e)}")

def main():
    # Create the main window with drag and drop support
    root = TkinterDnD.Tk()
    app = BrainTumorDetectionGUI(root)
    
    # Set window icon (optional)
    try:
        root.iconbitmap(default='brain_icon.ico')  # Add if you have an icon file
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    main()
