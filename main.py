#!/usr/bin/env python3
"""
industrial_image_stitcher_gui.py

Professional-grade GUI for robust multi-tile image stitching.
Features:
 - Modern industrial GUI design
 - Folder selection dialog
 - Real-time progress tracking
 - Result visualization with zoom/pan
 - Feature analysis display
 - Export functionality
 - Professional logging

Dependencies:
 - python3, numpy, opencv-python, tkinter, PIL
 - For full functionality: opencv-contrib-python
"""

import os
import cv2
import numpy as np
from glob import glob
import math
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

EPS = 1e-8

class ImageStitcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Image Stitcher v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        self.setup_styles()
        
        # Variables
        self.input_folder = tk.StringVar()
        self.output_file = tk.StringVar(value="panorama.png")
        self.feature_type = tk.StringVar(value="sift")
        self.min_inliers = tk.IntVar(value=30)
        self.resize_match = tk.DoubleVar(value=0.35)
        self.auto_crop = tk.BooleanVar(value=True)
        
        # Processing variables
        self.is_processing = False
        self.current_panorama = None
        self.processing_stats = {}
        self.feature_data = {}
        
        # Create GUI
        self.create_gui()
        
    def setup_styles(self):
        """Configure professional styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Heading.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00d4ff', 
                       font=('Segoe UI', 12, 'bold'))
        
        style.configure('Info.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Segoe UI', 10))
        
        style.configure('Success.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff88', 
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Error.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ff6b6b', 
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Professional.TButton',
                       background='#0066cc',
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        
        style.configure('Action.TButton',
                       background='#00aa44',
                       foreground='white',
                       font=('Segoe UI', 11, 'bold'),
                       padding=12)
        
    def create_gui(self):
        """Create the main GUI interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Professional Image Stitcher", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True)
        
        # Tab 1: Configuration
        self.config_frame = ttk.Frame(notebook)
        notebook.add(self.config_frame, text="Configuration")
        self.create_config_tab()
        
        # Tab 2: Processing
        self.process_frame = ttk.Frame(notebook)
        notebook.add(self.process_frame, text="Processing")
        self.create_process_tab()
        
        # Tab 3: Results
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="Results")
        self.create_results_tab()
        
        # Tab 4: Analysis
        self.analysis_frame = ttk.Frame(notebook)
        notebook.add(self.analysis_frame, text="Feature Analysis")
        self.create_analysis_tab()
        
    def create_config_tab(self):
        """Create configuration tab"""
        # Input section
        input_section = ttk.LabelFrame(self.config_frame, text="Input Configuration", padding=15)
        input_section.pack(fill='x', padx=10, pady=10)
        
        # Folder selection
        folder_frame = tk.Frame(input_section, bg='white')
        folder_frame.pack(fill='x', pady=5)
        
        ttk.Label(folder_frame, text="Input Folder:").pack(anchor='w')
        folder_select_frame = tk.Frame(folder_frame, bg='white')
        folder_select_frame.pack(fill='x', pady=5)
        
        self.folder_entry = ttk.Entry(folder_select_frame, textvariable=self.input_folder, width=60)
        self.folder_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        ttk.Button(folder_select_frame, text="Browse", 
                  command=self.select_folder,
                  style='Professional.TButton').pack(side='right')
        
        # Output file
        ttk.Label(input_section, text="Output Filename:").pack(anchor='w', pady=(15, 0))
        ttk.Entry(input_section, textvariable=self.output_file, width=60).pack(fill='x', pady=5)
        
        # Algorithm section
        algo_section = ttk.LabelFrame(self.config_frame, text="Algorithm Settings", padding=15)
        algo_section.pack(fill='x', padx=10, pady=10)
        
        # Feature detector
        ttk.Label(algo_section, text="Feature Detector:").pack(anchor='w')
        feature_frame = tk.Frame(algo_section, bg='white')
        feature_frame.pack(fill='x', pady=5)
        
        for feature in ['sift', 'akaze', 'orb']:
            ttk.Radiobutton(feature_frame, text=feature.upper(), 
                           variable=self.feature_type, 
                           value=feature).pack(side='left', padx=(0, 20))
        
        # Parameters
        params_frame = tk.Frame(algo_section, bg='white')
        params_frame.pack(fill='x', pady=10)
        
        # Min inliers
        ttk.Label(params_frame, text="Min Inliers:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        ttk.Scale(params_frame, from_=10, to=100, variable=self.min_inliers, 
                 orient='horizontal', length=200).grid(row=0, column=1, sticky='w')
        ttk.Label(params_frame, textvariable=self.min_inliers).grid(row=0, column=2, padx=(10, 0))
        
        # Resize factor
        ttk.Label(params_frame, text="Resize for Matching:").grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        ttk.Scale(params_frame, from_=0.1, to=1.0, variable=self.resize_match, 
                 orient='horizontal', length=200).grid(row=1, column=1, sticky='w', pady=(10, 0))
        resize_label = ttk.Label(params_frame, text="")
        resize_label.grid(row=1, column=2, padx=(10, 0), pady=(10, 0))
        
        def update_resize_label(*args):
            resize_label.config(text=f"{self.resize_match.get():.2f}")
        self.resize_match.trace('w', update_resize_label)
        update_resize_label()
        
        # Auto crop
        ttk.Checkbutton(algo_section, text="Auto-crop black borders", 
                       variable=self.auto_crop).pack(anchor='w', pady=10)
        
    def create_process_tab(self):
        """Create processing tab"""
        # Control section
        control_section = ttk.LabelFrame(self.process_frame, text="Processing Control", padding=15)
        control_section.pack(fill='x', padx=10, pady=10)
        
        button_frame = tk.Frame(control_section, bg='white')
        button_frame.pack()
        
        self.process_btn = ttk.Button(button_frame, text="🚀 START PROCESSING", 
                                     command=self.start_processing,
                                     style='Action.TButton')
        self.process_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ STOP", 
                                  command=self.stop_processing,
                                  style='Professional.TButton',
                                  state='disabled')
        self.stop_btn.pack(side='left')
        
        # Progress section
        progress_section = ttk.LabelFrame(self.process_frame, text="Progress", padding=15)
        progress_section.pack(fill='x', padx=10, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to process...")
        self.progress_label = ttk.Label(progress_section, textvariable=self.progress_var, style='Info.TLabel')
        self.progress_label.pack(anchor='w', pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_section, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Log section
        log_section = ttk.LabelFrame(self.process_frame, text="Processing Log", padding=15)
        log_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        log_frame = tk.Frame(log_section)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_frame, bg='#1e1e1e', fg='#ffffff', 
                               font=('Consolas', 10), wrap='word')
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scroll.pack(side='right', fill='y')
        
    def create_results_tab(self):
        """Create results tab"""
        # Control section
        control_section = ttk.LabelFrame(self.results_frame, text="Result Control", padding=15)
        control_section.pack(fill='x', padx=10, pady=10)
        
        button_frame = tk.Frame(control_section, bg='white')
        button_frame.pack()
        
        ttk.Button(button_frame, text="💾 Save Result", 
                  command=self.save_result,
                  style='Professional.TButton').pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="🔍 Zoom Fit", 
                  command=self.zoom_fit,
                  style='Professional.TButton').pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="🔄 Reset View", 
                  command=self.reset_view,
                  style='Professional.TButton').pack(side='left')
        
        # Image display
        display_section = ttk.LabelFrame(self.results_frame, text="Panorama Result", padding=15)
        display_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Canvas with scrollbars
        canvas_frame = tk.Frame(display_section)
        canvas_frame.pack(fill='both', expand=True)
        
        self.result_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.result_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal', command=self.result_canvas.xview)
        
        self.result_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.result_canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Bind mouse events for pan and zoom
        self.result_canvas.bind('<Button-1>', self.start_pan)
        self.result_canvas.bind('<B1-Motion>', self.do_pan)
        self.result_canvas.bind('<MouseWheel>', self.zoom_image)
        
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.zoom_factor = 1.0
        
    def create_analysis_tab(self):
        """Create feature analysis tab"""
        # Stats section
        stats_section = ttk.LabelFrame(self.analysis_frame, text="Processing Statistics", padding=15)
        stats_section.pack(fill='x', padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_section, height=8, bg='#1e1e1e', fg='#00ff88', 
                                 font=('Consolas', 11), wrap='word')
        self.stats_text.pack(fill='x', pady=5)
        
        # Feature details section
        features_section = ttk.LabelFrame(self.analysis_frame, text="Feature Matching Details", padding=15)
        features_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Treeview for feature data
        tree_frame = tk.Frame(features_section)
        tree_frame.pack(fill='both', expand=True)
        
        columns = ('Image Pair', 'Matches Found', 'Inliers', 'Inlier Ratio', 'Status')
        self.feature_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.feature_tree.heading(col, text=col)
            self.feature_tree.column(col, width=150, anchor='center')
        
        tree_scroll_v = ttk.Scrollbar(tree_frame, orient='vertical', command=self.feature_tree.yview)
        tree_scroll_h = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.feature_tree.xview)
        
        self.feature_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        self.feature_tree.grid(row=0, column=0, sticky='nsew')
        tree_scroll_v.grid(row=0, column=1, sticky='ns')
        tree_scroll_h.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
    def select_folder(self):
        """Open folder selection dialog"""
        folder = filedialog.askdirectory(title="Select Input Folder with Images")
        if folder:
            self.input_folder.set(folder)
            self.log_message(f"Selected input folder: {folder}")
            # Count images in folder
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
            paths = []
            for e in exts:
                paths.extend(glob(os.path.join(folder, e)))
            self.log_message(f"Found {len(paths)} images in folder")
    
    def log_message(self, message, level="INFO"):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert('end', formatted_msg)
        self.log_text.see('end')
        self.root.update_idletasks()
        
        # Color coding
        if level == "ERROR":
            self.log_text.tag_add("error", f"end-{len(formatted_msg)}c", 'end-1c')
            self.log_text.tag_config("error", foreground='#ff6b6b')
        elif level == "SUCCESS":
            self.log_text.tag_add("success", f"end-{len(formatted_msg)}c", 'end-1c')
            self.log_text.tag_config("success", foreground='#00ff88')
    
    def start_processing(self):
        """Start the stitching process in a separate thread"""
        if not self.input_folder.get():
            messagebox.showerror("Error", "Please select an input folder first!")
            return
            
        if self.is_processing:
            return
            
        self.is_processing = True
        self.process_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_bar.start()
        
        # Clear previous results
        self.log_text.delete(1.0, 'end')
        self.stats_text.delete(1.0, 'end')
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_images)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_processing(self):
        """Stop the processing"""
        self.is_processing = False
        self.progress_bar.stop()
        self.process_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_var.set("Processing stopped by user")
        self.log_message("Processing stopped by user", "INFO")
    
    def process_images(self):
        """Main processing function (runs in separate thread)"""
        try:
            self.progress_var.set("Loading images...")
            self.log_message("Starting image stitching process", "INFO")
            
            # Load images
            imgs, paths = self.load_images(self.input_folder.get())
            if len(imgs) == 0:
                self.log_message("No images found in selected folder", "ERROR")
                return
            
            self.log_message(f"Loaded {len(imgs)} images", "SUCCESS")
            
            # Process with current settings
            self.progress_var.set("Processing...")
            pano = self.stitch_images_gui(
                imgs,
                feature=self.feature_type.get(),
                min_inliers=self.min_inliers.get(),
                resize_for_match=self.resize_match.get() if self.resize_match.get() < 1.0 else None,
                crop=self.auto_crop.get()
            )
            
            if pano is not None and self.is_processing:
                self.current_panorama = pano
                self.progress_var.set("Stitching completed successfully!")
                self.log_message("Stitching completed successfully!", "SUCCESS")
                
                # Save result
                output_path = os.path.join(os.path.dirname(self.input_folder.get()), self.output_file.get())
                cv2.imwrite(output_path, pano)
                self.log_message(f"Result saved to: {output_path}", "SUCCESS")
                
                # Display result
                self.root.after(0, self.display_result)
                self.root.after(0, self.update_analysis)
                
            else:
                self.log_message("Stitching failed or was cancelled", "ERROR")
                
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}", "ERROR")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: [
                self.progress_bar.stop(),
                self.process_btn.config(state='normal'),
                self.stop_btn.config(state='disabled')
            ])
    
    def display_result(self):
        """Display the result panorama in the canvas"""
        if self.current_panorama is None:
            return
            
        # Convert OpenCV image to PIL and then to PhotoImage
        pano_rgb = cv2.cvtColor(self.current_panorama, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pano_rgb)
        
        # Calculate initial size to fit canvas
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate zoom to fit
            zoom_x = canvas_width / pil_image.width
            zoom_y = canvas_height / pil_image.height
            self.zoom_factor = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%
            
            if self.zoom_factor < 1.0:
                new_size = (int(pil_image.width * self.zoom_factor), 
                           int(pil_image.height * self.zoom_factor))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        
    def update_analysis(self):
        """Update the analysis tab with processing statistics"""
        if not self.processing_stats:
            return
            
        # Update statistics text
        stats_text = "PROCESSING STATISTICS\n"
        stats_text += "=" * 50 + "\n"
        for key, value in self.processing_stats.items():
            stats_text += f"{key}: {value}\n"
        
        self.stats_text.delete(1.0, 'end')
        self.stats_text.insert('end', stats_text)
        
        # Update feature tree
        for pair, data in self.feature_data.items():
            status = "✓ Good" if data['inliers'] >= self.min_inliers.get() else "✗ Rejected"
            ratio = f"{data['inliers']/max(data['matches'], 1)*100:.1f}%"
            
            self.feature_tree.insert('', 'end', values=(
                f"Image {pair[0]} ↔ {pair[1]}",
                data['matches'],
                data['inliers'],
                ratio,
                status
            ))
    
    # Canvas interaction methods
    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
    def do_pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.result_canvas.scan_dragto(dx, dy, gain=1)
        
    def zoom_image(self, event):
        # Mouse wheel zoom (Windows/Linux)
        if event.delta > 0 or event.num == 4:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))
        self.refresh_display()
        
    def zoom_fit(self):
        """Fit image to canvas"""
        if self.current_panorama is None:
            return
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        img_height, img_width = self.current_panorama.shape[:2]
        
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height
        self.zoom_factor = min(zoom_x, zoom_y)
        self.refresh_display()
        
    def reset_view(self):
        """Reset zoom to 100%"""
        self.zoom_factor = 1.0
        self.refresh_display()
        
    def refresh_display(self):
        """Refresh the displayed image with current zoom"""
        if self.current_panorama is None:
            return
            
        pano_rgb = cv2.cvtColor(self.current_panorama, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pano_rgb)
        
        if self.zoom_factor != 1.0:
            new_size = (int(pil_image.width * self.zoom_factor), 
                       int(pil_image.height * self.zoom_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
        
    def save_result(self):
        """Save the result to a chosen location"""
        if self.current_panorama is None:
            messagebox.showwarning("Warning", "No result to save!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filename:
            cv2.imwrite(filename, self.current_panorama)
            self.log_message(f"Result saved to: {filename}", "SUCCESS")
    
    # Core stitching functions (adapted from original with GUI integration)
    def load_images(self, folder):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for e in exts:
            paths.extend(sorted(glob(os.path.join(folder, e))))
        imgs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
        return imgs, paths

    def create_detector(self, name="sift", max_features=5000):
        name = name.lower()
        if name == "sift":
            try:
                return cv2.SIFT_create(nfeatures=max_features)
            except Exception:
                self.log_message("SIFT not available, falling back to AKAZE", "INFO")
                name = "akaze"
        if name == "akaze":
            return cv2.AKAZE_create()
        return cv2.ORB_create(nfeatures=max_features)

    def match_descriptors(self, desc1, desc2, detector_name):
        if desc1 is None or desc2 is None:
            return []

        if detector_name in ("sift", "akaze"):
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                matches = flann.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
            except Exception:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def stitch_images_gui(self, images, feature="sift", min_inliers=30, resize_for_match=None, crop=True):
        """Main stitching pipeline with GUI integration"""
        if not self.is_processing:
            return None
            
        n = len(images)
        if n == 0:
            raise ValueError("No images supplied.")

        self.processing_stats = {
            "Total Images": n,
            "Feature Detector": feature.upper(),
            "Min Inliers": min_inliers,
            "Resize Factor": resize_for_match or "None",
            "Auto Crop": crop
        }

        # Compute pairwise homographies
        self.progress_var.set("Computing feature matches...")
        H_dict, adjacency = self.compute_pairwise_homographies_gui(images, feature, min_inliers)
        
        if not self.is_processing:
            return None
            
        if len(adjacency) == 0:
            self.log_message("No reliable pairwise matches found", "ERROR")
            return None

        # Choose reference and compute global homographies
        self.progress_var.set("Computing global transformations...")
        ref = self.choose_reference_image(adjacency)
        H_to_ref = self.compute_global_homographies(H_dict, adjacency, ref, n)
        
        if not self.is_processing:
            return None

        # Filter unreachable images
        unreachable = [i for i, H in enumerate(H_to_ref) if H is None]
        if unreachable:
            self.log_message(f"Removing unreachable images: {unreachable}", "INFO")
            images = [img for i, img in enumerate(images) if H_to_ref[i] is not None]
            H_to_ref = [H for H in H_to_ref if H is not None]
            n = len(images)

        self.processing_stats["Connected Images"] = n
        self.processing_stats["Reference Image"] = ref

        # Compute canvas and stitch
        self.progress_var.set("Warping and blending images...")
        x_min, y_min, x_max, y_max, width, height, translation = self.warp_corners_and_compute_canvas(images, H_to_ref)
        
        self.processing_stats["Canvas Size"] = f"{width} x {height}"
        
        if not self.is_processing:
            return None

        # Accumulation with exposure compensation and blending
        pano = self.blend_images(images, H_to_ref, translation, width, height, adjacency, ref)
        
        if not self.is_processing:
            return None

        # Optional cropping
        if crop and pano is not None:
            self.progress_var.set("Cropping result...")
            pano = self.crop_black(pano)

        return pano

    def compute_pairwise_homographies_gui(self, images, detector_name="sift", min_inliers=30):
        """Compute matches and homographies with GUI updates"""
        n = len(images)
        self.log_message(f"Detecting keypoints & descriptors for {n} images with {detector_name}", "INFO")
        
        detector = self.create_detector(detector_name)
        kps = [None] * n
        descs = [None] * n
        
        for i, img in enumerate(images):
            if not self.is_processing:
                return {}, {}
            self.progress_var.set(f"Processing image {i+1}/{n}...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = detector.detectAndCompute(gray, None)
            kps[i], descs[i] = kp, desc
            self.log_message(f"Image {i}: {len(kp) if kp else 0} keypoints", "INFO")

        H_dict = dict()
        adjacency = defaultdict(list)
        self.feature_data = {}

        total_pairs = (n * (n - 1)) // 2
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if not self.is_processing:
                    return {}, {}
                    
                pair_count += 1
                self.progress_var.set(f"Matching pairs {pair_count}/{total_pairs}...")
                
                matches = self.match_descriptors(descs[j], descs[i], detector_name)
                self.feature_data[(i, j)] = {'matches': len(matches), 'inliers': 0}
                
                if len(matches) < 8:
                    continue
                    
                pts_j = np.float32([kps[j][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                pts_i = np.float32([kps[i][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(pts_j, pts_i, cv2.RANSAC, 5.0)
                
                if H is None:
                    continue
                    
                inliers = int(mask.sum())
                self.feature_data[(i, j)]['inliers'] = inliers
                
                if inliers >= min_inliers:
                    H_dict[(i, j)] = (H, mask, inliers)
                    try:
                        H_inv = np.linalg.inv(H)
                        H_dict[(j, i)] = (H_inv, mask.T, inliers)
                    except np.linalg.LinAlgError:
                        pass
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    self.log_message(f"Pair ({i},{j}) matched: {inliers} inliers", "SUCCESS")
                else:
                    self.log_message(f"Pair ({i},{j}) rejected: {inliers} inliers (< {min_inliers})", "INFO")

        self.processing_stats["Total Pairs"] = total_pairs
        self.processing_stats["Good Matches"] = len([k for k in H_dict.keys() if k[0] < k[1]])
        
        return H_dict, adjacency

    def choose_reference_image(self, adjacency):
        """Pick image with highest number of neighbors"""
        if not adjacency:
            return 0
        best = max(adjacency.keys(), key=lambda k: len(adjacency[k]))
        self.log_message(f"Chosen reference image: {best} (degree {len(adjacency[best])})", "INFO")
        return best

    def compute_global_homographies(self, H_dict, adjacency, ref_idx, n_images):
        """Compute global homographies using BFS"""
        H_to_ref = [None] * n_images
        H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)
        visited = set([ref_idx])
        q = deque([ref_idx])
        
        while q and self.is_processing:
            cur = q.popleft()
            for nbr in adjacency.get(cur, []):
                if nbr in visited:
                    continue
                if (cur, nbr) not in H_dict:
                    self.log_message(f"Missing homography for ({cur},{nbr})", "INFO")
                    continue
                H_nbr_to_cur = H_dict[(cur, nbr)][0]
                H_nbr_to_ref = H_to_ref[cur] @ H_nbr_to_cur
                H_to_ref[nbr] = H_nbr_to_ref
                visited.add(nbr)
                q.append(nbr)
                self.log_message(f"Computed H_{nbr}->ref via neighbor {cur}", "INFO")
        
        return H_to_ref

    def warp_corners_and_compute_canvas(self, images, H_to_ref):
        """Compute canvas bounds"""
        corners_all = []
        for img, H in zip(images, H_to_ref):
            if H is None:
                corners_all.append(None)
                continue
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H)
            corners_all.append(warped)
        
        pts = np.vstack([c.reshape(-1, 2) for c in corners_all if c is not None])
        x_min, y_min = np.floor(pts.min(axis=0) - 0.5).astype(int)
        x_max, y_max = np.ceil(pts.max(axis=0) + 0.5).astype(int)
        width = x_max - x_min
        height = y_max - y_min
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        
        self.log_message(f"Canvas bounds: x [{x_min},{x_max}], y [{y_min},{y_max}], size ({width} x {height})", "INFO")
        return (x_min, y_min, x_max, y_max, width, height, translation)

    def blend_images(self, images, H_to_ref, translation, width, height, adjacency, ref):
        """Blend images with exposure compensation and feathering"""
        acc_img = np.zeros((height, width, 3), dtype=np.float64)
        acc_weights = np.zeros((height, width), dtype=np.float64)

        # BFS order for processing
        order = []
        visited = set()
        q = deque([ref])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for nb in adjacency.get(node, []):
                if nb not in visited:
                    q.append(nb)

        order = [i for i in order if i < len(H_to_ref)]
        pano_mask = np.zeros((height, width), dtype=np.uint8)
        pano_image = np.zeros((height, width, 3), dtype=np.uint8)

        for idx in order:
            if not self.is_processing:
                return None
                
            self.progress_var.set(f"Blending image {idx+1}...")
            H = H_to_ref[idx]
            img = images[idx]
            H_trans = translation @ H
            warped = cv2.warpPerspective(img, H_trans, (width, height), flags=cv2.INTER_LINEAR)
            
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_mask = (gray > 0).astype(np.uint8) * 255

            # Exposure compensation
            if pano_mask.sum() > 0:
                warped = self.overlap_exposure_compensate(pano_image, pano_mask, warped, warped_mask)

            # Distance transform weights
            weights = self.distance_weight_mask(warped_mask)
            
            # Accumulate
            acc_img += (warped.astype(np.float64) * weights[..., None])
            acc_weights += weights

            # Update panorama approximation
            valid = acc_weights > 0
            if valid.any():
                pano_image[valid] = (acc_img[valid] / acc_weights[valid, None]).clip(0, 255).astype(np.uint8)
            pano_mask = (acc_weights > 0).astype(np.uint8) * 255

            self.log_message(f"Added image {idx} (coverage: {warped_mask.sum()} pixels)", "INFO")

        # Normalize
        denom = acc_weights[..., None] + EPS
        pano = (acc_img / denom).clip(0, 255).astype(np.uint8)
        
        return pano

    def overlap_exposure_compensate(self, pano_rgb, pano_mask, warped_rgb, warped_mask):
        """Exposure compensation in overlap regions"""
        overlap = (pano_mask > 0) & (warped_mask > 0)
        if not overlap.any():
            return warped_rgb
            
        warped = warped_rgb.astype(np.float32)
        pano = pano_rgb.astype(np.float32)
        
        for c in range(3):
            warped_chan = warped[..., c][overlap]
            pano_chan = pano[..., c][overlap]
            if warped_chan.size < 10:
                continue
            mean_w = warped_chan.mean()
            mean_p = pano_chan.mean()
            scale = (mean_p + EPS) / (mean_w + EPS)
            scale = np.clip(scale, 0.6, 1.6)
            warped[..., c] = np.clip(warped[..., c] * scale, 0, 255)
            
        return warped.astype(np.uint8)

    def distance_weight_mask(self, mask):
        """Create distance-based weights"""
        bmask = (mask > 0).astype(np.uint8)
        if bmask.sum() == 0:
            return bmask.astype(np.float32)
        dist = cv2.distanceTransform(bmask, cv2.DIST_L2, 5)
        maxd = dist.max() if dist.max() > 0 else 1.0
        weights = dist / (maxd + EPS)
        return weights.astype(np.float32)

    def crop_black(self, pano_img):
        """Crop black borders"""
        gray = cv2.cvtColor(pano_img, cv2.COLOR_BGR2GRAY)
        nonzero_rows = np.where(gray.sum(axis=1) > 0)[0]
        nonzero_cols = np.where(gray.sum(axis=0) > 0)[0]
        if nonzero_rows.size == 0 or nonzero_cols.size == 0:
            return pano_img
        r0, r1 = nonzero_rows[0], nonzero_rows[-1]
        c0, c1 = nonzero_cols[0], nonzero_cols[-1]
        return pano_img[r0:r1 + 1, c0:c1 + 1]


def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = ImageStitcherGUI(root)
    
    # Set window icon and additional properties
    root.resizable(True, True)
    root.minsize(1200, 800)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Handle window closing
    def on_closing():
        if hasattr(app, 'is_processing') and app.is_processing:
            if messagebox.askokcancel("Quit", "Processing is running. Do you want to quit?"):
                app.is_processing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()
