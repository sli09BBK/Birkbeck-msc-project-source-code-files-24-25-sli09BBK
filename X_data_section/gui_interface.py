#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xiaohongshu Data Analysis System - Graphical User Interface

Provides a user-friendly graphical interface to operate data processing and analysis functions.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, List, Optional
import queue
import subprocess
from datetime import datetime

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add scripts directory to path
scripts_dir = os.path.join(current_dir, 'scripts')
if os.path.exists(scripts_dir):
    sys.path.append(scripts_dir)

try:
    # Import modules directly, without scripts prefix
    from data_pipeline_main import DataPipeline
    from fixed_database_setup import DatabaseConfig, DatabaseSetup
    from fixed_enhanced_data_processor import EnhancedDataProcessor
    from fixed_advanced_user_analysis import AdvancedUserAnalysis

    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("📁 Current directory structure:")
    print(f"  Current directory: {current_dir}")
    if os.path.exists(scripts_dir):
        print(f"  Scripts directory: {scripts_dir}")
        print("  Python files in scripts directory:")
        for file in os.listdir(scripts_dir):
            if file.endswith('.py'):
                print(f"    - {file}")
    else:
        print("  ❌ Scripts directory does not exist")

    print("\nPython files in current directory:")
    for file in os.listdir(current_dir):
        if file.endswith('.py'):
            print(f"  - {file}")

    # Attempt alternate import solution
    try:
        print("\n🔄 Attempting alternate import solution...")
        sys.path.insert(0, current_dir)

        # Check and import available modules
        available_modules = {}

        # Check data_pipeline_main
        for possible_name in ['data_pipeline_main', 'main_data_pipeline']:
            try:
                module = __import__(possible_name)
                if hasattr(module, 'DataPipeline'):
                    available_modules['DataPipeline'] = getattr(module, 'DataPipeline')
                    print(f"  ✅ Found DataPipeline in: {possible_name}")
                    break
            except ImportError:
                continue

        # Check database_setup
        for possible_name in ['fixed_database_setup', 'database_setup']:
            try:
                module = __import__(possible_name)
                if hasattr(module, 'DatabaseConfig'):
                    available_modules['DatabaseConfig'] = getattr(module, 'DatabaseConfig')
                    available_modules['DatabaseSetup'] = getattr(module, 'DatabaseSetup')
                    print(f"  ✅ Found DatabaseConfig in: {possible_name}")
                    break
            except ImportError:
                continue

        # Check data_processor
        for possible_name in ['fixed_enhanced_data_processor', 'enhanced_data_processor']:
            try:
                module = __import__(possible_name)
                if hasattr(module, 'EnhancedDataProcessor'):
                    available_modules['EnhancedDataProcessor'] = getattr(module, 'EnhancedDataProcessor')
                    print(f"  ✅ Found EnhancedDataProcessor in: {possible_name}")
                    break
            except ImportError:
                continue

        # Check user_analysis
        for possible_name in ['fixed_advanced_user_analysis', 'advanced_user_analysis']:
            try:
                module = __import__(possible_name)
                if hasattr(module, 'AdvancedUserAnalysis'):
                    available_modules['AdvancedUserAnalysis'] = getattr(module, 'AdvancedUserAnalysis')
                    print(f"  ✅ Found AdvancedUserAnalysis in: {possible_name}")
                    break
            except ImportError:
                continue

        # Add found modules to global namespace
        if 'DataPipeline' in available_modules:
            DataPipeline = available_modules['DataPipeline']
        if 'DatabaseConfig' in available_modules:
            DatabaseConfig = available_modules['DatabaseConfig']
            DatabaseSetup = available_modules['DatabaseSetup']
        if 'EnhancedDataProcessor' in available_modules:
            EnhancedDataProcessor = available_modules['EnhancedDataProcessor']
        if 'AdvancedUserAnalysis' in available_modules:
            AdvancedUserAnalysis = available_modules['AdvancedUserAnalysis']

        missing_modules = []
        for required in ['DataPipeline', 'DatabaseConfig', 'EnhancedDataProcessor', 'AdvancedUserAnalysis']:
            if required not in available_modules:
                missing_modules.append(required)

        if missing_modules:
            print(f"\n❌ Still missing modules: {missing_modules}")
            print("Please ensure the following files exist in the current directory or scripts directory:")
            print("  - data_pipeline_main.py or main_data_pipeline.py")
            print("  - fixed_database_setup.py or database_setup.py")
            print("  - fixed_enhanced_data_processor.py or enhanced_data_processor.py")
            print("  - fixed_advanced_user_analysis.py or advanced_user_analysis.py")
        else:
            print("✅ Alternate import solution successful")

    except Exception as backup_error:
        print(f"❌ Alternate import solution also failed: {backup_error}")
        print("\n💡 Solution:")
        print("1. Ensure all Python files are in the current directory")
        print("2. Or place all files in the scripts subdirectory")
        print("3. Check if file names are correct")


class DataAnalysisGUI:
    """Data Analysis System Graphical User Interface"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Xiaohongshu Data Analysis System v2.0")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Set icon and styles
        self.setup_styles()

        # Initialize variables
        self.config_file = "X_data_section/database_config.json"
        self.current_config = None
        self.processing_thread = None
        self.log_queue = queue.Queue()
        self.stop_processing_flag = False

        # Create interface
        self.create_widgets()

        # Load configuration
        self.load_config()

        # Start log processing
        self.process_log_queue()

    def setup_styles(self):
        """Set interface styles"""
        style = ttk.Style()

        # Set theme
        try:
            style.theme_use('clam')
        except:
            pass

        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')

    def create_widgets(self):
        """Create interface components"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Xiaohongshu Data Analysis System", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Database Configuration Tab
        self.create_database_tab()

        # Data Processing Tab
        self.create_processing_tab()

        # Data Analysis Tab
        self.create_analysis_tab()

        # System Management Tab
        self.create_management_tab()

        # Log display area
        self.create_log_area(main_frame)

        # Status bar
        self.create_status_bar(main_frame)

    def create_database_tab(self):
        """Create database configuration tab"""
        db_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(db_frame, text="Database Configuration")

        # Configuration form
        config_frame = ttk.LabelFrame(db_frame, text="Database Connection Configuration", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Configuration fields
        self.db_vars = {}
        fields = [
            ('Host Address', 'host', '127.0.0.1'),
            ('Port', 'port', '3306'),
            ('Username', 'user', 'root'),
            ('Password', 'password', 'root'),
            ('Database Name', 'database', 'rednote')
        ]

        for i, (label, key, default) in enumerate(fields):
            ttk.Label(config_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)

            var = tk.StringVar(value=default)
            self.db_vars[key] = var

            if key == 'password':
                entry = ttk.Entry(config_frame, textvariable=var, show='*', width=30)
            else:
                entry = ttk.Entry(config_frame, textvariable=var, width=30)

            entry.grid(row=i, column=1, sticky=(tk.W, tk.E), pady=2, padx=(10, 0))

        config_frame.columnconfigure(1, weight=1)

        # Button frame
        btn_frame = ttk.Frame(db_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="Test Connection", command=self.test_connection).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Initialize Database", command=self.init_database).pack(side=tk.LEFT)

        # Connection status
        self.connection_status = ttk.Label(db_frame, text="Not Connected", style='Error.TLabel')
        self.connection_status.grid(row=2, column=0, columnspan=2, pady=10)

    def create_processing_tab(self):
        """Create data processing tab"""
        proc_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(proc_frame, text="Data Processing")

        # File selection
        file_frame = ttk.LabelFrame(proc_frame, text="Data File Selection", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                              padx=(10, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(10, 0))

        file_frame.columnconfigure(1, weight=1)

        # Processing options
        options_frame = ttk.LabelFrame(proc_frame, text="Processing Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.batch_size_var = tk.StringVar(value="1000")
        self.skip_analysis_var = tk.BooleanVar()
        self.cleanup_var = tk.BooleanVar()

        ttk.Label(options_frame, text="Batch Size:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(options_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, sticky=tk.W,
                                                                                  padx=(10, 0))

        ttk.Checkbutton(options_frame, text="Skip User Behavior Analysis", variable=self.skip_analysis_var).grid(row=1, column=0,
                                                                                                      columnspan=2,
                                                                                                      sticky=tk.W,
                                                                                                      pady=5)
        ttk.Checkbutton(options_frame, text="Clean Temporary Data After Processing", variable=self.cleanup_var).grid(row=2, column=0,
                                                                                                  columnspan=2,
                                                                                                  sticky=tk.W)

        # Processing buttons
        btn_frame = ttk.Frame(proc_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.process_btn = ttk.Button(btn_frame, text="Start Processing", command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(btn_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(proc_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

    def create_analysis_tab(self):
        """Create data analysis tab"""
        analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(analysis_frame, text="Data Analysis")

        # Analysis options
        options_frame = ttk.LabelFrame(analysis_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.clustering_var = tk.BooleanVar(value=True)
        self.prediction_var = tk.BooleanVar(value=True)
        self.visualization_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(options_frame, text="User Clustering Analysis", variable=self.clustering_var).grid(row=0, column=0,
                                                                                               sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Prediction Model Training", variable=self.prediction_var).grid(row=1, column=0,
                                                                                               sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Generate Visualizations", variable=self.visualization_var).grid(row=2, column=0,
                                                                                                    sticky=tk.W)

        # Analysis buttons
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="Start Analysis", command=self.start_analysis).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="View Results", command=self.view_results).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Export Report", command=self.export_report).pack(side=tk.LEFT)

        # Results display
        results_frame = ttk.LabelFrame(analysis_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        analysis_frame.rowconfigure(2, weight=1)

    def create_management_tab(self):
        """Create system management tab"""
        mgmt_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(mgmt_frame, text="System Management")

        # Database management
        db_mgmt_frame = ttk.LabelFrame(mgmt_frame, text="Database Management", padding="10")
        db_mgmt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(db_mgmt_frame, text="Backup Database", command=self.backup_database).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(db_mgmt_frame, text="Clean Data", command=self.cleanup_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(db_mgmt_frame, text="View Statistics", command=self.view_statistics).pack(side=tk.LEFT)

        # System information
        info_frame = ttk.LabelFrame(mgmt_frame, text="System Information", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Update system information
        self.update_system_info()

        mgmt_frame.columnconfigure(0, weight=1)
        mgmt_frame.rowconfigure(1, weight=1)

    def create_log_area(self, parent):
        """Create log display area"""
        log_frame = ttk.LabelFrame(parent, text="System Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Log control buttons
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(log_btn_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        ttk.Button(log_btn_frame, text="Save Log", command=self.save_log).pack(side=tk.LEFT, padx=(10, 0))

    def create_status_bar(self, parent):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))

    def log_message(self, message: str, level: str = "INFO"):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        self.log_queue.put(formatted_message)

    def process_log_queue(self):
        """Process log queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        # Check every 100ms
        self.root.after(100, self.process_log_queue)

    def update_status(self, message: str):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def browse_file(self):
        """Browse for file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)

    def load_config(self):
        """Load database configuration"""
        try:
            if os.path.exists(self.config_file):
                config = DatabaseConfig.load_config(self.config_file)

                for key, var in self.db_vars.items():
                    if key in config:
                        var.set(str(config[key]))

                self.current_config = config
                self.log_message("Configuration file loaded successfully")
            else:
                self.log_message("Configuration file does not exist, using default configuration", "WARNING")
                self.current_config = DatabaseConfig.get_default_config()
        except Exception as e:
            self.log_message(f"Failed to load configuration: {e}", "ERROR")
            self.current_config = DatabaseConfig.get_default_config()

    def save_config(self):
        """Save database configuration"""
        try:
            config = {}
            for key, var in self.db_vars.items():
                value = var.get()
                if key == 'port':
                    config[key] = int(value) if value else 3306
                else:
                    config[key] = value

            DatabaseConfig.save_config(config, self.config_file)
            self.current_config = config
            self.log_message("Configuration saved successfully")
            messagebox.showinfo("Success", "Database configuration saved")

        except Exception as e:
            self.log_message(f"Failed to save configuration: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def test_connection(self):
        """Test database connection"""
        try:
            import mysql.connector

            config = {}
            for key, var in self.db_vars.items():
                value = var.get()
                if key == 'port':
                    config[key] = int(value) if value else 3306
                else:
                    config[key] = value

            connection = mysql.connector.connect(
                host=config['host'],
                user=config['user'],
                password=config['password'],
                port=config['port'],
                charset='utf8mb4'
            )

            connection.close()

            self.connection_status.config(text="Connection Successful", style='Success.TLabel')
            self.log_message("Database connection test successful")
            messagebox.showinfo("Success", "Database connection test successful")

        except Exception as e:
            self.connection_status.config(text="Connection Failed", style='Error.TLabel')
            self.log_message(f"Database connection test failed: {e}", "ERROR")
            messagebox.showerror("Error", f"Database connection failed: {e}")

    def init_database(self):
        """Initialize database"""
        if not self.current_config:
            messagebox.showwarning("Warning", "Please save database configuration first")
            return

        try:
            self.update_status("Initializing database...")
            self.log_message("Starting database initialization")

            db_setup = DatabaseSetup(
                host=self.current_config['host'],
                user=self.current_config['user'],
                password=self.current_config['password'],
                port=int(self.current_config.get('port', 3306))
            )

            success = db_setup.setup_complete_database(self.current_config['database'])
            db_setup.close_connection()

            if success:
                self.log_message("Database initialization complete")
                self.update_status("Database initialization complete")
                messagebox.showinfo("Success", "Database initialization complete")
            else:
                self.log_message("Database initialization failed", "ERROR")
                self.update_status("Database initialization failed")
                messagebox.showerror("Error", "Database initialization failed")

        except Exception as e:
            self.log_message(f"Database initialization failed: {e}", "ERROR")
            self.update_status("Database initialization failed")
            messagebox.showerror("Error", f"Database initialization failed: {e}")

    def start_processing(self):
        """Start data processing"""
        if not self.file_path_var.get():
            messagebox.showwarning("Warning", "Please select a CSV file to process")
            return

        if not self.current_config:
            messagebox.showwarning("Warning", "Please configure database connection first")
            return

        # Disable process button, enable stop button
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Reset progress bar and stop flag
        self.progress_var.set(0)
        self.stop_processing_flag = False

        # Run processing in a new thread
        self.processing_thread = threading.Thread(target=self._process_data_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_data_thread(self):
        """Data processing thread"""
        error_message = None
        try:
            self.log_message("Starting data processing")
            self.update_status("Processing data...")

            # Create data pipeline
            pipeline = DataPipeline(self.config_file)

            # Processing parameters
            input_file = self.file_path_var.get()
            batch_size = int(self.batch_size_var.get())
            skip_analysis = self.skip_analysis_var.get()
            cleanup = self.cleanup_var.get()

            # Simulate progress update
            for i in range(0, 101, 10):
                if self.stop_processing_flag:
                    self.log_message("Processing interrupted by user", "WARNING")
                    return

                self.progress_var.set(i)
                self.root.update_idletasks()

                # Actual processing logic can be added here
                import time
                time.sleep(0.5)  # Simulate processing time

            # Run complete pipeline
            success = pipeline.run_complete_pipeline(
                csv_files=[input_file],
                setup_db=False,
                run_analysis=not skip_analysis,
                cleanup_data=cleanup,
                batch_size=batch_size
            )

            if success:
                self.log_message("Data processing complete")
                self.update_status("Data processing complete")
                # Display completion message in main thread
                self.root.after(0, lambda: messagebox.showinfo("Success", "Data processing complete"))
            else:
                self.log_message("Data processing failed", "ERROR")
                self.update_status("Data processing failed")
                self.root.after(0, lambda: messagebox.showerror("Error", "Data processing failed, please check logs"))

        except Exception as e:
            error_message = str(e)
            self.log_message(f"Data processing failed: {e}", "ERROR")
            self.update_status("Data processing failed")

        finally:
            # Restore button state
            self.root.after(0, self._reset_processing_buttons)

            # If there's an error, display it in the main thread
            if error_message:
                self.root.after(0, lambda msg=error_message: messagebox.showerror("Error", f"Data processing failed: {msg}"))

    def _reset_processing_buttons(self):
        """Reset processing button state"""
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_var.set(100)

    def stop_processing(self):
        """Stop data processing"""
        self.stop_processing_flag = True
        self.log_message("User requested to stop processing", "WARNING")
        self.update_status("Stopping processing...")

        # Reset button state
        self._reset_processing_buttons()

    def start_analysis(self):
        """Start data analysis"""
        if not self.current_config:
            messagebox.showwarning("Warning", "Please configure database connection first")
            return

        def analysis_thread():
            try:
                self.log_message("Starting user behavior analysis")
                self.update_status("Performing data analysis...")

                # Create analyzer
                analyzer = AdvancedUserAnalysis(self.current_config)

                # Connect to database
                if not analyzer.connect_to_database():
                    raise Exception("Unable to connect to database")

                # Run analysis
                results = analyzer.run_complete_analysis()

                # Display results
                if results:
                    self.root.after(0, lambda: self._display_analysis_results(results))
                    self.log_message("Data analysis complete")
                    self.update_status("Data analysis complete")
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Data analysis complete"))
                else:
                    self.log_message("Data analysis failed: no results returned", "ERROR")
                    self.update_status("Data analysis failed")
                    self.root.after(0, lambda: messagebox.showerror("Error", "Data analysis failed: no results returned"))

                analyzer.close_connection()

            except Exception as e:
                error_msg = str(e)
                self.log_message(f"Data analysis failed: {e}", "ERROR")
                self.update_status("Data analysis failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Data analysis failed: {error_msg}"))

        # Run analysis in a new thread
        analysis_thread_obj = threading.Thread(target=analysis_thread)
        analysis_thread_obj.daemon = True
        analysis_thread_obj.start()

    def _display_analysis_results(self, results):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)

        if isinstance(results, dict):
            # Format and display results
            formatted_results = json.dumps(results, indent=2, ensure_ascii=False, default=str)
            self.results_text.insert(tk.END, formatted_results)
        else:
            self.results_text.insert(tk.END, str(results))

    def view_results(self):
        """View analysis results"""
        try:
            results_dir = Path("Advanced User Analysis Results")
            if results_dir.exists():
                if sys.platform.startswith('win'):
                    os.startfile(str(results_dir))
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['open', str(results_dir)])
                else:
                    subprocess.run(['xdg-open', str(results_dir)])
            else:
                messagebox.showinfo("Tip", "Results directory does not exist, please run data analysis first")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open results directory: {e}")

    def export_report(self):
        """Export analysis report"""
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Report",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if filename:
            try:
                content = self.results_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.log_message(f"Report saved to: {filename}")
                messagebox.showinfo("Success", f"Report saved to: {filename}")

            except Exception as e:
                self.log_message(f"Failed to save report: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to save report: {e}")

    def backup_database(self):
        """Backup database"""
        if not self.current_config:
            messagebox.showwarning("Warning", "Please configure database connection first")
            return

        backup_dir = filedialog.askdirectory(title="Select Backup Directory")
        if backup_dir:
            try:
                self.log_message("Starting database backup")
                self.update_status("Backing up database...")

                db_setup = DatabaseSetup(
                    host=self.current_config['host'],
                    user=self.current_config['user'],
                    password=self.current_config['password'],
                    port=int(self.current_config.get('port', 3306))
                )

                backup_file = os.path.join(backup_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")
                success = db_setup.backup_database(backup_file)
                db_setup.close_connection()

                if success:
                    self.log_message(f"Database backup complete: {backup_file}")
                    self.update_status("Database backup complete")
                    messagebox.showinfo("Success", f"Database backup complete: {backup_file}")
                else:
                    self.log_message("Database backup failed", "ERROR")
                    self.update_status("Database backup failed")
                    messagebox.showerror("Error", "Database backup failed")

            except Exception as e:
                self.log_message(f"Database backup failed: {e}", "ERROR")
                self.update_status("Database backup failed")
                messagebox.showerror("Error", f"Database backup failed: {e}")

    def cleanup_data(self):
        """Clean up data"""
        result = messagebox.askyesno("Confirm", "Are you sure you want to clean temporary data? This operation is irreversible.")
        if result:
            try:
                self.log_message("Starting temporary data cleanup")

                # Clean temporary files
                temp_files = [
                    "data_pipeline.log",
                    "user_analysis.log",
                    "database_migration.log"
                ]

                cleaned_count = 0
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        cleaned_count += 1

                self.log_message(f"Temporary data cleanup complete, {cleaned_count} files cleaned")
                messagebox.showinfo("Success", f"Temporary data cleanup complete, {cleaned_count} files cleaned")

            except Exception as e:
                self.log_message(f"Data cleanup failed: {e}", "ERROR")
                messagebox.showerror("Error", f"Data cleanup failed: {e}")

    def view_statistics(self):
        """View data statistics"""
        if not self.current_config:
            messagebox.showwarning("Warning", "Please configure database connection first")
            return

        try:
            processor = EnhancedDataProcessor(self.current_config)
            if not processor.connect_to_database():
                raise Exception("Unable to connect to database")

            stats = processor.get_data_quality_report()
            processor.close_connection()

            # Display statistics
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Data Statistics")
            stats_window.geometry("600x400")

            stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
            stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            stats_text.insert(tk.END, json.dumps(stats, indent=2, ensure_ascii=False, default=str))

        except Exception as e:
            self.log_message(f"Failed to get statistics: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to get statistics: {e}")

    def update_system_info(self):
        """Update system information"""
        try:
            import platform
            import psutil

            info = f"""System Information:
Operating System: {platform.system()} {platform.release()}
Python Version: {sys.version}
Memory Usage: {psutil.virtual_memory().percent:.1f}%
CPU Usage: {psutil.cpu_percent():.1f}%

Project Information:
Project Path: {os.getcwd()}
Configuration File: {self.config_file}
Configuration Status: {'Loaded' if self.current_config else 'Not Loaded'}
"""

            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)

        except ImportError:
            info = f"""System Information:
Operating System: {platform.system()} {platform.release()}
Python Version: {sys.version}

Project Information:
Project Path: {os.getcwd()}
Configuration File: {self.config_file}
Configuration Status: {'Loaded' if self.current_config else 'Not Loaded'}

Note: psutil library needs to be installed to display system resource usage
"""
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)

    def clear_log(self):
        """Clear log"""
        self.log_text.delete(1.0, tk.END)

    def save_log(self):
        """Save log"""
        filename = filedialog.asksaveasfilename(
            title="Save Log File",
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if filename:
            try:
                content = self.log_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)

                messagebox.showinfo("Success", f"Log saved to: {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {e}")

    def run(self):
        """Run GUI"""
        self.log_message("Xiaohongshu Data Analysis System started")
        self.root.mainloop()


def main():
    """Main function"""
    try:
        app = DataAnalysisGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        try:
            messagebox.showerror("Error", f"Failed to start GUI: {e}")
        except:
            pass


if __name__ == "__main__":
    main()
