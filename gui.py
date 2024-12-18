import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QComboBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Assuming rv.py, jrv.py, and frv.py contain the necessary functions
import rv  # Assuming rv.py has functions: plot_dist, calc_stats, plot_mgf


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Variable Analyzer")

        self.layout = QVBoxLayout()
        self.create_rv_section()
        self.create_jrv_section()
        self.create_frv_section()

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def create_rv_section(self):
        section_layout = QVBoxLayout()
        section_layout.addWidget(QLabel("Section 1: Single Random Variable"))

        self.rv_filepath = QLineEdit()
        self.rv_filepath.setPlaceholderText("Path to .m file")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file_rv)

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.rv_filepath)
        file_layout.addWidget(browse_button)
        section_layout.addLayout(file_layout)


        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze_rv)
        section_layout.addWidget(analyze_button)

        self.rv_figure = Figure(figsize=(5, 4), dpi=100)
        self.rv_canvas = FigureCanvas(self.rv_figure)
        section_layout.addWidget(self.rv_canvas)

        self.layout.addLayout(section_layout)



    def create_jrv_section(self):
        # ... (Similar structure to create_rv_section)
        pass  # Implement the layout and functionality for joint random variables

    def create_frv_section(self):
        # ... (Similar structure to create_rv_section)
        pass  # Implement the layout and functionality for functions of random variables


    def browse_file_rv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open .m file", filter="*.m")
        if filepath:
            self.rv_filepath.setText(filepath)

    def analyze_rv(self):
        try:
            filepath = self.rv_filepath.text()
            samples = np.loadtxt(filepath) #  Load data assuming .m file is just data

            # Clear previous plots
            self.rv_figure.clear()
            ax = self.rv_figure.add_subplot(111)

            # Example usage of functions from rv.py (adapt as needed)
            rv.plot_dist(samples, ax)  # Plot distribution on the provided axes
            mean, variance, third_moment = rv.calc_stats(samples)
            # ... (Display the stats, plot MGF, etc.)
            self.rv_canvas.draw() # Redraw the canvas



        except FileNotFoundError:
            QMessageBox.warning(self, "Error", "File not found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
