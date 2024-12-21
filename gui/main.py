import sys

from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow

from gui import Ui_MainWindow
# from ..src import single_random_variable, joint_random_variable, function_of_random_variable, test_generator


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup_connections()
        self.file_path: str = ""
        self.results: str = ""
        self.analysis_mode: str = "Single Random Variable"

    def setup_connections(self):
        # Connect buttons and other UI elements to their respective slots
        self.OpenFileBt.clicked.connect(self.open_file)
        self.AnalyzeBt.clicked.connect(self.analyze)
        self.SaveResultsBt.clicked.connect(self.save_results)
        self.AnalysisModeBox.currentIndexChanged.connect(self.change_analysis_mode)

    def change_analysis_mode(self, index):
        # Change the analysis mode based on the selected index
        self.analysis_mode = self.AnalysisModeBox.itemText(index)
        print(f"Selected analysis mode: {self.analysis_mode}")
        if self.analysis_mode == "Single Randomo Variable":
            pass
        else:
            pass

    def disable_elements(self, elements):
        """takes a list of elements and disables them all"""
        for element in elements:
            element.setStyleSheet(
                """
                QLabel:disabled { color: #666666; }
            """
            )
            element.setEnabled(False)

    def open_file(self):
        """
        Open a file dialog to select samples .mat file
        """
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("MAT files (*.mat)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()[0]
            self.file_path = selected_files
            print(f"Selected file: {self.file_path}")
        else:
            print("No file selected")

    def analyze(self):
        # Logic for analyzing the file
        print("Analyze button clicked")

    def save_results(self):
        # Logic for saving results
        print("Save results button clicked")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
