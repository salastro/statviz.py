import sys
from io import StringIO

from PyQt6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from statviz.analysis import (functions_of_random_variables,
                              joint_random_variables, single_random_variable)
from statviz.gui.gui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup_connections()
        # Initialize the file path and results
        self.file_path: str = ""
        self.results: str = ""

        # Initialize the analysis mode
        self.analysis_mode: str = "Single Random Variable"
        self.change_analysis_mode(0)

        # Initialize the analysis parameters
        self.t_value: float = self.tValueNumber.value()
        self.Z_func: str = self.ZText.text()
        self.W_func: str = self.WText.text()

        # Disable editing of the results text box
        self.ResultsText.setReadOnly(True)

        print("Main window initialized")
        print("Analysis mode:", self.analysis_mode)
        print(
            "Analysis parameters: t_value =",
            self.t_value,
            ", Z_func =",
            self.Z_func,
            ", W_func =",
            self.W_func,
        )

    def setup_connections(self):
        # Connect buttons and other UI elements to their respective slots
        self.OpenFileBt.clicked.connect(self.open_file)
        self.AnalyzeBt.clicked.connect(self.analyze)
        self.SaveResultsBt.clicked.connect(self.save_results)
        self.AnalysisModeBox.currentIndexChanged.connect(self.change_analysis_mode)

    def change_analysis_mode(self, index):
        # Change the analysis mode based on the selected index
        single_mode_elements = [self.t_label, self.tValueNumber]
        joint_mode_elements = [self.Z_label, self.ZText, self.W_label, self.WText]
        self.analysis_mode = self.AnalysisModeBox.itemText(index)
        print(f"Selected analysis mode: {self.analysis_mode}")
        if self.analysis_mode == "Single Random Variable":
            self.enable_elements(single_mode_elements)
            self.disable_elements(joint_mode_elements)
        elif self.analysis_mode == "Joint Random Variable":
            self.disable_elements(joint_mode_elements)
            self.disable_elements(single_mode_elements)
        elif self.analysis_mode == "Function of Random Variable":
            self.enable_elements(joint_mode_elements)
            self.disable_elements(single_mode_elements)
        else:
            print("Invalid analysis mode selected")

    def enable_elements(self, elements):
        """takes a list of elements and enables them all"""
        for element in elements:
            element.setStyleSheet(
                """
                QLabel:enabled { color: #000000; }
            """
            )
            element.setEnabled(True)

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
        if self.analysis_mode == "Single Random Variable":
            self.single_random_variable()
        elif self.analysis_mode == "Joint Random Variable":
            self.joint_random_variable()
        elif self.analysis_mode == "Function of Random Variable":
            self.functions_of_random_variables()
        else:
            print("Invalid analysis mode selected")

    def save_results(self):
        default_filename = "output.txt"
        if not self.ResultsText.toPlainText().strip():
            QMessageBox.warning(
                self,
                "No Content",
                "There is no content to save. Please generate some output first.",
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Output", default_filename, "Text Files (*.txt);;All Files (*)"
        )

        if file_name:
            try:
                with open(file_name, "w", encoding="utf-8") as file:
                    # Write the content to the file
                    file.write(self.ResultsText.toPlainText())

                # Show success message
                QMessageBox.information(
                    self, "Success", f"Output has been saved to:\n{file_name}"
                )

            except Exception as e:
                # Show error message if something goes wrong
                QMessageBox.critical(
                    self, "Error", f"An error occurred while saving the file:\n{str(e)}"
                )
        # Logic for saving results
        print("Save results button clicked")

    def single_random_variable(self):
        sys.argv = [
            "single_random_variable.py",
            "-f",
            self.file_path,
            "-t",
            str(self.t_value),
        ]
        results = StringIO()
        single_random_variable.main(results)
        self.results = results.getvalue()
        self.ResultsText.setPlainText(self.results)

    def joint_random_variable(self):
        sys.argv = ["joint_random_variables.py", "-f", self.file_path]
        results = StringIO()
        joint_random_variables.main(results)
        self.results = results.getvalue()
        self.ResultsText.setPlainText(self.results)

    def functions_of_random_variables(self):
        sys.argv = [
            "function_of_random_variable.py",
            "-f",
            self.file_path,
            "-Z",
            self.Z_func,
            "-W",
            self.W_func,
        ]
        results = StringIO()
        functions_of_random_variables.main(results)
        self.results = results.getvalue()
        self.ResultsText.setPlainText(self.results)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
