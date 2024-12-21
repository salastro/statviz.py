import sys

from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow

from analysis import single_random_variable
from gui.gui import Ui_MainWindow


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
        elif (
            self.analysis_mode == "Joint Random Variable"
            or self.analysis_mode == "Function of Random Variable"
        ):
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
        elif self.analysis_mode == "Joint Random Variable" :
            self.joint_random_variable()
        else:
            print("Invalid analysis mode selected")


    def save_results(self):
        # Logic for saving results
        print("Save results button clicked")

    def single_random_variable(self):
        # Input file
        filename = self.file_path
        t_max = self.t_value
        X = single_random_variable.read_file_single(filename)
        P = single_random_variable.calc_prob(X)

        # Step 1: Plot Probability Distribution and CDF
        single_random_variable.plot_prob_cdf(X, P)

        # Step 2: Calculate and Display Statistical Measures
        mean_X, var_X, third_moment = single_random_variable.calc_stats(X, P)
        print("\n=== Results ===")
        print("\nStatistical Measures:")
        print(f"Mean = {mean_X:.4f}")
        print(f"Variance = {var_X:.4f}")
        print(f"Third Moment = {third_moment:.4f}")

        # Step 3: Plot MGF and Derivatives
        MGF, MGF_prime, MGF_double_prime = single_random_variable.calc_mgf_deriv(X, P, t_max)
        MGF_0, MGF_prime_0, MGF_double_prime_0 = MGF[0], MGF_prime[0], MGF_double_prime[0]
        print("\nValues at t = 0:")
        print(f"M(0) = {MGF_0:.4f}")
        print(f"M'(0) = {MGF_prime_0:.4f} (Mean)")
        print(f"M''(0) = {MGF_double_prime_0:.4f}")
        single_random_variable.plot_mgf_deriv(MGF, MGF_prime, MGF_double_prime, t_max)
        single_random_variable.plt.show(block=True)  # Keep the program alive until plots closed



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
