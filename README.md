# StatViz.py: A Random Variable Analysis Tool

This project, **StatViz.py**, provides a graphical user interface (GUI) for analyzing random variables and their statistics. It allows users to input random variables (single or multiple) and their functions, and then generate plots and statistical values to better understand their distributions. This project is developed for the Probability and Stochastic Processes (CIE 327) course.


![statviz-1](https://github.com/user-attachments/assets/615ef2e5-464d-4998-b179-6f18f17d6b0d)
![statviz-2](https://github.com/user-attachments/assets/b8f71757-43a7-4fc2-b420-32384b131810)

## Installation
StatViz uses `poetry` for the ease of packaging and updates
1. **Install poetry**
For linux machines
```pipx install poetry```
or
```pip install poetry```
For windows 

2. **Clone the repo and Build**
```
git clone https://github.com/salastro/statviz.py.git
cd statviz.py/src
poetry install 
```

3. **Run**
```poetry run statviz```

## Features

StatViz.py is designed to handle the following:

1.  **Single Random Variable Analysis:**
    *   Input: A single random variable defined by its sample space.
    *   Outputs:
        *   Probability distribution plot and cumulative distribution function (CDF) plot.
        *   Calculation and display of the mean, variance, and third moment.
        *   Moment generating function (MGF) plot with t varying from 0 to a user-defined `tmax`
        *   Plot the first and second derivatives of the MGF, and their values at `t = 0`.

2.  **Joint Random Variable Analysis:**
    *   Input: Two random variables defined as pairs in the sample space.
    *   Outputs:
        *   Plot of the joint probability distribution.
        *   Plots of the marginal probability distributions.
        *   Calculation of the covariance.
        *   Calculation of the correlation coefficient.

3.  **Functions of Random Variables:**
    *   Input: Two random variables, X and Y, provided as pairs.
    *   Outputs:
        *   (Bonus) User can define new random variables Z and W as functions of X and Y.
        *   Plot of the probability distribution of Z = 2X - 1.
        *   Plot of the probability distribution of W = 2 - 3Y.
        *   Plot of the joint probability distribution of Z and W.

## Usage

1.  **Execution:** Run `StatViz.py` (or its equivalent compiled executable) to launch the GUI.
2.  **Input:** The application allows you to input random variables according to the requirements in the specification.  The input sample space data can be loaded using example `.m` files that are provided.
3.  **Navigation:** Use the GUI to access different functions for analyzing random variables
4.  **Output:** Plots and statistical values are shown in their respective areas within the GUI.
5.  **Test Cases:** The GUI can be tested using these examples
    *   Single Random Variables:
        *   Use the sample file
        *   X2 ~ U(-5, 2)
        *   X3 ~ N(3, 4)
        *   X4 ~ Bin(5, 0.3)
        *   X5 ~ Poisson(10)
    *   Joint Random Variables
        *   Use the sample file
        *    X2 ~ N(3,4) and Y2 ~ N(-5, 2)
        *    X3 ~ Gamma(2,10) and Y3 ~ Bin(4, 0.5)
        *    X4 ~ Exp(0.05) and Y4 = 3X4 + 2
        *     X5 ∈ {-1,1} with uniform distribution, and Y5 = X5 + n, where n ~ N(0,0.5)

## Project Structure

This project is delivered as a package of files:

*   **Executable:** An executable file that starts the GUI.
*   **Source Code:** The source files written in Matlab or equivalent.
*   **Test Outputs:** Output from the GUI when run with the test cases.
*   **Sample Data:** The files used to generate random variables for the test cases.
*   **Project Report:** A PDF document with screenshots, results, and explanations from the application.
*   **Video:** A video demonstration of the GUI in action.

## Dependencies

The GUI is built using a Python library called `PyQt6` for the graphical interface. The statistical calculations and plotting are done using `numpy`, `scipy`, and `matplotlib` libraries. These libraries are required to run the application.

## Team Information

This project was developed by a team of students from the Department of Communication and Information Engineering at the University of Science and Technology in Zewail City, Egypt. The team members are: SalahDin A. Rezk, and Marwan B. Ragaa.
