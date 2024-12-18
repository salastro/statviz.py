# StatViz.py: A Random Variable Analysis Tool

This project, **StatViz.py**, provides a graphical user interface (GUI) for analyzing random variables and their statistics. It allows users to input random variables (single or multiple) and their functions, and then generate plots and statistical values to better understand their distributions. This project is developed for the Probability and Stochastic Processes (CIE 327) course.

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

The GUI is built using a Matlab or any other suitable software package.

## Team Information

This project was created by a team of 2-3 students from the Communications and Information Engineering Program, Zewail City.

## Grading Criteria

The project will be assessed based on the following criteria:

*   **Completeness and Correctness of Deliverables (50%):**  All features function as specified, and all calculations and visualizations are correct.
*   **GUI Design and Ease of Use (10%):** How clear and easy is it to use the GUI.
*   **Report Writing and Organization (20%):**  The quality and completeness of the project report with proper labeling and explanations.
*   **Comprehensiveness and Clarity of Content in Recorded Video (20%):**  A clear demonstration and explanation of the GUI functionality in the provided video.

## Grading Notes

*   The project will be evaluated for completeness, correctness, user interface, and report quality.
*   No sharing of reports is allowed between groups. If duplicate reports are submitted, both reports will be given a 0.
*   Late submissions will be penalized 10% per day, for up to 5 days late, after which, submissions will not be graded.

## Contact

For any questions or clarifications, please reach out to [insert contact information here]
