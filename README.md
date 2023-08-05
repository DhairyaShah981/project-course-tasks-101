# Project Description
## Task1:
Create a streamlit app for neural network having following features:
- It should have various controls such as number of neurons in each layer, lr, epochs etc.
- Show the fit on various datasets.
- Show the probabilities on the contour plot and not the discrete predictions.
- Have an option of including various transformations to the inputs (basis functions): Sine basis, Gaussian basis, Polynomial basis and show the effect of each on the fit via the same contour plots.
- Have an option of enabling MC dropout method for getting uncertainty.

#### Lint to the Hosted Streamlit:
```
https://project-course-tasks-101.streamlit.app/
```

#### To use in localhost
- In your terminal, go to directory containing streamlit_app.py.
- Install requirements.txt
- Run the following command in the terminal: streamlit run streamlit_app.py.

## Task2:
Fit a CNN model from scratch on MNIST dataset and try to get the maximum accuracy by hyperparameter tuning. Apply MC dropout method and compare your results again with non-MC dropout method.

a. For Non-MC dropout model, remove the final layer and plot TSNE plot from the fitted model's outputs.

b. Predict on some Fashion MNIST data points and check the predictive accuracy and uncertainty in predictions.

c. Repeat a. for MNIST + Fashion MNIST points and see if your see a pattern, something like if Fashion MNIST points are far from the other points.
