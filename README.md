# Machine Learning Fundamentals

A comprehensive collection of Jupyter notebooks covering the mathematical foundations and fundamental algorithms essential for machine learning. This repository provides both theoretical explanations and practical Python implementations using NumPy, Matplotlib, and SciPy.

## 📚 Contents

### 1. **Calculus and Optimization** (`calculus.ipynb`)
A deep dive into calculus concepts that form the mathematical backbone of machine learning optimization.

**Topics Covered:**
- **Limits and Continuity**: Understanding function behavior and convergence
- **Derivatives**: Rate of change and slope calculations
- **Partial Derivatives and Gradients**: Multivariate calculus for ML
- **Gradient Descent**: Core optimization algorithm for training ML models
- **Optimization Techniques**: Practical applications in minimizing loss functions

**Key Concepts:**
- Mathematical formulas with KaTeX notation
- Real-world ML applications
- Convergence analysis
- Loss function optimization

---

### 2. **Linear Algebra with NumPy** (`linear_algebra.ipynb`)
Comprehensive coverage of linear algebra operations essential for machine learning, data science, and scientific computing.

**Topics Covered:**
- **Systems of Linear Equations**: Matrix form (Ax = b) and solution methods
- **Matrix Operations**: Multiplication, inversion, and properties
- **Eigenvalues and Eigenvectors**: Finding principal directions and magnitudes
- **Determinants**: Matrix invertibility and properties
- **Matrix Properties**: Trace, characteristic polynomial, and transformations

**Techniques:**
- Matrix inversion method
- NumPy's efficient linear solvers (`np.linalg.solve()`)
- Eigenvalue decomposition
- Practical numerical computing

---

### 3. **Principal Component Analysis (PCA)** (`pca.ipynb`)
An in-depth exploration of PCA, one of the most powerful dimensionality reduction techniques in machine learning.

**Why PCA?**
- Combat the curse of dimensionality
- Reduce computational complexity
- Eliminate noise and redundant features
- Enable data visualization
- Extract meaningful patterns

**Mathematical Foundations:**
1. **Data Standardization**: Scale features uniformly
2. **Covariance Matrix**: Measure feature relationships
3. **Eigenvalue Decomposition**: Find principal components
4. **Component Selection**: Choose optimal k dimensions
5. **Data Projection**: Transform to lower-dimensional space

**Applications:**
- Image compression
- Stock market trend analysis
- Genetics and gene expression analysis
- Feature extraction
- Data visualization (2D/3D projections)

**Includes:**
- Step-by-step mathematical formulas
- Python implementation from scratch
- Explained variance analysis
- Practical examples with visualizations

---

### 4. **Perceptron Algorithm** (`perceptron.ipynb`)
Implementation of the perceptron, one of the fundamental building blocks of neural networks and deep learning.

**Exercises Included:**

#### **Exercise 1: OR Problem**
Implementation of perceptron for solving the OR logic gate problem.
- **Online Learning Method**: Updates weights after each sample
- **Offline Learning Method**: Batch updates using all samples

#### **Exercise 2: Multiple Perceptrons**
Advanced exercise using two perceptrons to solve more complex logical operations:
- One perceptron produces OR output
- Another produces NOR output
- Demonstrates multi-perceptron architectures

**Key Features:**
- Random weight initialization
- Bias handling
- Learning rate (α = 0.15)
- Error tracking and visualization
- Cost function minimization
- Epoch-based training

**Implementation Details:**
- NumPy for matrix operations
- Matplotlib for error convergence plots
- Step-by-step training process logging
- Both online and offline learning strategies

---

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computing and linear algebra
- **Matplotlib**: Data visualization and plotting
- **SciPy**: Scientific computing (optimization)

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:

```bash
pip install numpy matplotlib scipy jupyter
```

### Running the Notebooks

1. Clone this repository:
```bash
git clone https://github.com/CypherKingdom/Machine_Learning.git
cd Machine_Learning
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open any notebook file (`.ipynb`) and run the cells sequentially

---

## 📖 Learning Path

**Recommended order for beginners:**

1. **Start with `linear_algebra.ipynb`**: Build a strong foundation in matrix operations and linear systems
2. **Move to `calculus.ipynb`**: Understand derivatives, gradients, and optimization
3. **Explore `pca.ipynb`**: Apply linear algebra and statistics to dimensionality reduction
4. **Practice with `perceptron.ipynb`**: Implement your first machine learning algorithm

---

## 🎯 Learning Objectives

By working through these notebooks, you will:

- ✅ Master the mathematical foundations of machine learning
- ✅ Understand optimization algorithms like gradient descent
- ✅ Perform linear algebra operations using NumPy
- ✅ Implement dimensionality reduction with PCA
- ✅ Build and train basic neural network components (perceptron)
- ✅ Visualize algorithm convergence and results
- ✅ Apply mathematical concepts to real-world ML problems

---

## 📝 Notes

- All notebooks contain both **theory** and **practice**
- Mathematical formulas are rendered using **KaTeX** notation
- Code examples use **NumPy** for efficient computation
- Visualizations help understand algorithm behavior
- None of the cells have been executed yet - run them yourself to see the results!

---

## 🤝 Contributing

Feel free to:
- Add more examples
- Improve explanations
- Fix any issues
- Suggest new topics

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🔗 Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

---

**Happy Learning! 🎓**
