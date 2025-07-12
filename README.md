
<html lang="en">


  <h1>🧠 Artificial Neural Network: Customer Churn Prediction</h1>

  <p>This project uses an <strong>Artificial Neural Network (ANN)</strong> to predict whether a customer will churn (exit) from a bank. Implemented in both <strong>Python (TensorFlow)</strong> and <strong>R (H2O.ai)</strong>.</p>

  <h2>📁 Project Structure</h2>
  <pre><code>
ann-customer-churn-py-r/
│
├── dataset/
│   └── churn_modelling.csv
├── python/
│   └── ann.py
├── r/
│   └── ann.R
├── README.html
</code></pre>

  <h2>🎯 Objective</h2>
  <p>To classify whether a customer will churn using bank customer data based on features like credit score, balance, tenure, geography, and more.</p>

  <h2>🚀 Technologies Used</h2>
  <ul>
    <li><strong>Python</strong> – TensorFlow / Keras</li>
    <li><strong>R</strong> – H2O.ai</li>
    <li><strong>Preprocessing:</strong> OneHotEncoder, MinMaxScaler</li>
    <li><strong>Metrics:</strong> Accuracy, Confusion Matrix</li>
  </ul>

  <h2>🔍 Features in Dataset</h2>
  <ul>
    <li>CreditScore, Geography, Gender</li>
    <li>Age, Tenure, Balance</li>
    <li>Number of Products, Has Credit Card</li>
    <li>Is Active Member, Estimated Salary</li>
    <li><strong>Target:</strong> Exited (0 = No, 1 = Yes)</li>
  </ul>

  <h2>📈 Model Summary</h2>

  <h3>Python (TensorFlow)</h3>
  <ul>
    <li>Dense layers: [6, 6, 1]</li>
    <li>Activations: ReLU (hidden), Sigmoid (output)</li>
    <li>Loss: Binary Crossentropy</li>
    <li>Optimizer: Adam</li>
  </ul>

  <h3>R (H2O)</h3>
  <ul>
    <li>Activation: Rectifier</li>
    <li>Hidden layers: [6, 6]</li>
    <li>Epochs: 100</li>
    <li>Auto standardization</li>
  </ul>

  <h2>▶️ How to Run</h2>

  <h3>Python</h3>
  <ol>
    <li>Install dependencies:<br>
      <pre><code>pip install pandas numpy matplotlib seaborn scikit-learn tensorflow</code></pre>
    </li>
    <li>Run the model:<br>
      <pre><code>python python/ann_tensorflow.py</code></pre>
    </li>
  </ol>

  <h3>R</h3>
  <ol>
    <li>Install packages:<br>
      <pre><code>install.packages(c("caTools", "h2o"))</code></pre>
    </li>
    <li>Run the script:<br>
      <pre><code>source("r/ann_h2o.R")</code></pre>
    </li>
  </ol>

  <h2>📊 Output</h2>
  <ul>
    <li>Confusion Matrix (Python and R)</li>
    <li>Training Accuracy & Loss Curve (Python)</li>
    <li>H2O Deep Learning summary output (R)</li>
  </ul>

  <h2>👨‍💻 Author</h2>
  <ul>
    <li><strong>Name:</strong> Chukwuka Chijioke Jerry</li>
    <li><strong>Email:</strong> chukwuka.jerry@gmail.com | chukwuka_jerry@yahoo.com</li>
    <li><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/chukwukacj/" target="_blank">linkedin.com/in/chukwukajerry</a></li>
    <li><strong>Twitter (X):</strong> <a href="https://twitter.com/Mazimum_" target="_blank">@Mazimum_</a></li>
    <li><strong>WhatsApp:</strong> +2348038782912</li>
  </ul>

  <p>⭐ If you found this helpful, give the repo a star and share your thoughts!</p>

</body>
</html>
