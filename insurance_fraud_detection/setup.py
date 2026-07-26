from setuptools import setup, find_packages

setup(
    name="insurance_fraud_detection",
    version="1.0.0",
    description="Insurance Fraud Detection using Deep Neural Networks — PGCP-AI ACTS Pune",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "imbalanced-learn>=0.11.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
    ],
)
