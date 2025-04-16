import pandas as pd
import numpy as np

def generate_synthetic_dataset(n_samples=100, random_seed=50):
    # Set a random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random data with more features and some correlations
    feature1 = np.random.rand(n_samples)
    feature2 = np.random.rand(n_samples)
    feature3 = feature1 * 0.5 + np.random.rand(n_samples) * 0.5  # Correlated with feature1
    feature4 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)  # Normally distributed

    # Categorical variable with more categories
    categorical_variable = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

    # Target variable with some dependency on features
    target_variable = (feature1 + feature2 + np.random.rand(n_samples) > 1.5).astype(int)

    # Compile the data into a dictionary
    data = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'categorical_variable': categorical_variable,
        'target_variable': target_variable
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('enhanced_dataset.csv', index=False)

    print("Enhanced synthetic dataset created and saved as 'enhanced_dataset.csv'.")

# Generate the dataset
generate_synthetic_dataset(n_samples=200, random_seed=42)