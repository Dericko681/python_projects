import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

def generate_dataset(min_num=1, max_num=30, size=10000):
    numbers = np.random.randint(min_num, max_num+1, size=size)
    fib_numbers = np.array([fibonacci(n) for n in numbers], dtype=np.float32)
  
    fib_numbers = np.log(fib_numbers + 1e-8)
    
    return numbers.reshape(-1, 1), fib_numbers

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=[1]),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    X, y = generate_dataset(max_num=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = create_model()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f} (in log space)")
 
    test_numbers = np.array([[5], [10], [15], [20], [25], [30]])
    test_numbers_scaled = scaler.transform(test_numbers)
    
    predictions = model.predict(test_numbers_scaled)
    predictions = np.exp(predictions) - 1e-8 
    
    print("\nPrediction Examples:")
    for num, pred in zip(test_numbers, predictions):
        actual = fibonacci(num[0])
        print(f"Input: {num[0]}, Predicted: {pred[0]:.1f}, Actual: {actual}, Difference: {abs(pred[0]-actual)/actual*100:.1f}%")

if __name__ == "__main__":
c    main()