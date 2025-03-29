import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def generate_random_vector():
    random_number = random.randint(10, 100)
    print(f"Generated random number: {random_number}")
    vector = list(range(1, random_number+1))
    return vector


def fibonacci(n):
    if n<=0:
        return 0
    elif n== 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(n-1):
            a, b = b, a+b
        return b
def vector_fibonacci(numbers):
    return[fibonacci(num) for num in numbers]

def split_list(list):
    split_index = round(len(list)*0.6)
    training_set = list[:split_index]
    test_set = list[split_index:]
    return training_set, test_set

def train_model(xs):
    xs = np.array(xs, dtype=np.int64)
    ys = np.array(vector_fibonacci(xs), dtype=np.int64)
    model = Sequential([Dense(units=1, input_shape=[1])])
    model.compile(
        optimizer='sgd',
        loss='mean_squared_error'
    )
    model.fit(ys, xs, epochs=2)
    return model

def test_model(test_set, model):
    correct = 0
    total = len(test_set)

    for element in test_set:
        model_output = model.predict(np.array([[element]]))[0][0]
        fibonacci_number = fibonacci(element)
        if abs(model_output - fibonacci_number) <= 0.1:
            correct = correct+1
        else:
            print(f" Failed!  number:{element}, fibonacci:{fibonacci_number}, model's output:{model_output}")
    accuracy = (correct/total)*100
    print(f"correct={correct}  and total={total}")
    print(f"model's acuracy: {accuracy:.2f}%")
    return accuracy
   
    



generated_vector = generate_random_vector()
training_set, test_set = split_list(generated_vector)
xs =  training_set # np.array(training_set, dtype=np.int64)
ys = vector_fibonacci(xs)   # np.array(vector_fibonacci(xs), dtype=np.int64)

model = train_model(xs)
test_model(test_set, model)



