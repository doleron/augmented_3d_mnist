# augmented_3d_mnist
3D version of MNIST dataset with xyz rotation and colors

# Generating the dataset

```python
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x_3d = np.zeros((len(train_x), new_size, new_size, new_size, 3))
test_x_3d = np.zeros((len(test_x), new_size, new_size, new_size, 3))
        
for i in range(len(train_x)):
    train_x_3d[i] = transform_instance(train_x[i])

for i in range(len(test_x)):
    test_x_3d[i] = transform_instance(test_x[i])

print_grid(train_x_3d[2])
```

