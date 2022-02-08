# augmented_3d_mnist
3D version of MNIST dataset with xyz rotation and colors

![image](https://user-images.githubusercontent.com/9665358/153072218-0dcfc9fd-ed90-4b2c-ae67-a10909bc91b2.png)

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

print_grid(train_x_3d[42272])
```
![image](https://user-images.githubusercontent.com/9665358/153072156-274d4992-663c-48cf-8227-033a1344ff76.png)

