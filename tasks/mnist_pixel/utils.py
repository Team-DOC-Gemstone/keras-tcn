import numpy as np
from tensorflow.keras.utils import to_categorical
import preprocessing

def data_generator():
    print("Running data generator")
    # input image dimensions
    abnormal_scans, abnormal_labels = preprocessing.get_abnormal()
    normal_scans, normal_labels = preprocessing.get_normal()
    x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
    y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
    x_test = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
    y_test = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
    img_rows, img_cols = 512, 512
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, img_rows * img_cols, 1)
    x_test = x_test.reshape(-1, img_rows * img_cols, 1)

    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(data_generator())
