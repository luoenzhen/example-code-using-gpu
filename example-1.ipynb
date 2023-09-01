import tensorflow as tf
import numpy as np

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    # Create a simple TensorFlow computation on GPU
    with tf.device('/device:GPU:0'):
        print("using GPU..")
        tensor_size = 10000
        tensor_a = tf.random.normal((tensor_size, tensor_size))
        tensor_b = tf.random.normal((tensor_size, tensor_size))
        tensor_c = tf.random.normal((tensor_size, tensor_size))
        
        # Element-wise addition and multiplication
        result_add = tf.add(tensor_a, tensor_b)
        result_multiply = tf.multiply(tensor_a, tensor_b)
        
        # Dot product
        result_dot = tf.tensordot(tensor_a, tensor_b, axes=1)
        
        # Matrix-vector multiplication
        vector = tf.random.normal((tensor_size,))
        result_matvec = tf.linalg.matvec(tensor_a, vector)
        
        # Batch matrix multiplication
        batch_size = 10
        batch_tensors_a = tf.random.normal((batch_size, tensor_size, tensor_size))
        batch_tensors_b = tf.random.normal((batch_size, tensor_size, tensor_size))
        result_batch_matmul = tf.matmul(batch_tensors_a, batch_tensors_b)
        
        # Reduction along axes
        sum_along_axis0 = tf.reduce_sum(tensor_c, axis=0)
        mean_along_axis1 = tf.reduce_mean(tensor_c, axis=1)
        
        # Print results
        print("Element-wise addition result:\n", result_add)
        print("Element-wise multiplication result:\n", result_multiply)
        print("Dot product result:\n", result_dot)
        print("Matrix-vector multiplication result:\n", result_matvec)
        print("Batch matrix multiplication result:\n", result_batch_matmul)
        print("Sum along axis 0:\n", sum_along_axis0)
        print("Mean along axis 1:\n", mean_along_axis1)
