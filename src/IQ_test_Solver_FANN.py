import libfann

connection_rate = 1
learning_rate = 0.7
num_input = 2
num_hidden = 4
num_output = 1

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

ann = libfann.neural_net()
ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file('E:/FANN_workspace/xor.txt', max_iterations, iterations_between_reports, desired_error)

ann.save("E:/xor_py.net")