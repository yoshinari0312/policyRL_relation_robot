from accelerate import Accelerator
acc = Accelerator()
print("process_index:", acc.process_index, "num_processes:", acc.num_processes)