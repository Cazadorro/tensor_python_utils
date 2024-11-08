import numpy as np
import random
import sys
def generate_sparse_tensor_2d(sparsity_fraction : float, shape : tuple, fill_diagonal : bool = True) -> np.ndarray:
    # create an array of uniformly random values, set only sparsity fraction of the tensor to 1.
    empty_fraction =  np.clip(1.0 - sparsity_fraction, 0, 1.0)
    assert(len(shape) == 2)
    random_tensor = np.random.uniform(0.0, 1.0, size = shape)
    random_tensor[random_tensor < empty_fraction] = 0.0
    random_tensor[random_tensor >= empty_fraction] = 1.0


    if fill_diagonal:
        np.fill_diagonal(random_tensor, 1.0)

    return random_tensor

def generate_sparse_tensor_3d(sparsity_fraction : float, shape : tuple, slice_diagonal_fill_fraction : float)  -> np.ndarray:
    assert (len(shape) == 3)
    random_tensor = np.zeros(shape, dtype=float)
    shape_2d = shape[1:]
    for i in range(shape[0]):
        fill_diagonal = random.uniform(0, 1) < slice_diagonal_fill_fraction
        random_tensor[i, :] = generate_sparse_tensor_2d(sparsity_fraction, shape_2d, fill_diagonal)

    return random_tensor


class CooTensor:
    def __init__(self, values : np.ndarray, indices : np.ndarray, shape : tuple):
        assert(len(values) == len(indices))
        self.shape = shape
        self.values = values
        self.indices = indices


    def __len__(self):
        return len(self.values)

    @classmethod
    def from_file(cls, file_path : str) -> 'CooTensor':
        with open(file_path, "r") as file:
            mode_count : int = int(file.readline())
            shape : tuple[int,...] = tuple([int(size) for size in file.readline().split()])
            coo_entries = file.read()
            values = []
            indices = []
            for coo_entry in coo_entries:
                coo_split = coo_entry.split()
                values.append(float(coo_split[mode_count - 1]))
                indices.append([int(x) - 1 for x in coo_split[:mode_count - 1]])

            return cls(np.array(values), np.array(indices), shape)

    @classmethod
    def from_numpy(cls, array : np.ndarray) -> 'CooTensor':
        indices = np.array([index for index, value in np.ndenumerate(array) if value != 0.0])
        values = np.array([value for index, value in np.ndenumerate(array) if value != 0.0])
        return cls(values, indices, array.shape)

    def print_indices(self):
        with np.printoptions(threshold=sys.maxsize):
            print(self.indices)

    def print_values(self):
        with np.printoptions(threshold=sys.maxsize):
            print(self.values)

    def save(self, file_path : str, one_indexed : bool = True):
        with open(file_path, "w") as file:
            file.write("{}\n".format(len(self.shape)))
            file.write(" ".join([str(s) for s in self.shape]) + "\n")
            for index, value in zip(self.indices, self.values):
                if value == 0:
                    continue
                index_string = " ".join([str(s + 1 if one_indexed else s) for s in index])
                file.write("{} {}\n".format(index_string, value))

def numpy_to_tns(output_file_name: str, arr: np.ndarray, one_indexed: bool = True):
    with open(output_file_name, "w") as file:
        file.write("{}\n".format(len(arr.shape)))
        file.write(" ".join([str(s) for s in arr.shape]) + "\n")
        for index, value in np.ndenumerate(arr):
            # we don't store zeros in sparse matrix formats.
            if value == 0:
                continue
            index_string = " ".join([str(s + 1 if one_indexed else s) for s in index])
            file.write("{} {}\n".format(index_string, value))


def numpy_to_csr(output_file_name: str, arr: np.ndarray, one_indexed: bool = True):
    with open(output_file_name, "w") as file:
        file.write(" ".join([str(s) for s in arr.shape]) + " ")

        row_array = [0]
        col_array = []
        val_array = []
        for row in range(arr.shape[0]):
            row_array.append(row_array[row])
            for col in range(arr.shape[1]):
                value = arr[row, col]
                if value == 0:
                    continue
                row_array[row + 1] += 1
                col_array.append(col)
                val_array.append(1.0)
        for i in range(len(row_array)):
            row_array[i] += 1
        for i in range(len(col_array)):
            col_array[i] += 1
        file.write(" {} \n".format(len(col_array)))
        all_array = row_array + col_array + val_array
        file.write(" ".join([str(s) for s in all_array]))
def main():
    np_tensor_2d = generate_sparse_tensor_2d(0.01, (32,32))
    coo_2d = CooTensor.from_numpy(np_tensor_2d)

    coo_2d.print_indices()
    coo_2d.print_values()
    print(len(coo_2d.indices))

    np_tensor_3d = generate_sparse_tensor_3d(0.01, (4,32,32), 0.3)

    coo_3d = CooTensor.from_numpy(np_tensor_3d)

    coo_3d.print_indices()
    coo_2d.save("test_tensor_script_2d.tns")
    coo_3d.save("test_tensor_script_3d.tns")

    # 32x32x32, 32x32x512, 512x512x512, 1024,
    print(len(coo_3d))



if __name__ == '__main__':
    main()
