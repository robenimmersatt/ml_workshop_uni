import numpy as np

def test_train_test_split(test_func):
    data = np.random.rand(620, 4, 3, 2, 1)

    splits = [0.0, 0.3, 0.6, 1.0]
    for split in splits:
        print(f'Split = {split}')
        splitted_data = test_func(data, split)
        assert len(splitted_data) == 2
        data_test, data_train = splitted_data

        test_shape = data_test.shape
        train_shape = data_train.shape

        for i, (s1, s2) in enumerate(zip(test_shape, train_shape)):
            if i == 0:
                train_shape_asserted = int(data.shape[0] * split)
                test_shape_asserted = int(data.shape[0] * (1 - split))
                print(f'Assert Train Len.: {train_shape_asserted}, Erhaltene Train Len.: {s1}')
                print(f'Assert Test Len.: {test_shape_asserted}, Erhaltene Train Len.: {s2}')
                assert s1 == train_shape_asserted
                assert s2 == test_shape_asserted
            else:
                assert s1 == s2