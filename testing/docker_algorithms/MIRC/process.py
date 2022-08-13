import numpy as np
from settings import loader_settings
import medpy.io
import os, pathlib
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_fill_holes
from deepvoxnet2.components.mirc import Mirc as Mirc_, Dataset, Case, Record, ArrayModality
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel


class Mirc():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return

    @staticmethod
    def correct_zero_lesions_volume(arr, orig_vol, max_vol=1e14):
        if orig_vol > np.sum(arr):
            arr[...] = orig_vol / arr.size

        if np.sum(arr) > max_vol:
            arr *= max_vol / np.sum(arr)

        return arr

    @staticmethod
    def keep_n_labels(arr, n_labels, label_arr):
        iso_struct = generate_binary_structure(rank=3, connectivity=1)
        sort_idx = np.argsort([np.sum(label_arr == i) for i in range(1, np.max(label_arr) + 1)]) + 1
        for i in range(1, n_labels + 1):
            lesion = label_arr == sort_idx[-i]
            lesion_ = binary_dilation(lesion, iso_struct)
            arr[np.logical_xor(lesion, lesion_)] = 0
            arr[np.logical_xor(binary_fill_holes(lesion_), lesion_)] = 0

        return arr

    @staticmethod
    def improve_prediction(y_pred, min_threshold=0.1, binary_threshold=None, dummy=1e-21, max_lesion_count=1e14, max_vol=1e14):
        assert y_pred.ndim == 3
        assert min_threshold is None or binary_threshold is None and not (min_threshold is None and binary_threshold is None)
        vol = np.sum(y_pred)
        lesions, lesion_count = label(y_pred > (binary_threshold or 0.5))
        y_pred_ = np.where(y_pred > (binary_threshold or min_threshold), 1 if binary_threshold else y_pred, dummy).astype(np.float32)
        y_pred_ = Mirc.keep_n_labels(y_pred_, min(max_lesion_count, lesion_count) - 1, lesions)
        if lesion_count == 0:
            Mirc.correct_zero_lesions_volume(y_pred_, vol, max_vol)

        return y_pred_

    def process(self):
        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            dat, info = medpy.io.load(fil)  # dat is a numpy array
            # im_shape = dat.shape
            # dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # Convert 'dat' to Tensor, or as appropriate for your model.
            ###########
            arr = dat[None, ..., None]
            dataset = Dataset("_")
            case = Case("_")
            record = Record("_")
            record.add(ArrayModality("t1", arr))
            case.add(record)
            dataset.add(case)
            mirc = Mirc_(dataset)
            sampler = MircSampler(mirc)
            predictions = []
            for i in range(5):
                dvn_model = DvnModel.load_model(f"/dvn_models/dvn_model_final_{i}")
                predictions.append(dvn_model.predict("full_test_masked", sampler)[0][0][0][0, :, :, :, 0])

            prediction = np.mean(predictions, axis=0)
            prediction = Mirc.improve_prediction(prediction, min_threshold=None, binary_threshold=0.5, dummy=1e-21, max_lesion_count=7, max_vol=500000)
            dat = prediction
            ##############
            # dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            medpy.io.save(dat, out_filepath, hdr=info)
        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Mirc().process()
