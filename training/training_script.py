"""
A high-resolution (1 mm x 1 mm x 1 mm) No New-Net siamese/contra-lateral implementation for ATLAS challenge @ MICCAI 2022

The network that is used is the unet_generalized_v3 version, in which not only the U pathway is weight-shared, but also the common/fully-connected pathway.
Furthermore, this network outputs a prediction for each permutation of the inputs (concatenated according to this permutation right before the common/fully-connected pathway).

optional arguments:
- fold_i [0, 1, 2, 3, 4]: what fold to run according to the atlas dataset creation
"""

import os
import pickle
import shutil
import argparse
import numpy as np
from deepvoxnet2.components.mirc import Mirc, Dataset, Case, Record, NiftyFileModality
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.transformers import RandomCrop, Swap, NormalizeMask, Concat, Split, GaussianNoise, MircInput, Threshold, Flip, Group, Clip, IntensityTransform, Put, KerasModel, Multiply, Crop, AffineDeformation, Remove
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, LearningRateScheduler, DvnModelCheckpoint
from deepvoxnet2.factories.directory_structure import MircStructure
from training.unet_generalized_v3 import create_generalized_unet_v3_model


def atlas(data="train", fold_i=0, nb_folds=5, base_dir="/usr/local/micapollo01/MIC/DATA/SHARED/STAFF/eheyle3/ISLES/ATLAS/data"):
    if data in ["train", "val"]:
        data_dir = os.path.join(base_dir, "train/derivatives/ATLAS")

    elif data == "test":
        data_dir = os.path.join(base_dir, "test/derivatives/ATLAS")

    else:
        raise ValueError("Unknown data request. Please choose from: 'train', 'test' or 'val'.")

    subjects = [subject for subject in sorted(os.listdir(data_dir)) if subject.startswith("sub")]
    nb_subjects = len(subjects)
    np.random.seed(0)
    np.random.shuffle(subjects)
    if data in ["train", "val"]:
        assert isinstance(nb_folds, int) and fold_i in list(range(nb_folds))
        max_nb_cases_per_fold = int(np.ceil(nb_subjects / nb_folds))
        subjects_val = subjects[fold_i * max_nb_cases_per_fold:(fold_i + 1) * max_nb_cases_per_fold]
        subjects = subjects_val if data == "val" else [subject for subject in subjects if subject not in subjects_val]

    dataset = Dataset("ISLES", dataset_dir=data_dir)
    for subject in subjects:
        case_dir = os.path.join(data_dir, subject, "ses-1/anat")
        case = Case(subject, case_dir)
        record = Record("record_0")
        record.add(NiftyFileModality("t1", os.path.join(case_dir, f"{subject}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz")))
        if data in ["train", "val"]:
            record.add(NiftyFileModality("mask", os.path.join(case_dir, f"{subject}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")))
            record.add(NiftyFileModality("brain_mask", os.path.join(case_dir, f"mask.nii.gz")))

        case.add(record)
        dataset.add(case)

    return dataset


def main(base_dir, run_name, experiment_name, round_i=None, fold_i=0):
    #########################################
    # SETTING UP INPUT AND OUTPUT STRUCTURE #
    #########################################
    # training
    train_mirc = Mirc(atlas(data="train", fold_i=fold_i))
    train_sampler = MircSampler(train_mirc, mode="per_case", shuffle=True)
    # validation
    val_mirc = Mirc(atlas(data="val", fold_i=fold_i))
    val_sampler = MircSampler(val_mirc, mode="per_record", shuffle=False)
    # creating output directories
    structure = MircStructure(
        base_dir=os.path.join(base_dir, "Runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        round_i=round_i,
        fold_i=fold_i,
        validation_mirc=val_mirc
    )
    structure.create()
    shutil.copyfile(os.path.realpath(__file__), os.path.join(structure.fold_dir, "script.py"))

    #####################
    # KERAS MODEL SETUP #
    #####################
    output_size = (128, 128, 128)
    keras_model = create_generalized_unet_v3_model(
        number_input_features=1,
        subsample_factors_per_pathway=(
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8)
        ),
        kernel_sizes_per_pathway=(
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3)))
        ),
        number_features_per_pathway=(
            ((30, 30), (30, 30)),
            ((60, 60), (60, 30)),
            ((120, 120), (120, 60)),
            ((240, 240), (240, 120))
        ),
        kernel_sizes_common_pathway=((1, 1, 1),) * 2,
        number_features_common_pathway=(60, 1),
        dropout_common_pathway=(0, 0.2),
        extra_output_kernel_sizes=(((1, 1, 1),),),
        extra_output_number_features=((1,),),
        extra_output_dropout=((0.2,),),
        extra_output_at_common_pathway_layer=(1,),
        extra_output_activation_final_layer=("sigmoid",),
        output_size=output_size,
        dynamic_input_shapes=True,
        number_siam_pathways=2,
        instance_normalization=False,
        batch_normalization=True
    )
    keras_model_transformer = KerasModel(keras_model, output_to_input=[0, 0, 1, 1])

    ##########
    # INPUTS #
    ##########
    x_input = MircInput(["t1"], output_shapes=[(1, 197, 233, 189, 1)], n=1)
    y_input = MircInput(["mask"], output_shapes=[(1, 197, 233, 189, 1)], n=1)
    m_input = Threshold(1.2, 100)(x_input)

    ########################
    # PATH TRAINING INPUTS #
    ########################
    x_path, y_path, m_path = x_input, y_input, m_input
    x_path, y_path = AffineDeformation(
        x_path,
        rotation_window_width=(3.14 / 12, 3.14 / 12, 3.14 / 12),
        translation_window_width=(4, 4, 4),
        width_as_std=False,
        transform_probability=0.2,
        order=[1, 0])(x_path, y_path)
    x_path = GaussianNoise(0, 0.5)(x_path)
    x_path = IntensityTransform(0, 1, 1, 0.01)(x_path)
    x_path = Remove(0.01, 0, axis=3)(Remove(0.01, 0, axis=2)(Remove(0.01, 0, axis=1)(x_path)))
    x_path = Clip(1.2, 100)(x_path)
    x_path = NormalizeMask(Threshold(20, 80)(x_path), std_shift=1, std_scale=1)(x_path)
    cl_path, cly_path = Flip((1, 0, 0))(x_path, y_path)
    x_path, y_path, cl_path, cly_path = RandomCrop(
        y_path,
        segment_size=[output_size] * 4,
        subsample_factors=[(1, 1, 1)] * 4,
        n=1,
        nonzero=0.95,
        default_value=0)(x_path, y_path, cl_path, cly_path)
    x_path, y_path, cl_path, cly_path = Flip((0.5, 0, 0))(x_path, y_path, cl_path, cly_path)
    x_path, y_path = Group()([x_path, cl_path]), Group()([y_path, cly_path])
    x_path, y_path = Swap()(x_path, y_path)
    x_output_train = keras_model_transformer(x_path)
    y_output_train = Split(indices=(0, 0, 1, 1))(y_path)

    ##########################
    # PATH VALIDATION INPUTS #
    ##########################
    x_path, y_path, m_path = x_input, y_input, m_input
    x_path = Clip(1.2, 100)(x_path)
    x_path = NormalizeMask(Threshold(20, 80)(x_path))(x_path)
    cl_path, cly_path = Flip((1, 0, 0))(x_path, y_path)
    x_path, y_path, cl_path, cly_path = Crop(
        x_path,
        segment_size=[(192, 232, 184)] * 4,
        subsample_factors=[(1, 1, 1)] * 4,
        default_value=0)(x_path, y_path, cl_path, cly_path)
    x_path = Group()([x_path, cl_path])
    y_path = Group()([y_path, y_path, cly_path, cly_path])
    x_output_val = keras_model_transformer(x_path)
    y_output_val = y_path

    ################################
    # PATH FULL VALIDATION OUTPUTS #
    ################################
    x_path = x_output_val
    x_path = Concat(axis=0)([Split(indices=(0, 1))(x_path), Split(indices=(2, 3))(x_path)])
    x_path = Put(Group()([x_input, x_input]), keep_counts=True)(x_path)
    x_output_full_val = x_path
    x_output_full_val_masked = Multiply()([Group()([m_input, m_input]), x_path])
    y_output_full_val = Group()([y_input, y_input])

    ################################
    # DVN MODEL SETUP AND TRAINING #
    ################################
    dvn_model = DvnModel({
        "train": [x_output_train, y_output_train],
        "val": [x_output_val, y_output_val],
        "full_val": [x_output_full_val, y_output_full_val],
        "full_test": [x_output_full_val],
        "full_val_masked": [x_output_full_val_masked, y_output_full_val],
        "full_test_masked": [x_output_full_val_masked]
    })
    volume_difference = get_metric("volume_error", voxel_volume="auto")
    abs_volume_difference = get_metric("absolute_volume_error", voxel_volume="auto")
    dice_score = get_metric("dice_coefficient", threshold=0.5)
    cross_entropy = get_loss("cross_entropy")
    dice_loss = get_loss("dice_loss")
    dice_loss_ = get_loss("dice_loss", reduce_along_batch=True)
    learning_rates = [1e-4 * 2] * 200 + [1e-4 / 5 * 2] * 200 + [1e-4 / 25 * 2] * 200  # + [1e-4 / 125 * 2] * 200
    dvn_model.compile("train", optimizer=Adam(learning_rate=learning_rates[0]), losses=[[cross_entropy], [dice_loss, dice_loss_], [cross_entropy], [dice_loss, dice_loss_]])
    dvn_model.compile("val", losses=[[cross_entropy], [dice_loss, dice_loss_], [cross_entropy], [dice_loss, dice_loss_]])
    dvn_model.compile("full_val", losses=[[cross_entropy], [dice_loss, dice_loss_]], metrics=[[cross_entropy, dice_score, volume_difference, abs_volume_difference], [cross_entropy, dice_score, volume_difference, abs_volume_difference]])
    dvn_model.compile("full_val_masked", losses=[[cross_entropy], [dice_loss, dice_loss_]], metrics=[[cross_entropy, dice_score, volume_difference, abs_volume_difference], [cross_entropy, dice_score, volume_difference, abs_volume_difference]])
    callbacks = [
        LearningRateScheduler(lambda epoch, lr: learning_rates[epoch]),
        DvnModelEvaluator(dvn_model, "full_val", val_sampler, output_dirs=structure.val_images_output_dirs, freq=50, logs_dir=structure.logs_dir),
        DvnModelCheckpoint(dvn_model, structure.models_dir, freq=50),
    ]
    history = dvn_model.fit("train", train_sampler, epochs=len(learning_rates), batch_size=2, callbacks=callbacks, logs_dir=structure.logs_dir, shuffle_samples=False, steps_per_epoch=128, initial_epoch=0, num_parallel_calls=4, prefetch_size=64)
    with open(structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    dvn_model.evaluate("full_val_masked", val_sampler, output_dirs=structure.val_images_output_dirs)
    dvn_model.save(os.path.join(structure.models_dir, "dvn_model_final"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nnn-v3_ce-sdsd_cl_hlr')
    parser.add_argument('fold_i', type=int, default=0, nargs="?")
    args = parser.parse_args()
    main(
        base_dir="/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets",
        run_name="atlas",
        experiment_name=f"nnn-v3_ce-sdsd_cl_hlr",
        fold_i=args.fold_i
    )
