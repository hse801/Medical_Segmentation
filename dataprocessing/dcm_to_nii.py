import glob
import SimpleITK as sitk
import nibabel as nb
import os
import numpy as np
from scipy.interpolate import interpn


def crop_img(src_file: str, dst_file: str, crop_index):

    start_index, end_index = crop_index
    src_img = nb.load(src_file)

    zorigin = src_img.affine[2][3]
    zspacing = src_img.affine[2][2]

    src_img_data = src_img.get_fdata()
    src_img_data = src_img_data.astype(np.float32)

    affine = src_img.affine
    affine[2][3] = zorigin + zspacing * start_index
    crop_img_data = src_img_data[:, :, start_index: end_index]

    nb.save(nb.Nifti1Pair(crop_img_data, affine), dst_file)


def pad_img(src_file: str, img_dim: int, save_path: str):
    # pad the image to same size ( z slices )
    # size of spect image is 256 * 256 * ~256
    # pad every images to have 256 slices
    src_img = nb.load(src_file)
    affine = src_img.affine
    src_img_data = src_img.get_fdata()
    print(f'src_img_data.shape = {np.shape(src_img_data)}')
    new_arr = np.zeros((np.shape(src_img_data)[0], np.shape(src_img_data)[1], img_dim))
    print(f'new_arr.shape = {np.shape(new_arr)}')
    print(f'affine = {affine}')

    if np.shape(src_img_data)[2] < img_dim:
        new_arr[:, :, 0:np.shape(src_img_data)[2]] = src_img_data
    else:
        raise Exception('Image is bigger than dst dim')
    os.chdir(save_path)
    # nb.save(nb.Nifti1Pair(new_arr, affine), 'MASK_rsmpl.nii.gz')
    print(f'New file saved in {os.getcwd()}')

    return new_arr, affine


def pad_arr(src_arr, ref_affine, img_dim: int, save_path: str, file_name: str):
    # pad the image to same size ( z slices )
    # size of spect image is 256 * 256 * ~256
    # pad every images to have 256 slices
    src_img_data = src_arr
    print(f'src_img_data.shape = {np.shape(src_img_data)}')
    new_arr = np.zeros((np.shape(src_img_data)[0], np.shape(src_img_data)[1], img_dim))
    print(f'new_arr.shape = {np.shape(new_arr)}')
    print(f'affine = {affine}')

    if np.shape(src_img_data)[2] < img_dim:
        new_arr[:, :, 0:np.shape(src_img_data)[2]] = src_img_data
    else:
        raise Exception('Image is bigger than dst dim')
    os.chdir(save_path)
    nb.save(nb.Nifti1Pair(new_arr, affine), file_name)
    print(f'New file saved in {os.getcwd()}')

    return new_arr, affine


def reverse_along_axis(data: np.array, axis: int):
    if axis == 0:
        data = data[::-1, :, :]
    elif axis == 1:
        data = data[:, ::-1, :]
    else:
        data = data[:, :, ::-1]
    return data


def resample_nb(src_file: str, dst_file: str, ref_file: str, save_path: str):
    print(f'nii_resize_image: src={src_file} dst={dst_file} ref={ref_file}')
    src_img = nb.load(src_file)
    ref_img = nb.load(ref_file)
    src_coord = np.array([np.arange(d) for d in src_img.shape])
    ref_coord = np.array([np.arange(d) for d in ref_img.shape])
    print(f'src_coord = {src_coord.shape}, ref_coord = {ref_coord.shape}')
    src_img_data = src_img.get_fdata()

    for i in range(3):
        src_coord[i] = src_img.affine[i][i] * src_coord[i] + src_img.affine[i][3]
        ref_coord[i] = ref_img.affine[i][i] * ref_coord[i] + ref_img.affine[i][3]
        if src_img.affine[i][i] < 0:
            src_coord[i] = src_coord[i][::-1]
            src_img_data = reverse_along_axis(src_img_data, i)
        if ref_img.affine[i][i] < 0:
            ref_coord[i] = ref_coord[i][::-1]

    ref_mesh = np.rollaxis(np.array(np.meshgrid(*ref_coord)), 0, 4) # [xdim][ydim][zdim][3]
    src_resize_data = interpn(src_coord, src_img.get_fdata(), ref_mesh, bounds_error=False, fill_value=-1024)

    for i in range(3):
        if ref_img.affine[i][i] < 0:
            src_resize_data = reverse_along_axis(src_resize_data, i)
    src_resize_data = src_resize_data.astype(np.float32)

    import pathlib
    if pathlib.Path(dst_file).exists():
        print(f'{dst_file} already exists. will overwrite')
    src_resize_data = src_resize_data.swapaxes(0, 1)
    src_resize_data = src_resize_data[::-1, ::-1, :]

    os.chdir(f)    # nb.save(nb.Nifti1Pair(src_resize_data, ref_img.affine), dst_file)
    print(f'Resampled file {dst_file} saved in {os.getcwd()}')

    return src_resize_data


def resample_sitk(src_file: str, ref_file: str, dst_file_name: str, save_path):

    img_src = sitk.ReadImage(src_file)
    # print(img_src)
    img_src_data = sitk.GetArrayFromImage(img_src)
    # print(img_src_data)
    img_ref = sitk.ReadImage(ref_file)
    img_ref_data = sitk.GetArrayFromImage(img_ref)

    x_src = np.arange(-img_src.GetOrigin()[0],
                     -img_src.GetOrigin()[0] + (-img_src.GetSpacing()[0]) * img_src_data.shape[1],
                     step=-img_src.GetSpacing()[0])
    x_src = x_src[::-1]
    y_src = np.arange(-img_src.GetOrigin()[1],
                     -img_src.GetOrigin()[1] + img_src.GetSpacing()[1] * img_src_data.shape[2],
                     step=img_src.GetSpacing()[1])
    z_src = np.arange(img_src.GetOrigin()[2],
                     img_src.GetOrigin()[2] + img_src.GetSpacing()[2] * img_src_data.shape[0],
                     step=img_src.GetSpacing()[2])

    x_ref = np.arange(-img_ref.GetOrigin()[0],
                      -img_ref.GetOrigin()[0] + (-img_ref.GetSpacing()[0]) * img_ref_data.shape[1],
                      step=-img_ref.GetSpacing()[0])
    x_ref = x_ref[::-1]
    y_ref = np.arange(-img_ref.GetOrigin()[1],
                      -img_ref.GetOrigin()[1] + img_ref.GetSpacing()[1] * img_ref_data.shape[2],
                      step=img_ref.GetSpacing()[1])
    z_ref = np.arange(img_ref.GetOrigin()[2],
                      img_ref.GetOrigin()[2] + img_ref.GetSpacing()[2] * img_ref_data.shape[0],
                      step=img_ref.GetSpacing()[2])
    mesh_pet = np.array(np.meshgrid(z_ref, y_ref, x_ref))

    mesh_points = np.rollaxis(mesh_pet, 0, 4)
    mesh_points = np.rollaxis(mesh_points, 0, 2)
    interp = interpn((z_src, y_src, x_src), img_src_data[:, :, ::-1],
                     mesh_points, bounds_error=False, fill_value=-1024)

    src_rsmpl_img = sitk.GetImageFromArray(interp[:, :, ::-1])
    src_rsmpl_img.CopyInformation(img_ref[:, :, :])

    os.chdir(save_path)
    sitk.WriteImage(src_rsmpl_img, dst_file_name)
    # sitk.WriteImage(img_ref[:, :, 80:170], "PET_crop_h.nii.gz")
    print(f'Resampled file {dst_file_name} saved in {os.getcwd()}')


file_path = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/*/*/')

for f in file_path:
    # print(f'f = {f}')
    ct_list = glob.glob(f + 'CTCT StandardThyroid SPECT/IM1.DCM')
    mask_list = glob.glob(f + 'Q.Metrix Organs/*.DCM')
    spect_list = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.DCM')

    if len(ct_list) == 0:
        raise Exception('No CT files')

    if len(mask_list) != 1:
        raise Exception('Mask file # is not 1')

    if len(spect_list) != 1:
        raise Exception('Spect file # is not 1')

    # ct_list = glob.glob('D:/0902_Thyroid/ThyroidSPECT Dataset/10414125/*/CTCT StandardThyroid SPECT/IM1.DCM')
    # print(f'file list len = {len(ct_list)}')
    # print(f'ct = {ct_list[0]} \nmask = {mask_list[0]} \nspect = {spect_list[0]}')
    os.chdir('C:/Users/Bohye/Desktop/mricron')

    # os.system('dcm2nii.exe ' + ct_list[0])
    # os.system('dcm2nii.exe ' + mask_list[0])
    # os.system('dcm2nii.exe ' + spect_list[0])

    ct_nii = glob.glob(f + 'CTCT StandardThyroid SPECT/2*.nii.gz')
    mask_nii = glob.glob(f + 'Q.Metrix Organs/*.nii.gz')
    spect_nii = glob.glob(f + 'Q.Metrix_Transaxials_IsotropNM/*.nii.gz')

    # resize_arr, affine = pad_img(mask_nii[0], img_dim=256, save_path=f)
    # ref_interp = f + 'MASK_rsmpl.nii.gz'
    # ref_interp_img = nb.load(ref_interp)
    # ref_affine = ref_interp_img.affine
    # print(f'ref interp affine = {ref_affine}')

    # resample_sitk(src_file=ct_nii[0], ref_file=ref_interp, dst_file_name='CT_rsmpl.nii.gz', save_path=f)
    # resample_sitk(src_file=spect_nii[0], ref_file=ref_interp, dst_file_name='SPECT_rsmpl.nii.gz', save_path=f)

    src_img = nb.load(spect_nii[0])
    affine = src_img.affine
    src_img_data = src_img.get_fdata()
    #
    ct_arr = resample_nb(src_file=ct_nii[0], dst_file='CT_rsmpl.nii.gz', ref_file=spect_nii[0], save_path=f)
    # spect_arr = resample_nb(src_file=spect_nii[0], dst_file='SPECT_rsmpl.nii.gz', ref_file=spect_nii[0], save_path=f)
    mask_arr = resample_nb(src_file=mask_nii[0], dst_file='MASK_rsmpl.nii.gz', ref_file=spect_nii[0], save_path=f)

    pad_arr(src_img_data, ref_affine=affine, img_dim=256, save_path=f, file_name='SPECT_rsmpl.nii.gz')
    pad_arr(ct_arr, ref_affine=affine, img_dim=256, save_path=f, file_name='CT_rsmpl.nii.gz')
    pad_arr(mask_arr, ref_affine=affine, img_dim=256, save_path=f, file_name='MASK_rsmpl.nii.gz')

    # break
