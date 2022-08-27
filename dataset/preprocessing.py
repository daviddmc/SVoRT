import numpy as np
import nibabel as nib
import os
import ants

if __name__ == "__main__":

    data_dir = './feta_2.1/'
    atlas_dir = './CRL_Fetal_Brain_Atlas_2017v3/'
    out_dir = './registered'

    GA = np.genfromtxt(fname= os.path.join(data_dir, "participants.tsv"), skip_header=1)[:, -1]

    os.makedirs(out_dir, exist_ok=True)

    for n in range(1, 81):
        method = 'mial' if n <= 40 else 'irtk'
        f = os.path.join(data_dir, 'sub-%03d' % n, 'anat', 'sub-%03d_rec-%s_T2w.nii.gz' % (n, method))
        f_seg = os.path.join(data_dir, 'sub-%03d' % n, 'anat', 'sub-%03d_rec-%s_dseg.nii.gz' % (n, method))

        img = nib.load(f)
        seg = nib.load(f_seg)

        dtype = img.get_data_dtype()

        img_arr = img.get_fdata()
        seg_arr = seg.get_fdata()
        img_arr[seg_arr == 0] = 0

        masked_img = nib.Nifti1Image(img_arr.astype(img.get_data_dtype()), affine=img.affine, header=img.header)

        ga = min(max(int(np.around(GA[n-1])), 21), 38)

        f_atlas = ('STA%d' % ga) if ga < 36 else ('STA%dexp' % ga)
        f_atlas = os.path.join(atlas_dir, f_atlas + '.nii.gz')
        atlas = nib.load(f_atlas)

        masked_img = ants.utils.convert_nibabel.nifti_to_ants(masked_img)
        atlas = ants.utils.convert_nibabel.nifti_to_ants(atlas)

        reg = ants.registration(atlas, masked_img, type_of_transform='Similarity')

        img = ants.utils.convert_nibabel.nifti_to_ants(img)
        warped_img = ants.apply_transforms(fixed=atlas, moving=img, transformlist=reg['fwdtransforms'])

        warped_seg = []
        for label in range(int(seg_arr.max())+1):
            seg_label = nib.Nifti1Image((seg_arr == label).astype(seg.get_data_dtype()), affine=seg.affine, header=seg.header)
            seg_label = ants.utils.convert_nibabel.nifti_to_ants(seg_label)
            warped_seg_label = ants.apply_transforms(fixed=atlas, moving=seg_label, transformlist=reg['fwdtransforms'])
            warped_seg.append(warped_seg_label.numpy())
        
        warped_seg = np.stack(warped_seg, -1)
        warped_seg = np.argmax(warped_seg, -1)

        warped_seg = ants.from_numpy(warped_seg.astype(warped_seg_label.numpy().dtype), spacing=warped_seg_label.spacing,
                           origin=warped_seg_label.origin, direction=warped_seg_label.direction)

        ants.image_write(warped_img, os.path.join(out_dir, 'reg%d.nii.gz' % n))
        ants.image_write(warped_seg, os.path.join(out_dir, 'seg%d.nii.gz' % n))

