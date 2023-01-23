# general imports
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(281)


def display_random_images(imgs, labels):
    # note: this row/colum math is a bit fragile.  Best if ns_rows*ns_cols = n_samples. :(
    n_samples = 30
    ns_cols = 5
    ns_rows = int(n_samples / ns_cols)

    fig, axes = plt.subplots(nrows=ns_rows, ncols=ns_cols, figsize=(15, 15))

    for i, idx in enumerate(np.random.randint(len(imgs), size=n_samples)):
        ax = axes[int(i / ns_cols), i % ns_cols]
        ax.imshow(imgs[idx])
        ax.set_title(labels_used[labels[idx]])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])


def display_features(imgs, labels, color_reduction_factor, selected_images=[]):
    if len(selected_images) == 0:
        print(selected_images)
        n_samples = 10
        selected_images = np.random.randint(len(imgs), size=n_samples)

    n_samples = len(selected_images)

    n_cols = 9  # number of different feature images + 1
    if (color_reduction_factor != 1):
        n_cols = n_cols + 1
    n_rows = n_samples

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_samples * 2))

    for i, idx in enumerate(selected_images):
        img = imgs[idx]

        icol = 0
        # display base image
        ax = axes[i, icol]
        ax.imshow(imgs[idx])
        ax.set_ylabel(labels_used[labels[idx]])
        if (i == 0):
            ax.set_title("Original")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # color compression if it was done
        if (color_reduction_factor != 1):
            icol = icol + 1
            ax = axes[i, icol]
            ax.imshow(img)
            ax.set_ylabel(labels_used[labels[idx]])
            if (i == 0):
                ax.set_title("Reduce Palette")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

        # display average color
        icol = icol + 1
        ax = axes[i, icol]
        _, avg_img = average_color_feature(img)
        ax.imshow(avg_img)
        if (i == 0):
            ax.set_title("Avg Color")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display dominant colors
        icol = icol + 1
        ax = axes[i, icol]
        _, dom_img = calculate_n_dominant_colors(img, 8)
        ax.imshow(dom_img)
        if (i == 0):
            ax.set_title("Dominant Colors")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display color segmented image k = 3
        icol = icol + 1
        ax = axes[i, icol]
        _, seg_img = rp_kmeans_segmented(img, 3)
        ax.imshow(seg_img)
        if (i == 0):
            ax.set_title("Color Seg k=3")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display color segmented image k = 8
        icol = icol + 1
        ax = axes[i, icol]
        _, seg_img = rp_kmeans_segmented(img, 8)
        ax.imshow(seg_img)
        if (i == 0):
            ax.set_title("Color Seg k=8")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display HOG image
        icol = icol + 1
        ax = axes[i, icol]
        _, h_img = calculate_HOG_features(img)
        ax.imshow(h_img)
        if (i == 0):
            ax.set_title("HOG")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display lbd image
        icol = icol + 1
        ax = axes[i, icol]
        _, lbd_img = display_LBP(img)
        ax.imshow(lbd_img, cmap='gray')
        if (i == 0):
            ax.set_title("LBP")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display fft translated image
        icol = icol + 1
        ax = axes[i, icol]
        _, _, fft_img = calculate_fft_features(img)
        ax.imshow(fft_img)
        if (i == 0):
            ax.set_title("FFT")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        # display fft translated image
        icol = icol + 1
        ax = axes[i, icol]
        _, dartboard_img, _ = calculate_fft_features(img)
        ax.imshow(dartboard_img)
        if (i == 0):
            ax.set_title("FFT dart")
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

