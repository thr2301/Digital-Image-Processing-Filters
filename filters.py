import os  

import numpy as np  #version 1.22

import matplotlib.pyplot as plt  

from common import read_img, save_img 

from scipy.ndimage import convolve as test_convolve  


def image_patches(image, patch_size=(16, 16)):
    """
    --- Zitima 1.a ---
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    output = []  # Initialize an empty list to store image patches
    H, W = image.shape  # Get the height and width of the input image
    M, N = patch_size  # Get the height and width of the patch size
    for i in range(0, H, M):
        for j in range(0, W, M):
            patch = image[i:i+M, j:j+M]  # Extract the patch from the image
            if patch.shape == (M, N):  # Ensure the patch size is as expected
                patchMean = np.mean(patch)  # Calculate the mean of the patch
                patchStd = np.std(patch)  # Calculate the standard deviation of the patch
                if patchStd > 0:
                    patch = (patch - patchMean) / patchStd  # Normalize the patch if patchStd  is non-zero
                else:
                    patch = patch - patchMean  # If patchStd  is zero, just subtract the mean
                output.append(patch)  # Add the normalized patch to the output list
    return output  # Return the list of image patches

def convolve(image, kernel):
    """
    --- Zitima 2.b ---
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    # Get dimensions of the image and kernel
    imageHeight, imageWidth = image.shape
    kernelHeight, kernelWidth = kernel.shape
    # Flip the kernel for convolution
    kernel = np.flipud(np.fliplr(kernel))
    # Pad the image with zeros to handle borders
    padHeight = kernelHeight // 2
    padWidth = kernelWidth // 2
    paddedImage = np.pad(image, ((padHeight, padHeight), (padWidth, padWidth)), mode='constant')
    # Prepare the output array with the same size as the input image
    output = np.zeros_like(image)
    # Convolution operation
    for i in range(imageHeight):
        for j in range(imageWidth):
            region = paddedImage[i:i + kernelHeight, j:j + kernelWidth]  # Extract the region to be convolved
            output[i, j] = np.sum(region * kernel)  # Perform the convolution and store the result
    return output  # Return the convolved image

def edge_detection(image):
    """
    --- Zitima 2.f ---
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    kx = np.array([[1, 0, -1]])  # 1x3 kernel for x gradient
    ky = np.array([[1], [0], [-1]])  # 3x1 kernel for y gradient

    Ix = convolve(image, kx)  # Convolve the image with the x gradient kernel
    Iy = convolve(image, ky)  # Convolve the image with the y gradient kernel

    # Using Ix, Iy to calculate the gradient magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, grad_magnitude  # Return the gradients and gradient magnitude

def sobel_operator(image):
    """
    --- Zitima 3.b ---
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Sobel kernel for x gradient
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel kernel for y gradient
    
    # Gaussian filter for smoothing
    GS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    # Apply Gaussian filter
    filteredImage = convolve(image, GS)

    # Compute horizontal and vertical derivatives using Sobel filters
    Gx = convolve(filteredImage, Sx)
    Gy = convolve(filteredImage, Sy)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude  # Return the Sobel gradients and gradient magnitude

def main():
    # The main function to execute the image processing tasks
    img = read_img('./grace_hopper.png')  # Read the input image
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")  # Create directory if it doesn't exist
    patches = image_patches(img)  # Get image patches
    chosen_patches = patches[:3]  # Select first three patches
    stacked_patches = np.hstack(chosen_patches)  # Stack the patches horizontally
    save_img(stacked_patches, "./image_patches/q1_patch.png")  # Save the stacked patches
    
    # Plot the stacked patches
    plt.imshow(stacked_patches, cmap='gray')
    plt.title('Stacked Patches')
    plt.savefig("./image_patches/3_patches.png")
    plt.show()

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")  # Create directory if it doesn't exist
    s = 0.572  # Standard deviation for Gaussian filter
    size = 3  # Size of the Gaussian kernel
    kernel_gaussian = np.zeros((size, size))
    k = (size - 1) // 2
    for x in range(-k, k+1):
        for y in range(-k, k+1):
            kernel_gaussian[x+k, y+k] = (1/(2*np.pi*s**2)) * np.exp(-(x**2 + y**2)/(2*s**2))
    kernel_gaussian /= np.sum(kernel_gaussian)  # Normalize the Gaussian kernel
    filtered_gaussian = convolve(img, kernel_gaussian)  # Apply Gaussian filter to the image
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")  # Save the filtered image
    # Convolve using scipy for correctness check
    filtered_gaussian_test = test_convolve(img, kernel_gaussian, mode='constant')
    save_img(filtered_gaussian_test, "./gaussian_filter/q2_gaussian_test.png")  # Save the test filtered image
    # Verify the correctness by comparing the results
    if np.allclose(filtered_gaussian, filtered_gaussian_test):
        print("Test matches the Convolve.")
    else:
        print("Test failed.")
    print('Filtered Gaussian :')
    print(filtered_gaussian)   
    _, _, edge_detect = edge_detection(img)  # Perform edge detection on the original image
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")  # Save the edge detection result
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)  # Perform edge detection on the filtered image
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")  # Save the edge detection result on filtered image
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Edge Detection on Original Image")
    plt.imshow(edge_detect, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Edge Detection on Gaussian \nFiltered Image")
    plt.imshow(edge_with_gaussian, cmap='gray')
    plt.title("Edge Detection Results")
    plt.savefig("./gaussian_filter/edge_detection_comparison.png")
    plt.show()
    print("Gaussian Filter is done. ")

    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")  # Create directory if it doesn't exist
    Gx, Gy, grad_magnitude = sobel_operator(img)  # Apply Sobel operator
    save_img(Gx, "./sobel_operator/q2_Gx.png")  # Save the x gradient
    save_img(Gy, "./sobel_operator/q2_Gy.png")  # Save the y gradient
    save_img(grad_magnitude, "./sobel_operator/q2_grad_magnitude.png")  # Save the gradient magnitude
    # Plot the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(Gx, cmap='gray')
    plt.title('Gradient X')
    plt.subplot(1, 3, 2)
    plt.imshow(Gy, cmap='gray')
    plt.title('Gradient Y')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(grad_magnitude), cmap='gray')
    plt.title('Gradient Magnitude')
    plt.show()
    print("Sobel Operator is done. ")
    
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")  # Create directory if it doesn't exist
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian of Gaussian kernel 1
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])  # Laplacian of Gaussian kernel 2
    filtered_LoG1 = convolve(img, kernel_LoG1)  # Apply LoG kernel 1 to the image
    filtered_LoG2 = convolve(img, kernel_LoG2)  # Apply LoG kernel 2 to the image
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")  # Save the result of LoG kernel 1
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")  # Save the result of LoG kernel 2
    #print the results 
    print('filtered_LoG1:')
    print(filtered_LoG1)
    print('filtered_LoG2:')
    print(filtered_LoG2)
    print("Program is finished")

if __name__ == "__main__":
    main()  
