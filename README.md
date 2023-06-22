<h1>Photometric Stereo</h1>

<p>This project implements Photometric Stereo, a computer vision technique that reconstructs the shape and surface normals of an object from multiple images taken under different lighting conditions. It leverages the variations in pixel intensities caused by the changing light directions to estimate the surface properties.</p>

<h2>Project Overview</h2>

<p>The project includes the following components:</p>

<ol>
  <li><strong>Depth Map Estimation:</strong> The provided code processes a left image and corresponding disparity map to estimate the depth map of the scene. The depth values are normalized and converted to a range between 0 and 1.</li>
  
  <li><strong>3D Point Cloud Reconstruction:</strong> Using the camera intrinsic matrix and the estimated depth map, the code performs 2D-3D and 3D-2D reprojection to reconstruct a 3D point cloud of the scene. The 3D points are computed by triangulating the corresponding 2D image points.</li>
  
  <li><strong>Synthesized Image Creation:</strong> The original image is used to create a synthesized image by copying the RGB values from the original image to the corresponding reprojected 2D coordinates in the synthesized image. This step generates a visual representation of the estimated 3D shape.</li>
  
  <li><strong>Camera Position Manipulation:</strong> An additional step has been added to the code to generate a sequence of synthesized images by manipulating the camera positions along the baseline. This is achieved by applying a translation to the 3D points before performing the reprojection.</li>
</ol>

<h2>Usage</h2>

<p>To use the code, follow these steps:</p>

<ol>
  <li><strong>Prepare the input files:</strong>
    <ul>
      <li>Place the left image, disparity map, and depth map files in the appropriate directory (`Photometric_Stereo\data\example` in the provided code).</li>
      <li>Ensure the file names and formats match the code expectations (e.g., `im_left.jpg`, `disparity_map.jpg`, `depth_left.txt`).</li>
    </ul>
  </li>
  
  <li><strong>Set the camera parameters:</strong>
    <ul>
      <li>Adjust the camera intrinsic matrix `K` in the code to match the camera used for capturing the images.</li>
      <li>Configure the baseline and focal length values according to the camera setup.</li>
    </ul>
  </li>
  
  <li><strong>Run the code:</strong> Execute the code to estimate the depth map, reconstruct the 3D point cloud, and generate the synthesized images.</li>
  
  <li><strong>Review the results:</strong>
    <ul>
      <li>The synthesized images will be saved in the project directory and displayed on the screen for visual inspection.</li>
      <li>Adjust the camera positions or experiment with different input images to observe variations in the synthesized results.</li>
    </ul>
  </li>
</ol>

<h2>Dependencies</h2>

<p>The project relies on the following dependencies:</p>

<ul>
  <li>Python (version 3.x)</li>
  <li>OpenCV (cv2)</li>
  <li>NumPy</li>
</ul>

<p>Make sure to install these dependencies before running the code.</p>

<p>Feel free to customize the description to fit the specific details and structure of your project.</p>
