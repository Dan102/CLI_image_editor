This script provides command line interface for several useful operation with images. It implements <strong>not only basic operation such as "rotating", "mirroring"</strong> etc.. but also some <strong>more complex operations such as "reducing colours in image" by implementing K-means clustering</strong>. The purpose of this script was to allow well-defined operations on images without need of advanced GUI editor and to practice NUMPY.

<strong>Any number of operations</strong> can be specified at the same time. <strong>Order of the operation is preserved</strong> with the exception of same operations, which are done at the same time (eg. cropping image to 200x200 and then do 50x50 will only crop image to 50x50).


<h2>LIST OF OPERATIONS and their explanation:</h2>

<ul>
<li>Rotate image (to right by 90°)</li>
<li>Mirror image (by x axis)</li>
<li>Get inverse image (negativ)</li>
<li>Get image in the shades of grey</li>
<li>Lighten image</li>
<li>Darken image</li>
<li>Sharpen image</li>
<li>Blur image</li>
<li>Get single color image = reduce all other color components to zero, keep only one (r/g/b)</li>
<li>Reduce color variety = Reduce number of different colors in the picture to K, using heuristic called K-Meas-Clustering, which tries to choose the "most representing K colors out of the picture" and preserve only them.  This function has iteration limit which can be changed by argument.</li>
<li>Add gaussian noise = add randomly distributed noise to every pixel</li>
<li>Crop image</li>
<li>Crop image in predefined directions = Crop image so that the new image will be at the center/top/bottom/left/right</li>
</ul>

<h2>EXAMPLE USAGE:</h2>
<ul>
<li>"python3 image_processing.py -r -r source.jpg" = rotate image "source.jpg" 2 times to right and saved it into default "altered.jpg"</li>
<li>"python3 image_processing.py -r -r source.jpg target.jpg" = rotate image "source.jpg" 2 times to right and saved it into "target.jpg"</li>
<li>"python3 image_processing.py -m -r -l 80 -i -s source.jpg" = Mirror image "source.jpg", rotate it to right, lighten it by 80%, get its inverse and then sharpen it and saved it into default "altered.jpg"</li>
<li>"python3 image_processing.py -sc GREEN source.jpg" = remove both red and blue components from "source.jpg" (leave only green intact) and saved it into default "altered.jpg"</li>
<li>"python3 image_processing.py -rv 8 -mi 200 source.jpg" = Use algorithm K-MEANS-CLUSTERING to reduce color variety to 8 "closest"(almost closest, it minimizes squared distances, as its easier) colors, set maximum iterations to 200 (from default 100). If the algorithm doesnt find a solution in 200 iterations, it ends without reducing colors.</li>
<li>"python3 image_processing.py -c 200 200 100 100 source.jpg" = Crop image so that new image has height and width of 200 px and its left corner starts at [100px, 100px] of the old image</li>
<li>"python3 image_processing.py --cropsize 200 200 --cvp top source.jpg" = Crop image so that new image has height and width of 200 px and verticaly it starts from the top and horizontaly it cover center (default if --chp not set) of the old image</li>
</ul>


<h2>USAGE:</h2>
<code> image_processing.py [-h] [-mi MAX_ITERATION] [-r] [-m] [-i] [--bw]
                           [-l PERCENTAGE] [-d PERCENTAGE] [-s] [-b]
                           [-rv REDUCED_COLORS_NUMBER]
                           [-sc {'GREEN', 'RED', 'BLUE'}] [-gn PERCENTAGE]
                           [-c HEIGHT WIDTH Y_START X_START]
                           [--cropsize CROPSIZE CROPSIZE]
                           [--cvp {'bottom', 'center', 'top'}]
                           [--chp {'left', 'center', 'right'}] [--bf]
                           [--blend SECOND_IMAGE_PATH]
                           SOURCE_IMAGE_PATH [TARGET_IMAGE_PATH]
</code>


<h2>POSITIONAL ARGUMENTS:</h2>
<code>
  SOURCE_IMAGE_PATH<br/>
  Path (relative or absolute) to the source image which should be processed. THIS ARGUMENT IS NON-OPTIONAL<br/><br/>
<code>
</code>
  TARGET_IMAGE_PATH<br/>
  Path (relative or absolute) to where the result of source image processing will be saved. This argument is optional and will be defaulted to "altered.jpg" if not provided
</code>
<h2>OPTIONAL ARGUMENTS:</h2>
<code>
  -h, --help<br/>
  show this help message and exit<br/><br/> 
</code>
<code>
  -mi MAX_ITERATION, --max_iteration MAX_ITERATION<br/>
  Set different max iteration value for K-means algorithm used in reducing colors. Default value is 100<br/><br/> 
</code>
<code>
  -r, --rotate<br/>
  Rotate image by 90 degrees to right. Can be set multiple times in one run of program.<br/><br/> 
</code>
<code>
  -m, --mirror<br/>
  Mirror image by axis x.<br/><br/> 
</code>
<code>
  -i, --inverse<br/>
  Inverse all the pixels' colors in the image (create negativ).<br/><br/> 
</code>
<code>
  --bw<br/>
  Convert pixels' color in the image into shades of gray.<br/><br/> 
</code>
<code>
  -l PERCENTAGE, --lighten PERCENTAGE<br/>
  Lighten the image. The required parameter must be in percentage, and only integer in interval <0; 100>.<br/><br/> 
</code>
<code>
  -d PERCENTAGE, --darken PERCENTAGE<br/>
  Darken the image. The required parameter must be in percentage, and only integer in interval <0; 100>.<br/><br/> 
</code>
<code>
  -s, --sharpen<br/>
  Make the edges in the image sharper.<br/><br/> 
</code>
<code>
  -b, --blur<br/>
  Blur the image.<br/><br/> 
</code>
<code>
  -rv REDUCED_COLORS_NUMBER, --reducevariety REDUCED_COLORS_NUMBER<br/>
  Reduce number of colors in the image to the given number. K-mean clustering is used as algorithm to try to find reduced number of colors in the picture which represents the image best. The required parameter must be integer<br/><br/> 
</code>
<code>
  -sc {'GREEN', 'RED', 'BLUE'}, --singlecolor {'GREEN', 'RED', 'BLUE'}<br/>
  Leave only one color component in the image, reduce others to 0.<br/><br/> 
</code>
<code>
  -gn PERCENTAGE, --gaussiannoise PERCENTAGE<br/>
  Add random (gaussian) noise to every pixel in the image. Parameter is in percentage, which is relative to max value of each pixel (255), only integer in interval <0; 100> accepted.<br/><br/> 
</code>
<code>
  -c HEIGHT WIDTH Y_START X_START, --crop HEIGHT WIDTH Y_START X_START<br/>
  Crop the image. Given parameters are: height, width, y_start and x_start (top left corner and size of the new image, relative to the old image.<br/><br/> 
</code>
<code>
  --cropsize CROPSIZE CROPSIZE<br/>
  Predefined crop, set size of the new image and then with --cvp and --chp where to crop (horizontal and vertical). Default is to position the new cropped image to the center<br/><br/> 
</code>
<code>
  --cvp {'bottom', 'center', 'top'}<br/>
  Predefined crop, set vertical position for crop. To have effect, must be used with --cropsize parameter<br/><br/> 
</code>
<code>
  --chp {'left', 'center', 'right'}<br/>
  Predefined crop, set horizontal position for crop. To have effect, must be used with --cropsize parameter<br/><br/> 
</code>
<code>
  --bf <br/>
  Experiment with OpenCV which tries to automatically detect and blur faces presented in the image.<br/><br/> 
</code>
<code>
  --blend SECOND_IMAGE_PATH<br/>
  Blends two images into one, lighten both of them (alpha blending). Ideal for adding watermark into picture and using them for showcase purposes.
</code>