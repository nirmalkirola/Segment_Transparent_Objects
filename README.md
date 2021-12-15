# Segment_Transparent_Objects
Identify transparent objects in an image.

This notebook is the implementation of the work done by [xieenze | Segment_Transparent_Objects](https://github.com/xieenze/Segment_Transparent_Objects)

The link to the blog is : [Segmenting Transparent Objects in the Wild](https://xieenze.github.io/projects/TransLAB/TransLAB.html)

There are some modifications to the original work:
   - The original code is executed through CLI. In this project, the code is modified so that it could run after importing the python file.



   - The Original code was directory based, therefore too many files were there related to each other. In this project, all the code is packed into a single file named **pymasklib**. Only this file needs to be imported           in the notebook.
   
   
   
   - Previously, the images to be tested need to be stored in a directoryand output mask images were also stored in a directory. In this project, you can directly provide image urls and can see outputs in the notebook.
