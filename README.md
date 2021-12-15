# Segment_Transparent_Objects
Identify transparent objects in an image.

This notebook is the implementation of the work done by [xieenze | Segment_Transparent_Objects](https://github.com/xieenze/Segment_Transparent_Objects)

The link to the blog is : [Segmenting Transparent Objects in the Wild](https://xieenze.github.io/projects/TransLAB/TransLAB.html)

There are some modifications to the original work:
   - The original code is executed through CLI. In this project, the code is modified so that it could run after importing the python file.



   - The Original code was directory based, therefore too many files were there related to each other. In this project, all the code is packed into a single file named **pymasklib**. Only this file needs to be imported in the notebook.
   
   
   
   - Previously, the images to be tested need to be stored in a directoryand output mask images were also stored in a directory. In this project, you can directly provide image urls and can see outputs in the notebook.

## Environments

- python 3
- torch = 1.1.0 (>1.1.0 with cause performance drop, we can't find the reason)
- torchvision
- pyyaml
- Pillow
- numpy

### Requirements

Install the requirements through pip:
```
pip install -r requirements.txt
```


### Pretrained Model
Click on the link mentioned here to download the pre-trained model : [16.pth](https://drive.google.com/file/d/1moJ0B8ZhjN6679l3dwuxF0n2VKIoLBZt/view?usp=sharing)


After dowmnloading, copy the model in the directory where all other files of the project are present.


# License

For academic use, this project is licensed under the Apache License - see the LICENSE file for details. For commercial use, please contact the authors. 

## Citations

Citation for the original paper. BibTeX reference is as follows.

```
@article{xie2020segmenting,
  title={Segmenting Transparent Objects in the Wild},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Ding, Mingyu and Shen, Chunhua and Luo, Ping},
  journal={arXiv preprint arXiv:2003.13948},
  year={2020}
}
```
