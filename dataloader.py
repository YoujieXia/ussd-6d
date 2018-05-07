import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cPickle as pickle

num_image  = 1313
num_class  = 1
img_size   = 416.0
grid_size  = 13
h_ori      = 480
w_ori      = 640

root_path = 'data/train/'
class_path = []
img_path = []

for i in range(1, num_class+1):
    if i < 10:
        class_path.append(root_path + '0' + str(i) + '/')
        img_path.append(root_path + '0' + str(i) + '/rgb/')
    else:
        class_path.append(root_path + str(i) + '/')
        img_path.append(root_path + str(i) + '/rgb/')

def ctrl_pt_loader(class_index):
    """Load all the control points of one specific class.
    Args:
        class_index (int): index of a class (1-15)
    Returns:
        2D numpy array: 1313 * 18 
    """
    fr = open(class_path[class_index - 1] + 'bb.pkl')
    total_bbx = pickle.load(fr)
    fr.close()

    crtl_pt = []
    for i in range(0, num_image):
        bbx = total_bbx[str(class_index)]
        # print (bbx)
        scale = grid_size
        centriod = scale * 0.5
        pt = [centriod, centriod]
        for j in range(len(bbx)):
            temp_x = bbx[j][0][0] * scale * 1.0 / w_ori
            temp_y = bbx[j][1][0] * scale * 1.0 / h_ori
            pt.append(temp_x)
            pt.append(temp_y)
        crtl_pt.append(pt)
        
        # print (pt)
        # test 3D bbx
        # plt.scatter(x,y)
        # plt.plot([x[0],x[1]], [y[0],y[1]], '-o')
        # plt.plot([x[0],x[2]], [y[0],y[2]], '-o')
        # plt.plot([x[0],x[4]], [y[0],y[4]], '-o')
        # plt.plot([x[1],x[3]], [y[1],y[3]], '-o')
        # plt.plot([x[1],x[5]], [y[1],y[5]], '-o')
        # plt.plot([x[2],x[3]], [y[2],y[3]], '-o')
        # plt.plot([x[2],x[6]], [y[2],y[6]], '-o')
        # plt.plot([x[3],x[7]], [y[3],y[7]], '-o')
        # plt.plot([x[4],x[5]], [y[4],y[5]], '-o')
        # plt.plot([x[4],x[6]], [y[4],y[6]], '-o')
        # plt.plot([x[5],x[7]], [y[5],y[7]], '-o')
        # plt.plot([x[6],x[7]], [y[6],y[7]], '-o')
        # plt.show()
    return crtl_pt

def class_label_grid(class_index, img, pt):
    """Load the 'S X S X number_of_classes' class label grid of image.
    Args:
    	class_index (int): index of a class (1-15)
        img: image array
        pt: 2 * 9 point coordinates
    Returns:
        2D numpy array: S * S * 19
    """
    grid = np.zeros((grid_size, grid_size, 19))  
    step = int(img_size / grid_size)
    for x in range(grid_size):
        for y in range(grid_size):
            temp = img[y*step:(y+1)*step, x*step:(x+1)*step, :]
            if np.sum(temp) > 0:
                grid[y, x, 0] = class_index 
                grid[y, x, 1:] = pt
    return grid


def image_loader(class_index):
    """Load all the images of one specific class.
    Args:
        class_index (int): index of a class (1-15)
    Returns:
        Image - 4D numpy array: 1313 * 3 * 416 * 416
        Grid - 4D numpy array: 1313 * S * S * 19
    """
    pt = ctrl_pt_loader(class_index)

    image = np.zeros((num_image, 3, int(img_size), int(img_size))) 
    grid = []
    for i in range(0, num_image):
    	img_name_base = img_path[class_index - 1]
        if i < 10:
            img_name = img_name_base + '000' + str(i) + '.png'
        elif i < 100:
            img_name = img_name_base + '00' + str(i) + '.png'
        elif i < 1000:
            img_name = img_name_base + '0' + str(i) + '.png'
        else:
            img_name = img_name_base + str(i) + '.png'
        
        # do not know why can not read 0556.png, so weird..
        if i == 374 or i == 566:
            img_name = img_name_base + '0' + str(i+1) + '.png'

        # read image data
        img = Image.open(img_name)
        img = img.resize((int(img_size), int(img_size)), Image.ANTIALIAS)
        img.save(img_name)
        img = mpimg.imread(img_name)
        # plt.imshow(img)
        # plt.show()
        # print(img.shape)
        # print(img[0,0,:])
        image[i, 0, :, :] = img[:, :, 0]
        image[i, 1, :, :] = img[:, :, 1]
        image[i, 2, :, :] = img[:, :, 2]

        # generate grid 
        grid.append(class_label_grid(class_index, img, pt[i]))
    return image, grid



def total_image_loader(num):
    """Load all the images of all classes.
    Args:
        num (int): num of classes (1-15)
    Returns:
        Image - 4D numpy array: (num * 1313) * 3 * 416 * 416 
        Grid - 4D numpy array: (num * 1313) * S * S * 19
    """
    total_image = np.zeros((num_image * num, 3, int(img_size), int(img_size))) 
    total_grid = np.zeros((num_image * num, grid_size, grid_size, 19)) 
    for i in range(0, num):
        print ('Now Loading Class: ' + str(i+1) + '...')
        image, grid = image_loader(i+1)
        total_image[i * num_image : (i+1) * num_image ,:,:,:] = image
        total_grid[i * num_image : (i+1) * num_image ,:,:,:] = grid
        # total_image.append(image)
        # total_grid.append(grid)
    return total_image, total_grid


if __name__ == '__main__':
    image, grid = image_loader(1)
    pt = ctrl_pt_loader(1)
    print len(image)
    print image[0].shape
    print '----'
    print len(grid)
    print grid[0].shape
    print grid[0][6][6]
    print '----'
    total_image, total_grid = total_image_loader(7)
    print len(total_image)
    print total_image[0].shape
    print len(total_grid)
    print total_grid[0].shape

    # transfer numpy array to tensor
    # train_frame = np.array(train_frame)
    # train_frame = Variable(torch.FloatTensor(train_frame)) 
    # train_frame_label = np.array(train_frame_label)
    # train_frame_label = Variable(torch.LongTensor(train_frame_label)) 

