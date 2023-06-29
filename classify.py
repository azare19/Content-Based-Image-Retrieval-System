import cv2
import numpy as np
import math

#color bins
blue = 5
green = 6
red = 6

#texture bins
texture = 1000

#number of input images
num_imgs = 40

#loads data from crowd and author
crowd_data = np.loadtxt("gz2337_borda.txt")
my_data = np.loadtxt('gz2337.txt')

# variables for generating text for results display
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 80)
fontScale = 0.4
color = (255, 0, 0)
thickness = 1


def color_histogram(img_str, red = red, blue = blue, green = green):
    '''
    purpose: generates color histogram of image given rgb values

    inputs:
    img_str = string of image file to be analyzed
    red = number of red bins
    blue = number of blue bins
    green = number of red bins

    output:
    color_hist = histogram of image with colors binned according to givens
    '''
    #read image through opencv
    img = cv2.imread(img_str)

    #storage for histogram
    color_hist = {}

    #generate factors to reduce pixel values
    r_factor = math.floor(256/red)
    b_factor = math.floor(256/blue)
    g_factor = math.floor(256/green)

    #find size of image
    dim = img.shape

    #iterate through image rows
    for i in range(dim[0]):
        #iterate through image columns
        for j in range(dim[1]):

            #retrive pixel and BGR values
            pixel = img[i,j]
            b_val = pixel[0]
            g_val = pixel[1]
            r_val = pixel[2]

            #reduce values to requested number of bins
            b_val_fac = math.floor(b_val/b_factor)
            g_val_fac = math.floor(g_val/g_factor)
            r_val_fac = math.floor(r_val/r_factor)

            #want to zero index bins, in case of match to bin # reduce
            if b_val_fac == blue:
                b_val_fac = b_val_fac - 1

            if g_val_fac == green:
                g_val_fac = g_val_fac - 1

            if r_val_fac == red:
                r_val_fac = r_val_fac - 1           

            #generate color string
            color_str = str(b_val_fac) + "," + str(g_val_fac) + "," + str(r_val_fac)

            #if color already exists in image add to count, else add to histogram
            if color_str in color_hist:
                color_hist[color_str] += 1
            else:
                color_hist[color_str] = 1
    
    #return final histogram for image
    return color_hist 

def texture_histogram(img_str, texture = texture):
    '''
    purpose: generates texture histogram for input image

    inputs
    img_str = image to be analyzed
    texture = number of bins for texture values

    outputs
    texture_hist = histogram representing overall texture of image
    '''
    #read image to analyze and convert to grayscale
    img = cv2.imread(img_str)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #create storage for laplacian image conversion
    laplace = np.empty((60,89))

    #storage for histogram
    texture_hist = {}

    #calculates factor to reduce each pixel value by
    texture_fac = math.floor(2040/texture)

    #find size of image
    dim = gray.shape

    #iterate through rows of image
    for i in range(dim[0]):
        #iterate through columns of image
        for j in range(dim[1]):
            #find original value of each pixel
            pixel = gray[i][j]
            
            #initialize values for neighbors of pixel
            upper_left = 1
            upper_center = 1
            upper_right = 1
            bottom_left = 1
            bottom_center = 1
            bottom_right = 1
            center_left = 1
            center_right = 1

            #at top of image, make neighbors 0
            if i - 1 < 0:
                upper_left = 0
                upper_center = 0
                upper_right = 0
            else:
                #finds value of upper center neighbor
                upper_center = int(gray[i - 1][j])
            
            #at bottom of image, make neighbors 0
            if i + 1 > dim[0] - 1:
                bottom_left = 0
                bottom_center = 0
                bottom_right = 0
            else:
                #finds value of bottom center neighbor
                bottom_center = int(gray[i + 1][j])

            #at left of image, make neighbors 0
            if j - 1 < 0:
                upper_left = 0
                center_left = 0
                bottom_left = 0
            else:
                #finds value of center left neighbor
                center_left = int(gray[i][j - 1])

            #at right of image, make neighbors 0
            if j + 1 > dim[1] - 1:
                upper_right = 0
                center_right = 0
                bottom_right = 0
            else: 
                #finds value of center right neighbor
                center_right = int(gray[i][j + 1])

            #fill in values for corner neighbors if not determined to not exist
            if upper_left != 0:
                upper_left = int(gray[i - 1][j - 1])
            
            if upper_right != 0:
                upper_right = int(gray[i - 1][j + 1])

            if bottom_left != 0:
                bottom_left = int(gray[i + 1][j - 1])

            if bottom_right != 0:
                bottom_right = int(gray[i + 1][j + 1])

            #sum of neighbors to current pixel
            neighbors_sum = upper_center + upper_left + upper_right + bottom_center + bottom_left + bottom_right + center_left + center_right

            #calculates laplacian value of pixel and adds to new laplace img
            l_pix = (8*pixel) - (neighbors_sum)
            laplace[i][j] = abs(l_pix)

            #reduces value of pixel by number of bins
            texture_val_fac = math.floor(laplace[i][j]/texture_fac)

            #reduce if case where factor results in max value
            if texture_val_fac == texture:
                texture_val_fac = texture_val_fac - 1

            #if texture value already in image, increment, else add to histogram
            if texture_val_fac in texture_hist:
                texture_hist[texture_val_fac] += 1
            else:
                texture_hist[texture_val_fac] = 1

    #return final image histogram
    return texture_hist

def shape_overlap(img_str1, img_str2):
    '''
    purpose: sums the overlapping pixels of two images to find approximation of shape similarity

    inputs:
    img_str1 - first image to compare
    img_str2 - second image to compare

    outputs
    overlap - normalized value of summed matching pixels
    '''
    #convert images to grayscale, blur, then threshold to convert to binary
    #blur is neccessary to smooth shape edges, and the chosen threshold maximizes 
    #results for finding which pixels are background vs. foreground
    img1 = cv2.imread(img_str1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.blur(gray1, (30,30))
    ret, bw1 = cv2.threshold(blur1, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('blur', bw1)

    img2 = cv2.imread(img_str2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.blur(gray2, (30,30))
    ret, bw2 = cv2.threshold(blur2, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('blur2', bw2) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #find size of images, assumes images are both same size
    dim = bw1.shape

    #initialize summation value
    summation = 0

    #iterate through image pixels and compare
    #add to sum if they are equal
    for i in range(dim[0]):
        for j in range(dim[1]):
            if bw1[i][j] != bw2[i][j]:
                summation += 1

    #normalize summation by image size
    overlap = summation / (60*89)

    #return overlap distance calculated between the two images
    return overlap

def symmetry(img_str1):
    '''
    purpose: finds symmetry of image by folding image vertically and comparing pixel values

    inputs:
    img_str1: image to analyze

    outputs:
    symm: symmetry value normalized by image size
    '''
    #read image and convert to grayscale, blur and threshold to convert to binary
    #blur is used to soften edges of image and make less sensitive to slight changes
    #threshold value is chosen to maximize results
    img1 = cv2.imread(img_str1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.blur(gray1, (15,15))
    ret, bw1 = cv2.threshold(blur1, 140, 255, cv2.THRESH_BINARY)

    #initialize summation value
    summation = 0

    #iterate through columns comparing left and right sides of image
    for i in range(0,44):
        for k in range(0,60):
            #chose pixel in same row
            pixel1 = bw1[k][i]
            pixel2 = bw1[k][88 - i]

            #add to total sum if they match
            if pixel1 == pixel2:
                summation += 1
    
    #normalize summation by image size
    symm = summation / (60*44)

    #returns normalized symmetry value
    return symm


    
def color_distance(hist1, hist2, blue = blue, green = green, red = red):
    '''
    purpose: computes the distance between two image histograms by color

    inputs:
    hist1: first image histogram to compare
    hist2: second image histogram to compare
    blue: number of blue bins
    green: number of green bins
    red: nuber of red bins

    (assumes number of bgr bins are the same as those used to create the histogram)

    outputs:
    distance: normalized distance between the two image histograms
    '''
    #initialize summation value
    summation = 0

    #iterate through all possible color bins
    for i in range(blue):
        for j in range(green):
            for k in range(red):
                #colors stored as strings for key in dict
                color_str = str(i) + "," + str(j) + "," + str(k)

                #check if color in first image, if not set value to zero
                if color_str in hist1:
                    count1 = hist1[color_str]
                else:
                    count1 = 0
                
                #check if color in second image, if not set value to zero
                if color_str in hist2:
                    count2 = hist2[color_str]
                else:
                    count2 = 0

                #find absolute difference between two color counts
                diff = abs(count1 - count2)

                #add current color value difference to total summation
                summation = summation + diff

    #normalize value by image size
    distance = summation / (2 * 60 * 89) 

    #return final distance between two image histograms
    return distance

def texture_distance(hist1, hist2, texture = texture):
    '''
    purpose: find distance between two texture histograms

    inputs:
    hist1: first texture histogram to be analyzed
    hist2: second texture histogram to be analyzed
    texture: number of bins used to create histograms

    outputs:
    distance: normalized distance between the two image histograms
    '''
    #initialize summation value
    summation = 0

    #iterate through each texture bin
    for i in range(texture):

        #check if texture value is present in first image
        if i in hist1:
            count1 = hist1[i]
        else:
            count1 = 0
        
        #check if texture value is present in second image
        if i in hist2:
            count2 = hist2[i]
        else:
            count2 = 0

        #find absolute value of difference between images for this texture bin
        diff = abs(count1 - count2)

        #add to total distance summation
        summation = summation + diff

    #normalize distance by image size
    distance = summation / (2 * 60 * 89) 

    #return normalized distance between the two histograms
    return distance

def get_match_from_hist(color_or_text, texture = texture, blue = blue, green = green, red = red, num_imgs = num_imgs):
    '''
    purpose: find top three matches for all 40 images provided using either the generated color or texture histograms
    
    inputs
    color_or_text: used to determine if algorithm should compare by texture or color
    texture: texture bins
    blue: blue bins
    green: green bins
    red = red bins
    num_imgs: number of images to compare against one another

    outputs
    top_three: dict with each query image as a key and a nested dict containing the top three matches found 
    '''
    #initalize storage
    all_hist = {}
    top_three = {}

    #iterate through each image
    for i in range(1,num_imgs + 1):
        #generate img string
        if i < 10:
            i_str = "0" + str(i)
        else:
            i_str = str(i)

        img_str = "i" + i_str + ".jpg"
        
        #generate list of histograms for each input image to compare
        if color_or_text == 'color':
            all_hist[i] = color_histogram(img_str, blue, green, red)
        elif color_or_text == 'text':
            all_hist[i] = texture_histogram(img_str, texture)

    #counter variable used to track query image
    i = 1
    #iterate through each image histogram
    for key1 in all_hist:
        hist1 = all_hist[key1]

        #initalize match list
        dist_list = {1:0,2:0,3:0}

        #set distance to greatest possible
        match1 = 1.1
        match2 = 1.1
        match3 = 1.1

        #iterate through all other images
        for key2 in all_hist:
            if key1 != key2:
                hist2 = all_hist[key2]
                #compute distance between histograms by color or texture
                if color_or_text == 'color':
                    dist = color_distance(hist1, hist2, blue, green, red)
                elif color_or_text == 'text':
                    dist = texture_distance(hist1, hist2, texture)

                #find top three matches iteratively
                if dist < match1:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = match1
                    dist_list[2] = dist_list[1]
                    match1 = dist
                    dist_list[1] = key2
                elif dist < match2:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = dist
                    dist_list[2] = key2
                elif dist < match3:
                    match3 = dist
                    dist_list[3] = key2
        #add matches found for image to running dict
        top_three[i] = dist_list
        i += 1
    #return dict of matches for each input image
    return top_three

def get_match_from_shape( ):
    '''
    purpose: find top three matches by shape overlap

    input:
    none

    output:
    top_three: dict with each input image as key with a nested dict containing the top three matches for the query image
    '''
    #initalize
    top_three = {}

    #iterate through each input image, assumes 40
    for i in range(1,41):
        #initalize match to greater than highest possible
        match1 = 1.1
        match2 = 1.1
        match3 = 1.1
        dist_list = {1:0,2:0,3:0}
        #iterate through all other input images
        for j in range(1,41):
            if i != j:
                #generate image strings to read images from file
                if i < 10:
                    i_str = "0" + str(i)
                else:
                    i_str = str(i)

                str1 = "i" + i_str + ".jpg"

                if j < 10:
                    j_str = "0" + str(j)
                else:
                    j_str = str(j)

                str2 = "i" + j_str + ".jpg"

                #calculate distance between both images
                result = shape_overlap(str1, str2)

                #iteratively find top three matches for query image
                if result < match1:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = match1
                    dist_list[2] = dist_list[1]
                    match1 = result
                    #print("new match1:  " + str(j) + "at " + str(result))
                    dist_list[1] = j
                elif result < match2:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = result
                    #print("new match2:  " + str(j) + "at " + str(result))
                    dist_list[2] = j
                elif result < match3:
                    match3 = result
                    #print("new match3:   " + str(j) + "at " + str(result))
                    dist_list[3] = j

        top_three[i] = dist_list
    
    #return dict of top three matches for each input image
    return top_three

def get_match_from_symm():
    '''
    purpose: find top three matching images by symmetry
    
    inputs:
    none
    
    outputs
    top_three: dict with each input image as key with a nested dict containing the top three matches for the query image
    
    '''
    #find the symmetry value for each input image
    symm_list = {}
    for i in range(1,41):
        if i < 10:
            i_str = "0" + str(i)
        else:
            i_str = str(i)

        str1 = "i" + i_str + ".jpg"

        result = symmetry(str1)

        symm_list[i] = result
    
    #iterate through each to find top three matching images, see earlier functions for more detail
    top_three = {}
    for i in range(1,41):
        match1 = 1.1
        match2 = 1.1
        match3 = 1.1
        dist_list = {1:0, 2:0, 3:0}
        for j in range(1,41):
            if i != j:
                dist = abs(symm_list[i] - symm_list[j])

                if dist < match1:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = match1
                    dist_list[2] = dist_list[1]
                    match1 = dist
                    dist_list[1] = j
                elif dist < match2:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = dist
                    dist_list[2] = j
                elif dist < match3:
                    match3 = dist
                    dist_list[3] = j
        top_three[i] = dist_list

    #return dict of top three matches for each input image
    return top_three

def gestalt(c_fac, t_fac, sh_fac, symm_fac):
    '''
    purpose: uses color, texture, shape, and symmetry to find the top three matches between input images
    
    inputs
    c_fac: factor by which to multiply color distance
    t_fac: factor by which to multiply texture distance
    sh_fac: factor by which to multiply shape distance
    symm_fac: factor by which to multiply symmetry distance
    
    outputs:
    top_three: dict with each input image as key with a nested dict containing the top three matches for the query image
    '''
    
    top_three = {}

    #iterate through each input image, assumes 40
    for i in range(1,41):
        match1 = 1.1
        match2 = 1.1
        match3 = 1.1

        #generate first image string to read image
        if i < 10:
            i_str = "0" + str(i)
        else:
            i_str = str(i)

        str1 = "i" + i_str + ".jpg"

        dist_list = {1:0, 2:0, 3:0}

        #iterate through all other input images to find matches
        for j in range(1,41):
            #generate second image string
            if j < 10:
                j_str = "0" + str(j)
            else:
                j_str = str(j)
            
            str2 = "i" + j_str + ".jpg"

            #only compare different images
            if i != j:
                
                #find color distance between both images
                c_hist_1 = color_histogram(str1)
                c_hist_2 = color_histogram(str2)
                c_dist = color_distance(c_hist_1, c_hist_2)

                #find texture distance between both images
                t_hist_1 = texture_histogram(str1)
                t_hist_2 = texture_histogram(str2)
                t_dist = texture_distance(t_hist_1, t_hist_2)

                #find shape distance between both images
                sh_dist = shape_overlap(str1, str2)

                #find symmetry distance between both images
                symm_dist = abs(symmetry(str1) - symmetry(str2))

                #use given factors to calculate final distance between images
                dist = c_fac * c_dist + t_fac * t_dist + sh_fac * sh_dist + symm_fac * symm_dist

                #iteratively find top three matches
                if dist < match1:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = match1
                    dist_list[2] = dist_list[1]
                    match1 = dist
                    dist_list[1] = j
                elif dist < match2:
                    match3 = match2
                    dist_list[3] = dist_list[2]
                    match2 = dist
                    dist_list[2] = j
                elif dist < match3:
                    match3 = dist
                    dist_list[3] = j
        top_three[i] = dist_list

    #return dict of top three matches for each input image
    return top_three

def get_score(top_three):
    '''
    purpose: uses crowd sourced borda counts to find total score over all input images and their calculated matches
    
    inputs
    top_three: dict containing top three matches for each query image
    
    outputs
    total: sum of borda counts for each target image vs. query image
    '''
    total = 0
    #iterate through each input image
    for key in top_three:
        #calculate total score across all three matches
        score = crowd_data[key - 1, top_three[key][1] - 1] + crowd_data[key - 1, top_three[key][2] - 1] + crowd_data[key - 1, top_three[key][3] - 1]
        #print(str(score))
        total = total + score
    #returns total score across all input images
    return total

def concat_vh(list_2d):
    '''
    purpose: vertically and horizontally concatenates images of the same size

    source: https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
        
    This is listed as opensource and uses openCV to perform its intended function.
    '''
    
    #concatenates images horizontally first then vertically
    return cv2.vconcat([cv2.hconcat(list_h) 
                        for list_h in list_2d])


def generate_visual(results, name):
    '''
    purpose: generates 40x4 images with each row representing a query image and its top three matches with columns 1-3 being the number 1,2, and 3
    matches respectively. Also calculates scores for each target image, row, and overall score for all images.

    inputs
    results: dict containing top three matches for each query image
    name: file name to save generated image to

    outputs
    none
    '''
    h_img_list_half1 = []

    bg = np.zeros([30,89,3],dtype=np.uint8)
    bg.fill(255)
    total_score = get_score(results)

    for query in results:
        row_image_list = []
        if query < 10:
            query_image_str = 'i0' + str(query) + '.jpg'
        else:
            query_image_str = 'i' + str(query) + '.jpg'
        
        row_score = int(crowd_data[query - 1, results[query][1] - 1] + crowd_data[query - 1, results[query][2] - 1] + crowd_data[query - 1, results[query][3] - 1])
        query_img = cv2.imread(query_image_str)
        query_img = cv2.copyMakeBorder(query_img, 10, 20, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if query == 1:
            query_img = cv2.putText(query_img, 'tot_score=', (10,10), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        query_img = cv2.putText(query_img, 'q(' + str(query) + ')=' + str(row_score), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        row_image_list.append(query_img)

        for key in results[query]:
            if results[query][key] < 10:
                img_str = 'i0' + str(results[query][key]) + '.jpg'
            else:
                img_str = 'i' + str(results[query][key]) + '.jpg'
            
            score = int(crowd_data[query - 1, results[query][key] - 1])
            img = cv2.imread(img_str)
            img = cv2.copyMakeBorder(img, 10, 20, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            if query == 1 and key == 1:
                img = cv2.putText(img, str(total_score), (10,10), font, 
                    fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.putText(img, str(results[query][key]) + ', s=' + str(score), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
            row_image_list.append(img)
        
        h_img_list_half1.append(row_image_list)



    img_tile1 = concat_vh(h_img_list_half1)

    cv2.imwrite(name, img_tile1)

def happiness(results):
    '''
    purpose: calculates the set intersection of a generated top three matches versus the author's personal results
    
    inputs
    results: dict containing top three matches for each query image
     
    outputs
    set_int: value of number of images matching over all input images
    '''
    set_int = 0
    for i in range(1,41):
        img_tt = results[i]
        for j in range(1,4):
            if my_data[i - 1][j] in img_tt.values():
                set_int += 1

    print(str(set_int))
    return set_int

generate_visual(gestalt(.45, .13, .26, .20), 'me.jpg')
