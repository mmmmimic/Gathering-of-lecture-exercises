'''
Weekly Project
Implementation of stereo block matching
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

class Stereo():
    def __init__(self):
        self.pic = []

    def addPic(self, namelist):
        for name in namelist:
            im = imread(name)
            self.pic.append(im)
    
    @staticmethod
    def SAD(img1, img2):
        # return the absolute difference of two pictures
        return np.sum(np.abs(img1-img2))
    
    def match(self, templ, method, sel="min"):
        res = []
        sim = 0
        pic = self.pic
        for im in pic:
            sim = eval("self."+method+"(templ,im)")
            res.append(sim)

        if sel=="min":
            for i in range(len(res)):
                if res[i]==min(res):
                    break
        else:
            for i in range(len(res)):
                if res[i]==min(res):
                    break
        return pic[i], i
    
    def locateInRow(self, templ, row, method, minmax="min", step=1, padding=False, padding_method="zeros"):
        # generate a filter
        sim_row = np.ones((row.shape[1],1))*-1 # Similarity
        if row.shape[0]!=templ.shape[0]:
            print("Error, template doesn't match!")
            sys.exit(0)
        inter = int(np.floor((row.shape[1]-templ.shape[1])/step))
        left = int((templ.shape[1]-1)/2)
        for i in range(inter+1):
            vis = row[:,(0+step*i):(templ.shape[0]+step*i)]
            sim = eval("self."+method+"(templ,vis)")
            sim_row[left+i*step] = sim
        for i in range(len(sim_row)):
            if minmax=="min":
                if sim_row[i]==sim_row[sim_row!=-1].min() and sim_row[i]!=-1:
                    break
            else:
                if sim_row[i]==sim_row[sim_row!=-1].max() and sim_row[i]!=-1:
                    break
        #sim_row = [int((each-min(sim_row))/(max(sim_row)-min(sim_row))*255) for each in sim_row]
        return i   

    def imgLocate(self, im1, im2, mask, method, templ_step=7, step=7, padding=False, padding_method="zeros"):
        # firstly, create the template
        # It's not good here to use 3 loops since the time complexity is O(n^3)
        if im1.shape!=im2.shape:
            print("Error, image size doesn't match!")
            sys.exit(0)
        value = np.ones_like(im1)
        inter_y = int(np.floor((im1.shape[0]-mask[0])/templ_step))
        inter_x = int(np.floor((im1.shape[1]-mask[1])/templ_step))
        left_y = int((mask[0]-1)/2)
        left_x = int((mask[1]-1)/2)
        for k in range(inter_y+1):
            #print('Current Row: '+str(k)+' /'+str(inter_y))
            row = im2[(0+templ_step*k):(mask[0]+templ_step*k),:]
            imm1 = im2.copy()
            cv2.rectangle(imm1, (0, 0+templ_step*k), (row.shape[1], mask[0]+templ_step*k), (255, 0, 0))
            for j in range(inter_x+1):
                templ = im1[(0+templ_step*k):(mask[0]+templ_step*k),
                (0+templ_step*j):(mask[1]+templ_step*j)]
                imm2 = im1.copy()
                cv2.rectangle(imm2, (0+templ_step*j, 0+templ_step*k), (mask[0]+templ_step*j, mask[1]+templ_step*k), (255, 0, 0))
                imm3 = cv2.hconcat((imm1, imm2))
                idx = self.locateInRow(templ, row, method)
                # disparity
                v = np.abs(idx-j)
                value[left_y+k*templ_step:left_y+(k+1)*templ_step, left_x+(j-1)*templ_step:left_x+(j+1)*templ_step]=v 
                imm3 = cv2.hconcat((imm3, value))
                cv2.imshow('rec', imm3)
                cv2.waitKey(500)
        return value

def imread(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

if __name__=="__main__":
    s = Stereo()
    nose_left = imread("nose_left.png")
    nose_right = imread("nose_right.png")
    #print(s.SAD(nose_left, nose_right))
    namelist = ["nose1.png", "nose2.png", "nose3.png"]
    s.addPic(namelist)
    im, i = s.match(nose_left, "SAD")
    nose_span = imread("nose_span.png")
    i = s.locateInRow(nose_left, nose_span, "SAD", step=1)
    print(i)
    #cv2.rectangle(nose_span,(i-3,0),(i+3,7),(0, 255, 0), 1)
    #f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18,18))
    #ax_left.imshow(nose_span,cmap="gray")
    #ax_right.imshow(nose_left, cmap="gray")
    #plt.show()
    tl = imread("tsukuba_left.png")
    tr = imread("tsukuba_right.png")
    tl = cv2.resize(tl, (128, 128))
    tr = cv2.resize(tr, (128, 128))
    v, I = s.imgLocate(tl, tr, [7, 7], "SAD")
    #v[v==999999] = max(v[v!=999999])
    #for i in range(v.shape[0]):
    #    for j in range(v.shape[1]):
    #        v[i,j] = 255-int((v[i,j]-v.min())/(v.max()-v.min()))*255 
    #plt.imshow(I)
    print(v)
    cv2.imshow('disparity map', v)
    cv2.destroyAllWindows()