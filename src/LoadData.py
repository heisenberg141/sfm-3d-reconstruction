import numpy as np
import cv2
import sys
import os


def main():
    correspondences, _ = LoadFeatureCorrespondences("../SFM_data/")
    removeDuplicateCorrespondences(correspondences)
    # image1_points = np.array(correspondences[(1,2)])[:,0]
    # image2_points =  np.array(correspondences[(1,2)])[:,1]
    # print(image1_points[:8])
    # print(image2_points[:8])


    
def removeDuplicateCorrespondences(correspondences):
    for image_pair in correspondences:
        image_correspondences = correspondences[image_pair]
        unique_data = np.unique(image_correspondences, axis=0)
        print(len(image_correspondences), len(unique_data))
        correspondences[image_pair] = unique_data
    
    return correspondences


def LoadFeatureCorrespondences(dir):
    '''
    Dictionary structure 
    {
        (imageid1,imageid2) : [ [featureimageid1,featureimageid2],
                                .
                                .
                                .
                            ]
                .
                .
                .
    }
    
    '''
    
    correspondances = dict()
    rect_images = [filename for filename in os.listdir(dir) if filename.endswith("png")]
    matching_files = [filename for filename in os.listdir(dir) if filename.startswith("matching")]
    for i in range(len(rect_images)):
        for j in range(i,len(rect_images)):
            if i!=j:
                correspondances[(i+1,j+1)] = list()
    
    for file in matching_files:
        with open(os.path.join(dir,file)) as f:
            current_img_id = int(file[8])
            
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i>0:
                    raw_correspondance = np.fromstring(line, dtype=float, sep=' ')
                    # print("RAW: ",raw_correspondance)
                    current_image_coordinate = [raw_correspondance[4],raw_correspondance[5]]
                    matches = raw_correspondance[6:,].reshape([-1,3])
                    for match in matches:
                        corresponding_image_coordinate = [match[1],match[2]]
                        correspondances[(current_img_id,int(match[0]))].append(current_image_coordinate+corresponding_image_coordinate)
                    # raw_correspondance = raw_correspondance[1:,].reshape([n_correspondances,-1])
                    # break
            # break
    for key in correspondances:
        correspondances[key] = np.array(correspondances[key])
    for key in correspondances:
        print("for", key, ":")
        print(len(correspondances[key]), len(set(map(tuple, correspondances[key]))))


    
    return correspondances, rect_images
                  




if __name__=='__main__':
    print("Note: Run from the SFM directory.")
    main()