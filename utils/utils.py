import os

def xywh_2_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [(x-w/2), (y-h/2), (x+w/2), (y+h/2)]

def xyxy_2_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [((x1+x2)/2), ((y1+y2)/2), (x2-x1), (y2-y1)]

def normalized_2_actual_pixel_values(bbox, image_width, image_height):
    return [bbox[0]*image_width, bbox[1]*image_height, bbox[2]*image_width, bbox[3]*image_height]

def get_xyxy_bboxes_from_YOLO_format_txt(label_path):
    #This function returns bboxes in xyxy format EXCLUDING the class label
    with open(label_path, "r") as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            line = line.strip()
            if(len(line) == 0):
                continue
            line = line.split(" ")
            line = [float(x) for x in line[1:]]
            bboxes.append(line)
        bboxes = [xywh_2_xyxy(bbox) for bbox in bboxes]
        return bboxes
    

def get_IoU(bbox1, bbox2):
    #bbox1 and bbox2 are in xyxy format
    #returns IoU
    x1_1, y1_1, x1_2, y1_2 = bbox1
    x2_1, y2_1, x2_2, y2_2 = bbox2
    
    x1_INTERSECTION = max(x1_1, x2_1)
    y1_INTERSECTION = max(y1_1, y2_1)
    x2_INTERSECTION = min(x1_2, x2_2)
    y2_INTERSECTION = min(y1_2, y2_2)
    
    intersection_area = max(0, x2_INTERSECTION-x1_INTERSECTION) * max(0, y2_INTERSECTION-y1_INTERSECTION)
    
    bbox1_area = (x1_2-x1_1) * (y1_2-y1_1)
    bbox2_area = (x2_2-x2_1) * (y2_2-y2_1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area


def calculate_TP_FP_FN_from_label_path(predictions, label_path, IoU_threshold=0.5):
    #predictions should be ordered by their confidence level
    #predictions should be a list where its elements are in xyxy format lists
    #label_path should be a txt file in YOLO format
    #returns TP, FP, FN counts for the given predictions and label_path

    TP, FP, FN = 0, 0, 0

    label_bboxes = get_xyxy_bboxes_from_YOLO_format_txt(label_path)


    for prediction in predictions:
        assert(len(prediction) == 4); "Prediction should be in xyxy format"
        assert(prediction[0] < prediction[2]); "x1 should be smaller than x2"
        assert(prediction[1] < prediction[3]); "y1 should be smaller than y2"
        
        max_IoU = -1
        max_IoU_index = -1

        for current_index, label_bbox in enumerate(label_bboxes):
            assert(len(label_bbox) == 4); "Label bbox should be in xyxy format"
            assert(label_bbox[0] < label_bbox[2]); "x1 should be smaller than x2"
            assert(label_bbox[1] < label_bbox[3]); "y1 should be smaller than y2"

            current_IoU = get_IoU(prediction, label_bbox)

            
            if(current_IoU > max_IoU):
                max_IoU = current_IoU
                max_IoU_index = current_index

        if(max_IoU >= IoU_threshold):
            TP += 1
            label_bboxes.pop(max_IoU_index)
        else:
            FP += 1
    
    FN = len(label_bboxes)

    return TP, FP, FN


def calculate_TP_FP_FN(predictions, label_bboxes, IoU_threshold=0.5):
    #predictions should be ordered by their confidence level
    #predictions should be a list where its elements are in xyxy format lists
    #label_bboxes should be a list where its elements are in xyxy format lists
    #returns TP, FP, FN counts for the given predictions and label_path

    TP, FP, FN = 0, 0, 0


    for prediction in predictions:
        assert(len(prediction) == 4); "Prediction should be in xyxy format"
        assert(prediction[0] < prediction[2]); "x1 should be smaller than x2"
        assert(prediction[1] < prediction[3]); "y1 should be smaller than y2"
        
        max_IoU = -1
        max_IoU_index = -1

        for current_index, label_bbox in enumerate(label_bboxes):
            assert(len(label_bbox) == 4); "Label bbox should be in xyxy format"
            assert(label_bbox[0] < label_bbox[2]); "x1 should be smaller than x2"
            assert(label_bbox[1] < label_bbox[3]); "y1 should be smaller than y2"

            current_IoU = get_IoU(prediction, label_bbox)

            
            if(current_IoU > max_IoU):
                max_IoU = current_IoU
                max_IoU_index = current_index

        if(max_IoU >= IoU_threshold):
            TP += 1
            label_bboxes.pop(max_IoU_index)
        else:
            FP += 1
    
    FN = len(label_bboxes)

    return TP, FP, FN



def calculate_TP_FP_FN_all_images_from_label_path(all_predictions, label_paths, IoU_threshold=0.5):
    #all_predictions should be a list of lists where its elements are in xyxy format lists
    #label_paths should be a list of txt files in YOLO format
    #returns TP, FP, FN counts for the given predictions and label_paths

    TP, FP, FN = 0, 0, 0
    for prediction, label_path in zip(all_predictions, label_paths):
        current_TP, current_FP, current_FN = calculate_TP_FP_FN_from_label_path(prediction, label_path, IoU_threshold=IoU_threshold)
        TP += current_TP
        FP += current_FP
        FN += current_FN

    return TP, FP, FN


def calculate_TP_FP_FN_all_images(all_predictions, labels, IoU_threshold=0.5):
    #all_predictions should be a list of lists where its elements are in xyxy format lists
    #labels should be a list of lists where its elements are in xyxy format lists
    #returns TP, FP, FN counts for the given predictions and label_paths

    TP, FP, FN = 0, 0, 0
    for prediction, label in zip(all_predictions, labels):
        current_TP, current_FP, current_FN = calculate_TP_FP_FN(prediction, list(label), IoU_threshold=IoU_threshold)
        TP += current_TP
        FP += current_FP
        FN += current_FN

    return TP, FP, FN




def calculate_recall_and_precision_from_TP_FP_FN(TP,FP,FN):
    #returns recall and precision
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    return recall, precision


def calculate_recall_and_precision(all_predictions, label_paths, IoU_threshold=0.5):
    TP, FP, FN = calculate_TP_FP_FN_all_images(all_predictions, label_paths, IoU_threshold)
    return calculate_recall_and_precision_from_TP_FP_FN(TP, FP, FN)

def get_F1_score_from_recall_and_precision(recall, precision):
    #harmonic mean of recall and precision
    return (2 * recall * precision) / (recall + precision)


if("__main__" == __name__):
    x = get_xyxy_bboxes_from_YOLO_format_txt("/home/umut/Desktop/thermal-disaster-dataset/HIT_UAV_and_NII_CU_dataset/test/labels/0002.txt")
    print(x)


