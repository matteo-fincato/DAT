import json
import os

root = "/mnt/beegfs/homes/mfincato/Tracking/trackformer/data/Panoptic/annotations"
file_name = "val2.json"

def convert(filename_in, filename_out):
    with open(filename_in, 'r') as f :
        data_in = json.load(f)

        data_out = {}
        data_out["images"] = data_in["images"]
        data_out["videos"] = data_in["videos"]
        data_out["categories"] = data_in["categories"]
        data_out["info"] = data_in["info"]
        data_out["annotations"] = []
        for ann in data_in["annotations"]:
            dict_tmp = {}
            dict_tmp["id"] = ann["id"]
            dict_tmp["image_id"] = ann["image_id"]
            dict_tmp["category_id"] = ann["category_id"]
            dict_tmp["iscrowd"] = ann["iscrowd"]
            dict_tmp["bbox"] = ann["bbox"]
            dict_tmp["area"] = ann["area"]
            # dict_tmp["segmentation"] = ann["segmentation"]
            dict_tmp["num_keypoints"] = ann["num_keypoints"]
            dict_tmp["keypoints"] = ann["keypoints"]
            dict_tmp["keypoints_3d"] = ann["keypoints_3d"]
            dict_tmp["track_id"] = ann["instance_id"]
            dict_tmp["visibility"] = ann["vis"]
            dict_tmp["ignore"] = 0 if ann["vis"] > 0.25 else 1
            data_out["annotations"].append(dict_tmp)
    
    with open(filename_out, 'w') as f :
        json.dump(data_out, f)

def add_sequences(filename_in, filename_out):
    with open(filename_in, 'r') as f :
        data_in = json.load(f)
        data_in["sequences"] = []
        for ann in data_in["annotations"]:
            id = ann["id"]
            image_id = ann["image_id"]
            imgann= data_in["images"][image_id]
            if imgann['id']!=image_id:
                print("ERROR")
            filename = imgann["file_name"]
            seqname=filename.split("/")[0]+"_"+filename.split("/")[2]
            if seqname not in data_in["sequences"]:
                data_in["sequences"].append(seqname)
            data_in["annotations"][id]["seq"] = seqname
            
    with open(filename_out, 'w') as f :
        json.dump(data_in, f)
        
    
# convert(os.path.join(root, file_name), os.path.join(root, "train2_converted.json"))
add_sequences(os.path.join(root, file_name), os.path.join(root, "val_converted.json"))