import csv


FIELDS = [

"frame",
"tracker_id",
"identity_id",
"recognized_identity",
"recognition_score",
"recognition_status",

"recognition_correct_by_tracker",

"snapshot_path",

"face_detected",
"tracker_active",
"bbox",

"area_ratio",
"brightness",
"landmark_conf",

"scale_pass",
"brightness_pass",
"landmark_pass",

"status",
"failure",
"tracker_invalid"
]

class CSVLogger:

    def __init__(self,path):

        self.f = open(path,"w",newline="")
        self.w = csv.DictWriter(self.f,fieldnames=FIELDS)

        self.w.writeheader()

    def log(self,row):
        self.w.writerow(row)