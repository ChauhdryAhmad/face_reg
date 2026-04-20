def iou(a,b):

    xA = max(a[0],b[0])
    yA = max(a[1],b[1])
    xB = min(a[2],b[2])
    yB = min(a[3],b[3])

    inter = max(0,xB-xA)*max(0,yB-yA)

    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])

    union = areaA + areaB - inter

    if union==0:
        return 0

    return inter/union