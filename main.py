## conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
## pip install ultralytics


import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# Load a model
model = YOLO('yolov5n.pt')  # load an official model
# model = YOLO('yolov5n-seg.pt')  # load an official model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('yolov8n-seg.pt')  # load an official model


###MODEL OPTIMIZE
#model.export(format="onnx")
# model = YOLO('yolov8n-seg.onnx')  # load an official model

# model.export(format="engine")
# model = YOLO('yolov8n-seg.engine')  # load an official model



# model.overrides['conf'] = 0.3  # NMS confidence threshold
# model.overrides['iou'] = 0.4  # NMS IoU threshold
# model.overrides['agnostic_nms'] = False  # NMS class-agnostic
# model.overrides['max_det'] = 1000  # maximum number of detections per image
# model.overrides['classes'] = 2,3,0 ## define classes
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

tracking_trajectories = {}
def process(image, track=False):

    if track is True:
        results = model.track(image, verbose=False, device=0, persist=True, tracker="bytetrack.yaml")

        for id_ in list(tracking_trajectories.keys()):
            if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                del tracking_trajectories[id_]

        for predictions in results:
            if predictions is None:
                continue

            ### instance segmentations
            if predictions.masks is not None:
                if predictions.boxes is None or predictions.masks is None or predictions.boxes.id is None:
                    continue
                for bbox, masks in zip(predictions.boxes, predictions.masks):
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)

                        label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        centroid_x = (xmin + xmax) / 2
                        centroid_y = (ymin + ymax) / 2

                        # Append centroid to tracking_points
                        if id_ is not None and int(id_) not in tracking_trajectories:
                            tracking_trajectories[int(id_)] = deque(maxlen=5)
                        if id_ is not None:
                            tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                    # Draw trajectories
                    for id_, trajectory in tracking_trajectories.items():
                        for i in range(1, len(trajectory)):
                            cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), 2)

                    for mask in masks.xy:
                        polygon = mask
                        cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

                        color_ = [int(c) for c in colors[int(classes)]]
                        # cv2.fillPoly(image, [np.int32(polygon)], color_) 
                        mask = image.copy()
                        cv2.fillPoly(mask, [np.int32(polygon)], color_) 
                        alpha = 0.8  # Adjust the transparency level
                        blended_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
                        image = blended_image.copy()


            ### body keypoints
            elif predictions.keypoints is not None:
                if predictions.boxes is None or predictions.keypoints is None or predictions.boxes.id is None:
                    continue

                for bboxs, keypoints, name in zip(predictions.boxes, predictions.keypoints, predictions.names):

                    for keypoints_ in keypoints.xy:
                        for kp in keypoints_:
                            cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                    # print(bboxs)
                    for bbox, conf, id_ in zip(bboxs.xyxy, bboxs.conf, bboxs.id):
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 225), 2)
                        label = ' Person '+str(int(id_))+' '+'{:.2f}'.format(conf)+' %'
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), ((int(bbox[0]) + dim[0] //3) - 40, int(bbox[1]) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(bbox[0]), int(bbox[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


            ### object detections
            else:
                if predictions.boxes is None or predictions.boxes.id is None:
                    continue
                for bbox_ in zip(predictions.boxes):
                    for bbox in bbox_:
                        for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                            xmin    = bbox_coords[0]
                            ymin    = bbox_coords[1]
                            xmax    = bbox_coords[2]
                            ymax    = bbox_coords[3]
                            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)

                            label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                            dim, baseline = text_size[0], text_size[1]
                            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                            cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2

                            # Append centroid to tracking_points
                            if id_ is not None and int(id_) not in tracking_trajectories:
                                tracking_trajectories[int(id_)] = deque(maxlen=5)
                            if id_ is not None:
                                tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                        # Draw trajectories
                        for id_, trajectory in tracking_trajectories.items():
                            for i in range(1, len(trajectory)):
                                cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), 2)




    if not track:
        results = model.predict(image, verbose=False, device=0)  # predict on an image
        for predictions in results:
            if predictions is None:
                continue  # Skip this image if YOLO fails to detect any objects
            
            ### instance segmentations
            if predictions.masks is not None:
                if predictions.boxes is None or predictions.masks is None:
                    continue  # Skip this image if there are no boxes or masks

                for bbox, masks in zip(predictions.boxes, predictions.masks):              
                    for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)

                        label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    for mask in masks.xy:
                        polygon = mask
                        cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

                        color_ = [int(c) for c in colors[int(classes)]]
                        # cv2.fillPoly(image, [np.int32(polygon)], color_) 
                        mask = image.copy()
                        cv2.fillPoly(mask, [np.int32(polygon)], color_) 
                        alpha = 0.8  # Adjust the transparency level
                        blended_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
                        image = blended_image.copy()


            ### body keypoints
            elif predictions.keypoints is not None:
                if predictions.boxes is None or predictions.keypoints is None:
                    continue

                for bboxs, keypoints, name in zip(predictions.boxes, predictions.keypoints, predictions.names):

                    for keypoints_ in keypoints.xy:
                        for kp in keypoints_:
                            cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                    # print(bboxs)
                    for bbox, conf in zip(bboxs.xyxy, bboxs.conf):
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 225), 2)
                        label = ' Person '+' '+'{:.2f}'.format(conf)+' %'
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), ((int(bbox[0]) + dim[0] //3) - 40, int(bbox[1]) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(bbox[0]), int(bbox[1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)



            ### object detections
            else:
                if predictions.boxes is None:
                    continue  # Skip this image if there are no boxes

                for bbox_ in zip(predictions.boxes):
                    for bbox in bbox_:             
                        for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                            xmin    = bbox_coords[0]
                            ymin    = bbox_coords[1]
                            xmax    = bbox_coords[2]
                            ymax    = bbox_coords[3]
                            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)

                            label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                            dim, baseline = text_size[0], text_size[1]
                            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                            cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


    return image


if __name__ == '__main__':
    # cap = cv2.VideoCapture('/home/bharath/Downloads/test_codes/3Dbbox/kitti/test_videos/2011_10_03_drive_0034_sync_video.mp4')
    # cap = cv2.VideoCapture('/home/bharath/Downloads/output_2.mp4')
    # cap = cv2.VideoCapture('/home/bharath/Downloads/test/suman/controlsys_api_v1.5/dec_12/output_5.mp4')
    cap = cv2.VideoCapture(0)


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 15, (frame_width, frame_height))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frameId = 0
    start_time = time.time()
    fps = str()

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame = process(frame, track=True)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
            fps = f'FPS: {fps_current:.2f}'
            start_time = time.time()  # Reset start_time for the next 20 frames

        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("yolo", frame)
        out.write(frame)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    # Release the video capture featuresect
    cap.release()
    out.release()
    cv2.destroyAllWindows()
