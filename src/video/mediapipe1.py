import json
import os
import pathlib
from threading import Thread
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd

# image = mp.Image.create_from_file("image.png")

model_task_path = (
    pathlib.Path(os.path.abspath(__file__)).parent
    / "face_landmarker_v2_with_blendshapes.task"
)


class StopVideoStream(Exception):
    pass


class VideoStreamWidget(object):
    def __init__(self, src=0, live=False, save_data=False):
        self.live = live
        self.save_data = save_data
        self.base_options = python.BaseOptions(
            # model_asset_path="./mediapipe1/face_landmarker_v2_with_blendshapes.task",
            model_asset_path=model_task_path,
        )
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        self.src = src
        self.capture = cv2.VideoCapture(self.src)
        # self.final_data = pd.DataFrame(columns=["start_time", "blendshapes"])
        self.final_data = []
        self.timestamp = 0
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [
            face_blendshapes_category.category_name
            for face_blendshapes_category in face_blendshapes
            if face_blendshapes_category.score > 0.2
        ]
        # face_blendshapes_scores = [
        #     face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]

        face_blendshapes_scores = [
            face_blendshapes_category.score
            for face_blendshapes_category in face_blendshapes
            if face_blendshapes_category.score > 0.2
        ]

        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(
            face_blendshapes_ranks,
            face_blendshapes_scores,
            label=[str(x) for x in face_blendshapes_ranks],
        )
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(
                patch.get_x() + patch.get_width(),
                patch.get_y(),
                f"{score:.4f}",
                va="top",
            )

        ax.set_xlabel("Score")
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        # plt.ion()
        plt.show()
        plt.close("all")

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image

    def append_blendshapes(self, data):
        # face_blendshapes = [
        #     face_blendshapes_category for face_blendshapes_category in data if face_blendshapes_category.score > 0.2]

        face_blendshapes = {
            face_blendshapes_category.category_name: face_blendshapes_category.score
            for face_blendshapes_category in data
            if face_blendshapes_category.score > 0.2
        }

        # self.final_data.loc[-1] = [self.timestamp, face_blendshapes]
        # self.final_data.index = self.final_data.index + 1
        # self.final_data = self.final_data.sort_index()
        # self.final_data.append([self.timestamp, face_blendshapes])
        self.final_data.append({"ts": self.timestamp, "actions": face_blendshapes})
        print(self.final_data[-1])
        return None

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame1) = self.capture.read()
                if self.status:
                    self.frame = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=self.frame1,
                        # data=cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGBA)
                    )
                    self.timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            # self.frame = mp.Image.
            if self.live:
                time.sleep(0.01)
            # time.sleep(1)

    def show_frame(self):
        # Display frames in main program
        detection_result = self.detector.detect(self.frame)
        annotated_image = self.draw_landmarks_on_image(
            self.frame.numpy_view(), detection_result
        )

        # print(detection_result.face_landmarks)
        # print(detection_result.face_blendshapes)

        if len(detection_result.face_blendshapes) > 0:
            if len(detection_result.face_blendshapes[0]) > 0:

                self.append_blendshapes(detection_result.face_blendshapes[0])

        # print(self.timestamp/1000)
        if self.live:
            cv2.imshow("frame", annotated_image)
        key = cv2.waitKey(1)
        if self.status != True or key == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()

            if self.save_data:
                # self.final_data.to_csv(f'./{self.src}.csv')
                data_df = pd.DataFrame(self.final_data, columns=["ts", "actions"])
                data_df.sort_values("ts", ascending=True, inplace=True)
                data_df.to_csv(f"{self.src}.csv", index=False)
                # with open(f"./{self.src}.json", "w") as file:
                # with open(f"{self.src}.json", "w") as file:
                #     json.dump(self.final_data, file)

            raise StopVideoStream
            # exit(1)

    def run(self):
        while True:
            try:
                self.show_frame()
            except AttributeError as e:
                print(e)
            except StopVideoStream as e:
                print(e)
                break


if __name__ == "__main__":
    video_path = r"C:\Projects\Panga\eeg-artifact-filtering\data\ANMR002.mp4"
    # video_stream_widget = VideoStreamWidget(src="../../data/ANMR002.mp4", live=False)
    video_stream_widget = VideoStreamWidget(src=video_path, live=False)
    # video_stream_widget = VideoStreamWidget(live=True)

    video_stream_widget.run()
    print(len(video_stream_widget.final_data))

    # while True:
    #     try:
    #         video_stream_widget.show_frame()
    #     except AttributeError as e:
    #         print(e)
    #     except Exception as e:
    #         print(e)
