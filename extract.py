import numpy as np
import cv2


x = np.load("data/initial_scene.npy", allow_pickle=True).item()
print(x.keys())
for key in x["rgbs"].keys():
    cv2.imwrite(f"{key}.png", x["rgbs"][key][:, :, ::-1])
