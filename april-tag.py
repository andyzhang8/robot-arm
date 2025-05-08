import cv2
import numpy as np
import argparse
import cv2.aruco as aruco
import os



def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibration file not found: {path}")

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = fs.getNode("camera_matrix").mat()
    
    dist_coeffs = fs.getNode("dist_coeff").mat()
    
    fs.release()
    return camera_matrix, dist_coeffs


def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    if sy > 1e-6:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])

    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0

    return np.degrees([roll, pitch, yaw])


def compute_metrics(tvec):
    x, y, z = tvec.flatten()

    lateral   =  x
    forward   =  z
    vertical  = -y

    range_m    = np.hypot(x, z)
    bearing    = np.degrees(np.arctan2(x, z))
    elevation  = np.degrees(np.arctan2(vertical, range_m))

    return lateral, forward, vertical, range_m, bearing, elevation


def draw_overlay(frame, text, pos=(10,30), color=(0,255,0)):
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (pos[0], pos[1]+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def detect_and_pose(frame, aruco_dict, parameters, camera_matrix,
                    dist_coeffs, tag_size):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is None:
        return frame

    aruco.drawDetectedMarkers(frame, corners)
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, tag_size, camera_matrix, dist_coeffs)

    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        tag_id = ids.flatten()[idx]
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, tag_size*0.5)

        roll, pitch, yaw = rvec_to_euler(rvec)
        lat, fwd, vert, rng, bear, ele = compute_metrics(tvec)

        info = (
            f"ID {tag_id}: X={lat:.2f}m Y={fwd:.2f}m Z={vert:.2f}m\n"
            f"Roll={roll:.1f}° Pitch={pitch:.1f}° Yaw={yaw:.1f}°\n"
            f"Range={rng:.2f}m Bearing={bear:.1f}° Elevation={ele:.1f}°"
        )


        draw_overlay(frame, info, pos=(10, 30 + idx*80))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', required=True,
                        help='calib file')
    parser.add_argument('--tag-size', type=float, default=0.05,
                        help='tag side len')
    parser.add_argument('--camera', type=int, default=0,
                        help='cam')
    args = parser.parse_args()


    cam_mtx, dist = load_calibration(args.calib)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()


    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"can't open camera {args.camera}")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detect_and_pose(frame, aruco_dict, params,
                              cam_mtx, dist, args.tag_size)
                              
        cv2.imshow('This is what the arm sees', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
