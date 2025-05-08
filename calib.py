import cv2
import numpy as np
import argparse
import sys


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True,
                        help='calibration file')

    parser.add_argument('--board-width', type=int, required=True,
                        help='inner corners per row')

    parser.add_argument('--board-height', type=int, required=True,
                        help='inner corners per column')

    parser.add_argument('--square-size', type=float, default=0.024,
                        help='in meters')

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera')

    return parser.parse_args()


def main():
    args = parse_args()

    objp = np.zeros((args.board_height*args.board_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_width, 0:args.board_height].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"can't open camera {args.camera}")
        sys.exit(1)

    print("space to capture, esc to leave")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray,(args.board_width, args.board_height),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, (args.board_width, args.board_height), corners, found)


        cv2.imshow('Calibration', display)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32 and found:

            objpoints.append(objp)
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
            imgpoints.append(corners_subpix)

            print(f"capped frame #{len(objpoints)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Need 5 captures, you got ", len(objpoints))
        sys.exit(1)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration RMS error: {ret:.4f}")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())

    fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("dist_coeff", dist)
    fs.release()
    print(f"Saved calibration to {args.output}")


if __name__ == '__main__':
    main()
