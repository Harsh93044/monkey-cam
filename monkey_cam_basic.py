import cv2
import os

IMAGE_FOLDER = "images"
WEBCAM_INDEX = 0  # Codespaces will use virtual cam; if not available, we can adapt later


def load_monkey_images(folder):
    images = {}
    for i in range(1, 13):
        path = os.path.join(folder, f"{i}.jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not load {path}")
        else:
            images[i] = cv2.resize(img, (500, 500))
    return images


def get_position(face_box, frame_width):
    x, y, w, h = face_box
    center_x = x + w / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x > 2 * frame_width / 3:
        return "right"
    else:
        return "center"


def choose_monkey(is_smiling, position, monkeys):
    """
    Very simple mapping:
      - is_smiling = True / False
      - position = left / center / right
    Returns one of your monkey images.
    """

    if is_smiling:
        # happy / goofy set
        mapping = {
            "left": 5,     # tongue out
            "center": 4,   # big smile + finger up
            "right": 9,    # rich monkey with money
        }
    else:
        # neutral / angry / thinking set
        mapping = {
            "left": 1,     # thinking
            "center": 6,   # blank face
            "right": 11,   # angry arms crossed
        }

    num = mapping.get(position, 6)
    return monkeys.get(num, list(monkeys.values())[0])


def main():
    monkeys = load_monkey_images(IMAGE_FOLDER)
    if not monkeys:
        print("No monkey images loaded! Check your images folder.")
        return

    # Haar cascades for face + smile
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Are you in an environment with a camera?")
        return

    print("Press 'q' to quit.")

    current_monkey = monkeys[6]  # start with neutral

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )

        label_text = "No face detected"
        h, w, _ = frame.shape

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_roi_gray = gray[y:y + fh, x:x + fw]

            smiles = smile_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.8,
                minNeighbors=20
            )

            is_smiling = len(smiles) > 0
            position = get_position((x, y, fw, fh), w)
            current_monkey = choose_monkey(is_smiling, position, monkeys)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 255, 255), 2)

            label_text = f"{'SMILE' if is_smiling else 'NO SMILE'} | {position.upper()}"

        cv2.putText(
            frame,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Webcam", frame)
        cv2.imshow("Monkey Reaction", current_monkey)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
