import cv2

def draw_sequence(Text, Bounding_boxes, canvas):
    test_seq = []
    for i, text in enumerate(Text):
        # 序号
        seq = i + 1

        test_seq.append((seq, text))

        # 圆形气泡
        Bounding_box = Bounding_boxes[i].astype("int32")
        # print(Bounding_box.shape)
        # print(Bounding_box)

        right_upper = Bounding_box[1]
        # print(type(right_upper))
        # print(right_upper)

        bubble_pt = (right_upper[0] + 20,right_upper[1] - 20)
        # print(bubble_pt)
        # print(type(bubble_pt))

        cv2.circle(canvas, bubble_pt, 10, color=(250,125,120), thickness=-1)

        if seq < 10:
            text_pt = (bubble_pt[0] - 5, bubble_pt[1] + 5)
        else:
            text_pt = (bubble_pt[0] - 10, bubble_pt[1] + 5)
        cv2.putText(canvas, str(seq), text_pt, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=None)

    return test_seq