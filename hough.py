import numpy as np
import cv2
import matplotlib.pyplot as plt
import pillow_heif

INPUT = "./input/test_img2.HEIC"

def main():
    # adding file check later for other extensions
    heif_file = pillow_heif.open_heif(INPUT, convert_hdr_to_8bit=False, bgr_mode=True)
    image = np.asarray(heif_file)

    # Convert to grayscale, blurred it and detect edges from the black&white input
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 200)

    horizontal, vertical = extract_lines(edges)
    
    horizontal_cluster = cluster_lines(
        horizontal,
        "y",
        1, 3,
    )
    vertical_cluster = cluster_lines(
        vertical,
        "x",
        0, 2,
    )

    output = image.copy()

    height = image.shape[0]
    hscores = score_clusters(horizontal_cluster, height)
    
    output = draw_score(hscores, output)
    cv2.imwrite("./output/hough.jpg", output)

def extract_lines(edges):
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=200,
        maxLineGap=50
    )
    print(lines[1])
    lines = sorted(lines, key=lambda x: x[0])
    

    horizontal = []
    vertical = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1

            angle = np.degrees(np.arctan2(dy, dx))

            length = np.hypot(
                        x2-x1,
                        y2-y1)

            if abs(angle) < 20:
                horizontal.append({
                    "line": line,
                    "y": (y1+y2)/2,
                    "length": length
                })
            if abs(angle) > 70 and abs(angle) < 110:
                vertical.append({
                    "line": line,
                    "x": (x1+x2)/2,
                    "length": length
                })
    return horizontal, vertical

def cluster_lines(lines, key, idx1, idx2, threshold=40):
    clusters = []

    for item in lines:

        value = item[key]

        if not clusters:
            clusters.append({
                "lines": [item],
            })

        elif abs(value - clusters[-1]["center"]) < threshold:

            clusters[-1]["lines"].append(item)

        else:
            clusters.append({
                "lines": [item]
            })

        cluster = clusters[-1]

        values = [ l[key] for l in cluster["lines"]]

        lengths = [ l["length"] for l in cluster["lines"]]

        # update values
        cluster["center"] = np.average(
            values,
            weights=lengths
        )
        cluster["total_length"] = sum(lengths)

    return clusters

def score_clusters(clusters, height):
    scores = []

    for c in clusters:
        y = c["center"]
        position_score = abs(
            y - height/2
        ) / (height/2)
        score = c["total_length"] * position_score
        scores.append((score, c))
    scores = sorted(scores, key=lambda x: x[0])
    return scores

def draw_score(clusters, output):

    for _, c in clusters[-5:]:
        for l in c["lines"]:
            x1, y1, x2, y2 = l["line"]
            cv2.line(
                output,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                4
            )
    return output
if __name__ == '__main__':
    main()