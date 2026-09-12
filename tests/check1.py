import cv2
import numpy as np
from mrz.generator.td3 import TD3CodeGenerator


def generate_td3_lines() -> tuple[str, str]:
    generator = TD3CodeGenerator(
        document_type="P",
        country_code="UTO",
        surname="ERIKSSON",
        given_names="ANNA MARIA",
        document_number="L898902C3",
        nationality="UTO",
        birth_date="740812",
        sex="F",
        expiry_date="240415",
        optional_data="ZE184226B",
    )
    lines = str(generator).strip().split("\n")
    return lines[0], lines[1]


def create_passport_with_contours(output_path: str = "passport.png"):
    line1, line2 = generate_td3_lines()

    # 1. Total Canvas with desktop/scanner margin
    canvas_w = 1600
    canvas_h = 2200
    margin_x = (canvas_w - 1476) // 2  # 62 px
    margin_y = (canvas_h - 2079) // 2  # 60 px

    doc_w = 1476
    doc_h = 2079

    # Darker background table/desk surface to give clear edge contrast
    img = np.full((canvas_h, canvas_w, 3), 180, dtype=np.uint8)

    # 2. Main Passport Page Body
    p_x1, p_y1 = margin_x, margin_y
    p_x2, p_y2 = margin_x + doc_w, margin_y + doc_h
    cv2.rectangle(img, (p_x1, p_y1), (p_x2, p_y2), (245, 245, 245), -1)

    # === CONTOUR 1: Outer Document Perimeter Box ===
    # A distinct border for document-level contour detectors (findContours / minAreaRect)
    cv2.rectangle(img, (p_x1, p_y1), (p_x2, p_y2), (30, 30, 30), thickness=4)

    # Center booklet spine / fold line
    spine_y = p_y1 + (doc_h // 2)
    cv2.line(img, (p_x1, spine_y), (p_x2, spine_y), (140, 140, 140), thickness=2)

    # Top Page (Observations / Visas)
    cv2.rectangle(
        img,
        (p_x1 + 60, p_y1 + 60),
        (p_x2 - 60, spine_y - 60),
        (230, 230, 230),
        -1,
    )
    cv2.putText(
        img,
        "TOP PAGE (OBSERVATIONS / VISAS)",
        (p_x1 + 380, p_y1 + 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (150, 150, 150),
        2,
        cv2.LINE_AA,
    )

    # Bottom Page Photo Box
    photo_x = p_x1 + 80
    photo_y = spine_y + 120
    cv2.rectangle(
        img,
        (photo_x, photo_y),
        (photo_x + 400, photo_y + 520),
        (215, 215, 215),
        -1,
    )
    cv2.putText(
        img,
        "PHOTO",
        (photo_x + 130, photo_y + 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (130, 130, 130),
        2,
        cv2.LINE_AA,
    )

    # Bottom Page Text Fields
    field_y = spine_y + 170
    for field in [
        "Type: P",
        "Code: UTO",
        "Passport No: L898902C3",
        "Surname: ERIKSSON",
        "Given Names: ANNA MARIA",
        "Nationality: UTO",
    ]:
        cv2.putText(
            img,
            field,
            (p_x1 + 540, field_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        field_y += 75

    # === CONTOUR 2: Distinct MRZ Bounding Box ===
    # High-contrast border and pure white fill to easily trigger MRZ region-of-interest detectors
    mrz_x1 = p_x1 + 40
    mrz_y1 = p_y2 - 340
    mrz_x2 = p_x2 - 40
    mrz_y2 = p_y2 - 40

    # Fill MRZ background
    cv2.rectangle(img, (mrz_x1, mrz_y1), (mrz_x2, mrz_y2), (255, 255, 255), -1)
    # Draw high-contrast contour border (thickness=3)
    cv2.rectangle(img, (mrz_x1, mrz_y1), (mrz_x2, mrz_y2), (20, 20, 20), thickness=3)

    # Render standard MRZ text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    thickness = 2
    color = (0, 0, 0)

    cv2.putText(
        img,
        line1,
        (mrz_x1 + 35, mrz_y1 + 120),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        line2,
        (mrz_x1 + 35, mrz_y1 + 240),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite(output_path, img)
    print(f"[+] Saved synthetic passport with contour boxes to {output_path}")


if __name__ == "__main__":
    create_passport_with_contours()