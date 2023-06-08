import os
import requests
from io import BytesIO
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_bytes


def save_buffer_as_pdf(buffer, output_file_path):
    with open(output_file_path, 'wb') as output_file:
        output_file.write(buffer)


def download_pdf_from_url(url):
    response = requests.get(url)
    buffer = BytesIO()
    buffer.write(response.content)
    output_file_path = 'output.pdf'
    save_buffer_as_pdf(buffer, output_file_path)
    return buffer


def download_pdf(url, destination):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    with open(destination, 'wb') as file:
        file.write(response.content)


def read_barcode(barcode_file):
    img = cv2.imread(barcode_file)
    barcodes = pyzbar.decode(img)
    for barcode in barcodes:
        barcode = barcode.data.decode("utf-8")
        return barcode


def divide_pdf_pages(pdf_buffer):
    pdf_reader = PdfFileReader(pdf_buffer)
    num_pages = pdf_reader.getNumPages()
    page_buffers = []

    for page in range(num_pages):
        pdf_writer = PdfFileWriter()
        current_page = pdf_reader.getPage(page)
        pdf_writer.addPage(current_page)
        page_buffer = BytesIO()
        pdf_writer.write(page_buffer)
        page_buffers.append(page_buffer)

    return page_buffers


def process_exam2(image_path):
    # Read the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    img = img[height // 5:, :]
    height, width, _ = img.shape

    # Divide the image into 8 equal parts
    # 4 rows and 2 columns
    # put the divided images in an array
    squares = np.zeros((1, 4))
    for i in range(4):
        for j in range(2):
            x1 = width // 2 * j
            x2 = width // 2 * (j + 1)
            y1 = height // 4 * i
            y2 = height // 4 * (i + 1)
            squares = np.append(squares, [[x1, y1, x2, y2]], axis=0)

    # Remove the first element of the array
    squares = np.delete(squares, 0, 0)

    # loop through the squares and find the big rectangle containing 8 smaller rectangles
    answers = np.zeros((1, 4))
    extractedAnswers = []
    answersArray = []
    for i, sq in enumerate(squares):
        x1, y1, x2, y2 = sq.astype(int)
        # Extract the region of interest (ROI)
        roi = img[y1:y2, x1:x2]
        # Convert the ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to the image
        _, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over each contour and extract the bounding rectangle
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Show the image with the rectangles drawn
            cv2.imshow("Image with rectangles", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # crop the rectangle
            crop = roi[y:y + h, x:x + w]

            # Check if the cropped region has valid dimensions
            if crop.size != 0:
                # add the rectangle to the answers array with height and width relative to the original image
                answers = np.append(answers, [[x1 + x, y1 + y, w, h]], axis=0)
                cv2.imshow("Cropped rectangle", crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                answers = np.append(answers, [[x1 + x, y1 + y, x1 + x + w, y1 + y + h]], axis=0)

                # look for 8 answers in the answer square
                # divide the rectangle into 8 equal parts
                # 2 rows and 4 columns
                # put the divided images in an array
                squares = np.zeros((1, 4))
                for i in range(2):
                    for j in range(4):
                        x1 = w // 4 * j
                        x2 = w // 4 * (j + 1)
                        y1 = h // 2 * i
                        y2 = h // 2 * (i + 1)
                        squares = np.append(squares, [[x1, y1, x2, y2]], axis=0)

                # Remove the first element of the array
                squares = np.delete(squares, 0, 0)
                print(squares)
                # loop through the squares and see if they are 50% black
                for i, sq in enumerate(squares):
                    x1, y1, x2, y2 = sq.astype(int)
                    # Extract the region of interest (ROI)
                    roi = crop[y1:y2, x1:x2]
                    # Convert the ROI to grayscale
                    cv2.imshow("Cropped rectangle", roi)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Apply a binary threshold to the image
                    _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
                    # Count the number of pixels that are black or near black by 25 percent
                    black_pixels = np.sum(binary == 255)
                    # Count the number of pixels in the image
                    total_pixels = roi.shape[0] * roi.shape[1]
                    # Calculate the percentage of black pixels
                    percentage = black_pixels / total_pixels
                    print(percentage)
                    if percentage > 0.5:
                        print("Black")
                    else:
                        print("White")
            else:
                print("Invalid region of interest. Skipping display.")

    results = {}
    for index, answer in enumerate(answersArray):
        results[index] = answer

    return results


def process_exam(image_path):
    img = cv2.imread(image_path)
    height, _, _ = img.shape
    cropped_img = img[height // 5:, :]

    cell_height = cropped_img.shape[0] // 4
    cell_width = cropped_img.shape[1] // 2
    results = {}

    for row in range(4):
        for col in range(2):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width
            cell = cropped_img[y1:y2, x1:x2]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if contours list is not empty
            if not contours:
                continue

            largest_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)
            answer_box = cell[y:y + h, x:x + w]
            answer_cell_height = answer_box.shape[0] // 2
            answer_cell_width = answer_box.shape[1] // 4

            answer_array = []

            for ans_row in range(2):
                for ans_col in range(4):
                    y1_ans = ans_row * answer_cell_height
                    y2_ans = (ans_row + 1) * answer_cell_height
                    x1_ans = ans_col * answer_cell_width
                    x2_ans = (ans_col + 1) * answer_cell_width
                    answer_cell = answer_box[y1_ans:y2_ans, x1_ans:x2_ans]

                    gray_ans = cv2.cvtColor(answer_cell, cv2.COLOR_BGR2GRAY)
                    _, thresh_ans = cv2.threshold(gray_ans, 120, 255, cv2.THRESH_BINARY_INV)
                    black_pixels = np.sum(thresh_ans == 255)
                    total_pixels = thresh_ans.size
                    percentage = black_pixels / total_pixels
                    if percentage > 0.3:
                        answer_array.append(1)
                    else:
                        answer_array.append(0)

            question_number = len(results)
            results[question_number] = answer_array

    return results

def process_exam_2(image_path):
    img = cv2.imread(image_path)
    height, _, _ = img.shape
    cropped_img = img[height // 5:, :]

    cell_height = cropped_img.shape[0] // 4
    cell_width = cropped_img.shape[1] // 4
    results = {}

    for row in range(4):
        for col in range(4):
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width
            cell = cropped_img[y1:y2, x1:x2]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if contours list is not empty
            if not contours:
                continue

            largest_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)
            answer_box = cell[y:y + h, x:x + w]
            answer_cell_height = answer_box.shape[0] // 2
            answer_cell_width = answer_box.shape[1] // 4

            answer_array = []

            for ans_row in range(2):
                for ans_col in range(4):
                    y1_ans = ans_row * answer_cell_height
                    y2_ans = (ans_row + 1) * answer_cell_height
                    x1_ans = ans_col * answer_cell_width
                    x2_ans = (ans_col + 1) * answer_cell_width
                    answer_cell = answer_box[y1_ans:y2_ans, x1_ans:x2_ans]

                    gray_ans = cv2.cvtColor(answer_cell, cv2.COLOR_BGR2GRAY)
                    _, thresh_ans = cv2.threshold(gray_ans, 120, 255, cv2.THRESH_BINARY_INV)
                    black_pixels = np.sum(thresh_ans == 255)
                    total_pixels = thresh_ans.size
                    percentage = black_pixels / total_pixels
                    if percentage > 0.3:
                        answer_array.append(1)
                    else:
                        answer_array.append(0)

            question_number = len(results)
            results[question_number] = answer_array

    return results



def pdf_buffer_to_images(pdf_path):
    pdf_buffer = open(pdf_path, 'rb')
    pdf_buffer.seek(0)  # reset the buffer position to the beginning
    images = convert_from_bytes(pdf_buffer.read())
    # base path
    base_path = 'Temp\TempImg'
    # save the images
    imagePaths = []
    for i in range(len(images)):
        image_path = os.path.join(base_path, f'{i}.png')
        images[i].save(image_path, 'PNG')
        imagePaths.append(image_path)
    return imagePaths


def get_final_grade(image_buffer):
    # Decode the image buffer
    img = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image for barcode detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block_size = 11
    barcodeImg = gray[160:220, 950:2000]
    binary = cv2.adaptiveThreshold(barcodeImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, 2)

    # Decode the barcode
    barcodes = pyzbar.decode(binary)
    barcodeData = ''
    for barcode in barcodes:
        barcodeData = barcode.data.decode("utf-8")

    # Extract grade markings
    noteImg = img[0:220, 600:2000]
    notesArry = []
    coords = [
        (120, 160, 15, 40), (120, 160, 50, 75), (120, 160, 83, 108), (120, 160, 117, 142), (120, 160, 152, 175),
        (160, 200, 15, 40), (160, 200, 50, 75), (160, 200, 83, 108), (160, 200, 117, 142), (160, 200, 152, 175),
        (160, 200, 272, 293), (160, 200, 303, 325)
    ]

    for y1, y2, x1, x2 in coords:
        note = noteImg[y1:y2, x1:x2]
        notesArry.append(note)

    # Check if each grade is marked
    grades = []
    for item in notesArry:
        gray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        black_pixels = np.sum(binary == 255)
        total_pixels = item.shape[0] * item.shape[1]
        percentage = black_pixels / total_pixels

        if percentage > 0.001:
            grades.append(True)
        else:
            grades.append(False)

    # Calculate the final grade based on the marked grades
    final_grade = 0
    for i, marked in enumerate(grades):
        if marked:
            final_grade += i + 1

    # Return an object with the final grade and the barcode data
    result = {
        'final_grade': final_grade,
        'barcode_data': barcodeData
    }
    return result


def process_exam_barcode(img_Path):
    # Decode the image buffer
    img = cv2.imread(img_Path)
    # use Pyzbar to decode the barcode
    barcodes = pyzbar.decode(img)
    barcodeData = ''
    for barcode in barcodes:
        barcodeData = barcode.data.decode("utf-8")

    # process the barcode data we have two ids in the barcode seperated by a _
    # the first id is the student id and the second is the exam id
    ids = barcodeData.split('_')
    student_id = ids[0]
    exam_id = ids[1]
    paperSize = ids[2]
    examType = ids[3]
    # return the student id and the exam id
    return student_id, exam_id, paperSize, examType


def CompareResults(results, correction):
    # compare the results with the correction
    # return the number of correct answers
    correct = 0
    for i in range(len(results)):
        if results[i] == correction[i]:
            correct += 1
    return correct


def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False

    for key, value in dict1.items():
        if key not in dict2 or dict2[key] != value:
            return False

    return True


def calculate_grade(correction, answers):
    total_questions = len(correction)
    correct_answers = 0

    for key in correction:
        if correction[key] == answers[int(key)]:
            correct_answers += 1

    grade = (correct_answers / total_questions) * 20
    return grade
