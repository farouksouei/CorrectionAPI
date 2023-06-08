from fastapi import FastAPI
from requests import post
from utils import download_pdf_from_url, pdf_buffer_to_images, process_exam, download_pdf, process_exam2, \
    process_exam_barcode, CompareResults, compare_dicts, calculate_grade
import json

app = FastAPI()


@app.get("/extractAnswers/{tempExamId}/")
async def root(tempExamId: str):
    # get the url from the request
    print(tempExamId)
    url = f"http://localhost:5003/api/smartExam/TemporaryExamUploads/getExamUrlById/"
    url2 = f"http://localhost:5003/api/smartExam/exam/correct_responses_by_id/"
    url3 = f"http://localhost:5003/api/smartExam/grades/updateGrade/"
    payload = {"id": tempExamId}
    response = post(url, json=payload)
    print(response.json())
    pdf_url = response.json()
    # download the pdf
    save_path = "Temp/output.pdf"
    download_pdf(pdf_url, save_path)
    # divide the pdf into pages
    img_paths = pdf_buffer_to_images(save_path)
    # process each page
    results = []
    for img_path in img_paths:
        studentId, examId = process_exam_barcode(img_path)
        print(examId)
        print(studentId)
        # fetch the correct answers from the database
        payload2 = {"id": examId}
        response2 = post(url2, json=payload2)
        correctAnswers = response2.json()
        examAnswers = process_exam(img_path)
        # compare the results
        correctAnswers = correctAnswers['data'][0]
        print(calculate_grade(correctAnswers, examAnswers))
        payload3 = {"examId": examId, "studentId": studentId, "grade": calculate_grade(correctAnswers, examAnswers)}
        response3 = post(url3, json=payload3)
        print(response3.json())
    return {"json_results"}


@app.get("/extractCorrection/{ExamId}")
async def say_hello(ExamId: str):
    # get the url from the request
    print(ExamId)
    url = f"http://localhost:5003/api/smartExam/exam/getCorrectedfile/"
    payload = {"id": ExamId}
    response = post(url, json=payload)
    responseJson = response.json()
    pdf_url = responseJson['data'][0]['link']
    print(pdf_url)
    # download the pdf
    save_path = "Temp/TempExamCorrection/output.pdf"
    download_pdf(pdf_url, save_path)
    # divide the pdf into pages
    img_paths = pdf_buffer_to_images(save_path)
    # process each page
    results = []
    for img_path in img_paths:
        results.append(process_exam(img_path))
    print(results)
    # return the results
    json_results = json.dumps(results)
    print(json_results)
    return {json_results}
