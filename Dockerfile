FROM python:3.7

RUN mkdir /staticsiteflask5
WORKDIR /staticsiteflask5
COPY usersDb.db /staticsiteflask5
COPY test_accuracy_RF.py /staticsiteflask5
COPY table.py /staticsiteflask5
COPY run.py /staticsiteflask5
COPY requirements.txt /staticsiteflask5
COPY model_RF.py /staticsiteflask5
COPY invetigate_RF.py /staticsiteflask5
COPY extract_frames_from_video.py /staticsiteflask5
COPY dummy.py /staticsiteflask5
COPY Dockerfile /staticsiteflask5
COPY data /staticsiteflask5/data
COPY generated_frames_test /staticsiteflask5/generated_frames_test
COPY generated_frames_train /staticsiteflask5/generated_frames_train
COPY generated_frames_train_raw /staticsiteflask5/generated_frames_train_raw
COPY model /staticsiteflask5/model
COPY static /staticsiteflask5/static
COPY templates /staticsiteflask5/templates
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "/staticsiteflask5/run.py"]