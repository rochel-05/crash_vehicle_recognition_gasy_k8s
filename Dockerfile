FROM python:3.7

RUN mkdir /staticsiteflask6
WORKDIR /staticsiteflask6
COPY usersDb.db /staticsiteflask6
COPY test_accuracy_RF.py /staticsiteflask6
COPY table.py /staticsiteflask6
COPY run.py /staticsiteflask6
COPY requirements.txt /staticsiteflask6
COPY model_RF.py /staticsiteflask6
COPY invetigate_RF.py /staticsiteflask6
COPY extract_frames_from_video.py /staticsiteflask6
COPY dummy.py /staticsiteflask6
COPY Dockerfile /staticsiteflask6
COPY data /staticsiteflask6/data
COPY generated_frames_test /staticsiteflask6/generated_frames_test
COPY generated_frames_train /staticsiteflask6/generated_frames_train
COPY generated_frames_train_raw /staticsiteflask6/generated_frames_train_raw
COPY model /staticsiteflask6/model
COPY static /staticsiteflask6/static
COPY templates /staticsiteflask6/templates
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "/staticsiteflask6/run.py"]
