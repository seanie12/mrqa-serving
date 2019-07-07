# mrqa-serving

## Docker setting

- Docker hub: adieujw/mrqa

  - How to build
  ```
  $ docker login
  $ cd docker
  $ docker build -t adieujw/mrqa:latest .
  $ docker push adieujw/mrqa:latest
  ```

  - 용량은 대략 2GB

## How to do

1. Weight 파일을 config/save에 저장
    - src/serve.py에서 자동으로 weight file name 파악하도록 세팅함(os.listdir('./config/save/')[0])
    - 단! 반드시 save 폴더 안에는 파일이 하나만 있어야 함

2. 가상환경 python2.7로 만들기(virtualenv나 conda)
    - codalab이 python2에서만 돌아간다.
    ```
    $ virtualenv -p python2.7 venv
    $ source venv/bin/activate
    $ pip install -U codalab
    ```

3. Data upload (upload_data.sh)
    - 우선 Worksheet를 변경해야 함
        - CMD: cl work <worksheet_name>
        - upload_data.sh의 상단에 자기가 사용할 worksheet명 반드시 바꾸기
    - mrqa2019에서 기본적으로 제공한 3개의 파일 업로드하기(내 local에서 올리는 게 아님)
        1. mrqa_dev_data
        2. predict_server.py
        3. mrqa_official_eval.py
        - **문제는 이 python 파일들이 allennlp를 사용하고 있기에, docker파일에서도 반드시 allennlp를 깔아놔야 한다.**
    - 그 후 src와 config 폴더 올리기
        - Web CLI에서 업로드할 시에는 시간이 오래 걸려서 local에 codalab 파이썬 패키지를 설치해서 local CLI를 사용하는 것임.

4. dev set에서 prediction 진행(run-predictions.sh)
    - **만일 3번 단계를 건너뛰고 4번으로 올 경우에는 반드시 cl work <worksheet_name>을 해줘야 함**
    - Docker: adieujw/mrqa:latest
        - 해당 도커파일은 기본적으로 codalab/default-gpu를 가져와서 내용 추가함
        - https://github.com/codalab/codalab-worksheets/blob/master/docker/dockerfiles/Dockerfile.default-gpu
    - 이 bundle의 이름은 run-predictions로 설정함
    - --request-gpus 1 옵션 추가
    - 이 작업이 끝나면 run-predictions 번들 안에 prediction.json이 생겨남

5. submission용 bundle만들기(create-submission-bundle.sh)
    - sh 파일 상단에, 내가 정하고 싶은 모델명을 지정해주기
        - **모델명에 whitespace나 special character 들어가지 않도록 주의!**
    - 쉘 파일을 돌리면 cl make run-predictions/predictions.json -n predictions-{MODELNAME}이 돌아감

6. Submission(submit.sh)
    1. web에서 codalab worksheet로 가서 **predictions-{MODELNAME}** 번들의 Description 변경하기
        - 이거 웹에 가서 직접해야 함.
        - {Name_of_model} (Institution) 의 형태로 description 변경해야 함.
        - e.g. BiDAF + Self Attention + ELMo (Allen Institute for Artificial Intelligence [modified by Stanford])
    2. predictions-{MODELNAME} 번들에 **mrqa2019-submit** flag를 달아주기
        - submit.sh 돌리기 전에 MODEL_NAME 인자 바꿔주기
        - CMD: cl edit predictions-{MODELNAME} --tags mrqa2019-submit
    3. 해당 번들의 링크를 이메일 보내기
        - Send a short email to mrforqa@gmail.com with a link to your predictions-{MODELNAME} bundle, not to the worksheet!. We will verify that your submission is in the appropriate format.