# mrqa-serving

## Docker setting

- Docker hub: adieujw/mrqa

  - How to build
  ```
  $ docker login
  $ cd docker
  $ docker build -t mrqa .
  $ docker tag mrqa adieujw/mrqa:v1
  $ docker push adieujw/mrqa:v1
  ```

  - 용량은 대략 2GB