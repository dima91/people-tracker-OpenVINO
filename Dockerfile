
FROM dockerdima/clea_ai_base

WORKDIR /root/ws/people-tracker-OpenVINO
ADD demo_runner.sh .

ENTRYPOINT ["bash", "demo_runner.sh"]
CMD [ "GPU" ]