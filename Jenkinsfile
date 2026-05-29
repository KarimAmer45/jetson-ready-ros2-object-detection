pipeline {
  agent any
  options {
    timestamps()
  }
  environment {
    IMAGE_NAME = 'ros2-object-detection'
  }
  stages {
    stage('Python sanity') {
      steps {
        sh 'python3 -m compileall object_detection_ros2 tests'
      }
    }
    stage('Unit tests') {
      steps {
        sh 'python3 -m pytest -q tests'
      }
    }
    stage('Docker build') {
      when {
        expression { fileExists('Dockerfile') }
      }
      steps {
        sh 'docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .'
      }
    }
  }
}