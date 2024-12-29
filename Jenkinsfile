pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                // Clone the repository
                checkout scm
            }
        }
        stage('Set Permissions') {
            steps {
                // Set execute permission
                sh 'chmod +x list_files.sh'
            }
        }
        stage('Execute Script') {
            steps {
                // Execute the bash script
                sh './list_files.sh'
            }
        }
    }
}
