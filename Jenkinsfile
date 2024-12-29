pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                // Clone the repository
                checkout scm
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
