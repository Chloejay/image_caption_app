# detection_api

## 
this repo is used to build an end to end computer vision app. I did object detection model and use AWS SageMaker as endpoint to test image sets data
stored in AWS S3 bucket, can see example from ipython file. This time, I will use Flask as a frontend tool and MLflow to log training versions.
For this demo I used already trained model which I trained one year ago. So the purpose is not get the highest accuracy of model.

## automate workflow
- Makefile 

## Setup 
```
pip install -r requirements.txt

setup.py 
pip install -e . 
```

## CI/CD
```
#Use Github Actions.
mkdir -p .github/workflows
cd .github/workflows
touch cml.yaml
```

#### Use git branch to experiment 
```
git checkout -b experiment_v1
git add .
git commit -m""
git push origin experiment_v1
```
## LICENSE 
```
#auto generate LICENSE doc
wget -c https://www.gnu.org/licenses/gpl-3.0.txt -O LICENSE
```

## Resource

