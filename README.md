# detection_api

This repo is used to build an end to end computer vision app. I did object detection model and use AWS SageMaker as endpoint(URI) to test image sets data stored in AWS S3 bucket, can see example from ipython file. This time, I will use Flask as a frontend tool and MLflow to log training versions.

For this demo I originally planned to use already trained model which I trained one year ago, so the purpose is not get the highest accuracy of model. However, the latest trained tensorflow model is not saved on my local machine, so I decide to rebuild an interesting model, related with CV. Base model options are: 
- mmdetection 
- Mask - RCNN 

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
[MIT](https://opensource.org/licenses/MIT)
<s>#auto generate LICENSE doc</s>
<s>wget -c https://www.gnu.org/licenses/gpl-3.0.txt -O LICENSE</s>


## Resource

