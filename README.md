# Image Classification using AWS SageMaker
This project uses the dog breed classification data set to train a pretrained model for image classification called vgg19 using AWS Sagemaker profiling, debugger, hyperparameter tuning, etc.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
[dogbreed classification dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).


### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I choose the VGG19 fro this experiment since it is an advanced CNN with pre-trained layers and a great understanding of what defines an image in terms of shape, color, and structure. 

**1. Screenshot of completed training jobs **![hpo.png](https://d-ehlicnlmgqny.studio.us-east-1.sagemaker.aws/jupyter/default/files/soothsayerworkflow/ref2/project/image_classification_project/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/hpo.png?_xsrf=2%7Cf17ea0a5%7C59b349f0f19a4676c2f89ae3150c860f%7C1664774910)
**2. Logs metrics during the training process**

**3. Tune at least two hyperparameters**
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "train_batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
**4. Retrieve the best best hyperparameters from all your training jobs**
{'_tuning_objective_metric': '"average test loss"',
 'lr': '0.004277615192677857',
 'momentum': '0.9',
 'num_classes': '133',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"pytorch-training-2022-10-21-10-01-17-755"',
 'sagemaker_program': '"hpo.py"',
 'sagemaker_region': '"us-east-1"',
 'sagemaker_submit_directory': '"s3://sagemaker-deployment-project/model-artifacts/pytorch-training-2022-10-21-10-01-17-755/source/sourcedir.tar.gz"',
 'test_batch_size': '128',
 'train_batch_size': '"512"'}

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
