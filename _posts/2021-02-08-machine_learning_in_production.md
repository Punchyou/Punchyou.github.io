---
layout: post
title: Machine Learning in Production
author: Maria Pantsiou
date: '2021-02-08 14:35:23 +0530'
category: AI
summary: Machine Learning in Production
thumbnail: light_circles2.png

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            displayMath: [['$$','$$']],
            inlineMath: [['$','$']],
        },
    });
</script>

# Machine Learning in Production


## Endpoints and REST APIs

The interface (endpoint) facilitates an ease of communication between the model and the applicationion

One way to think of the endpoint that acts as this interface, is to think of a Python program where:

- The endpoint itself is like a function call
- The function itself would be the model and
- The Python program is the application.


```py
""" The whole script is the application"""
def main():
	input_user_data = get_user_data()

	# this is the interface
	predictions = ml_model(input_user_data)


def ml_model(data):
	""" This is the model"""
	loaded_data = load_user_data(data)
```

### REST API
Communication between the application and the model is done through the endpoint (interface), where the endpoint is an Application Programming Interface (API). The **REST API** is one that uses HTTP requests and responses to enable communication between the application and the model through the endpoint (interface).

- Endpoint

This endpoint will be in the form of a URL, Uniform Resource Locator, which is commonly known as a web address.

### HTTP Methods:
- GET: READ request action (retrieve information - of found is returned)
- POST: CREATE request action (create new info - once is created it's returned as a reposnse)
- PUT: UPDATE request action (there is also PATCH for partial update)
- DELETE: DELETE request action

The HTTP response sent from your model to your application is composed of three parts:

- HTTP Status Code

If the model successfully received and processed the user’s data that was sent in the message, the status code should start with a 2, like 200.

- HTTP Headers

The headers will contain additional information, like the format of the data within the message, that’s passed to the receiving program.

- Message (Data or Body)
What’s returned as the data within the message is the prediction that’s provided by the model. The user Data might need to be formatted (csv or json). The HTTP response message might need translations to be readable for the application's user (like csv or json).

The HTTP request that’s sent from your application to your model uses a POST HTTP Method.

## Containers

The model and the application can each be run in a container computing environment. The containers are created using a script that contains instructions on which software packages, libraries, and other computing attributes are needed in order to run a software application, in our case either the model or the application. 

A container can be thought of as a standardized collection/bundle of software that is to be used for the specific purpose of running an application. 

Three continer running three different applications:
![container](../assets/img/container.png)

Example of a [dockerfile](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile), a file of instructions about how a container ccan be created.

Notes on containers:
- Containers are sharing the kernel with underlying OS, like VMs. They are though lighter.
- In order to install new software, we can update the build script for the container and run them again.
- With container we use microservices and we brek down things in smaller and managable units.

## Production environment
Deployment to production can simply be thought of as a method that integrates a machine learning model into an existing production environment so that the model can be used to make decisions or predictions based upon data input into this model. A production evironment can be a mobile or web application among others.

In machine learning, a **hyperparameter** is a parameter whose value cannot be estimated from the data. It is not directly learned through the estimators; therefore, their value must be set by the model developer. Often cloud platform machine learning services provide methods that allow for automatic hyperparameter tuning for use with model training.

### Model Deployment Characteristics
- Model Versioning
- Model Monitoring
- Model Updating (from change in the data) and Routing (the deployment platform should support routing differing proportions of user requests to the deployed models; to allow comparison of performance between the deployed model variants.

Routing in this way allows for a test of a model performance as compared to other model variants.)
- Model predictions: On-demand (real-time, as responses from a request, needs to have low latency) or Batch prediction (asynchronous, like high volume of requests with periodic submitions)


### SageMaker
- There are at least fifteen built-in algorithms that are easily used within SageMaker. Contains algorithms like linear learner or XGBoost, item recommendations using factorization machine, grouping based upon attributes using K-Means and more. You can also create custome algorithms, using popular ML algorithms.
- Make use of jupyter notebooks
- Has tuning tools and monitoring
- You must leave it running to provide predictions, so it's faster.

Other Notes
-  TensorFlow can be used for creating, training, and deploying machine learning and deep learning models. Keras is a higher level API written in Python that runs on top of TensorFlow, that's easier to use and allows for faster development. 
-  Scikit-learn and an XGBoost Python package can be used together for creating, training, and deploying machine learning models. 


## AWS Quotas and commands

### List your quotas, type pn AWL CLI:

`list-service-quotas` and `list-aws-default-service-quotas`

### Increase your quotas 
- Use amazon Service Quotas service. This service consolidates your account-specific values for quotas across all AWS services for improved manageability. Service Quotas is available at no additional charge. You can directly try logging into Service Quotas console here.
- Using AWS Support Center - You can create a case for support from AWS.
   command: `request-service-quota-increase`
- **Amazon SageMaker ML Instance Types**:characterized by a combination of CPU, memory, GPU, GPU memory, and networking capacity.
- **Shut Down SageMaker Instances, if not in use**. You can re-install it later.
- Read about the limits [here](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html).

## Set up Instances
### Notebooks
- Amazon Saze Maker &rarr; Notebook Instances &rarr; Creat Notebook Instance
- Give the notebook a name, and a role (like a security acess level) under **Permissions and Encryption** &rarr; Create a new role &rarr; S3 buckets you specify: *None* &rarr; Create Notebook Instance
- Once the notebook instance is ready, amazon will automatically start it &rarr; **Stop it if you won't use it right away** - costs increases by the time

#### Notebooks with SageMaker
- When training in notebooks, sagemaker will create  VM with the characteristics we chose, then the VM will load an image in the form of a docker container, which contined the code to use XGBoost.
- The VM needs access to the data, which should be available in S3 (amazon data storage facilities), so need to upload our datasets there (by saving the dataset from the script). Then we upload by using sagemaker package:
- Sagemaker needs to be installed (can be done within a notebook with `!pip`)

```py
import sagemaker
session = sagemaker.Session()
session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
```
This uploads the file to the S3 bucket associated with this session.

- You can also print your role (defines how data that your notebook uses/creates will be stotred)
`print(role)`

- Different containers are created for different regions and models we choose to use. Amazon provides us with a function that can print the container uri, if we pass the region and the model as arguments:

```py
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(session.boto_region_name, 'xgboost')
```
This will return something like:

```
'811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:1'
```
Then we can train our model:
```py
# we create an estimator
# we want the estimator to have the appropriate role to access the data
xgb = sagemaker.estimator.Estimator(container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

```
Then we can set the hyper parameters by:
```py
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10, # in case the model is overfitting and starts to make it worse in the validation set - we should sytop the model then
                        num_round=200)
```
Learn more about estimators [here](https://sagemaker.readthedocs.io/en/latest/estimators.html).
Then we can fit the model, by providing the location of the training/validation datasets:
```py
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```
Lastly we fit the model. We send all the data to sagemaker, and it will decide how to split the data into train/val/test sets.

```py
# create the transformer
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
# start the transform job
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
# check how the transformation jos is progressing
xgb_transformer.wait()
```
After the training is complete, we need to bring the output to our notebook. We'll use AWS's functionality to do that (still inside the notebook):

```
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

At the end we should remove the directory with the data (saved in `data_dir`), to free up space for the rest of our prpojects:

```
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```
## Launch a project
An example is [here](https://github.com/udacity/sagemaker-deployment).  


### Modeling on SageMaker
Notes:

- XGBoost is a tree based method, so prone to overfitting. Having a validation set for those kind of models can improve results.
- After uploading data in S3, the files can be found here:
AWS &rarr; S3 &rarr; Choose bucket. There should be the folder with the datasets uploaded.uploaded
- A sagemaker model is a collection of information that includes both a link to the model artifacts (saved files created by the training job, if we were fitting a linear model then the coefficients that were fit would be saved as model artifacts) and some information in how those srtifacts should be used.
We can also add all the nessesary params in a dictionary and unpack them in the job func:
- We can create training jobs. After a trining job is completed, we can build a sagemaker model.
- After the model is complete, we can test it, by using transform (like before).

Training job:
```py
training_job = sessions.sagemaker_client.create_training_job(**training_params)
# to check the logs of the session
session.logs_for_job(training_job_name)
```

Testing job:
```py
# use the transform_request dict with all the informtion about the continer, VM and model
transform_response = session.agemaker_client.create_transform_job(**transform_request)
# check the logs - transform name to be unique, str
transform_desk = ession.wait_for_transform_job(transform_job_name)
```
#### What happend when a model is fit using SageMaker?
When a model is fit using SageMaker, the process is as follows.

First, a compute instance (basically a server somewhere) is started up with the properties that we specified.

Next, when the compute instance is ready, the code, in the form of a container, that is used to fit the model is loaded and executed. When this code is executed, it is provided access to the training (and possibly validation) data stored on S3.

Once the compute instance has finished fitting the model, the resulting model artifacts are stored on S3 and the compute instance is shut down.

You can check the jobs in the AWS console: Underneath the Notebook Instances in AWS Sagemaker, click om Training Jobs. Click on a training job, will reaveal information about the job.There, theres is also a `View logs` option.

## Deploy a model with SageMaker
Instead of testing the model with the transform method, we will deploy our model and then send the test data to the endpoint.
Deployment means creating an endpoint, which is just a URL that we can send data to.
The only thing we need to do to deploy it in an application, is to just call `deploy` in our existing model (after training):
```py
# we need to pass the number of VMs we want to use, and the type of the VM
xgb_preedictor = xgb.deployt(initial_instance_count=1, instance_type='ml.m4.xlarge')
```
SageMaker has now created a VM that has ran the trained model, that it now accessed by an endpoint (URL).

In order to send data to the model's endpoint, we can use the `predict` method of the xgd_predictor object that returned above. The data the we provide to the predictor object need to serialized first. We can do that with the sagemaker's `csv_serializer`. the returned results is also serialized (a string), so we'll need to convert it back to numpy.array:

```py
# We need to tell the endpoint what format the data we are sending is in
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
# predictions is currently a comma delimited string and so we would like to break it up
# as a numpy array.
Y_pred = np.fromstring(Y_pred, sep=',')
```

Lastly we need to shut down the deployed model, as it  will run, waiting for data to be send to it:

```py
xgb_predictor.delete_endpoint()
```

To do the above in mode detail (configure the endpoint), go to the low-level deployment tutorial of the course.
Endpoit configuration objects are also available in **SageMaker menu** &rarr; **Endpoint configurations**. After its creation, we can use the same configuration as many times as we want.

Once the endpoint is created, we can see it Under **SageMaker menu** &rarr; **Endpoints**, where we can see the URL, or delete our endpoint. Then we can test the model (send the data to endpoint)
Notes
- SageMaker gets the data to the endpoint, and if we want it to, split the data amongst different models.
- In case you need to aplit the data before feeding them into the model, use :
```py
# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    
    return np.fromstring(predictions[1:], sep=',')
```

# Hyperparameters Tuning

- SageMaker can run a bunch of models and choose the best one. We have to define how many models to runand the best model method (like rmse).
- To check the output of the model, check **Cloud Watch**. (logs I mentioned earlier)

Wee need a model as we have done so far, and we set the hyper parameters like before. This will be our basline model:

```py
# set a baseline object
xgb.set_hyper_parameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
			num_round=200)
```

Then we create another object that will tune the parameters above and we fit the tuner:

```py
xgb_hyperparameter_tuner = HyperparameterTuner(estimator = xgb, # The estimator object to use as the basis for the training jobs.
                                               objective_metric_name = 'validation:rmse', # The metric used to compare trained models.
                                               objective_type = 'Minimize', # Whether we wish to minimize or maximize the metric.
                                               max_jobs = 20, # The total number of models to train
                                               max_parallel_jobs = 3, # The number of models to train in parallel
                                               hyperparameter_ranges = {
                                                    'max_depth': IntegerParameter(3, 12),
                                                    'eta'      : ContinuousParameter(0.05, 0.5),
                                                    'min_child_weight': IntegerParameter(2, 8),
                                                    'subsample': ContinuousParameter(0.5, 0.9),
                                                    'gamma': ContinuousParameter(0, 10),
                                               })
# load data tp s3
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')
# fit the tuner
xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
```
 To check the nest model:
```py
xgb_hyperparameter_tuner.best_training_job()
```

At the moment we have trained the tuner, but not the initial estimator. In order to do that, we can attach the tuner to the estimator. We can choose the best training model:
```py
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())
```

Then we can test our best model.

If we want to use the API, we can do it by defining more details option for the container and the model, the only difference will be that instead of setting up the parameters of the baseline mode, we will only refer to the ones that will remain static, and later one we can define the ranges for the parameters that will vary.


## Deploy a Sentiment Analysis Model
### Bag of words
In the project, we'll use the bag-of-words approach to determinte if a movie feedback is positive or negative.

- Turn each document (or a sentense) into a vector of numbers:
![bag of words vector](../assets/img/bag_of_words.png)
We can compare two documents based on how many words they have in common. The mathematical repressentation is:

$$a\cdotb = \Sigma(a_0 b_0i + a_1 b_1 + ... + a_n b_n) = 1$$

The greater the product, the mode similar the documents. The issue is that different pairs my end up having the same product. In that case, we the *cosine similarity*:

$$cos(θ) = \frac{a\cdot b}{|a|\cdot|b|}$$

## Web app for the model to be deployed

Issues you need to overcome:
- The endpoind takes encoded data as input, but the user input on the web app is a string.
- Security - endpoints provided by sagemaker have limited access.
