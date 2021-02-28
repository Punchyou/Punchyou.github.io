---
layout: post
title: Machine Learning Case Studies
author: Maria Pantsiou
date: '2021-02-08 14:35:23 +0530'
category: AI
summary: Machine Learning Case Studies
thumbnail: blue_light_cude2.png

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

# Machine Learning Case Studies
## Population Segmentation
Thie is a classi clustering problem, that can be completed with  methods like k-means (for clustering) and PCA (for dimentionality reduction, before clustering). To learn more abou k-means, you can check the articles [k-means Clustering](https://punchyou.github.io/datascience/2020/06/22/kmeans-analysis-notes/#/) and [k-means vs DBSCAN](https://punchyou.github.io/datascience/2020/06/25/keams-vs-dbscan/#/).

### Dimentionality Reduction with PCA exqmple
If we have too many dimentions in our data, we would probably need to  perform some kind od dimentinoality reduction. We can find the first x principle components, depends on how many dimentions we want to end up with, that explain the (most part of the) variance of the data. Then we can project all the datapoints on that surface.

The following are examples derived by the notebook you can find [here](https://github.com/Punchyou/ML_SageMaker_Studies/blob/master/Population_Segmentation/Pop_Segmentation_Solution.ipynb).

To use data already stored in AWS (S3 bucket) we can use:
```py
s3_client = boto3.client('s3')
bucket_name = 'aws-bucket-name'

# list objects in that bucket
obj_list = s3_client.list_objects(Bucket=bucket_name)

# print all objects
files = []
for contents in obj_list['Contents']:
	files.append(contents['Key'])

# I want to get the first file
files_name = files[0]
data_object = s3_client.get_object(Bucket=bucket_name, Key=file_name)
data_object # Assume 'Body' is a stream, and file format is csv

# the streaming data is what we want from that object
data_body = data_object["Body"].read()
type(data_body) # the data is bytes

# read the bytes
data_stream = io.BytesIO(data_body)

# create a dataframe from the stream with read_csv
df = pd.read_csv(data_stream)
```

**TIP**: Turning data to *RecordSet* format allows models to perform really fast, and it's a requirement for all SageMaker build-in models. To do the conversion, do the following:

```py
# convert the data to a numpy array
train_data_np = df.to_numpy(dtype='float32')

# use a PCA model example
formatted_data = pca_model.record_set(train_data_np)

```

The model data are saved as a `tar` file, so we need to get that file back from the S3 location stored, and unzip it:

```py
model_key = 'path_to_the_model_output'

# download the model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzip and rename
os.system('tar -zxvf model.tar.gz')
os.system('unzip my_model')

# load model as ndarray
import mxnet as mx # build-in sagemaker pkg
model_params = mx.ndarray.load('my_model')
model.parameters
```


# Payment Fraud Detection
## SageMeker as a Tool & The Future of ML
## Deploying Custom Models
## Time-series Forecasting
## Project: Plagiarism Detector
