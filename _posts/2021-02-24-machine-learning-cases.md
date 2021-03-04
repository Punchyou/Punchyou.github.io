---
layout: post
title: Machine Learning Case Studies
author: Maria Pantsiou
date: '2021-02-24 14:35:23 +0530'
category: AI
summary: Machine Learning Case Studies
thumbnail: blue_light_purple_cude.png

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

At some point, we might want to see which is the optimal number of components that explain most of the variance in the data. For example, to calculate the explained variance for the top 5 components, calculate s squared for *each* of the top 5 components, add those up and normalize by the sum of *all* squared s values, according to this formula:

\begin{equation*}
\frac{\sum_{5}^{ } s_n^2}{\sum s^2}
\end{equation*}

The corresponding python function is the following:
```py
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components  ## 33-3 = 30, for example
    # calculate approx variance
    exp_variance = np.square(s.iloc[start_idx:,:]).sum()/np.square(s).sum()
    
    return exp_variance[0]
```

Now do some trial and error with different number of components to find the higher percentage of variance explaining:
```py
# test cell
n_top_components = 7 # select a value for the number of top components

# calculate the explained variance
exp_variance = explained_variance(s, n_top_components)
print('Explained variance: ', exp_variance)
```
We can now examine the makeup of each PCA component based on the weightings of the original features that are included in the component. Could be used in a feature selectino process:

```py
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()
```
 
Keep in mind that some of the data in certain categories (columns) for each component will be left out.

You can now deploy and predict with `model.deploy()` and `model.predict()`

We would also need to transform the data in order to use k-means, based on the components we have for each feature. To do that, use the following function:

```py
# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    # create new dataframe to add data to
    counties_transformed=pd.DataFrame()

    # for each of our new, transformed data points
    # append the component values to the dataframe
    for data in train_pca:
        # get component values for each data point
        components=data.label['projection'].float32_tensor.values
        counties_transformed=counties_transformed.append([list(components)])

    # index by county, just like counties_scaled
    counties_transformed.index=counties_scaled.index

    # keep only the top n components
    start_idx = N_COMPONENTS - n_top_components
    counties_transformed = counties_transformed.iloc[:,start_idx:]
    
    # reverse columns, component order     
    return counties_transformed.iloc[:, ::-1]
```
After your dataset is created, make sure you delete all endpoints. We are now ready to apply our dataset to k-means. k-means deployment is very similar to the pca model deployment above. Find more on how to deploy a k-means model with SageMaker in the notebook mentioned above.

# SageMeker as a Tool & The Future of ML
## Deploying Custom Models
## Time-series Forecasting

For linear models, we can use SageMaker's `LinerLearner`. We can always calculate the precision, recal and total accuracy to evaluate our models. En example of a linear regression model with SageMaker is presented [here](https://github.com/Punchyou/ML_SageMaker_Studies/blob/master/Payment_Fraud_Detection/Fraud_Detection_Solution.ipynb).

### DeepAR
A SageMeker build-in RNN model for predictive modeling for timeseries. We will use energy consumption data from kaggle to use tha model. The dataset used can be found [here](wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/March/5c88a3f1_household-electric-power-consumption/household-electric-power-consumption.zip
). The data needs to be preprocessed.

DeepAR expect JSON (dict) as data input, with the fields *start* (string in datetime index format that defines the starting date and time), *target* and *cat* (encoded €categorical). We need to upload the data to S3, as we did before.

for the training, first we create an image, which we will eventually pass into the estimator:
```py
from sagemaker.amazon.amazon_estimator import get_image_uri

image_name = get_image_uri(boto3.Session().region_name, # get the region
                           'forecasting-deepar') # specify image
```

After training, we create a `predictor` in a similar way. The response is JSON of predictions, organized in quantiles (0.1, 0.5 (mean), 0.9). We will need to decode the JSON data.
Find full code example [here](https://github.com/Punchyou/ML_SageMaker_Studies/blob/master/Time_Series_Forecasting/Energy_Consumption_Solution.ipynb).

## Project: Plagiarism Detector

Sources:
1. [ML Case Studies repo](https://github.com/Punchyou/ML_SageMaker_Studies/)

Check more on SageMaker [here](https://github.com/aws/amazon-sagemaker-examples).
