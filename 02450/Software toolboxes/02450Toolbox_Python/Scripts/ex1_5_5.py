## exercise 1.5.5
import numpy as np
import matplotlib.pyplot as plt

# In this exercise we will rely on pandas for some of the processing steps:
import pandas as pd

# We start by defining the path to the file that we're we need to load.
# Upon inspection, we saw that the messy_data.data was infact a file in the
# format of a CSV-file with a ".data" extention instead.  
file_path = '../Data/messy_data/messy_data.data'
# First of we simply read the file in using readtable, however, we need to
# tell the function that the file is tab-seperated. We also need to specify
# that the header is in the second row:
messy_data = pd.read_csv(file_path, sep='\t', header=1)

# We also need to remove the added header line in the .data file which seems
# to have included a shortend form the variables (check messy_data.head()):
messy_data = messy_data.drop(index=0) 

# We extract the attribute names:
attributeNames = np.asarray(messy_data.columns)

# As we progress through this script, we might change which attributes are
# stored where. For simplicity in presenting the processing steps, we wont
# keep track of those changes in attributeNames in this example script.

# The last column is a unique string for each observation defining the
# car make and model. We decide to extract this in a variable for itself
# for now, and then remove it from messy_data:
car_names = np.array(messy_data.carname)
messy_data = messy_data.drop(['carname'], axis=1)

# Inspect messy data by e.g.:
#print(messy_data.to_string())

# At this point, youll see that some of the missing values from the data
# has already been represented as NaNs (in the displacement column). 
# However, these were only the places where an empty element was in the file.
# First off, we remove the question marks in displacement and replace
# them with not a number, NaN:
messy_data.displacement = messy_data.displacement.str.replace('?','NaN')

# Similarly, we remove the formatting for a thousand seperator that is
# present for the weight attribute:
messy_data.weight = messy_data.weight.str.replace("'", '')
# And lastly, we replace all the commas that were used as decimal seperatos
# in the accceleration attribute with dots:
messy_data.acceleration = messy_data.acceleration.str.replace(",", '.')

# the data has some zero values that the README.txt tolds us were missing
# values - this was specifically for the attributes mpg and displacement,
# so we're careful only to replace the zeros in these attributes, since a
# zero might be correct for some other variables:
messy_data.mpg = messy_data.mpg.replace({'0': np.nan})
messy_data.displacement = messy_data.displacement.replace({'0': np.nan})

# We later on find out that a value of 99 for the mpg is not value that is
# within reason for the MPG of the cars in this dataset. The observations
# that has this value of MPG is therefore incorrect, and we should treat
# the value as missing. How would you add a line of code to this data
# cleanup script to account for this information?

## X,y-format
# If the modelling problem of interest was a classification problem where
# we wanted to classify the origin attribute, we could now identify obtain
# the data in the X,y-format as so:
data = np.array(messy_data.values, dtype=np.float64)
X_c = data[:, :-1].copy()
y_c = data[:, -1].copy()

# However, if the problem of interest was to model the MPG based on the
# other attributes (a regression problem), then the X,y-format is
# obtained as:
X_r = data[:, 1:].copy()
y_r = data[:, 0].copy()

# Since origin is categorical variable, we can do as in previos exercises
# and do a one-out-of-K encoding:
origin = np.array(X_r[:, -1], dtype=int).T-1
K = origin.max()+1
origin_encoding = np.zeros((origin.size, K))
origin_encoding[np.arange(origin.size), origin] = 1
X_r = np.concatenate((X_r[:, :-1], origin_encoding),axis=1)
# Since the README.txt doesn't supply a lot of information about what the
# levels in the origin variable mean, you'd have to either make an educated
# guess based on the values in the context, or preferably obtain the
# information from any papers that might be references in the README.
# In this case, you can inspect origin and car_names, to see that (north)
# american makes are all value 0 (try looking at car_names[origin == 0],
# whereas origin value 1 is European, and value 2 is Asian.

## Missing values
# In the above X,y-matrices, we still have the missing values. In the
# following we will go through how you could go about handling the missing
# values before making your X,y-matrices as above.

# Once we have identified all the missing data, we have to handle it
# some way. Various apporaches can be used, but it is important
# to keep it mind to never do any of them blindly. Keep a record of what
# you do, and consider/discuss how it might affect your modelling.

# The simplest way of handling missing values is to drop any records 
# that display them, we do this by first determining where there are
# missing values:
missing_idx = np.isnan(data)
# Observations with missing values have a row-sum in missing_idx
# which is greater than zero:
obs_w_missing = np.sum(missing_idx, 1) > 0
data_drop_missing_obs = data[np.logical_not(obs_w_missing), :]
# This reduces us to 15 observations of the original 29.

# Another approach is to first investigate where the missing values are.
# A quick way to do this is to visually look at the missing_idx:
plt.title('Visual inspection of missing values')
plt.imshow(missing_idx)
plt.ylabel('Observations'); plt.xlabel('Attributes');
plt.show()

# From such a plot, we can see that the issue is the third column, the
# displacement attribute. This can be confirmed by e.g. doing:
#np.sum(missing_idx, 0)
# which shows that 12 observations are missing a value in the third column. 
# Therefore, another way to move forward is to disregard displacement 
# (for now) and remove the attribute. We then remove the few
# remaining observations with missing values:
cols = np.ones((data.shape[1]), dtype=bool)
cols[2] = 0
data_wo_displacement = data[:, cols] 
obs_w_missing_wo_displacement = np.sum(np.isnan(data_wo_displacement),1)>0
data_drop_disp_then_missing = data[np.logical_not(obs_w_missing_wo_displacement), :]
# Now we have kept all but two of the observations. This however, doesn't
# necesarrily mean that this approach is superior to the previous one,
# since we have now also lost any and all information that we could have
# gotten from the displacement attribute. 

# One could impute the missing values - "guess them", in some
# sense - while trying to minimize the impact of the guess.
# A simply way of imputing them is to replace the missing values
# with the median of the attribute. We would have to do this for the
# missing values for attributes 1 and 3:
data_imputed = data.copy();
for att in [0, 2]:
     # We use nanmedian to ignore the nan values
    impute_val = np.nanmedian(data[:, att])
    idx = missing_idx[:, att]
    data_imputed[idx, att] = impute_val;
