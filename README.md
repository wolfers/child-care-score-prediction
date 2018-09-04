# Provider Ratings

#### The data used for this model is confidential. Any data provided in this repo or used as examples is all fake data and does not reflect the actual data.

## Child Care Aware Washington background info

Child Care Aware provided me confidential data for provider CCQB(baseline) ratings, and their official ratings. Child Care Aware Washington sends out a representative to give a provider a baseline rating and advice on how to proceed. Then an official rater from University of Washington goes ut and gives the provider their official rating.

There are two different tests, CLASS and ERS. Each measure different things. In each of those tests there are 3 subtests for the type of classroom they are rating. 

## What this project hopes to achieve

This project set out to create a model that predicts individual score for the official ratings based on the ccqb and then emasure the variance between baseline and predicted. It then recommends some areas that have high variance as a suggestion for providers to focus efforts in those areas.

## EDA and feature engineering

#### initial information

The data was fairly messy and for this project to work I needed to clean up a lot of things. There are a lot of baselines for some providers, and there are several official ratings that took place before any baselines. 

The CCQB data contained some basic provider information, the scores, and the notes and feedback form the person rating the provider.

In the official ratings data, there is a lot of columns that represent the same thing but are named differently and contain different data.

Official ratings data contained a little provider information and all of the ratings columns.

#### feature engineering

There was a lot of feature engineering done for this model. I started by dropping any baseline ratings that were too far away from the first baseline. Any that were farther are likely just practice and aren't very relevant to this problem. I seperated the baselines by their subtype and averaged together any of the same type. I filled NaNs with column averages and created dummy columns for the missing NaNs, the location, the subtest, and the type of care. I ended up dropping any of the columns that contained notes and feedback.

In the official ratings data I dropped any ratings that were taken before the first baseline for that provider. This data was difficult to work with. I needed to map the columns from the CCQB data to the columns from the official ratings and then combine any that were duplicates and rename them to match up. I ended up filling the zeroes with NaNs for this part. I averaged the columns that were mapped to the same column in the CCQB data. In the CLASS data, the columns were a bit more confusing. There were a few main catergories and several subcatergories under those main few. The CCQB CLASS data had several columns that didn't match but a few that matched the main catergories. I combined all the subcatergories into the main ones and mapped those with the CCQB. This left the CLASS official ratings fairly small column wise.

To get the models for prediction I created a DataFrame for each column in the official ratings data, setting the official rating column as the target. I dropped anycolumns that contained NaNs in the target data.

#### A little bit about assumptions I made during feature engineering

I assumed that the CCQBs far away from the first ones were not relevant. In this model I decided to drop the columns containing text, but they could be relevant and contain good information for a model.

In the official ratings data I made a lot of assumptions. I assumed that columns with similar names or some typos were all the same set of data and could be combined. The zeroes in the ratings might have been significant, I didnt' see anything that indicated that so I turned them to NaNs to be dropped. I dropped any official rating data that was before a ccqb rating, but there may be other things that are possible with that data. espeiclaly if there is an offical rating, a ccqb, and an offical rating afterwords. Those could be weighed differently.

## The Models

There were a few models I attempted to try, I tested Ridge regression, Gradiant Boosting, and Random Forest. I ended up going with Random Forest as it was the fastest to train and gave good results.

For testing I did a lot of cross validation. Wtih the parameters I settled on the models were getting around 1-4 mean squared error. A few of them had around 8, but for the most part they were in the smaller range.

Overall there were 40 models for the ERS data and 4 models for the CLASS data.

## performance

The model is difficult to test, as it's only predicting areas to focus on for providers.

### TO DO

website information

furute plans

things to do better

add link to slides

create README for testing directiory(it's old stuff)

create README in app directory
