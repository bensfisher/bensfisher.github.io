Title: Random Forests in R
Date: 02-25-2015

Lately, I've seen more and more social scientists include random forests in their
research. In order to make things easier for those new to random forests, I thought
I'd provide a quick guide to getting started with them in R. Keep in mind that 
this introduction assumes you're already familiar with the basics of how decision trees 
and random forests work.

Let's say you just read some article using random forests, and you're interested
in using them in your own work. One of the first things you'll want to know
is whether there's a software implementation available. If you're an R user like me,
this would involve googling something along the lines of "random forest R package."
The first thing you'll notice is that this search yields not one, but two packages:
randomForest and randomForestSRC. A little more digging will yield another popular
package called party. So, which one to use? In this post, I'm going to walk through
the three main random forest packages: randomForest, randomForestSRC, and party.
I'll cover how each of them works, useful functions in each package, and the
pros and cons of using one over the other two.[1]  

I'll be using data on militarized interstate disputes (MIDs) taken from "Improving
Quantitative Studies of International Conflict" (Beck, King, and Zeng 2000) to 
illustrate the various functions in each package. The data are in a dyad-year
format (i.e. state-state-year) and cover all dyads from 1947 to 1989.The dependent 
variable, the onset of a MID, is a dichotomous variable indicating whether a dispute occurred
between two states in a given year. Independent variables include whether the two
states in a dyad border each other (contig), whether they are allies (ally), their relative military
capabilities, similarity in preferences based on alliance portfolio (sq), polity 
score for country a (dema), polity score for country b (demb), and the natural
log of the number of peaceful years since the last dispute (logpy). There are 
23,529 observations total, and the size of the data is about 378 KB.

## Background ##

randomForest is the original random forest package in R. It's essentially a port
of the original Fortran code by Breiman and Cutler. As a result, this is where
most R users start with randomForest. If you're looking to ONLY do prediction and
are not interested in variable importance measures, you can stop reading here and
just use this one.

The party package was designed to correct for statistical issues with the original
random forest algorithm - specifically bias in variable selection and inflation
of importance scores for correlated variables. It uses conditional inference
trees as the base learner. The main difference between these and decision trees
in a standard random forest model is that conditional inference trees use a significance
test at each split to determine which variable to split the data on.

The final package is randomForestSRC. It uses the same base random forest algorithm
(though not based on the original Fortran code) as randomForest, but can also 
handle right-censored survival data in addition to regression and classification
problems. It also makes several small, but important improvements on the original
package.

## Speed ##

For those interested in purely predictive tasks, the main concern - since accuracy
is largely the same across the three - is going to be speed. In this regard, randomForest 
is far and away the fastest of the 3. Fitting a randomForest object of 1000 trees on the MID data
takes about 50 seconds. Party and randomForestSRC take approximately 15 and 5 minutes
respectively. Fortunately, randomForestSRC implements fairly straightforward 
parallelization based on OpenMP. This actually makes it a better option over the 
other two for researchers working with a large amount of data.

```
ptm = proc.time()
rf.mod = randomForest(fmla, bkz, ntree=1000, mtry=3, importance=TRUE)
rf.time = proc.time() - ptm

ptm = proc.time()
cf.mod = cforest(fmla, bkz, controls=cforest_unbiased(ntree=1000, mtry=3))
cf.time = proc.time() - ptm

ptm = proc.time()
src.mod = rfsrc(fmla, bkz, ntree=1000, mtry=3, importance='permute', var.used='all.trees')
src.time = proc.time() - ptm

packages = c('randomForest', 'party', 'randomForestSRC')
time = c(rf.time, cf.time, src.time)
barplot(time, names=packages, xlab="Package", ylab="Time in seconds", ylim=c(0,900), main="Speed Comparison")
```


## Variable Importance ##




Notes
-----
[1] These aren't the only 3 forest packages in R, but they are the most widely
used in my experience.
